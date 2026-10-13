"""Sequenced, persisted run events with race-free SSE replay (4D.1).

Integrity model (one channel per run = one producer):

- ``publish`` assigns the next sequence number, DURABLY persists the
  row and only then fans it out to live subscribers — all under the
  channel lock. A subscriber can therefore never observe an event
  that a reconnect would fail to replay from the database.
- ``subscribe`` snapshots the current sequence and attaches the
  subscriber queue under the SAME lock. The streamer then replays
  durable rows up to that snapshot (``Last-Event-ID`` excluded) and
  drains the queue for everything published afterwards. Events
  published concurrently with replay land in the queue, so they can
  be neither lost nor duplicated: replay covers ``seq <= cursor``,
  the live queue only ever carries ``seq > cursor``.
- Subscribing is read-only: reconnecting, replaying or double-
  subscribing can never create a research run (runs are created
  only by POST, idempotently).
- A closed channel delivers everything durable up to its final
  sequence and then a sentinel — a client of a finished run always
  receives the complete history and a clean end-of-stream.

Events are content-free by construction: the publisher whitelists
the payload keys per event type and sanitizes values (see
``RunManager._sanitize``). The persisted ``payload`` column is JSON.
"""
from __future__ import annotations

import json
import queue
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.db import Database

#: Sentinel pushed to subscribers when the run's channel closes.
_CLOSED = object()


@dataclass(frozen=True)
class RunEvent:
    """One persisted, sequenced event."""

    seq: int
    ts: str
    type: str
    payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "ts": self.ts,
            "type": self.type,
            "payload": self.payload,
        }


@dataclass
class Subscription:
    """A subscriber's queue plus the cursor it attached at."""

    events: "queue.Queue[Any]"
    cursor: int
    closed: bool


class RunEventChannel:
    """Single-producer sequenced event stream for ONE run."""

    def __init__(self, db: Database, run_id: str) -> None:
        self._db = db
        self._run_id = run_id
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue[Any]] = []
        self._closed = False
        self._last_seq = self._load_last_seq()

    def _load_last_seq(self) -> int:
        with self._db.cursor() as cursor:
            cursor.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM run_events WHERE run_id = ?",
                (self._run_id,),
            )
            row = cursor.fetchone()
        return int(row[0]) if row else 0

    @property
    def last_seq(self) -> int:
        with self._lock:
            return self._last_seq

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed

    def publish(self, event_type: str, payload: dict[str, Any]) -> RunEvent:
        """Persist then fan out; safe from any thread, single producer
        per run by construction (the run's worker owns the channel)."""
        event = RunEvent(
            seq=0,
            ts=datetime.now(timezone.utc).isoformat(),
            type=event_type,
            payload=payload,
        )
        with self._lock:
            if self._closed:
                # Terminal events must precede close(); ignore late
                # publishes rather than breaking the seq contract.
                raise RuntimeError(
                    f"channel for run {self._run_id} is closed"
                )
            seq = self._last_seq + 1
            with self._db.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO run_events (run_id, seq, ts, type, payload) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        self._run_id,
                        seq,
                        event.ts,
                        event.type,
                        json.dumps(event.payload),
                    ),
                )
            self._last_seq = seq
            event = RunEvent(
                seq=seq, ts=event.ts, type=event.type, payload=event.payload
            )
            for subscriber in self._subscribers:
                subscriber.put(event)
            return event

    def close(self) -> None:
        """End the live stream; durable history stays replayable."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for subscriber in self._subscribers:
                subscriber.put(_CLOSED)

    def subscribe(self) -> Subscription:
        """Attach a live queue and snapshot the replay cursor.

        Race-free by construction: the snapshot and the attachment
        happen under the lock that ``publish`` also holds, so no
        event can fall between "durable" and "queued" for this
        subscriber.
        """
        with self._lock:
            events: queue.Queue[Any] = queue.Queue()
            closed = self._closed
            if not closed:
                self._subscribers.append(events)
            return Subscription(
                events=events, cursor=self._last_seq, closed=closed
            )

    def detach(self, subscription: Subscription) -> None:
        with self._lock:
            if subscription.events in self._subscribers:
                self._subscribers.remove(subscription.events)

    def replay(self, after_seq: int, upto: int | None = None) -> list[RunEvent]:
        """Durable events with ``after_seq < seq <= upto`` (ordered)."""
        with self._db.cursor() as cursor:
            if upto is None:
                cursor.execute(
                    "SELECT seq, ts, type, payload FROM run_events "
                    "WHERE run_id = ? AND seq > ? ORDER BY seq",
                    (self._run_id, after_seq),
                )
            else:
                cursor.execute(
                    "SELECT seq, ts, type, payload FROM run_events "
                    "WHERE run_id = ? AND seq > ? AND seq <= ? ORDER BY seq",
                    (self._run_id, after_seq, upto),
                )
            rows = cursor.fetchall()
        return [
            RunEvent(
                seq=int(row["seq"]),
                ts=str(row["ts"]),
                type=str(row["type"]),
                payload=json.loads(row["payload"]),
            )
            for row in rows
        ]
