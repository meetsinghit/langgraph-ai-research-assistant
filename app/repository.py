"""Row-level persistence for sessions, runs and sources (4D.1).

Plain ``sqlite3`` statements — the schema is small and explicit, so
an ORM would add a dependency and a mapping layer without buying
anything. All timestamps are ISO-8601 UTC strings.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from app.db import Database

#: Run lifecycle: queued → running → terminal; ``interrupted`` is set
#: ONLY by startup recovery when a process died mid-run.
RUN_STATUS_QUEUED = "queued"
RUN_STATUS_RUNNING = "running"
RUN_STATUS_CANCELLING = "cancelling"
RUN_STATUS_COMPLETED = "completed"
RUN_STATUS_FAILED = "failed"
RUN_STATUS_CANCELLED = "cancelled"
RUN_STATUS_INTERRUPTED = "interrupted"

TERMINAL_STATUSES = frozenset(
    {
        RUN_STATUS_COMPLETED,
        RUN_STATUS_FAILED,
        RUN_STATUS_CANCELLED,
    }
)


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return str(uuid.uuid4())


class Repository:
    def __init__(self, db: Database) -> None:
        self._db = db

    # -- sessions -----------------------------------------------------

    def create_session(self, title: str) -> dict[str, Any]:
        session_id = new_id()
        now = utcnow()
        with self._db.cursor() as cursor:
            cursor.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, title, now, now),
            )
        return {
            "id": session_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
        }

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._db.cursor() as cursor:
            cursor.execute(
                "SELECT s.id, s.title, s.created_at, s.updated_at, "
                "(SELECT COUNT(*) FROM runs r WHERE r.session_id = s.id) "
                "AS run_count FROM sessions s WHERE s.id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
        return dict(row) if row else None

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._db.cursor() as cursor:
            cursor.execute(
                "SELECT s.id, s.title, s.created_at, s.updated_at, "
                "(SELECT COUNT(*) FROM runs r WHERE r.session_id = s.id) "
                "AS run_count FROM sessions s ORDER BY s.updated_at DESC"
            )
            rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def touch_session(self, session_id: str) -> None:
        with self._db.cursor() as cursor:
            cursor.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (utcnow(), session_id),
            )

    def delete_session(self, session_id: str) -> list[str]:
        """Delete a session and its rows (cascade); returns run ids
        whose checkpoints the caller must prune."""
        with self._db.cursor() as cursor:
            cursor.execute(
                "SELECT id FROM runs WHERE session_id = ?", (session_id,)
            )
            run_ids = [str(row["id"]) for row in cursor.fetchall()]
            cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return run_ids

    # -- runs ---------------------------------------------------------

    def create_run(
        self,
        *,
        session_id: str,
        question: str,
        conversation_context: str,
        idempotency_key: str | None,
        parent_run_id: str | None = None,
    ) -> dict[str, Any]:
        run_id = new_id()
        now = utcnow()
        with self._db.cursor() as cursor:
            cursor.execute(
                "INSERT INTO runs (id, session_id, idempotency_key, "
                "parent_run_id, question, conversation_context, status, "
                "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    session_id,
                    idempotency_key,
                    parent_run_id,
                    question,
                    conversation_context,
                    RUN_STATUS_QUEUED,
                    now,
                ),
            )
        return self.get_run(run_id)  # type: ignore[return-value]

    def find_run_by_idempotency_key(
        self, session_id: str, key: str
    ) -> dict[str, Any] | None:
        with self._db.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM runs WHERE session_id = ? AND idempotency_key = ?",
                (session_id, key),
            )
            row = cursor.fetchone()
        return dict(row) if row else None

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._db.cursor() as cursor:
            cursor.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
            row = cursor.fetchone()
        return dict(row) if row else None

    def list_runs(self, session_id: str) -> list[dict[str, Any]]:
        with self._db.cursor() as cursor:
            cursor.execute(
                "SELECT * FROM runs WHERE session_id = ? ORDER BY created_at",
                (session_id,),
            )
            rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def claim_for_execution(self, run_id: str) -> bool:
        """Atomic queued→running CAS: exactly one worker can win."""
        with self._db.cursor() as cursor:
            cursor.execute(
                "UPDATE runs SET status = ?, started_at = ? "
                "WHERE id = ? AND status = ?",
                (RUN_STATUS_RUNNING, utcnow(), run_id, RUN_STATUS_QUEUED),
            )
            return cursor.rowcount == 1

    def mark_cancelling(self, run_id: str) -> bool:
        """running→cancelling CAS (the worker records the terminal
        CANCELLED state when the graph stops)."""
        with self._db.cursor() as cursor:
            cursor.execute(
                "UPDATE runs SET status = ? WHERE id = ? AND status = ?",
                (RUN_STATUS_CANCELLING, run_id, RUN_STATUS_RUNNING),
            )
            return cursor.rowcount == 1

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        answer: str | None = None,
        review_status: str | None = None,
        outcome_status: str | None = None,
        outcome_warnings: list[str] | None = None,
        error_category: str | None = None,
        iterations: int = 0,
        tokens_in: int = 0,
        tokens_out: int = 0,
        estimated_cost_usd: float = 0.0,
        total_execution_seconds: float = 0.0,
        metrics: dict[str, Any] | None = None,
    ) -> None:
        with self._db.cursor() as cursor:
            cursor.execute(
                "UPDATE runs SET status = ?, answer = ?, review_status = ?, "
                "outcome_status = ?, outcome_warnings = ?, error_category = ?, "
                "iterations = ?, tokens_in = ?, tokens_out = ?, "
                "estimated_cost_usd = ?, total_execution_seconds = ?, "
                "metrics_json = ?, finished_at = ? WHERE id = ?",
                (
                    status,
                    answer,
                    review_status,
                    outcome_status,
                    json.dumps(outcome_warnings or []),
                    error_category,
                    iterations,
                    tokens_in,
                    tokens_out,
                    estimated_cost_usd,
                    total_execution_seconds,
                    json.dumps(metrics) if metrics is not None else None,
                    utcnow(),
                    run_id,
                ),
            )

    def mark_interrupted(self, run_ids: list[str]) -> None:
        if not run_ids:
            return
        placeholders = ",".join("?" for _ in run_ids)
        with self._db.cursor() as cursor:
            cursor.execute(
                f"UPDATE runs SET status = ? WHERE status IN (?, ?, ?) "
                f"AND id IN ({placeholders})",
                (
                    RUN_STATUS_INTERRUPTED,
                    RUN_STATUS_QUEUED,
                    RUN_STATUS_RUNNING,
                    RUN_STATUS_CANCELLING,
                    *run_ids,
                ),
            )

    def recover_interrupted_run_ids(self) -> list[str]:
        """Run ids left non-terminal by a previous process."""
        with self._db.cursor() as cursor:
            cursor.execute(
                "SELECT id FROM runs WHERE status IN (?, ?, ?)",
                (
                    RUN_STATUS_QUEUED,
                    RUN_STATUS_RUNNING,
                    RUN_STATUS_CANCELLING,
                ),
            )
            rows = cursor.fetchall()
        return [str(row["id"]) for row in rows]

    def cas_status(
        self, run_id: str, *, from_status: str, to_status: str
    ) -> bool:
        """Generic atomic status transition (e.g. interrupted→queued)."""
        with self._db.cursor() as cursor:
            cursor.execute(
                "UPDATE runs SET status = ? WHERE id = ? AND status = ?",
                (to_status, run_id, from_status),
            )
            return cursor.rowcount == 1

    # -- sources ------------------------------------------------------

    def replace_sources(
        self, run_id: str, sources: list[dict[str, Any]]
    ) -> None:
        with self._db.cursor() as cursor:
            cursor.execute(
                "DELETE FROM run_sources WHERE run_id = ?", (run_id,)
            )
            for position, source in enumerate(sources):
                url = str(source["url"])
                cursor.execute(
                    "INSERT INTO run_sources (run_id, sid, url, host, "
                    "evidence, position) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        run_id,
                        str(source["sid"]),
                        url,
                        urlparse(url).hostname or "",
                        str(source["evidence"]),
                        position,
                    ),
                )

    def list_sources(self, run_id: str) -> list[dict[str, Any]]:
        with self._db.cursor() as cursor:
            cursor.execute(
                "SELECT sid, url, host, evidence, position FROM run_sources "
                "WHERE run_id = ? ORDER BY position",
                (run_id,),
            )
            rows = cursor.fetchall()
        return [dict(row) for row in rows]

    # -- follow-up context ---------------------------------------------

    def recent_turns(
        self, session_id: str, *, exclude_run_id: str, limit: int
    ) -> list[dict[str, Any]]:
        """Most recent ANSWERED runs of a session (newest first)."""
        with self._db.cursor() as cursor:
            cursor.execute(
                "SELECT question, answer FROM runs WHERE session_id = ? "
                "AND id != ? AND answer IS NOT NULL "
                "ORDER BY created_at DESC LIMIT ?",
                (session_id, exclude_run_id, limit),
            )
            rows = cursor.fetchall()
        return [dict(row) for row in rows]
