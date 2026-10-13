"""Sequenced events, SSE replay and reconnect races (Phase 4D.1).

Covers the architecture corrections for stream integrity:

- persisted sequence numbers with durable-before-publish ordering;
- Last-Event-ID replay that neither drops nor duplicates events;
- race-free subscribe-vs-publish (union of replay and live queue is
  exactly the event sequence, every time);
- reconnect during active execution;
- client disconnect detaches the subscriber;
- subscribing/replaying is read-only (no run can be created by a
  stream request).

All offline: real SQLite files in tmp dirs, scripted fakes.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent.graph import ResearchPlan, ReviewVerdict
from app.events import RunEventChannel, _CLOSED
from app.main import create_app
from tests.app_fakes import TEST_TOKEN, make_settings, model_factory, tools_factory
from tests.test_app_api import AUTH, SEARCH_RESULTS, PAGES, _wait_status, _new_session
from tests.test_tool_schema import _tool_call_response

from langchain_core.messages import AIMessage

URL = "https://example.com/doc"


# -- channel-level guarantees ----------------------------------------------


def _channel(tmp_path: Path, run_id: str = "run-1") -> RunEventChannel:
    from app.db import Database
    from app.repository import Repository

    db = Database(tmp_path / "events.db")
    repo = Repository(db)
    # run_events rows need a parent run (FK)
    session = repo.create_session("s")
    repo.create_run(
        session_id=session["id"],
        question="q",
        conversation_context="",
        idempotency_key=None,
    )
    return RunEventChannel(db, repo.list_runs(session["id"])[0]["id"])


def test_events_are_sequenced_persisted_and_replayable(tmp_path: Path):
    channel = _channel(tmp_path)
    for i in range(5):
        channel.publish("stage.started", {"node": "plan" if i else "start"})
    # durable rows survive a NEW channel instance on the same file
    reopened = RunEventChannel(channel._db, channel._run_id)  # noqa: SLF001
    assert reopened.last_seq == 5
    replayed = reopened.replay(-1)
    assert [event.seq for event in replayed] == [1, 2, 3, 4, 5]
    assert all(event.type == "stage.started" for event in replayed)
    # Last-Event-ID semantics: strictly-after, no duplicates
    assert [event.seq for event in reopened.replay(3)] == [4, 5]
    assert [event.seq for event in reopened.replay(99)] == []


def test_subscribe_then_publish_delivers_exactly_once(tmp_path: Path):
    channel = _channel(tmp_path)
    channel.publish("run.started", {})
    subscription = channel.subscribe()
    assert subscription.cursor == 1
    assert not subscription.closed
    channel.publish("stage.started", {"node": "plan"})
    channel.publish("stage.finished", {"node": "plan"})
    received = []
    while not subscription.events.empty():
        received.append(subscription.events.get_nowait())
    # replay covers <= cursor, queue covers > cursor: disjoint, complete
    replayed = channel.replay(-1, upto=subscription.cursor)
    assert [event.seq for event in replayed] == [1]
    assert [event.seq for event in received] == [2, 3]
    assert len({event.seq for event in replayed} | {event.seq for event in received}) == 3


def test_subscribe_on_closed_channel_is_replay_only(tmp_path: Path):
    channel = _channel(tmp_path)
    channel.publish("run.started", {})
    channel.publish("run.finished", {"outcome": "COMPLETED"})
    channel.close()
    subscription = channel.subscribe()
    assert subscription.closed is True
    assert subscription.events.empty()
    assert [event.seq for event in channel.replay(-1, upto=subscription.cursor)] == [1, 2]
    # close() is idempotent
    channel.close()


def test_close_delivers_sentinel_to_live_subscribers(tmp_path: Path):
    channel = _channel(tmp_path)
    subscription = channel.subscribe()
    channel.publish("run.started", {})
    channel.close()
    items = []
    while not subscription.events.empty():
        items.append(subscription.events.get_nowait())
    assert [item.seq for item in items if item is not _CLOSED] == [1]
    assert items[-1] is _CLOSED
    # publishing after close is refused (terminal contract)
    with pytest.raises(RuntimeError):
        channel.publish("run.started", {})


def test_concurrent_publish_and_subscribe_has_no_gap_or_duplicate(tmp_path: Path):
    """Race-free replay-vs-subscribe under real concurrency."""
    channel = _channel(tmp_path)
    total = 200

    def producer():
        for i in range(total):
            channel.publish("stage.started", {"node": "research"})

    thread = threading.Thread(target=producer)
    thread.start()
    # A subscriber attaches mid-stream repeatedly. For EACH
    # subscription the contract is: replay covers seq <= cursor, the
    # queue covers seq > cursor — the two must be disjoint (no
    # duplicates within one client session) and gap-free (nothing
    # between them falls through).
    attaches = 0
    deadline = time.monotonic() + 10
    while thread.is_alive() and time.monotonic() < deadline:
        subscription = channel.subscribe()
        attaches += 1
        replayed = [
            event.seq for event in channel.replay(-1, upto=subscription.cursor)
        ]
        queued: list[int] = []
        if not subscription.closed:
            while not subscription.events.empty():
                event = subscription.events.get_nowait()
                if event is not _CLOSED:
                    queued.append(event.seq)
        channel.detach(subscription)
        overlap = set(replayed) & set(queued)
        assert not overlap, f"replay/queue overlap: {sorted(overlap)[:10]}"
        merged = replayed + queued
        assert merged == sorted(merged), merged
        assert len(set(merged)) == len(merged), merged
    thread.join(timeout=10)
    assert attaches > 0
    # after the producer finishes, durable replay covers everything
    final = [event.seq for event in channel.replay(-1)]
    assert final == list(range(1, total + 1))


# -- HTTP SSE ---------------------------------------------------------------


def _sse_events(body: str) -> list[dict]:
    events = []
    for frame in body.split("\n\n"):
        if not frame.strip():
            continue
        data_lines = [
            line[len("data: ") :]
            for line in frame.split("\n")
            if line.startswith("data: ")
        ]
        if data_lines:
            events.append(json.loads("\n".join(data_lines)))
    return events


def _happy_app(tmp_path: Path):
    models: list = []
    tools: list = []
    app = create_app(
        make_settings(tmp_path),
        db_path=tmp_path / "app.db",
        token=TEST_TOKEN,
        checkpointer=None,
        model_factory=model_factory(
            models,
            responses=[
                _tool_call_response("call_1", "web_search", {"query": "q"}),
                AIMessage(content="done"),
            ],
            structured_outputs=[
                ResearchPlan(queries=["q"]),
                ReviewVerdict(status="approved", reason="ok"),
            ],
        ),
        tools_factory=tools_factory(
            tools, search_results=SEARCH_RESULTS, pages=PAGES
        ),
    )
    return app, models, tools


def test_sse_replays_full_history_of_a_finished_run(tmp_path: Path):
    app, _, _ = _happy_app(tmp_path)
    with TestClient(app) as client:
        session_id = _new_session(client)
        run_id = client.post(
            f"/api/sessions/{session_id}/runs",
            json={"question": "question"},
            headers=AUTH,
        ).json()["run_id"]
        app.state.manager.wait_idle()
        _wait_status(client, run_id, "completed")

        response = client.get(f"/api/runs/{run_id}/events", headers=AUTH)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _sse_events(response.text)
        seqs = [event["seq"] for event in events]
        assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
        types = [event["type"] for event in events]
        assert "run.started" in types
        assert "run.finished" in types
        assert "stage.started" in types
        assert types.index("run.started") < types.index("run.finished")

        # Last-Event-ID: strictly-after replay, never from scratch
        middle = seqs[len(seqs) // 2]
        partial = client.get(
            f"/api/runs/{run_id}/events",
            headers={**AUTH, "Last-Event-ID": str(middle)},
        )
        tail = _sse_events(partial.text)
        assert [event["seq"] for event in tail] == [
            seq for seq in seqs if seq > middle
        ]
        # an up-to-date client gets nothing new (but a clean stream)
        empty = client.get(
            f"/api/runs/{run_id}/events",
            headers={**AUTH, "Last-Event-ID": str(seqs[-1])},
        )
        assert _sse_events(empty.text) == []
    app.state.manager.shutdown()


def test_sse_unknown_run_is_404(tmp_path: Path):
    app, _, _ = _happy_app(tmp_path)
    with TestClient(app) as client:
        response = client.get("/api/runs/nope/events", headers=AUTH)
        assert response.status_code == 404
        assert response.json()["error"]["category"] == "not_found"
    app.state.manager.shutdown()


def test_stream_request_cannot_create_a_run(tmp_path: Path):
    """Read-only guarantee: replay/subscribe side effects = none."""
    app, _, _ = _happy_app(tmp_path)
    with TestClient(app) as client:
        session_id = _new_session(client)
        before = len(client.get(f"/api/sessions/{session_id}", headers=AUTH).json()["runs"])
        client.get("/api/runs/any-id/events", headers=AUTH)
        client.get("/api/runs/other-id/events", headers=AUTH)
        after = client.get(f"/api/sessions/{session_id}", headers=AUTH).json()["runs"]
        assert len(after) == before == 0
    app.state.manager.shutdown()


def test_live_tail_then_clean_end_of_stream(tmp_path: Path):
    """Reconnect-equivalent: attach during execution, drain to close."""
    from types import SimpleNamespace

    from app.api import _event_stream

    models: list = []
    tools: list = []
    entered, release = threading.Event(), threading.Event()

    from tests.test_app_api import _GatedTools

    def gated_tools_factory(_):
        return _GatedTools(
            entered, release, search_results=SEARCH_RESULTS, pages=PAGES
        )

    app = create_app(
        make_settings(tmp_path),
        db_path=tmp_path / "app.db",
        token=TEST_TOKEN,
        checkpointer=None,
        model_factory=model_factory(
            models,
            responses=[
                _tool_call_response("call_1", "web_search", {"query": "q"}),
                AIMessage(content="done"),
            ],
            structured_outputs=[
                ResearchPlan(queries=["q"]),
                ReviewVerdict(status="approved", reason="ok"),
            ],
        ),
        tools_factory=gated_tools_factory,
    )
    with TestClient(app) as client:
        session_id = _new_session(client)
        run_id = client.post(
            f"/api/sessions/{session_id}/runs",
            json={"question": "question"},
            headers=AUTH,
        ).json()["run_id"]
        assert entered.wait(timeout=10)

        request = SimpleNamespace(app=app)
        stream = _event_stream(request, run_id, -1, heartbeat=0.2)
        # attach live while the run is mid-tool-call
        first = next(stream)
        assert "run.started" in first or "status.changed" in first

        release.set()
        app.state.manager.wait_idle()
        _wait_status(client, run_id, "completed")

        frames = [first]
        for frame in stream:
            if frame.startswith(": heartbeat"):
                continue
            frames.append(frame)
        stream.close()

        events = _sse_events("".join(frames))
        seqs = [event["seq"] for event in events]
        assert seqs == sorted(set(seqs)), seqs
        types = [event["type"] for event in events]
        assert "run.finished" in types
        # detached after close: no subscriber leak
        channel = app.state.manager.channel(run_id)
        assert channel._subscribers == []  # noqa: SLF001
        # durable history equals what the live stream delivered
        durable = [event.seq for event in channel.replay(-1)]
        assert durable == seqs
    app.state.manager.shutdown()


def test_client_disconnect_mid_stream_detaches(tmp_path: Path):
    from types import SimpleNamespace

    from app.api import _event_stream

    app, _, _ = _happy_app(tmp_path)
    with TestClient(app) as client:
        session_id = _new_session(client)
        run_id = client.post(
            f"/api/sessions/{session_id}/runs",
            json={"question": "question"},
            headers=AUTH,
        ).json()["run_id"]
        app.state.manager.wait_idle()
        _wait_status(client, run_id, "completed")

        request = SimpleNamespace(app=app)
        stream = _event_stream(request, run_id, -1, heartbeat=0.2)
        next(stream)  # client reads one event, then goes away
        stream.close()  # GeneratorExit -> finally -> detach
        channel = app.state.manager.channel(run_id)
        assert channel._subscribers == []  # noqa: SLF001
        # the same client can reconnect and replay from scratch
        response = client.get(f"/api/runs/{run_id}/events", headers=AUTH)
        assert _sse_events(response.text)
    app.state.manager.shutdown()
