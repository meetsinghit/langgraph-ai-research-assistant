"""Crash recovery, capability-gated resume, resume concurrency (4D.1).

These tests are the evidence for the resume capability — a checkpoint
round-trip alone does NOT prove crash recovery, so every test here

- simulates process death with an exception that ESCAPES the graph
  (``BaseException`` subclass — nothing in the node/worker error
  handling catches it, so no terminal cleanup runs, exactly like a
  killed process),
- builds a SEPARATE application instance over the SAME database and
  checkpoint files afterwards, and
- asserts the boundary: completed nodes and tool actions are NOT
  replayed, provenance/security state survives, the resumed run
  gets a FRESH live deadline, and concurrent resumes cannot start a
  second worker.

Offline throughout: scripted fakes, temp dirs, synthetic token.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ConfigDict

from agent.graph import _REVIEW_SYSTEM, ResearchPlan, ReviewVerdict
from app.main import create_app
from tests.app_fakes import TEST_TOKEN, make_settings, model_factory, tools_factory
from tests.fakes import ScriptedChatModel
from tests.test_app_api import AUTH, PAGES, SEARCH_RESULTS, _wait_status, _new_session
from tests.test_tool_schema import _tool_call_response

from langchain_core.messages import AIMessage

URL = "https://example.com/doc"


class SimulatedProcessDeath(BaseException):
    """Escapes every ``except Exception`` in node, retry and worker
    code — the run is left exactly as a killed process would leave
    it: status ``running``, channel open, checkpoint intact."""


class _GatedModel(ScriptedChatModel):
    """Scripted model that blocks inside the model call — lets a test
    inspect the LIVE runtime context mid-resume."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    gate_entered: object | None = None
    gate_release: object | None = None

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.gate_entered is not None:
            self.gate_entered.set()  # type: ignore[union-attr]
            self.gate_release.wait(timeout=10)  # type: ignore[union-attr]
        return super()._generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )


def _wait_until(predicate, timeout: float = 10.0, message: str = "condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.001)
    raise AssertionError(f"timed out waiting for {message}")


def _crash_first_app(tmp_path: Path):
    """Run a study to completion of research, crash in synthesize.

    Returns (app, client, session_id, run_id, tools) — the app has a
    durable checkpoint and a NON-terminal run row.
    """
    models: list = []
    tools: list = []
    app = create_app(
        make_settings(tmp_path),
        db_path=tmp_path / "app.db",
        token=TEST_TOKEN,
        checkpointer="auto",
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
            # plain invoke #0/#1 = research; #2 = synthesize dies
            raise_on_call={2: SimulatedProcessDeath("killed")},
        ),
        tools_factory=tools_factory(
            tools, search_results=SEARCH_RESULTS, pages=PAGES
        ),
    )
    client = TestClient(app).__enter__()
    session_id = _new_session(client)
    run_id = client.post(
        f"/api/sessions/{session_id}/runs",
        json={"question": "question"},
        headers=AUTH,
    ).json()["run_id"]
    manager = app.state.manager
    # the context is registered before the graph runs and removed in
    # the worker's finally — its disappearance (while the row is still
    # 'running') is the crash moment
    _wait_until(
        lambda: run_id in manager._contexts,  # noqa: SLF001
        message="worker to start",
    )
    _wait_until(
        lambda: run_id not in manager._contexts,  # noqa: SLF001
        message="simulated crash",
    )
    return app, client, session_id, run_id, tools


def test_crash_recovery_across_separate_app_instances(tmp_path: Path):
    # ---- instance 1: executes research, dies mid-synthesize --------
    app1, client1, _session_id, run_id, tools1 = _crash_first_app(tmp_path)
    repo1 = app1.state.repo
    row = repo1.get_run(run_id)
    assert row["status"] == "running"  # never finalized — it "died"
    assert tools1[0].search_calls, "research must have run pre-crash"
    assert app1.state.manager._has_checkpoint(run_id)  # noqa: SLF001
    client1.__exit__(None, None, None)
    app1.state.manager.shutdown()

    # ---- instance 2: fresh app, same DB + checkpoint files ---------
    models2: list = []
    tools2: list = []
    entered, release = threading.Event(), threading.Event()
    resumed_model = _GatedModel(
        responses=[AIMessage(content="resumed answer")],
        structured_outputs=[ReviewVerdict(status="approved", reason="ok")],
        gate_entered=entered,
        gate_release=release,
    )
    app2 = create_app(
        make_settings(tmp_path),
        db_path=tmp_path / "app.db",
        token=TEST_TOKEN,
        checkpointer="auto",
        model_factory=lambda _: _record(models2, resumed_model),
        tools_factory=tools_factory(
            tools2, search_results=SEARCH_RESULTS, pages=PAGES
        ),
    )
    with TestClient(app2) as client2:
        # startup recovery: the dead process's row becomes interrupted
        row = app2.state.repo.get_run(run_id)
        assert row["status"] == "interrupted"
        assert app2.state.manager._has_checkpoint(run_id)  # noqa: SLF001

        response = client2.post(f"/api/runs/{run_id}/resume", headers=AUTH)
        assert response.status_code == 202, response.text

        # while the resumed synthesize is in flight, the LIVE context
        # carries a FRESH deadline (full budget, not cancelled)
        assert entered.wait(timeout=10)
        manager = app2.state.manager
        with manager._registry_lock:  # noqa: SLF001
            context = manager._contexts.get(run_id)  # noqa: SLF001
        assert context is not None
        assert context.deadline is not None
        assert not context.deadline.cancelled
        assert context.deadline.remaining() > 0
        assert (
            context.deadline.total_seconds
            == make_settings(tmp_path).max_total_execution_seconds
        )
        release.set()

        app2.state.manager.wait_idle()
        _wait_status(client2, run_id, "completed")

        detail = client2.get(f"/api/runs/{run_id}", headers=AUTH).json()
        assert detail["answer"] == "resumed answer"
        assert detail["review_status"] == "approved"
        assert detail["outcome_status"] in (
            "COMPLETED",
            "COMPLETED_WITH_WARNINGS",
        )
        # provenance from BEFORE the crash survived the restart
        assert [source["url"] for source in detail["sources"]] == [URL]
        assert detail["sources"][0]["evidence"] == "retrieved"
        # completed nodes were NOT replayed: after resume the only
        # structured call is the REVIEW (plan would be a second one
        # with the plan system prompt)
        assert len(models2) == 1
        structured = models2[0].seen_structured_messages
        assert len(structured) == 1, f"unexpected structured calls: {len(structured)}"
        assert structured[0][0].content == _REVIEW_SYSTEM, (
            "plan must not re-run on resume"
        )
        assert all(not t.search_calls and not t.fetch_calls for t in tools2), (
            "research tool actions must not replay on resume"
        )
        # terminal checkpoint pruning (bounded content retention)
        assert not app2.state.manager._has_checkpoint(run_id)  # noqa: SLF001
        # event history is one contiguous sequence across the restart
        events = [
            event.seq
            for event in app2.state.manager.channel(run_id).replay(-1)
        ]
        assert events == list(range(1, len(events) + 1))
        full = app2.state.manager.channel(run_id).replay(-1)
        types = [event.type for event in full]
        # one run.started per WORKER EXECUTION: the original (crashed)
        # one plus the resumed one; exactly one run.finished — only
        # the resumed execution completed the run.
        assert types.count("run.started") == 2, types
        assert types.count("run.finished") == 1
        assert "status.changed" in types

        # WS-D (Phase 4E.2): the persisted diagnostics are reconciled
        # across both segments — the resumed collector itself saw NO
        # tool calls (research ran pre-crash), so without the event-
        # log reconciliation these would read 0 and contradict the
        # progress timeline, which shows the pre-crash search.
        diagnostics = detail["diagnostics"]
        assert diagnostics["segments"] == 2
        counts = diagnostics["counts"]
        assert counts["searches"] == 1, counts  # pre-crash search
        assert counts["failed_searches"] == 0
        # token usage of the crashed segment died with its collector —
        # honestly scoped, never fabricated
        assert diagnostics["token_usage"]["scope"] == "since_resume"


def _record(models: list, model):
    models.append(model)
    return model


def test_startup_recovery_marks_only_non_terminal_runs(tmp_path: Path):
    settings = make_settings(tmp_path)
    app = create_app(
        settings,
        db_path=tmp_path / "app.db",
        token=TEST_TOKEN,
        checkpointer=None,
        recover=False,
    )
    repo = app.state.repo
    session = repo.create_session("s")
    queued = repo.create_run(
        session_id=session["id"], question="q", conversation_context="",
        idempotency_key=None,
    )
    running = repo.create_run(
        session_id=session["id"], question="q", conversation_context="",
        idempotency_key=None,
    )
    assert repo.claim_for_execution(running["id"])
    finished = repo.create_run(
        session_id=session["id"], question="q", conversation_context="",
        idempotency_key=None,
    )
    assert repo.cas_status(
        finished["id"], from_status="queued", to_status="completed"
    )
    app.state.manager.shutdown()

    app2 = create_app(
        settings,
        db_path=tmp_path / "app.db",
        token=TEST_TOKEN,
        checkpointer=None,
        recover=True,
    )
    assert repo.get_run(queued["id"])["status"] == "interrupted"
    assert repo.get_run(running["id"])["status"] == "interrupted"
    # terminal rows are untouched
    assert repo.get_run(finished["id"])["status"] == "completed"
    # recovery is reflected in the event history
    events = app2.state.manager.channel(running["id"]).replay(-1)
    assert events[-1].payload == {"status": "interrupted"}
    app2.state.manager.shutdown()


# -- resume capability matrix (409s) ---------------------------------------


def _fabricated_run(app, status: str = "interrupted"):
    """A run row in ``status`` with NO worker and (usually) no
    checkpoint — for capability/status gating tests."""
    repo = app.state.repo
    session = repo.create_session("s")
    run = repo.create_run(
        session_id=session["id"], question="q", conversation_context="",
        idempotency_key=None,
    )
    assert repo.cas_status(
        run["id"], from_status="queued", to_status=status
    )
    return run["id"]


def _resume_reason(client: TestClient, run_id: str) -> str:
    response = client.post(f"/api/runs/{run_id}/resume", headers=AUTH)
    assert response.status_code == 409, response.text
    return response.json()["error"]["message"]


def test_resume_without_checkpointing_is_unverified(tmp_path: Path):
    app = create_app(
        make_settings(tmp_path),
        db_path=tmp_path / "app.db",
        token=TEST_TOKEN,
        checkpointer=None,  # checkpointing disabled
    )
    with TestClient(app) as client:
        run_id = _fabricated_run(app)
        reason = _resume_reason(client, run_id)
        assert "resume_unverified" in reason
        assert app.state.manager.resume_capability is False
        # config advertises the disabled capability honestly
        config = client.get("/api/config", headers=AUTH).json()
        assert config["resume_enabled"] is False
    app.state.manager.shutdown()


def test_resume_without_a_checkpoint_is_refused(tmp_path: Path):
    app = create_app(
        make_settings(tmp_path),
        db_path=tmp_path / "app.db",
        token=TEST_TOKEN,
        checkpointer="auto",
    )
    with TestClient(app) as client:
        run_id = _fabricated_run(app)  # interrupted, but never checked in
        assert "no_checkpoint" in _resume_reason(client, run_id)
        # still resumable later: the refusal changed nothing
        assert app.state.repo.get_run(run_id)["status"] == "interrupted"
    app.state.manager.shutdown()


@pytest.mark.parametrize("status", ["queued", "completed", "failed", "cancelled"])
def test_resume_of_non_interrupted_runs_is_refused(tmp_path: Path, status: str):
    app = create_app(
        make_settings(tmp_path),
        db_path=tmp_path / "app.db",
        token=TEST_TOKEN,
        checkpointer="auto",
    )
    with TestClient(app) as client:
        run_id = _fabricated_run(app, status=status)
        assert "not_interrupted" in _resume_reason(client, run_id)
        assert app.state.repo.get_run(run_id)["status"] == status
    app.state.manager.shutdown()


def test_resume_of_unknown_run_is_404(tmp_path: Path):
    app = create_app(
        make_settings(tmp_path),
        db_path=tmp_path / "app.db",
        token=TEST_TOKEN,
        checkpointer="auto",
    )
    with TestClient(app) as client:
        response = client.post("/api/runs/missing/resume", headers=AUTH)
        assert response.status_code == 404
    app.state.manager.shutdown()


def test_concurrent_resume_cannot_start_two_workers(tmp_path: Path):
    app1, client1, _session_id, run_id, _tools = _crash_first_app(tmp_path)
    client1.__exit__(None, None, None)
    app1.state.manager.shutdown()

    models: list = []
    tools: list = []
    app2 = create_app(
        make_settings(tmp_path),
        db_path=tmp_path / "app.db",
        token=TEST_TOKEN,
        checkpointer="auto",
        model_factory=model_factory(
            models,
            responses=[AIMessage(content="resumed answer")],
            structured_outputs=[ReviewVerdict(status="approved", reason="ok")],
        ),
        tools_factory=tools_factory(
            tools, search_results=SEARCH_RESULTS, pages=PAGES
        ),
    )
    with TestClient(app2) as client2:
        assert app2.state.repo.get_run(run_id)["status"] == "interrupted"
        results: list[int] = []
        barrier = threading.Barrier(2)

        def attempt():
            barrier.wait(timeout=10)
            response = client2.post(f"/api/runs/{run_id}/resume", headers=AUTH)
            results.append(response.status_code)

        threads = [
            threading.Thread(target=attempt) for _ in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        assert sorted(results) == [202, 409], results

        app2.state.manager.wait_idle()
        _wait_status(client2, run_id, "completed")
        # exactly ONE worker executed the resumed work
        assert len(models) == 1
        assert all(not t.search_calls for t in tools)
        detail = client2.get(f"/api/runs/{run_id}", headers=AUTH).json()
        assert detail["answer"] == "resumed answer"
        events = app2.state.manager.channel(run_id).replay(-1)
        started = [e for e in events if e.type == "run.started"]
        # one per worker execution: original crash + single resume
        assert len(started) == 2, [e.seq for e in started]
    app2.state.manager.shutdown()
