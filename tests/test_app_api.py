"""API contract, auth, idempotency and run lifecycle (Phase 4D.1).

Offline: scripted model + in-memory tools via dependency injection;
per-test temp data dirs; synthetic token only.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent.graph import ResearchPlan, ReviewVerdict
from app.main import create_app
from tests.app_fakes import (
    TEST_TOKEN,
    make_settings,
    model_factory,
    tools_factory,
)
from tests.fakes import FakeTools
from tests.test_tool_schema import _tool_call_response

from langchain_core.messages import AIMessage

URL = "https://example.com/doc"
AUTH = {"Authorization": f"Bearer {TEST_TOKEN}"}

SEARCH_RESULTS = [{"title": "Doc", "snippet": "S", "url": URL}]
PAGES = {URL: "Fetched page content."}


def happy_scripts() -> dict:
    return {
        "responses": [
            _tool_call_response("call_1", "web_search", {"query": "q"}),
            AIMessage(content="done"),
        ],
        "structured_outputs": [
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="ok"),
        ],
    }


def build_app(tmp_path: Path, **kwargs):
    models: list = []
    tools: list = []
    scripts = kwargs.pop("scripts", None) or happy_scripts()
    settings = kwargs.pop("settings", None) or make_settings(tmp_path)
    app = create_app(
        settings,
        db_path=tmp_path / "app.db",
        token=TEST_TOKEN,
        checkpointer=kwargs.pop("checkpointer", None),
        model_factory=model_factory(models, **scripts),
        tools_factory=tools_factory(
            tools,
            search_results=SEARCH_RESULTS,
            pages=PAGES,
        ),
        **kwargs,
    )
    return app, models, tools


@pytest.fixture()
def client(tmp_path: Path):
    app, models, tools = build_app(tmp_path)
    with TestClient(app) as test_client:
        test_client.app = app  # expose for manager access in tests
        yield test_client
    app.state.manager.shutdown()


def _new_session(client: TestClient, title: str = "Session") -> str:
    response = client.post("/api/sessions", json={"title": title}, headers=AUTH)
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _wait_status(client: TestClient, run_id: str, status: str, timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = client.app.state.repo.get_run(run_id)
        if row and row["status"] == status:
            return row
        time.sleep(0.02)
    row = client.app.state.repo.get_run(run_id)
    raise AssertionError(f"run never reached {status}; last status {row and row['status']}")


# -- auth -------------------------------------------------------------------


def test_health_needs_no_token(tmp_path: Path):
    app, _, _ = build_app(tmp_path)
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}
    app.state.manager.shutdown()


def test_every_api_route_requires_a_bearer_token(tmp_path: Path):
    app, _, _ = build_app(tmp_path)
    with TestClient(app) as client:
        assert client.get("/api/sessions").status_code == 401
        assert (
            client.get(
                "/api/sessions", headers={"Authorization": "Bearer wrong"}
            ).status_code
            == 401
        )
        assert (
            client.get("/api/sessions", headers=AUTH).status_code == 200
        )
        # SSE endpoints are covered by the same rule
        assert (
            client.get("/api/runs/whatever/events").status_code == 401
        )
    app.state.manager.shutdown()


def test_config_endpoint_never_exposes_secrets(tmp_path: Path):
    settings = make_settings(
        tmp_path, gemini_api_key="CANARY-" + "secret-key-value"
    )
    app, _, _ = build_app(tmp_path, settings=settings)
    with TestClient(app) as client:
        response = client.get("/api/config", headers=AUTH)
    assert response.status_code == 200
    body = response.text
    assert "CANARY" not in body
    assert TEST_TOKEN not in body
    payload = response.json()
    assert payload["provider"] == settings.llm_provider
    assert payload["model"] == settings.llm_model
    app.state.manager.shutdown()


# -- sessions ---------------------------------------------------------------


def test_session_crud_roundtrip(tmp_path: Path):
    app, _, _ = build_app(tmp_path)
    with TestClient(app) as client:
        created = _new_session(client, "My research")
        listed = client.get("/api/sessions", headers=AUTH).json()
        assert [s["id"] for s in listed] == [created]
        assert listed[0]["run_count"] == 0
        detail = client.get(f"/api/sessions/{created}", headers=AUTH).json()
        assert detail["session"]["title"] == "My research"
        assert detail["runs"] == []
        assert (
            client.delete(f"/api/sessions/{created}", headers=AUTH).status_code
            == 204
        )
        assert (
            client.get(f"/api/sessions/{created}", headers=AUTH).status_code
            == 404
        )
        assert (
            client.delete(f"/api/sessions/{created}", headers=AUTH).status_code
            == 404
        )
    app.state.manager.shutdown()


@pytest.mark.parametrize(
    "question", ["", "   ", "x" * 2001]
)
def test_run_question_is_validated(tmp_path: Path, question: str):
    app, _, _ = build_app(tmp_path)
    with TestClient(app) as client:
        session_id = _new_session(client)
        response = client.post(
            f"/api/sessions/{session_id}/runs",
            json={"question": question},
            headers=AUTH,
        )
        assert response.status_code == 422
    app.state.manager.shutdown()


def test_oversized_request_body_is_rejected(tmp_path: Path):
    app, _, _ = build_app(tmp_path)
    with TestClient(app) as client:
        session_id = _new_session(client)
        response = client.post(
            f"/api/sessions/{session_id}/runs",
            content=b"x" * (70 * 1024),
            headers={**AUTH, "Content-Type": "application/json"},
        )
        assert response.status_code == 413
    app.state.manager.shutdown()


# -- run lifecycle ----------------------------------------------------------


def test_run_lifecycle_happy_path(tmp_path: Path):
    app, models, tools = build_app(tmp_path)
    with TestClient(app) as client:
        session_id = _new_session(client)
        response = client.post(
            f"/api/sessions/{session_id}/runs",
            json={"question": "question"},
            headers={**AUTH, "Idempotency-Key": "key-1"},
        )
        assert response.status_code == 202, response.text
        run_id = response.json()["run_id"]
        assert response.json()["events_url"].endswith("/events")

        app.state.manager.wait_idle()
        _wait_status(client, run_id, "completed")

        detail = client.get(f"/api/runs/{run_id}", headers=AUTH).json()
        assert detail["status"] == "completed"
        assert detail["answer"] == "done"
        assert detail["review_status"] == "approved"
        assert detail["outcome_status"] in (
            "COMPLETED",
            "COMPLETED_WITH_WARNINGS",
        )
        assert detail["sources"] == [
            {
                "sid": "S1",
                "url": URL,
                "host": "example.com",
                "evidence": "retrieved",
                "position": 0,
            }
        ]
        diagnostics = detail["diagnostics"]
        assert diagnostics["outcome"]["status"] == detail["outcome_status"]
        assert diagnostics["totals"]["model_calls"] >= 3
        assert detail["error_category"] is None
        # the session summary reflects the run
        summary = client.get(f"/api/sessions/{session_id}", headers=AUTH).json()
        assert summary["runs"][0]["id"] == run_id
        assert summary["runs"][0]["has_answer"] is True
        assert summary["session"]["run_count"] == 1
    app.state.manager.shutdown()


def test_run_creation_is_idempotent_per_key(tmp_path: Path):
    app, _, _ = build_app(tmp_path)
    with TestClient(app) as client:
        session_id = _new_session(client)
        first = client.post(
            f"/api/sessions/{session_id}/runs",
            json={"question": "question"},
            headers={**AUTH, "Idempotency-Key": "same"},
        )
        second = client.post(
            f"/api/sessions/{session_id}/runs",
            json={"question": "question"},
            headers={**AUTH, "Idempotency-Key": "same"},
        )
        assert first.status_code == second.status_code == 202
        assert first.json()["run_id"] == second.json()["run_id"]
        app.state.manager.wait_idle()
        summary = client.get(f"/api/sessions/{session_id}", headers=AUTH).json()
        assert len(summary["runs"]) == 1
        # a different key (or none) creates a distinct run
        third = client.post(
            f"/api/sessions/{session_id}/runs",
            json={"question": "question"},
            headers={**AUTH, "Idempotency-Key": "other"},
        )
        assert third.json()["run_id"] != first.json()["run_id"]
    app.state.manager.shutdown()


def test_followup_context_is_server_derived_and_bounded(tmp_path: Path):
    app, models, _ = build_app(tmp_path)
    with TestClient(app) as client:
        session_id = _new_session(client)
        first = client.post(
            f"/api/sessions/{session_id}/runs",
            json={"question": "first question"},
            headers=AUTH,
        )
        app.state.manager.wait_idle()
        _wait_status(client, first.json()["run_id"], "completed")

        second = client.post(
            f"/api/sessions/{session_id}/runs",
            json={
                "question": "follow-up",
                # client-supplied context is IGNORED by the schema
                "conversation_context": "INJECTED-CLIENT-TEXT",
            },
            headers=AUTH,
        )
        app.state.manager.wait_idle()
        _wait_status(client, second.json()["run_id"], "completed")

        row = app.state.repo.get_run(second.json()["run_id"])
        assert "INJECTED-CLIENT-TEXT" not in (row["conversation_context"] or "")
        assert "first question" in row["conversation_context"]
        assert "done" in row["conversation_context"]
        # the plan prompt of the follow-up run saw the context
        plan_prompt = models[-1].seen_structured_messages[0][1].content
        assert "first question" in plan_prompt
        assert "NOT evidence" in plan_prompt
    app.state.manager.shutdown()


def test_sessions_are_isolated_for_context(tmp_path: Path):
    app, _, _ = build_app(tmp_path)
    with TestClient(app) as client:
        session_a = _new_session(client, "A")
        session_b = _new_session(client, "B")
        run_a = client.post(
            f"/api/sessions/{session_a}/runs",
            json={"question": "topic A"},
            headers=AUTH,
        ).json()["run_id"]
        app.state.manager.wait_idle()
        _wait_status(client, run_a, "completed")
        run_b = client.post(
            f"/api/sessions/{session_b}/runs",
            json={"question": "topic B"},
            headers=AUTH,
        ).json()["run_id"]
        app.state.manager.wait_idle()
        _wait_status(client, run_b, "completed")
        row = app.state.repo.get_run(run_b)
        # session B has no prior turns of its own
        assert (row["conversation_context"] or "") == ""
    app.state.manager.shutdown()


def test_delete_session_cascades_its_runs(tmp_path: Path):
    app, _, _ = build_app(tmp_path)
    with TestClient(app) as client:
        session_id = _new_session(client)
        run_id = client.post(
            f"/api/sessions/{session_id}/runs",
            json={"question": "question"},
            headers=AUTH,
        ).json()["run_id"]
        app.state.manager.wait_idle()
        _wait_status(client, run_id, "completed")
        assert client.delete(f"/api/sessions/{session_id}", headers=AUTH).status_code == 204
        assert client.get(f"/api/runs/{run_id}", headers=AUTH).status_code == 404
        assert app.state.repo.list_sources(run_id) == []
    app.state.manager.shutdown()


@pytest.mark.parametrize("endpoint", ["/api/runs/nope", "/api/runs/nope/cancel", "/api/runs/nope/resume"])
def test_unknown_run_is_404(tmp_path: Path, endpoint: str):
    app, _, _ = build_app(tmp_path)
    with TestClient(app) as client:
        method = client.post if endpoint.endswith(("cancel", "resume")) else client.get
        response = method(endpoint, headers=AUTH)
        assert response.status_code == 404
        assert response.json()["error"]["category"] == "not_found"
    app.state.manager.shutdown()


# -- cancellation -----------------------------------------------------------


class _GatedTools(FakeTools):
    """Search blocks until released — models an in-flight tool call."""

    def __init__(self, entered: threading.Event, release: threading.Event, **kwargs):
        super().__init__(**kwargs)
        self.entered = entered
        self.release = release

    def web_search(self, query: str):
        self.entered.set()
        self.release.wait(timeout=10)
        return super().web_search(query)


def test_cancel_running_run_is_cooperative_and_reported(tmp_path: Path):
    models: list = []
    tools: list = []
    entered, release = threading.Event(), threading.Event()

    def gated_tools_factory(_):
        instance = _GatedTools(
            entered,
            release,
            search_results=SEARCH_RESULTS,
            pages=PAGES,
        )
        tools.append(instance)
        return instance

    scripts = happy_scripts()
    settings = make_settings(tmp_path)
    app = create_app(
        settings,
        db_path=tmp_path / "app.db",
        token=TEST_TOKEN,
        checkpointer=None,
        model_factory=model_factory(models, **scripts),
        tools_factory=gated_tools_factory,
    )
    with TestClient(app) as client:
        session_id = _new_session(client)
        run_id = client.post(
            f"/api/sessions/{session_id}/runs",
            json={"question": "question"},
            headers=AUTH,
        ).json()["run_id"]
        assert entered.wait(timeout=10), "research never started a search"
        response = client.post(f"/api/runs/{run_id}/cancel", headers=AUTH)
        assert response.status_code == 200
        assert response.json()["status"] == "cancelling"
        release.set()
        app.state.manager.wait_idle()
        _wait_status(client, run_id, "cancelled")
        detail = client.get(f"/api/runs/{run_id}", headers=AUTH).json()
        assert detail["status"] == "cancelled"
        assert detail["outcome_status"] == "CANCELLED"
        assert "cancelled" in (detail["answer"] or "").lower()
        # a second cancel of a terminal run is a conflict
        again = client.post(f"/api/runs/{run_id}/cancel", headers=AUTH)
        assert again.status_code == 409
    app.state.manager.shutdown()
