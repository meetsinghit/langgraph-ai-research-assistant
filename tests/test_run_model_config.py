"""Phase 4E.3: per-run model/thinking configuration (WS-B/C).

Offline: scripted models record the ``Settings`` they were built
with, so each test asserts the frozen configuration a run ACTUALLY
used — for fresh runs, concurrently, for legacy rows (NULL config),
for rows whose model was retired afterwards, and across crash+resume.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from agent.config import Settings
from agent.graph import ReviewVerdict, build_research_graph
from app.main import create_app
from tests.app_fakes import (
    TEST_TOKEN,
    make_settings,
    tools_factory,
)
from tests.fakes import ScriptedChatModel
from tests.test_app_api import (
    PAGES,
    SEARCH_RESULTS,
    _wait_status,
    build_app,
    happy_scripts,
)
from tests.test_app_recovery import _crash_first_app

from langchain_core.messages import AIMessage

AUTH = {"Authorization": f"Bearer {TEST_TOKEN}"}


def _scripts() -> dict[str, Any]:
    return happy_scripts()


def _capturing_factory(
    captures: list[Settings],
    models: list[ScriptedChatModel],
    lock: threading.Lock | None = None,
):
    """A model factory that records the Settings object each model was
    constructed with (the frozen per-run view)."""

    def factory(settings: Settings) -> ScriptedChatModel:
        scripts = _scripts()
        model = ScriptedChatModel(
            responses=list(scripts["responses"]),
            structured_outputs=list(scripts["structured_outputs"]),
        )
        if lock:
            with lock:
                captures.append(settings)
                models.append(model)
        else:
            captures.append(settings)
            models.append(model)
        return model

    return factory


def _client(app) -> TestClient:
    client = TestClient(app).__enter__()
    client.app = app  # type: ignore[attr-defined]
    return client


def _new_session(client: TestClient) -> str:
    response = client.post(
        "/api/sessions", json={"title": "Session"}, headers=AUTH
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


# ---------- fresh runs freeze their configuration ----------


def test_run_factory_receives_frozen_model_settings(tmp_path: Path):
    captures: list[Settings] = []
    models: list[ScriptedChatModel] = []
    app = create_app(
        make_settings(tmp_path),
        db_path=tmp_path / "app.db",
        token=TEST_TOKEN,
        checkpointer=None,
        model_factory=_capturing_factory(captures, models),
        tools_factory=tools_factory(
            [], search_results=SEARCH_RESULTS, pages=PAGES
        ),
    )
    client = _client(app)
    session_id = _new_session(client)
    run_id = client.post(
        f"/api/sessions/{session_id}/runs",
        json={
            "question": "question",
            "model_id": "gemini-3.6-flash",
            "thinking_level": "high",
            "research_depth": "deep",
            "evidence_policy": "verified",
        },
        headers=AUTH,
    ).json()["run_id"]
    app.state.manager.wait_idle()
    _wait_status(client, run_id, "completed")

    # the model was built with the run's frozen settings...
    assert captures, "model factory must have been called"
    run_settings = captures[0]
    assert run_settings.llm_model == "gemini-3.6-flash"
    assert run_settings.llm_thinking_level == "high"
    # ...on a per-run COPY, never mutating the shared settings
    assert run_settings is not app.state.settings
    assert app.state.settings.llm_model == "gemini-3.5-flash-lite"

    # history shows what was actually used
    detail = client.get(f"/api/runs/{run_id}", headers=AUTH).json()
    assert detail["model_id"] == "gemini-3.6-flash"
    assert detail["thinking_level"] == "high"
    assert detail["research_depth"] == "deep"
    assert detail["evidence_policy"] == "verified"
    # diagnostics carry the frozen config for display (WS-G)
    config = detail["diagnostics"]["run_config"]
    assert config == {
        "model_id": "gemini-3.6-flash",
        "thinking_level": "high",
        "research_depth": "deep",
        "evidence_policy": "verified",
    }
    app.state.manager.shutdown()


def test_defaults_are_frozen_into_the_row(tmp_path: Path):
    """An old-style request (question only) still records the config
    it resolved to, so history keeps its original settings."""
    app, _, _ = build_app(tmp_path)
    client = _client(app)
    session_id = _new_session(client)
    run_id = client.post(
        f"/api/sessions/{session_id}/runs",
        json={"question": "question"},
        headers=AUTH,
    ).json()["run_id"]
    app.state.manager.wait_idle()
    _wait_status(client, run_id, "completed")
    row = app.state.repo.get_run(run_id)
    assert row["model_id"] == "gemini-3.5-flash-lite"
    assert row["thinking_level"] == "medium"  # historical default
    app.state.manager.shutdown()


# ---------- concurrency: no shared mutable model state ----------


def test_concurrent_runs_use_isolated_model_settings(tmp_path: Path):
    settings = make_settings(tmp_path, max_concurrent_runs=3)
    captures: list[Settings] = []
    graph_settings: list[Settings] = []
    models: list[ScriptedChatModel] = []
    lock = threading.Lock()

    def graph_factory(
        run_settings: Settings, model: Any, tools: Any, metrics: Any = None
    ) -> Any:
        graph_settings.append(run_settings)
        return build_research_graph(
            run_settings, model, tools, metrics=metrics, checkpointer=None
        )

    app = create_app(
        settings,
        db_path=tmp_path / "app.db",
        token=TEST_TOKEN,
        checkpointer=None,
        model_factory=_capturing_factory(captures, models, lock=lock),
        tools_factory=tools_factory(
            [], search_results=SEARCH_RESULTS, pages=PAGES
        ),
        graph_factory=graph_factory,
    )
    client = _client(app)
    session_id = _new_session(client)
    selections = [
        ("gemini-3.6-flash", "minimal"),
        ("gemini-2.5-flash", "high"),
        ("gemini-3.8-flash", "auto"),
    ]
    run_ids = [
        client.post(
            f"/api/sessions/{session_id}/runs",
            json={
                "question": f"question {i}",
                "model_id": model_id,
                "thinking_level": level,
            },
            headers=AUTH,
        ).json()["run_id"]
        for i, (model_id, level) in enumerate(selections)
    ]
    app.state.manager.wait_idle()
    for run_id in run_ids:
        _wait_status(client, run_id, "completed")

    # every run got its OWN settings view with ITS frozen values
    seen = {(s.llm_model, s.llm_thinking_level) for s in captures}
    assert seen == set(selections), seen
    assert all(s is not app.state.settings for s in captures)
    # the graph received the same per-run view as the model
    assert len(graph_settings) == len(captures)
    assert {(s.llm_model, s.llm_thinking_level) for s in graph_settings} == set(
        selections
    )
    # the shared settings object was never mutated
    assert app.state.settings.llm_model == "gemini-3.5-flash-lite"
    assert app.state.settings.llm_thinking_level is None
    # and each historical run reports its own selection
    for run_id, (model_id, level) in zip(run_ids, selections, strict=True):
        row = app.state.repo.get_run(run_id)
        assert row["model_id"] == model_id
        assert row["thinking_level"] == level
    app.state.manager.shutdown()


# ---------- legacy rows (NULL config) ----------


def test_legacy_row_executes_with_configured_defaults(tmp_path: Path):
    captures: list[Settings] = []
    models: list[ScriptedChatModel] = []
    app = create_app(
        make_settings(tmp_path),
        db_path=tmp_path / "app.db",
        token=TEST_TOKEN,
        checkpointer=None,
        model_factory=_capturing_factory(captures, models),
        tools_factory=tools_factory(
            [], search_results=SEARCH_RESULTS, pages=PAGES
        ),
    )
    repo = app.state.repo
    session = repo.create_session("legacy")
    run = repo.create_run(
        session_id=session["id"],
        question="old question",
        conversation_context="",
        idempotency_key=None,
        # pre-4E.3 row: model_id/thinking_level left NULL
    )
    assert run["model_id"] is None and run["thinking_level"] is None

    app.state.manager._execute(run["id"])  # noqa: SLF001
    row = repo.get_run(run["id"])
    assert row["status"] == "completed"
    # legacy path: the configured settings are used untouched
    assert captures and captures[0] is app.state.settings
    # the row is NOT back-filled with a guess afterwards
    assert repo.get_run(run["id"])["model_id"] is None
    # ...and diagnostics honestly report nulls for this old run
    diagnostics = row["metrics_json"]
    assert diagnostics is not None
    config = json.loads(diagnostics)["run_config"]
    assert config["model_id"] is None
    assert config["thinking_level"] is None
    assert config["research_depth"] == "standard"
    app.state.manager.shutdown()


# ---------- retired / invalid frozen configuration fails clearly ----------


def test_retired_model_fails_fast_without_substitution(tmp_path: Path):
    captures: list[Settings] = []
    models: list[ScriptedChatModel] = []
    app = create_app(
        make_settings(tmp_path),
        db_path=tmp_path / "app.db",
        token=TEST_TOKEN,
        checkpointer=None,
        model_factory=_capturing_factory(captures, models),
        tools_factory=tools_factory(
            [], search_results=SEARCH_RESULTS, pages=PAGES
        ),
    )
    repo = app.state.repo
    session = repo.create_session("s")
    # a run created while the model was alive, executed after it was
    # retired (simulated by writing the row directly — the API would
    # reject this selection for NEW runs)
    run = repo.create_run(
        session_id=session["id"],
        question="q",
        conversation_context="",
        idempotency_key=None,
        model_id="gemini-2.0-flash",
        thinking_level="medium",
    )
    app.state.manager._execute(run["id"])  # noqa: SLF001

    row = repo.get_run(run["id"])
    assert row["status"] == "failed"
    assert row["error_category"] == "model_unavailable"
    # NO model was ever constructed and nothing was substituted
    assert captures == [], "must fail before building any model"
    app.state.manager.shutdown()


def test_unsupported_frozen_thinking_level_fails_as_configuration(
    tmp_path: Path,
):
    captures: list[Settings] = []
    models: list[ScriptedChatModel] = []
    app = create_app(
        make_settings(tmp_path),
        db_path=tmp_path / "app.db",
        token=TEST_TOKEN,
        checkpointer=None,
        model_factory=_capturing_factory(captures, models),
        tools_factory=tools_factory(
            [], search_results=SEARCH_RESULTS, pages=PAGES
        ),
    )
    repo = app.state.repo
    session = repo.create_session("s")
    # e.g. the registry changed after this run was persisted
    run = repo.create_run(
        session_id=session["id"],
        question="q",
        conversation_context="",
        idempotency_key=None,
        model_id="gemini-3.8-flash",
        thinking_level="minimal",  # 3.8-flash does not support it
    )
    app.state.manager._execute(run["id"])  # noqa: SLF001

    row = repo.get_run(run["id"])
    assert row["status"] == "failed"
    assert row["error_category"] == "configuration"
    assert captures == [], "must fail before building any model"
    app.state.manager.shutdown()


# ---------- resume restores the ORIGINAL configuration ----------


def test_resume_uses_the_runs_original_model_config(tmp_path: Path):
    # instance 1 crashes mid-synthesize on a run created with an
    # explicit non-default model + thinking level
    app1, client1, _sid, run_id, _tools = _crash_first_app(
        tmp_path,
        payload={
            "question": "question",
            "model_id": "gemini-3.6-flash",
            "thinking_level": "high",
        },
    )
    row = app1.state.repo.get_run(run_id)
    assert row["model_id"] == "gemini-3.6-flash"
    assert row["thinking_level"] == "high"
    client1.__exit__(None, None, None)
    app1.state.manager.shutdown()

    # instance 2: same DB, captures what resume builds the model with
    captures: list[Settings] = []
    models2: list[ScriptedChatModel] = []

    def factory(settings: Settings) -> ScriptedChatModel:
        captures.append(settings)
        return ScriptedChatModel(
            responses=[AIMessage(content="resumed answer")],
            structured_outputs=[ReviewVerdict(status="approved", reason="ok")],
        )

    app2 = create_app(
        make_settings(tmp_path),
        db_path=tmp_path / "app.db",
        token=TEST_TOKEN,
        checkpointer="auto",
        model_factory=factory,
        tools_factory=tools_factory(
            [], search_results=SEARCH_RESULTS, pages=PAGES
        ),
    )
    with TestClient(app2) as client2:
        client2.app = app2  # type: ignore[attr-defined]
        assert app2.state.repo.get_run(run_id)["status"] == "interrupted"
        response = client2.post(f"/api/runs/{run_id}/resume", headers=AUTH)
        assert response.status_code == 202, response.text
        app2.state.manager.wait_idle()
        detail = client2.get(f"/api/runs/{run_id}", headers=AUTH).json()
        assert detail["status"] == "completed"
        assert detail["model_id"] == "gemini-3.6-flash"  # still frozen
        assert detail["thinking_level"] == "high"
        # the resumed execution built the model with the ORIGINAL
        # configuration — not the configured default
        assert captures, "resume must construct a model"
        for settings in captures:
            assert settings.llm_model == "gemini-3.6-flash"
            assert settings.llm_thinking_level == "high"
        # diagnostics stamped the frozen config onto the resumed run
        assert detail["diagnostics"]["run_config"]["model_id"] == (
            "gemini-3.6-flash"
        )
    app2.state.manager.shutdown()
