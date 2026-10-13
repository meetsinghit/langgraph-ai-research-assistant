"""API security invariants (Phase 4D.1).

Canary-based negative tests — synthetic secrets only:

- configuration secrets never appear in any response, event stream
  or on-disk artifact;
- provider exception text (which may quote credentials) never
  reaches an API response or event — category only;
- answer credential redaction still applies on the API path;
- progress events are whitelisted and sanitized (no arguments, no
  page text, no free-form titles/URLs);
- engine-side tool policy (DLP/fetch policy) still fires end-to-end
  through the API;
- system prompts never leak through the API.
"""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from agent.graph import (
    _PLAN_SYSTEM,
    _RESEARCH_SYSTEM,
    _REVIEW_SYSTEM,
    _SYNTHESIZE_SYSTEM,
    ResearchPlan,
    ReviewVerdict,
)
from app.main import create_app
from tests.app_fakes import (
    TEST_TOKEN,
    make_settings,
    model_factory,
    tools_factory,
)
from tests.test_app_api import AUTH, PAGES, SEARCH_RESULTS, _wait_status, _new_session
from tests.test_tool_schema import _tool_call_response

from langchain_core.messages import AIMessage

URL = "https://example.com/doc"
FAKE_KEY = "AIza" + "Sy0123456789abcdefghijklmnopqrstuvw"
CANARY_SETTING_KEY = "CANARY-" + "config-secret-value"


def _app(tmp_path: Path, **kwargs):
    models: list = []
    tools: list = []
    settings = kwargs.pop("settings", None) or make_settings(
        tmp_path, gemini_api_key=CANARY_SETTING_KEY
    )
    scripts = kwargs.pop(
        "scripts",
        {
            "responses": [
                _tool_call_response("call_1", "web_search", {"query": "q"}),
                AIMessage(content="done"),
            ],
            "structured_outputs": [
                ResearchPlan(queries=["q"]),
                ReviewVerdict(status="approved", reason="ok"),
            ],
        },
    )
    app = create_app(
        settings,
        db_path=tmp_path / "app.db",
        token=TEST_TOKEN,
        checkpointer=kwargs.pop("checkpointer", None),
        model_factory=model_factory(models, **scripts),
        tools_factory=tools_factory(
            tools, search_results=SEARCH_RESULTS, pages=PAGES
        ),
        **kwargs,
    )
    return app, models, tools


def _run_to_completion(app, client: TestClient, question: str = "question") -> str:
    session_id = _new_session(client)
    run_id = client.post(
        f"/api/sessions/{session_id}/runs",
        json={"question": question},
        headers=AUTH,
    ).json()["run_id"]
    app.state.manager.wait_idle()
    _wait_status(client, run_id, "completed")
    return run_id


def _db_bytes(tmp_path: Path) -> bytes:
    blobs = []
    for path in sorted(tmp_path.glob("**/*")):
        if path.is_file() and path.suffix in (".db", ".sqlite") or (
            path.is_file() and path.name.endswith(("-wal", "-shm"))
        ):
            blobs.append(path.read_bytes())
    return b"".join(blobs)


def test_config_secrets_never_leave_the_process(tmp_path: Path):
    app, _, _ = _app(tmp_path)
    with TestClient(app) as client:
        run_id = _run_to_completion(app, client)
        surfaces = [
            client.get("/api/health").text,
            client.get("/api/config", headers=AUTH).text,
            client.get("/api/sessions", headers=AUTH).text,
            client.get(f"/api/runs/{run_id}", headers=AUTH).text,
            client.get(f"/api/runs/{run_id}/events", headers=AUTH).text,
        ]
        for body in surfaces:
            assert CANARY_SETTING_KEY not in body
            assert TEST_TOKEN not in body
    # nor in any persisted artifact (app db, WAL, token not stored)
    assert CANARY_SETTING_KEY.encode() not in _db_bytes(tmp_path)
    app.state.manager.shutdown()


def test_answer_credentials_are_redacted_on_the_api_path(tmp_path: Path):
    app, _, _ = _app(
        tmp_path,
        scripts={
            "responses": [
                _tool_call_response("call_1", "web_search", {"query": "q"}),
                AIMessage(content=f"answer with key {FAKE_KEY} inline"),
            ],
            "structured_outputs": [
                ResearchPlan(queries=["q"]),
                ReviewVerdict(status="approved", reason="ok"),
            ],
        },
    )
    with TestClient(app) as client:
        run_id = _run_to_completion(app, client)
        detail = client.get(f"/api/runs/{run_id}", headers=AUTH).json()
        answer = detail["answer"]
        assert FAKE_KEY not in answer
        assert "[REDACTED-CREDENTIAL]" in answer
        events = client.get(f"/api/runs/{run_id}/events", headers=AUTH).text
        assert FAKE_KEY not in events
    assert FAKE_KEY.encode() not in _db_bytes(tmp_path)
    app.state.manager.shutdown()


def test_provider_exception_text_never_leaks(tmp_path: Path):
    """A provider failure quoting a credential must surface as
    category + generic message only — in responses AND events."""
    app, _, _ = _app(
        tmp_path,
        scripts={
            "responses": [AIMessage(content="never used")],
            "structured_outputs": [
                ResearchPlan(queries=["q"]),
                ReviewVerdict(status="approved", reason="ok"),
            ],
            # research's first plain invoke dies with a leaky message
            "raise_on_call": {0: RuntimeError(f"boom {FAKE_KEY} endpoint")},
        },
    )
    with TestClient(app) as client:
        session_id = _new_session(client)
        run_id = client.post(
            f"/api/sessions/{session_id}/runs",
            json={"question": "question"},
            headers=AUTH,
        ).json()["run_id"]
        app.state.manager.wait_idle()
        _wait_status(client, run_id, "failed")

        detail = client.get(f"/api/runs/{run_id}", headers=AUTH).text
        assert FAKE_KEY not in detail
        assert "boom" not in detail
        events = client.get(f"/api/runs/{run_id}/events", headers=AUTH).text
        assert FAKE_KEY not in events
        assert "boom" not in events
        payload = client.get(f"/api/runs/{run_id}", headers=AUTH).json()
        assert payload["status"] == "failed"
        assert payload["error_category"]  # a category, not an exception
        # error message is the fixed contract text
        assert payload.get("answer") is None or "Traceback" not in payload["answer"]
    assert FAKE_KEY.encode() not in _db_bytes(tmp_path)
    app.state.manager.shutdown()


def test_system_prompts_never_appear_in_responses(tmp_path: Path):
    app, _, _ = _app(tmp_path)
    with TestClient(app) as client:
        run_id = _run_to_completion(app, client)
        bodies = [
            client.get("/api/config", headers=AUTH).text,
            client.get(f"/api/runs/{run_id}", headers=AUTH).text,
            client.get(f"/api/runs/{run_id}/events", headers=AUTH).text,
        ]
    for system_prompt in (
        _PLAN_SYSTEM,
        _RESEARCH_SYSTEM,
        _SYNTHESIZE_SYSTEM,
        _REVIEW_SYSTEM,
    ):
        fragment = system_prompt.strip().splitlines()[0]
        for body in bodies:
            assert fragment not in body, fragment
    app.state.manager.shutdown()


def test_progress_events_are_whitelisted_and_sanitized(tmp_path: Path):
    app, _, _ = _app(tmp_path)
    with TestClient(app) as client:
        session_id = _new_session(client)
        run_id = client.post(
            f"/api/sessions/{session_id}/runs",
            json={"question": "question"},
            headers=AUTH,
        ).json()["run_id"]
        # sanitizer behavior, pushed straight at the pipeline
        manager = app.state.manager
        manager._publish(  # noqa: SLF001
            run_id,
            "source.discovered",
            {
                "url": "javascript:alert(1)",  # scheme rejected
                "title": "ok\x00\x1b[31mtitle\u202e" + "x" * 500,
                "page_text": "SMUGGLED-PAGE-CONTENT",  # not whitelisted
            },
        )
        manager._publish(  # noqa: SLF001  — unknown type dropped
            run_id, "totally.unknown", {"foo": "bar"}
        )
        manager._publish(  # noqa: SLF001  — bad node/tool dropped
            run_id, "stage.started", {"node": "../../etc", "tool": "exec"}
        )
        app.state.manager.wait_idle()
        _wait_status(client, run_id, "completed")

        events = [
            event
            for event in app.state.manager.channel(run_id).replay(-1)
        ]
        body = str([event.as_dict() for event in events])
        assert "SMUGGLED-PAGE-CONTENT" not in body
        assert "javascript:" not in body
        assert "totally.unknown" not in body
        assert "../../etc" not in body
        for event in events:
            assert set(event.payload) <= {
                "node",
                "tool",
                "ok",
                "error_category",
                "latency_ms",
                "url",
                "title",
                "outcome",
                "review_status",
                "status",
            }, (event.type, event.payload)
            if event.type == "source.discovered":
                assert set(event.payload) <= {"url", "title"}
            if "title" in event.payload:
                assert len(event.payload["title"]) <= 200
                assert "\x00" not in event.payload["title"]
        # page text from the tool never reaches the stream either
        stream = client.get(f"/api/runs/{run_id}/events", headers=AUTH).text
        assert "Fetched page content." not in stream
        assert "SMUGGLED-PAGE-CONTENT" not in stream
    app.state.manager.shutdown()


def test_engine_tool_policy_still_fires_through_the_api(tmp_path: Path):
    """A fetch of a URL the research never discovered is refused by
    the unchanged Phase 4C policy, visible only as a category count
    in the content-free diagnostics."""
    app, _, _ = _app(
        tmp_path,
        scripts={
            "responses": [
                _tool_call_response(
                    "call_1",
                    "fetch_page",
                    {"url": "https://evil-not-discovered.example/data"},
                ),
                AIMessage(content="done"),
            ],
            "structured_outputs": [
                ResearchPlan(queries=["q"]),
                ReviewVerdict(status="approved", reason="ok"),
            ],
        },
    )
    with TestClient(app) as client:
        run_id = _run_to_completion(app, client)
        detail = client.get(f"/api/runs/{run_id}", headers=AUTH).json()
        diagnostics = detail["diagnostics"]
        refusals = diagnostics["security"]["tool_refusals"]
        assert sum(refusals.values()) >= 1, refusals
        # the refused target never becomes a source
        assert all(
            "evil-not-discovered" not in source["url"]
            for source in detail["sources"]
        )
        stream = client.get(f"/api/runs/{run_id}/events", headers=AUTH).text
        assert "evil-not-discovered.example" not in stream
    app.state.manager.shutdown()


def test_question_control_characters_are_stripped(tmp_path: Path):
    app, _, _ = _app(tmp_path)
    with TestClient(app) as client:
        session_id = _new_session(client)
        response = client.post(
            f"/api/sessions/{session_id}/runs",
            json={"question": "ok\x00\x1b[31mquestion\u202e"},
            headers=AUTH,
        )
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        row = app.state.repo.get_run(run_id)
        assert "\x00" not in row["question"]
        assert "\x1b" not in row["question"]
        assert "\u202e" not in row["question"]
        assert "ok" in row["question"] and "question" in row["question"]
    app.state.manager.shutdown()
