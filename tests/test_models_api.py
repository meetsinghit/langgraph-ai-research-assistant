"""Phase 4E.3: /api/models, selection validation, idempotency with
changed settings, and session rename (WS-A/B/C/F).

Offline: registry-backed catalogue (discovery disabled by default),
scripted models for run creation, synthetic token only.
"""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from tests.app_fakes import (
    TEST_TOKEN,
    make_settings,
    model_factory,
    tools_factory,
)
from tests.test_app_api import (
    PAGES,
    SEARCH_RESULTS,
    _new_session,
    happy_scripts,
)

AUTH = {"Authorization": f"Bearer {TEST_TOKEN}"}


def _client(tmp_path: Path, **kwargs) -> TestClient:
    settings = kwargs.pop("settings", None) or make_settings(tmp_path)
    scripts = happy_scripts()
    app = create_app(
        settings,
        db_path=tmp_path / "app.db",
        token=TEST_TOKEN,
        checkpointer=None,
        model_factory=model_factory(
            [],
            responses=scripts["responses"],
            structured_outputs=scripts["structured_outputs"],
        ),
        tools_factory=tools_factory(
            [], search_results=SEARCH_RESULTS, pages=PAGES
        ),
        **kwargs,
    )
    client = TestClient(app).__enter__()
    client.app = app  # type: ignore[attr-defined]
    return client


# ---------- GET /api/models ----------


def test_models_endpoint_requires_auth(tmp_path: Path):
    client = _client(tmp_path)
    assert client.get("/api/models").status_code == 401
    assert client.get("/api/models", headers=AUTH).status_code == 200
    client.__exit__(None, None, None)
    client.app.state.manager.shutdown()


def test_models_endpoint_shape_and_selector_rules(tmp_path: Path):
    settings = make_settings(tmp_path, gemini_api_key="fake-key-abc123")
    client = _client(tmp_path, settings=settings)
    response = client.get("/api/models", headers=AUTH)
    assert response.status_code == 200, response.text
    body = response.json()

    # defaults mirror the configuration
    assert body["default_model_id"] == "gemini-3.8-flash"
    assert body["default_thinking_level"] == "medium"
    # discovery is opt-in: reported as disabled, nothing was fetched
    assert body["discovery"]["status"] == "disabled"
    assert body["discovery"]["checked_at"] is None
    assert body["discovery"]["error_category"] is None

    models = body["models"]
    by_id = {m["model_id"]: m for m in models}
    # the configured default is listed first
    assert models[0]["model_id"] == "gemini-3.8-flash"
    # every free-tier-eligible, non-retired model is selectable...
    default = by_id["gemini-3.8-flash"]
    assert default["selectable"] is True
    assert default["free_tier"] == "eligible"
    assert default["thinking"] == {
        "control": "level",
        "levels": ["auto", "low", "medium", "high"],
        "default": "medium",
    }
    assert default["verified_on"] == "2026-10-14"
    assert default["docs_url"].startswith("https://ai.google.dev/")
    # ...and nothing else is, always with a stated reason
    for model in models:
        if not model["selectable"]:
            assert model["exclusion_reason"], model
    assert by_id["gemini-4-argon"]["selectable"] is False
    assert by_id["gemini-4-argon"]["exclusion_reason"] == "paid_tier"
    assert by_id["gemini-3.7-flash"]["exclusion_reason"] == "eligibility_unknown"
    assert by_id["gemini-2.0-flash"]["exclusion_reason"] == "shutdown"
    # budget-controlled family reports its control type
    assert by_id["gemini-2.5-flash"]["thinking"]["control"] == "budget"
    # discovery-only observations (none here) live apart from models
    assert body["unregistered"] == []
    # no secrets anywhere in the payload
    assert "fake-key-abc123" not in response.text
    client.__exit__(None, None, None)
    client.app.state.manager.shutdown()


# ---------- selection validation before any run exists ----------


def test_invalid_model_selections_are_rejected_with_categories(
    tmp_path: Path,
):
    client = _client(tmp_path)
    session_id = _new_session(client)
    cases = (
        ({"model_id": "gpt-imaginary"}, 400, "unknown_model"),
        ({"model_id": "gemini-4-argon"}, 400, "model_not_selectable"),
        ({"model_id": "gemini-3.7-flash"}, 400, "model_not_selectable"),
        ({"model_id": "gemini-2.0-flash"}, 400, "model_not_selectable"),
        (
            {"model_id": "gemini-3.8-flash", "thinking_level": "minimal"},
            400,
            "unsupported_thinking_level",
        ),
        # Literal-validated vocabulary (422 like depth/evidence)
        ({"thinking_level": "banana"}, 422, None),
        ({"model_id": "a" * 200}, 422, None),
        ({"model_id": "bad\u0000id"}, 422, None),
    )
    for payload, status, category in cases:
        response = client.post(
            f"/api/sessions/{session_id}/runs",
            json={"question": "question", **payload},
            headers=AUTH,
        )
        assert response.status_code == status, (payload, response.text)
        if category:
            assert response.json()["error"]["category"] == category
    # none of the rejections created a run
    summary = client.get(f"/api/sessions/{session_id}", headers=AUTH).json()
    assert summary["runs"] == []
    client.__exit__(None, None, None)
    client.app.state.manager.shutdown()


def test_empty_model_id_falls_back_to_the_configured_default(
    tmp_path: Path,
):
    client = _client(tmp_path)
    session_id = _new_session(client)
    response = client.post(
        f"/api/sessions/{session_id}/runs",
        json={"question": "question", "model_id": "   "},
        headers=AUTH,
    )
    assert response.status_code == 202, response.text
    client.app.state.manager.wait_idle()
    row = client.app.state.repo.get_run(response.json()["run_id"])
    assert row["model_id"] == "gemini-3.8-flash"
    client.__exit__(None, None, None)
    client.app.state.manager.shutdown()


# ---------- idempotency with changed settings ----------


def test_idempotency_conflict_when_settings_change(tmp_path: Path):
    client = _client(tmp_path)
    session_id = _new_session(client)
    headers = {**AUTH, "Idempotency-Key": "key-1"}
    first = client.post(
        f"/api/sessions/{session_id}/runs",
        json={"question": "question", "model_id": "gemini-3.6-flash"},
        headers=headers,
    )
    assert first.status_code == 202
    run_id = first.json()["run_id"]

    # same key + identical payload -> replay of the SAME run
    replay = client.post(
        f"/api/sessions/{session_id}/runs",
        json={"question": "question", "model_id": "gemini-3.6-flash"},
        headers=headers,
    )
    assert replay.status_code == 202
    assert replay.json()["run_id"] == run_id

    # same key + CHANGED settings -> clear conflict, no silent replay
    # and no silent second run
    for changed in (
        {"question": "question", "model_id": "gemini-2.5-flash"},
        {"question": "question", "model_id": "gemini-3.6-flash",
         "thinking_level": "high"},
        {"question": "question", "model_id": "gemini-3.6-flash",
         "research_depth": "deep"},
        {"question": "question", "model_id": "gemini-3.6-flash",
         "evidence_policy": "verified"},
        {"question": "different question",
         "model_id": "gemini-3.6-flash"},
    ):
        response = client.post(
            f"/api/sessions/{session_id}/runs",
            json=changed,
            headers=headers,
        )
        assert response.status_code == 409, (changed, response.text)
        assert response.json()["error"]["category"] == "idempotency_conflict"

    # still exactly one run for that key
    summary = client.get(f"/api/sessions/{session_id}", headers=AUTH).json()
    assert len(summary["runs"]) == 1
    assert summary["runs"][0]["id"] == run_id
    client.__exit__(None, None, None)
    client.app.state.manager.shutdown()


def test_replaying_a_legacy_row_with_default_settings_still_matches(
    tmp_path: Path,
):
    """A pre-4E.3 row (NULL model/thinking) replays cleanly for a
    default-settings retry — old clients are not broken by the new
    conflict rule."""
    client = _client(tmp_path)
    session_id = _new_session(client)
    # simulate a row persisted by an older version
    db = client.app.state.db
    with db.cursor() as cursor:
        cursor.execute(
            "INSERT INTO runs (id, session_id, idempotency_key, question, "
            "conversation_context, status, created_at) "
            "VALUES ('legacy-run', ?, 'legacy-key', 'question', '', "
            "'completed', '2026-01-01')",
            (session_id,),
        )
    response = client.post(
        f"/api/sessions/{session_id}/runs",
        json={"question": "question"},
        headers={**AUTH, "Idempotency-Key": "legacy-key"},
    )
    assert response.status_code == 202, response.text
    assert response.json()["run_id"] == "legacy-run"
    client.__exit__(None, None, None)
    client.app.state.manager.shutdown()


# ---------- session rename (WS-F) ----------


def test_rename_session_roundtrip(tmp_path: Path):
    client = _client(tmp_path)
    session_id = _new_session(client, "Original title")
    before = client.get(f"/api/sessions/{session_id}", headers=AUTH).json()
    response = client.patch(
        f"/api/sessions/{session_id}",
        json={"title": "  Renamed  \u0007 title "},
        headers=AUTH,
    )
    assert response.status_code == 200, response.text
    # control characters are stripped, whitespace trimmed (leading
    # and trailing only — internal runs survive verbatim)
    assert response.json()["title"] == "Renamed   title"
    after = client.get(f"/api/sessions/{session_id}", headers=AUTH).json()
    assert after["session"]["title"] == "Renamed   title"
    # renaming does NOT bump updated_at or reorder anything
    assert after["session"]["updated_at"] == before["session"]["updated_at"]
    assert after["session"]["created_at"] == before["session"]["created_at"]
    # the sidebar list shows the new title
    listed = client.get("/api/sessions", headers=AUTH).json()
    assert [s["title"] for s in listed] == ["Renamed   title"]
    client.__exit__(None, None, None)
    client.app.state.manager.shutdown()


def test_rename_session_validation_and_errors(tmp_path: Path):
    client = _client(tmp_path)
    session_id = _new_session(client)
    for payload, status in (
        ({"title": ""}, 422),
        ({"title": "   "}, 422),
        ({"title": "x" * 121}, 422),
        ({}, 422),  # title is required for an explicit rename
    ):
        response = client.patch(
            f"/api/sessions/{session_id}", json=payload, headers=AUTH
        )
        assert response.status_code == status, (payload, response.text)
    # unknown session -> 404 with a fixed category
    missing = client.patch(
        "/api/sessions/nope", json={"title": "x"}, headers=AUTH
    )
    assert missing.status_code == 404
    assert missing.json()["error"]["category"] == "not_found"
    # rename is authenticated like every other mutating route
    assert (
        client.patch(
            f"/api/sessions/{session_id}", json={"title": "x"}
        ).status_code
        == 401
    )
    client.__exit__(None, None, None)
    client.app.state.manager.shutdown()


def test_rename_does_not_leak_secrets_or_html(tmp_path: Path):
    client = _client(tmp_path)
    session_id = _new_session(client)
    response = client.patch(
        f"/api/sessions/{session_id}",
        json={"title": "Research <script>alert(1)</script>"},
        headers=AUTH,
    )
    # printable characters (including markup) are preserved verbatim
    # as TITLE DATA — rendering is the frontend's job (escaped)
    assert response.status_code == 200
    assert response.json()["title"] == "Research <script>alert(1)</script>"
    client.__exit__(None, None, None)
    client.app.state.manager.shutdown()
