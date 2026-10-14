"""Phase 4E.3: GET /api/models catalogue + bounded discovery (WS-A).

All discovery is INJECTED (fetch callable) or disabled — the real
``models.list`` SDK path is never executed, so this suite is fully
offline by construction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agent.gemini_models import GEMINI_MODEL_REGISTRY
from app.model_discovery import ModelCatalog
from tests.app_fakes import make_settings


def _settings(tmp_path: Path, **overrides: Any):
    overrides.setdefault("model_discovery_enabled", False)
    return make_settings(tmp_path, **overrides)


def _listing(**extra: Any) -> dict[str, dict[str, Any]]:
    base = {
        "gemini-3.8-flash": {
            "display_name": "Gemini 3.8 Flash",
            "actions": ("generateContent", "countTokens"),
        },
        "gemini-2.0-flash": {
            "display_name": "Gemini 2.0 Flash",
            "actions": ("generateContent",),
        },
        "gemini-4-argon": {
            "display_name": "Gemini 4 Argon",
            "actions": ("generateContent",),
        },
        "imagen-5": {
            "display_name": "Imagen 5",
            "actions": ("generateImages",),
        },
    }
    base.update(extra)
    return base


# ---------- disabled by default: no external calls ----------


def test_discovery_is_disabled_by_default_and_never_fetches(tmp_path: Path):
    calls: list[int] = []

    def fetch() -> dict:
        calls.append(1)
        return _listing()

    settings = _settings(tmp_path)  # no overrides -> disabled
    assert settings.model_discovery_enabled is False
    catalog = ModelCatalog(settings, fetch=fetch)
    snapshot = catalog.snapshot()
    assert calls == []  # zero network attempts
    assert snapshot.discovery.status == "disabled"
    assert snapshot.discovery.checked_at is None
    assert snapshot.discovery.model_count is None
    assert all(m.availability == "not_checked" for m in snapshot.models)


# ---------- registry-backed shape ----------


def test_snapshot_lists_registry_with_selectable_flags(tmp_path: Path):
    catalog = ModelCatalog(_settings(tmp_path), fetch=lambda: pytest.fail("no"))
    snapshot = catalog.snapshot()
    assert snapshot.default_model_id == "gemini-3.8-flash"
    assert snapshot.default_thinking_level == "medium"
    listed = {m.model_id: m for m in snapshot.models}
    assert set(GEMINI_MODEL_REGISTRY) <= set(listed)
    # default sorts first
    assert snapshot.models[0].model_id == "gemini-3.8-flash"
    # free-only selector: only confirmed-eligible, non-retired models
    assert listed["gemini-3.8-flash"].selectable is True
    assert listed["gemini-4-argon"].selectable is False
    assert listed["gemini-4-argon"].exclusion_reason == "paid_tier"
    assert listed["gemini-3.7-flash"].selectable is False
    assert listed["gemini-3.7-flash"].exclusion_reason == "eligibility_unknown"
    assert listed["gemini-2.0-flash"].exclusion_reason == "shutdown"
    # capabilities are carried for the thinking selector
    thinking = listed["gemini-3.8-flash"].thinking
    assert thinking is not None
    assert thinking.control == "level"
    assert thinking.levels == ["auto", "low", "medium", "high"]
    budget = listed["gemini-2.5-flash"].thinking
    assert budget is not None and budget.control == "budget"
    # non-selectable models still explain WHY (transparency)
    for model in snapshot.models:
        if not model.selectable:
            assert model.exclusion_reason


def test_snapshot_contains_no_secrets(tmp_path: Path):
    settings = _settings(tmp_path, gemini_api_key="super-secret-key-xyz")
    snapshot = ModelCatalog(settings, fetch=lambda: {}).snapshot()
    import json

    body = json.dumps(snapshot.model_dump())
    assert "super-secret-key-xyz" not in body
    assert "super-secret-key-xyz" not in body.lower()


# ---------- enabled + mocked fetch ----------


def test_discovery_annotates_availability_only(tmp_path: Path):
    calls: list[int] = []

    def fetch() -> dict:
        calls.append(1)
        return _listing()

    settings = _settings(tmp_path, model_discovery_enabled=True)
    snapshot = ModelCatalog(settings, fetch=fetch).snapshot()
    assert calls == [1]
    assert snapshot.discovery.status == "available"
    assert snapshot.discovery.checked_at is not None
    assert snapshot.discovery.model_count == 4
    listed = {m.model_id: m for m in snapshot.models}
    # present in the listing -> confirmed; absent -> not_listed
    assert listed["gemini-3.8-flash"].availability == "confirmed"
    assert listed["gemini-3.6-flash"].availability == "not_listed"
    # a paid model being listed does NOT make it selectable
    assert listed["gemini-4-argon"].selectable is False
    # registry models missing from the listing stay selectable: the
    # registry is the selection authority, discovery is advisory
    assert listed["gemini-3.6-flash"].selectable is True


def test_discovered_models_never_become_selectable(tmp_path: Path):
    """Listing-only models are reported as observations, with unknown
    eligibility and thinking capabilities, never selectable."""

    def fetch() -> dict:
        return _listing(
            **{
                "gemini-9-concept": {
                    "display_name": "Gemini 9 Concept",
                    "actions": ("generateContent",),
                }
            }
        )

    settings = _settings(tmp_path, model_discovery_enabled=True)
    snapshot = ModelCatalog(settings, fetch=fetch).snapshot()
    by_id = {m.model_id: m for m in snapshot.unregistered}
    assert "gemini-9-concept" in by_id
    entry = by_id["gemini-9-concept"]
    assert entry.selectable is False
    assert entry.free_tier == "unknown"
    assert entry.exclusion_reason == "eligibility_unknown"
    assert entry.thinking is None  # capabilities unknown, not guessed
    assert entry.source == "discovery"
    # non-text models are out of scope entirely
    assert "imagen-5" not in by_id
    # discovery-only observations never appear in the catalogue's
    # selectable list, and registry entries keep their vetted flags
    # regardless of what the listing showed (advisory only)
    assert "gemini-9-concept" not in {m.model_id for m in snapshot.models}
    listed = {m.model_id: m for m in snapshot.models}
    assert listed["gemini-4-argon"].selectable is False
    assert listed["gemini-3.8-flash"].selectable is True


def test_ttl_cache_bounds_discovery_calls(tmp_path: Path):
    calls: list[int] = []

    def fetch() -> dict:
        calls.append(1)
        return _listing()

    settings = _settings(
        tmp_path, model_discovery_enabled=True, model_discovery_ttl_seconds=3600
    )
    catalog = ModelCatalog(settings, fetch=fetch)
    catalog.snapshot()
    catalog.snapshot()
    catalog.snapshot()
    assert len(calls) == 1  # one call per TTL, not per page load


def test_discovery_failures_report_fixed_categories(tmp_path: Path):
    class FakeAPIError(Exception):
        def __init__(self) -> None:
            super().__init__(
                "raw provider detail with secret-looking text " "abc123"
            )
            self.code = 429

    def fetch() -> dict:
        raise FakeAPIError()

    settings = _settings(tmp_path, model_discovery_enabled=True)
    catalog = ModelCatalog(settings, fetch=fetch)
    snapshot = catalog.snapshot()
    assert snapshot.discovery.status == "unavailable"
    assert snapshot.discovery.error_category == "rate_limit"
    # raw provider text never leaks
    import json

    body = json.dumps(snapshot.model_dump())
    assert "secret-looking" not in body and "abc123" not in body
    # the registry is still fully served
    assert any(m.model_id == "gemini-3.8-flash" for m in snapshot.models)


def test_discovery_error_categories(tmp_path: Path):
    from app.model_discovery import _categorize_discovery_error

    class Timeout(Exception):
        pass

    class Err(Exception):
        pass

    err = Err("x")
    err.code = 403  # type: ignore[attr-defined]
    assert _categorize_discovery_error(Timeout("t")) == "timeout"
    assert _categorize_discovery_error(err) == "auth"
    assert _categorize_discovery_error(ValueError("bad")) == "provider"
    assert _categorize_discovery_error(ConnectionError()) == "network"


def test_failure_backs_off_without_bounded_retry_storm(tmp_path: Path):
    calls: list[int] = []

    def fetch() -> dict:
        calls.append(1)
        raise ValueError("nope")

    settings = _settings(
        tmp_path,
        model_discovery_enabled=True,
        model_discovery_backoff_seconds=3600,
        model_discovery_ttl_seconds=3600,
    )
    catalog = ModelCatalog(settings, fetch=fetch)
    catalog.snapshot()
    catalog.snapshot()
    catalog.snapshot()
    assert len(calls) == 1  # one failure, then backoff


def test_fetch_result_is_capped(tmp_path: Path):
    """A hostile/huge listing cannot balloon memory or the response."""
    from app.model_discovery import MAX_DISCOVERY_MODELS, MAX_UNREGISTERED

    def fetch() -> dict:
        return {
            f"gemini-x-{i:04d}": {
                "display_name": f"X {i}",
                "actions": ("generateContent",),
            }
            for i in range(MAX_DISCOVERY_MODELS + 100)
        }

    settings = _settings(tmp_path, model_discovery_enabled=True)
    snapshot = ModelCatalog(settings, fetch=fetch).snapshot()
    assert len(snapshot.unregistered) <= MAX_UNREGISTERED
    assert snapshot.discovery.model_count is not None
    assert snapshot.discovery.model_count <= MAX_DISCOVERY_MODELS
