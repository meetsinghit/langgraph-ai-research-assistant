"""Phase 4E.3: Gemini model capability + free-tier registry (WS-A/B).

Fully offline: the registry is maintained data, validation is pure
logic and thinking mapping never constructs a client — no discovery
call, no API key, no network.
"""
from __future__ import annotations

import pytest

from agent.gemini_models import (
    BUDGET_BY_LEVEL,
    GEMINI_MODEL_REGISTRY,
    THINKING_LEVELS,
    ModelSelectionError,
    ModelUnavailableError,
    assert_runnable,
    control_by_family,
    family_of,
    is_selectable,
    resolve_run_model_config,
    selectable_models,
    spec_for,
    thinking_kwargs,
)


# ---------- registry integrity ----------


def test_registry_ids_are_unique_and_clean():
    ids = list(GEMINI_MODEL_REGISTRY)
    assert len(ids) == len(set(ids))
    for model_id in ids:
        assert model_id == model_id.strip()
        assert model_id.startswith("gemini-")
        assert "/" not in model_id


def test_registry_entries_carry_documented_provenance():
    for spec in GEMINI_MODEL_REGISTRY.values():
        assert spec.docs_url.startswith("https://ai.google.dev/")
        assert spec.verified_on == "2026-10-14"
        assert spec.display_name
        assert spec.notes  # eligibility/capability caveats are stated
        assert spec.thinking_levels  # at least "auto"
        for level in spec.thinking_levels:
            assert level in THINKING_LEVELS


def test_default_model_is_selectable_and_free():
    spec = spec_for("gemini-3.8-flash")
    assert spec is not None
    assert is_selectable(spec)
    assert spec.free_tier == "eligible"
    # minimal is NOT supported on 3.8-flash (official docs: 400)
    assert "minimal" not in spec.thinking_levels
    assert "low" in spec.thinking_levels and "medium" in spec.thinking_levels


def test_paid_and_unknown_models_are_never_selectable():
    # explicitly not on the free tier
    argon = spec_for("gemini-4-argon")
    assert argon is not None and argon.free_tier == "paid_only"
    assert not is_selectable(argon)
    # free-tier eligibility unconfirmed -> excluded (never assumed free)
    for model_id in ("gemini-3.7-flash", "gemini-3.5-flash"):
        spec = spec_for(model_id)
        assert spec is not None and spec.free_tier == "unknown"
        assert not is_selectable(spec)


def test_retired_models_exist_as_tombstones():
    for model_id in ("gemini-2.0-flash", "gemini-2.0-flash-lite"):
        spec = spec_for(model_id)
        assert spec is not None
        assert spec.status == "shutdown"
        assert spec.retired_on == "2026-06-01"
        assert not is_selectable(spec)


def test_free_only_selector_excludes_everything_unverified():
    selectable_ids = {s.model_id for s in selectable_models()}
    for spec in GEMINI_MODEL_REGISTRY.values():
        if spec.free_tier != "eligible":
            assert spec.model_id not in selectable_ids
        if spec.status in ("deprecated", "shutdown"):
            assert spec.model_id not in selectable_ids
    # the configured default is always in the selector
    assert "gemini-3.8-flash" in selectable_ids
    assert all(s.free_tier == "eligible" for s in selectable_models())


def test_budget_family_levels_match_documented_mapping():
    for spec in GEMINI_MODEL_REGISTRY.values():
        if spec.thinking_control == "budget":
            assert spec.family == "gemini-2.5"
            assert set(BUDGET_BY_LEVEL) <= set(spec.thinking_levels)


# ---------- resolution / validation ----------


def _resolve(model_id, thinking, **kwargs):
    kwargs.setdefault("default_model_id", "gemini-3.8-flash")
    return resolve_run_model_config(model_id, thinking, **kwargs)


def test_defaults_reproduce_historical_behavior():
    config = _resolve(None, None)
    assert config.model_id == "gemini-3.8-flash"
    assert config.thinking_level == "medium"
    # a configured thinking override becomes the run default
    config = _resolve(None, None, default_thinking_level="high")
    assert config.thinking_level == "high"
    # explicit "auto" is honored (model default on the wire)
    assert _resolve(None, "auto").thinking_level == "auto"


def test_unknown_model_id_is_rejected():
    with pytest.raises(ModelSelectionError) as exc:
        _resolve("gpt-99-fantasy", None, explicit_selection=True)
    assert exc.value.category == "unknown_model"


def test_paid_model_selection_is_rejected_with_reason():
    with pytest.raises(ModelSelectionError) as exc:
        _resolve("gemini-4-argon", None, explicit_selection=True)
    assert exc.value.category == "model_not_selectable"
    assert "free tier" in str(exc.value)


def test_unknown_eligibility_model_is_rejected_with_reason():
    with pytest.raises(ModelSelectionError) as exc:
        _resolve("gemini-3.7-flash", None, explicit_selection=True)
    assert exc.value.category == "model_not_selectable"
    assert "eligibility" in str(exc.value)


def test_retired_model_cannot_be_selected():
    with pytest.raises(ModelSelectionError) as exc:
        _resolve("gemini-2.0-flash", None, explicit_selection=True)
    assert exc.value.category == "model_not_selectable"


def test_explicit_unknown_thinking_value_is_rejected():
    with pytest.raises(ModelSelectionError) as exc:
        _resolve(None, "extreme", explicit_selection=True)
    assert exc.value.category == "unsupported_thinking_level"


def test_minimal_rejected_when_model_does_not_support_it():
    with pytest.raises(ModelSelectionError) as exc:
        _resolve("gemini-3.8-flash", "minimal", explicit_selection=True)
    assert exc.value.category == "unsupported_thinking_level"
    message = str(exc.value)
    assert "minimal" in message and "low" in message  # supported list


def test_minimal_accepted_where_documented():
    config = _resolve("gemini-3.6-flash", "minimal", explicit_selection=True)
    assert config.thinking_level == "minimal"


def test_auto_is_always_accepted():
    config = _resolve("gemini-3.8-flash", "auto", explicit_selection=True)
    assert config.thinking_level == "auto"


def test_values_are_normalized():
    config = _resolve("  gemini-3.6-flash  ", "  MEDIUM ", explicit_selection=True)
    assert config.model_id == "gemini-3.6-flash"
    assert config.thinking_level == "medium"


def test_config_derived_invalid_combo_still_rejected():
    """An admin config that contradicts the model must surface as a
    clear error, never reach the provider as a 400."""
    with pytest.raises(ModelSelectionError) as exc:
        _resolve("gemini-3.8-flash", "minimal", explicit_selection=False)
    assert exc.value.category == "unsupported_thinking_level"


def test_custom_configured_model_stays_usable():
    """A model outside the registry (admin .env) keeps working — and
    the configured default itself is always nameable."""
    config = _resolve(
        "my-private-model", "high", default_model_id="my-private-model"
    )
    assert config.model_id == "my-private-model"
    # naming the configured default explicitly is allowed...
    assert (
        _resolve(
            "my-private-model",
            None,
            default_model_id="my-private-model",
            explicit_selection=True,
        ).model_id
        == "my-private-model"
    )
    # ...but an arbitrary non-registry ID as a user selection is not
    with pytest.raises(ModelSelectionError) as exc:
        _resolve("some-other-model", None, explicit_selection=True)
    assert exc.value.category == "unknown_model"


# ---------- execution-time safety ----------


def test_retired_model_fails_fast_with_clear_error():
    with pytest.raises(ModelUnavailableError) as exc:
        assert_runnable("gemini-2.0-flash")
    message = str(exc.value)
    assert "model unavailable" in message
    assert "2026-06-01" in message
    # the error must not promise or hint at any substitution
    assert "fallback" not in message and "instead" not in message


def test_live_and_custom_models_pass_runnable_check():
    assert assert_runnable("gemini-3.8-flash") is not None
    assert assert_runnable("my-private-model") is None


# ---------- wire-level thinking mapping ----------


def test_level_family_uses_thinking_level():
    assert thinking_kwargs("gemini-3.8-flash", level="low") == {
        "thinking_level": "low"
    }
    assert thinking_kwargs("gemini-3.6-flash", level="minimal") == {
        "thinking_level": "minimal"
    }


def test_auto_sends_no_thinking_parameter():
    for model_id in ("gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash"):
        assert thinking_kwargs(model_id, level="auto") == {}


def test_unsupported_level_never_reaches_the_api():
    # gemini-3.8-flash does not support minimal: omission, not a 400
    assert thinking_kwargs("gemini-3.8-flash", level="minimal") == {}


def test_budget_family_uses_documented_token_budgets():
    expected = {"minimal": 1024, "low": 1024, "medium": 8192, "high": 24576}
    for level, budget in expected.items():
        kwargs = thinking_kwargs("gemini-2.5-flash", level=level)
        assert kwargs == {"thinking_budget": budget}


def test_level_and_budget_are_never_combined():
    for model_id in GEMINI_MODEL_REGISTRY:
        for level in THINKING_LEVELS:
            kwargs = thinking_kwargs(model_id, level=level)
            assert not (
                "thinking_level" in kwargs and "thinking_budget" in kwargs
            )
        kwargs = thinking_kwargs(model_id, level="low", verification=True)
        assert not (
            "thinking_level" in kwargs and "thinking_budget" in kwargs
        )


def test_models_without_thinking_controls_send_nothing():
    assert thinking_kwargs("gemini-2.0-flash", level="high") == {}


def test_unknown_models_use_family_heuristics_only():
    assert control_by_family("gemini-3.4-flash") == "level"
    assert control_by_family("gemini-2.5-flash-preview") == "budget"
    assert control_by_family("gemini-2.0-flash") == "none"
    assert control_by_family("totally-custom") == "none"
    assert family_of("models/gemini-3.9-flash") == "gemini-3"
    # custom IDs keep working through the family mapping
    assert thinking_kwargs("gemini-3.4-flash", level="high") == {
        "thinking_level": "high"
    }
    assert thinking_kwargs("gemini-2.5-flash-preview", level="low") == {
        "thinking_budget": 1024
    }
    assert thinking_kwargs("totally-custom", level="low") == {}


def test_verification_path_is_cheap_and_never_research_config():
    # level family: lowest level
    assert thinking_kwargs("gemini-3.8-flash", level="low", verification=True) == {
        "thinking_level": "low"
    }
    # budget family: thinking OFF (verification-only constraint)
    assert thinking_kwargs(
        "gemini-2.5-flash", level="low", verification=True
    ) == {"thinking_budget": 0}
    # no thinking controls: nothing
    assert thinking_kwargs("gemini-2.0-flash", level="low", verification=True) == {}
