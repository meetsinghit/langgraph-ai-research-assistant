"""Tests for research depth and evidence policy resolution (4E.1).

WS-G: depth drives explicit, configurable budgets and guidance —
never a numerical confidence score, never a length gate, and never
the global deadline (which stays the hard ceiling). WS-A: the
evidence policy is a separate axis with its own default.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.config import Settings
from agent.depth import (
    DEFAULT_EVIDENCE_POLICY,
    DEFAULT_RESEARCH_DEPTH,
    DEPTH_WORD_TARGETS,
    EVIDENCE_POLICIES,
    RESEARCH_DEPTHS,
    DepthPolicy,
    depth_policy,
    resolve_depth,
    resolve_evidence_policy,
    review_depth_expectation,
    synthesize_depth_guidance,
)


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


# --- resolution ----------------------------------------------------


def test_defaults_are_standard_and_balanced():
    assert DEFAULT_RESEARCH_DEPTH == "standard"
    assert DEFAULT_EVIDENCE_POLICY == "balanced"
    assert RESEARCH_DEPTHS == ("quick", "standard", "deep")
    assert EVIDENCE_POLICIES == ("balanced", "verified")


@pytest.mark.parametrize("value", [None, "", "  ", "extreme", 7, True])
def test_resolve_depth_falls_back_to_standard(value):
    assert resolve_depth(value) == "standard"


@pytest.mark.parametrize("value", [None, "", "strict", 7])
def test_resolve_evidence_policy_falls_back_to_balanced(value):
    assert resolve_evidence_policy(value) == "balanced"


def test_resolve_is_case_insensitive_but_exact():
    assert resolve_depth(" Quick ") == "quick"
    assert resolve_depth("DEEP") == "deep"
    # not a prefix match: "quicksilver" is not a depth
    assert resolve_depth("quicksilver") == "standard"
    assert resolve_evidence_policy("VERIFIED") == "verified"


# --- budgets -------------------------------------------------------


def test_standard_policy_is_the_historical_settings_unchanged():
    """Backward compatibility: standard == the pre-4E.1 numbers."""
    s = _settings()
    policy = depth_policy(s, "standard")
    assert policy.plan_query_cap == s.max_plan_queries
    assert policy.max_iterations == s.max_research_iterations
    assert policy.max_tool_calls == s.max_tool_calls_per_research
    assert policy.min_findings == s.min_findings
    assert policy.min_retrieved_sources == s.min_retrieved_sources
    assert policy.max_evidence_chars == s.max_evidence_chars
    assert policy.word_target == (500, 900)


def test_quick_policy_reads_the_explicit_quick_knobs():
    s = _settings(
        quick_max_plan_queries=2,
        quick_max_research_iterations=1,
        quick_max_tool_calls_per_research=3,
        quick_min_findings=2,
        quick_min_retrieved_sources=1,
    )
    policy = depth_policy(s, "quick")
    assert policy.plan_query_cap == 2
    assert policy.max_iterations == 1
    assert policy.max_tool_calls == 3
    assert policy.min_findings == 2
    assert policy.min_retrieved_sources == 1
    # never exceeds the standard budgets
    standard = depth_policy(s, "standard")
    assert policy.max_tool_calls <= standard.max_tool_calls
    assert policy.max_iterations <= standard.max_iterations


def test_deep_policy_reads_the_explicit_deep_knobs():
    s = _settings(
        deep_max_plan_queries=6,
        deep_max_research_iterations=5,
        deep_max_tool_calls_per_research=8,
        deep_min_findings=6,
        deep_min_retrieved_sources=3,
        deep_max_evidence_chars=8000,
    )
    policy = depth_policy(s, "deep")
    assert policy.plan_query_cap == 6
    assert policy.max_iterations == 5
    assert policy.max_tool_calls == 8
    assert policy.min_findings == 6
    assert policy.min_retrieved_sources == 3
    assert policy.max_evidence_chars == 8000
    # deeper than standard on every axis
    standard = depth_policy(s, "standard")
    assert policy.max_iterations > standard.max_iterations
    assert policy.min_retrieved_sources > standard.min_retrieved_sources


def test_depth_policy_is_immutable_and_typed():
    policy = depth_policy(_settings(), "deep")
    assert isinstance(policy, DepthPolicy)
    with pytest.raises(Exception):
        policy.max_iterations = 99  # frozen dataclass


def test_word_targets_are_ordered_and_ranged():
    quick, standard, deep = (
        DEPTH_WORD_TARGETS["quick"],
        DEPTH_WORD_TARGETS["standard"],
        DEPTH_WORD_TARGETS["deep"],
    )
    assert quick[1] < standard[0]
    assert standard[1] < deep[0]
    assert quick == (150, 300)
    assert standard == (500, 900)
    assert deep == (1200, 2500)


def test_invalid_budget_settings_are_rejected():
    with pytest.raises(ValidationError):
        _settings(deep_max_research_iterations=0)
    with pytest.raises(ValidationError):
        _settings(quick_max_tool_calls_per_research=-1)
    with pytest.raises(ValidationError):
        _settings(deep_max_evidence_chars=10)


# --- guidance ------------------------------------------------------


def test_guidance_is_a_range_not_a_quota_and_forbids_padding():
    for depth in RESEARCH_DEPTHS:
        text = synthesize_depth_guidance(depth_policy(_settings(), depth))
        low, high = DEPTH_WORD_TARGETS[depth]
        assert f"{low}-{high}" in text
        assert "Never pad" in text
        assert "Research depth:" in text
    # quick never asks for report sections
    quick = synthesize_depth_guidance(depth_policy(_settings(), "quick"))
    assert "no long report sections" in quick


def test_depth_guidance_is_orthogonal_to_evidence_policy():
    """Depth guidance never mentions the evidence policy, and the
    review expectation never turns length into a gate."""
    deep = synthesize_depth_guidance(depth_policy(_settings(), "deep"))
    assert "snippet" not in deep.lower()
    for depth in RESEARCH_DEPTHS:
        expectation = review_depth_expectation(
            depth_policy(_settings(), depth)
        )
        assert "Never approve because" in expectation
        assert "never reject a fully substantive answer" in expectation


def test_review_expectation_states_the_mode_target():
    text = review_depth_expectation(depth_policy(_settings(), "quick"))
    assert "QUICK" in text
    assert "150-300" in text
    assert "do not demand a long report" in text


def test_unknown_depth_resolves_to_standard_budgets():
    s = _settings()
    assert depth_policy(s, "nonsense") == depth_policy(s, "standard")
    assert depth_policy(s, None) == depth_policy(s)


# --- CLI surface ---------------------------------------------------


def test_cli_exposes_backward_compatible_depth_flags(capsys):
    """`python main.py "question"` still works unchanged; the new
    flags are optional and documented in --help."""
    import main as cli

    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "--depth" in out
    assert "--evidence-policy" in out
