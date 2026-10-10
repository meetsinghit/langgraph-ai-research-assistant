"""Unit tests for the deterministic topic-coverage check.

Keyword coverage only — these tests never claim coverage proves
correctness; that semantic judgement stays with the LLM reviewer.
"""
from agent.completeness import (
    check_completeness,
    format_completeness_report,
)


def test_framing_words_and_short_tokens_are_not_topics():
    report = check_completeness("What is LangGraph?", "LangGraph is a framework.")
    assert report.requested_terms == ("langgraph",)
    assert report.ok


def test_missing_requested_topic_is_reported():
    """Live regression: the question asked for conditional routing;
    the draft never mentioned it and the reviewer still approved."""
    report = check_completeness(
        "Explain LangGraph conditional routing",
        "LangGraph builds graphs from nodes and edges.",
    )
    assert report.requested_terms == ("langgraph", "conditional", "routing")
    assert report.missing == ("conditional", "routing")
    assert not report.ok


def test_covered_topic_passes_case_insensitively():
    report = check_completeness(
        "Explain LangGraph conditional routing",
        "Conditional routing in LangGraph means edges with predicates.",
    )
    assert report.ok


def test_plural_and_variant_forms_are_tolerated():
    report = check_completeness(
        "List the routes in the app", "The app defines a single route."
    )
    assert report.requested_terms == ("route",)
    assert report.ok


def test_no_distinctive_terms_trivially_passes():
    report = check_completeness("What is it?", "Any answer at all.")
    assert report.requested_terms == ()
    assert report.ok


def test_report_disclaims_correctness():
    report = check_completeness(
        "Explain conditional routing", "Nothing relevant here."
    )
    text = format_completeness_report(report)
    assert "does NOT verify correctness" in text
    assert "FAIL" in text
    assert "conditional, routing" in text
    passing = format_completeness_report(
        check_completeness("Explain conditional routing", "Conditional routing explained.")
    )
    assert "PASS" in passing
