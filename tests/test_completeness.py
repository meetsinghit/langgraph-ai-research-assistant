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


def test_disclaimer_only_mention_is_not_coverage():
    """Live regression: the draft said conditional routing details
    'were not available' — an unavailability disclaimer, not an
    explanation. Pure keyword presence counted it as covered."""
    report = check_completeness(
        "Explain LangGraph conditional routing",
        "LangGraph state flows through checkpointers. Conditional "
        "routing details were not available in the retrieved sources.",
    )
    assert report.missing == ("conditional", "routing")
    assert not report.ok


def test_topic_explained_in_one_sentence_survives_a_disclaimer_elsewhere():
    """A disclaimer elsewhere must not erase a real explanation of
    the topic in another sentence."""
    report = check_completeness(
        "Explain LangGraph conditional routing",
        "LangGraph conditional routing means edges with predicates. "
        "Retry backoff details were not available in the sources.",
    )
    assert report.ok


def test_alternative_disclaimer_phrasings_are_not_coverage():
    for answer in (
        "LangGraph conditional routing was unavailable in the sources.",
        "LangGraph conditional routing could not be found in the pages.",
        "No details on conditional routing were provided.",
        "LangGraph conditional routing is not covered by the findings.",
    ):
        report = check_completeness(
            "Explain LangGraph conditional routing", answer
        )
        assert not report.ok, answer
        assert "routing" in report.missing, answer


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
    assert "LEXICAL" in text
    assert "does NOT verify" in text
    assert "FAIL" in text
    assert "conditional, routing" in text
    passing = format_completeness_report(
        check_completeness("Explain conditional routing", "Conditional routing explained.")
    )
    assert "PASS" in passing
