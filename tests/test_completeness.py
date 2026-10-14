"""Unit tests for the deterministic topic-coverage check.

Keyword coverage only — these tests never claim coverage proves
correctness; that semantic judgement stays with the LLM reviewer.
"""
from agent.completeness import (
    ExplanationReport,
    check_completeness,
    check_explanation_depth,
    format_completeness_report,
    format_explanation_report,
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


def test_not_present_disclaimer_is_not_coverage():
    """Live regression: the exact phrasing observed — 'Specific
    details regarding conditional routing syntax were not present
    in the retrieved text excerpts.' — is an unavailability
    disclaimer, not an explanation of conditional routing."""
    report = check_completeness(
        "Explain LangGraph conditional routing syntax",
        "Specific details regarding conditional routing syntax were "
        "not present in the retrieved text excerpts. LangGraph syntax "
        "uses nodes and edges.",
    )
    assert not report.ok
    assert "conditional" in report.missing
    assert "routing" in report.missing
    # the unrelated substantive sentence still counts as coverage
    assert "langgraph" not in report.missing
    assert "syntax" not in report.missing


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


# --- explanation depth (Phase 4E.1, WS-B) -------------------------
#
# Lexical heuristic tests: "named but never explained". These never
# claim the explanation is CORRECT — semantic adequacy stays with
# the reviewer.

_RETRIEVED = [
    {
        "text": "Conditional routing branches execution along "
        "edges guarded by predicate functions that read the state.",
        "source_url": "https://wiki.example.org/routing",
        "evidence": "retrieved",
    }
]
_WS4_QUESTION = (
    "Explain LangGraph state management and conditional routing "
    "using official documentation."
)
_WS4_ANSWER = (
    "LangGraph state management persists state through "
    "checkpointers that snapshot the state at each node. "
    "Conditional routing is a core LangGraph capability for "
    "directing flow. See [S1] and [S2]."
)


def test_named_only_topic_is_flagged_as_shallow():
    """The WS4 failure: one topic explained, one merely classified."""
    report = check_explanation_depth(
        _WS4_QUESTION, _WS4_ANSWER, _RETRIEVED
    )
    assert report.shallow == ("conditional", "routing")
    assert not report.ok


def test_substantive_explanations_are_not_flagged():
    answer = (
        "LangGraph state management persists state through "
        "checkpointers that snapshot the state at each node. "
        "Conditional routing branches execution along edges "
        "guarded by predicate functions that read the state, "
        "per [S1]."
    )
    report = check_explanation_depth(_WS4_QUESTION, answer, _RETRIEVED)
    assert report.shallow == ()
    assert report.ok


def test_check_is_skipped_without_retrieved_evidence():
    """Honest UNVERIFIED answers (snippets only) keep the existing
    verification governance — depth is never used to pile on."""
    snippets = [
        {
            "text": "snippet",
            "source_url": "https://wiki.example.org/routing",
            "evidence": "snippet",
        }
    ]
    report = check_explanation_depth(
        _WS4_QUESTION, _WS4_ANSWER, snippets
    )
    assert report.shallow == ()
    # ... and with no findings at all
    assert check_explanation_depth(_WS4_QUESTION, _WS4_ANSWER).ok


def test_check_is_skipped_without_explanation_intent():
    """A resource request answered with a pointer to the right
    source is complete — it is not an explanation demand."""
    report = check_explanation_depth(
        "LangGraph official documentation",
        "LangGraph docs, per [S1].",
        _RETRIEVED,
    )
    assert report.shallow == ()


def test_simple_definition_questions_with_substantive_answers_pass():
    """Legitimate short answers to 'What is X?' are preserved."""
    findings = [
        {
            "text": "SecOps integrates security and IT operations "
            "teams and workflows.",
            "source_url": "https://secops.example.com/overview",
            "evidence": "retrieved",
        }
    ]
    answer = (
        "SecOps (security operations) integrates security and IT "
        "operations teams so threats are detected and responded to "
        "faster, per [S1]."
    )
    report = check_explanation_depth("What is SecOps?", answer, findings)
    assert report.shallow == ()


def test_echoing_the_question_is_not_explanation():
    """A sentence made of question vocabulary + a generic category
    noun is naming-only, even when the topic word itself looks like
    a verb ('routing')."""
    answer = "Conditional routing is a core capability for routing."
    report = check_explanation_depth(
        _WS4_QUESTION, answer, _RETRIEVED
    )
    assert "routing" in report.shallow


def test_disclaimer_sentences_never_count_as_explanations():
    answer = (
        "LangGraph state management persists state through "
        "checkpointers that snapshot the state at each node. "
        "Conditional routing details were not available in the "
        "retrieved sources."
    )
    report = check_explanation_depth(_WS4_QUESTION, answer, _RETRIEVED)
    # covered-only-by-a-disclaimer terms belong to check_completeness
    # (missing), not to this check (shallow)
    assert report.shallow == ()


def test_explanation_report_formatting_states_limits():
    report = check_explanation_depth(
        _WS4_QUESTION, _WS4_ANSWER, _RETRIEVED
    )
    text = format_explanation_report(report)
    assert "LEXICAL" in text
    assert "does NOT judge whether any explanation is correct" in text
    assert "FAIL" in text
    assert "conditional, routing" in text
    assert "PASS" in format_explanation_report(ExplanationReport(()))
