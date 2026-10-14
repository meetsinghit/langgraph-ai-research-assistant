"""Unit tests for the deterministic topic-coverage check.

Keyword coverage only — these tests never claim coverage proves
correctness; that semantic judgement stays with the LLM reviewer.
"""
from agent.completeness import (
    ExplanationReport,
    check_code_example,
    check_completeness,
    check_explanation_depth,
    format_code_example_report,
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


# --- live-acceptance Problem B: phrases, instruction verbs, deliverables --
#
# The observed browser run flagged "management" and "include" as missing
# from an answer that substantively covered state management. Root cause:
# isolated-token matching treated "include" (an instruction verb) as a
# topic, split the phrase "state management" into two independent
# required words, and matched requested deliverables ("a practical
# example", "citations") as bare keywords instead of checking whether
# they were substantively present.

_OBSERVED_Q = (
    "Explain how LangGraph state management and conditional routing work. "
    "Include a practical example, implementation considerations, and "
    "citations to official documentation."
)

_COMPLETE_ANSWER = (
    "LangGraph state management stores shared data in a TypedDict. State "
    "management works by defining a schema every node reads and writes, so "
    "state persists across super-steps. Conditional routing decides the next "
    "edge at runtime: a router returns a key and add_conditional_edges maps "
    "it.\n\n"
    "For example:\n```python\nfrom langgraph.graph import StateGraph, "
    "START, END\nclass S(TypedDict):\n    value: int\n"
    "def plan(state: S) -> S:\n    return {'value': state['value'] + 1}\n"
    "g = StateGraph(S)\ng.add_node('plan', plan)\ng.add_edge(START, "
    "'plan')\napp = g.compile()\napp.invoke({'value': 0})\n```\n\n"
    "Implementation considerations: prefer reducers for concurrent writes "
    "and keep state serializable. See "
    "https://langchain-ai.github.io/langgraph/ and [S1]."
)


def test_observed_complete_answer_is_not_flagged():
    """Positive fixture: an answer that covers both phrases AND delivers
    the example/considerations/citations passes."""
    report = check_completeness(_OBSERVED_Q, _COMPLETE_ANSWER)
    assert report.ok, report.missing
    # "include" is an instruction verb, never a topic
    assert "include" not in report.requested_terms
    # deliverable words are absorbed out of ordinary topics
    for word in ("practical", "example", "citation", "consideration"):
        assert word not in report.requested_terms


def test_observed_incomplete_answer_is_still_flagged():
    """Negative fixture: dropping conditional routing AND the code example
    must not read as complete."""
    answer = (
        "LangGraph state management stores shared data in a TypedDict, and "
        "each node updates it. State persists across super-steps. "
        "Implementation considerations: prefer reducers."
    )
    report = check_completeness(_OBSERVED_Q, answer)
    assert not report.ok
    assert any("conditional" in m or "routing" in m for m in report.missing)
    # the requested example was never substantively provided
    assert any("example" in m for m in report.missing)


def test_compound_phrase_is_matched_as_a_unit_not_isolated_tokens():
    """'state management' is covered when the answer substantively
    discusses state — without echoing the literal token 'management'."""
    answer = (
        "In LangGraph, state is a shared TypedDict that every node reads and "
        "updates; the graph merges partial deltas between super-steps. "
        "Conditional routing picks the next node at runtime via a router "
        "callable. For example: ```x = build(); x.run()``` Consider that "
        "state should stay serializable. https://example.org/docs [S1]"
    )
    report = check_completeness(_OBSERVED_Q, answer)
    assert "management" not in report.missing
    assert report.ok, report.missing


def test_phrase_recognition_is_generalizable_not_overfit_to_one_word():
    """A DIFFERENT compound phrase behaves the same way — the fix is not
    special-cased to 'management'."""
    question = "Explain how database connection pooling and retry backoff work."
    answer = (
        "Connection pooling reuses live database connections so a request "
        "borrows one instead of opening a new socket. Retry backoff waits "
        "progressively longer after a failure."
    )
    report = check_completeness(question, answer)
    assert "pooling" not in report.missing  # covered via its phrase
    assert report.ok, report.missing
    # but a question whose phrase is entirely absent is still flagged
    missing_all = check_completeness(question, "It stores data in tables.")
    assert not missing_all.ok


def test_instruction_verbs_are_not_topics():
    """The defect: 'Include' was flagged as a missing topic. Instruction
    verbs — a whole class — are never topics."""
    cases = [
        ("Explain caching. Include details.", ("include", "detail")),
        ("Describe load balancing. Provide an outline.", ("provide", "outline")),
        ("Explain sharding. List the options.", ("list",)),
        ("Explain routing. Show the steps.", ("show",)),
        ("Explain tracing. Demonstrate the setup.", ("demonstrate",)),
    ]
    for question, verbs in cases:
        report = check_completeness(question, "Caching reuses results.")
        for verb in verbs:
            assert verb not in report.requested_terms, (question, verb)
            assert verb not in report.missing, (question, verb)


def test_requested_example_is_verified_structurally():
    """An answer that discusses everything but gives NO example when one
    was explicitly requested is flagged — and a code fence satisfies it."""
    topics = "Conditional routing picks the next node based on state."
    no_example = topics + " There are several approaches to consider."
    assert any(
        "example" in m
        for m in check_completeness(_OBSERVED_Q, no_example).missing
    )
    with_code = topics + " For example: ```route(state)``` Keep it small. [S1] https://x.dev"
    report = check_completeness(_OBSERVED_Q, with_code)
    assert not any("example" in m for m in report.missing)


def test_requested_citations_are_verified_structurally():
    """A citation deliverable is satisfied by markers/URLs, never by the
    word 'citations'."""
    answer = (
        "Conditional routing picks the next node based on state. For "
        "example: ```route(state)``` Consider trade-offs. "
        "See https://langchain-ai.github.io/langgraph/"
    )
    assert not any("citations" in m for m in check_completeness(_OBSERVED_Q, answer).missing)
    # no markers/URL at all -> citation deliverable missing
    bare = (
        "Conditional routing picks the next node based on state. For "
        "example: ```route(state)``` Consider trade-offs."
    )
    assert any("citations" in m for m in check_completeness(_OBSERVED_Q, bare).missing)


# --- live-acceptance Problem C: a requested example must be runnable ----
#
# The observed defect asked for a practical working example and the
# drafted LangGraph code commented out the required node registrations
# and never established a complete graph entry path — it read like an
# example but could not run. check_code_example reports the structural
# shape of the code text (never claiming it executed).

_EX_Q = (
    "Explain how LangGraph conditional routing works and include a "
    "practical working example."
)

_VALID_EXAMPLE = """Conditional routing picks the next edge at runtime.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class S(TypedDict):
    value: int

def decide(state: S) -> str:
    return "big" if state["value"] > 5 else "small"

def big(state: S) -> S:
    return {"value": state["value"] * 2}

g = StateGraph(S)
g.add_node("big", big)
g.add_conditional_edges("decide", decide, {"big": "big"})
app = g.compile()
app.invoke({"value": 3})
```

See https://langchain-ai.github.io/langgraph/ and [S1]."""

_INCOMPLETE_EXAMPLE = """Conditional routing picks the next edge at runtime.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class S(TypedDict):
    value: int

def decide(state: S) -> str:
    return "big" if state["value"] > 5 else "small"

# g.add_node("decide", decide)
# g.add_edge(START, "decide")
# app = g.compile()
g = StateGraph(S)
```

See https://langchain-ai.github.io/langgraph/ and [S1]."""


def test_valid_runnable_example_passes_the_structural_check():
    report = check_code_example(_EX_Q, _VALID_EXAMPLE)
    assert report.ok
    assert report.requested and report.blocks == 1
    assert report.has_import and report.has_definition and report.has_invocation
    assert not report.commented_out
    assert "PASS" in format_code_example_report(report)


def test_commented_out_wiring_with_no_entry_path_is_flagged():
    """The observed defect: node registrations commented out and no
    invocation — illustrative, not runnable."""
    report = check_code_example(_EX_Q, _INCOMPLETE_EXAMPLE)
    assert not report.ok
    assert report.commented_out and not report.has_invocation
    assert any("commented out" in r for r in report.reasons)
    assert "FAIL" in format_code_example_report(report)


def test_definitions_that_never_run_are_flagged():
    answer = (
        "A router returns the next node.\n\n"
        "```python\ndef route(state):\n    return 'b' if state['n'] > 1 else 'a'\n```\n\n"
        "See https://example.org/docs [S1]."
    )
    report = check_code_example(_EX_Q, answer)
    assert not report.ok
    assert report.has_definition and not report.has_invocation


def test_bare_invocation_snippet_is_not_flagged():
    answer = (
        "Use the compiled graph.\n\n```python\napp.invoke({'n': 5})\n```\n\n"
        "https://x.dev [S1]"
    )
    report = check_code_example(_EX_Q, answer)
    assert report.ok  # it invokes; there are no dead definitions


def test_check_only_runs_when_an_example_is_requested_and_code_present():
    # no example requested -> untouched
    assert check_code_example("Explain LangGraph conditional routing", _VALID_EXAMPLE).ok
    # example requested but no code block -> left to the coverage check
    prose = (
        "Conditional routing picks the next node, e.g. via a router. "
        "See https://x.dev [S1]"
    )
    report = check_code_example(_EX_Q, prose)
    assert report.ok and report.requested and report.blocks == 0


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
