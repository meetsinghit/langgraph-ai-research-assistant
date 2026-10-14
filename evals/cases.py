"""Offline evaluation benchmark for the AI Research Assistant.

Every case is a fully deterministic fixture: scripted model
responses, canned search results and page texts, per-URL
fetch failures — NO network, NO API keys, NO live provider
calls. The harness (``evals.harness``) drives the real
LangGraph graph with these fixtures and scores the run
against each case's expectations.

Case categories (per the Phase 4A brief):

- official technical documentation
- state management and conditional routing
- multi-topic research questions
- general technical research
- questions requiring multiple verified sources
- requests with ambiguous wording
- unavailable or contradictory evidence
- questions where an incomplete answer is the correct outcome
- tool failure and source-retrieval failure cases

Semantic scoring is fixture-based (human-authored substantive
markers per topic) — it scores fixture answers against
rubric criteria. It is NOT a claim of factual correctness;
the optional LLM judge (``EVAL_LLM_JUDGE=1``) is disabled
by default and never runs in tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "CaseExpectations",
    "CaseSecurity",
    "EvalCase",
    "ResearchStep",
    "TopicRubric",
    "benchmark_cases",
]


@dataclass(frozen=True)
class CaseSecurity:
    """Security-scenario metadata for a case (Phase 4C, WS-G).

    Every security case documents WHY it exists and how far its
    result can be trusted:

    - ``defense_layer``: ``"deterministic"`` when the pass criterion
      is enforced entirely in code (policy refusals, schema
      validation, redaction, citation gates); ``"model_behavior"``
      when the case probes how a MODEL responds to hostile content.
      A scripted model that imitates a compromised model CANNOT
      prove real-model prompt-injection resistance — those cases
      are kept to make the limitation visible, and their pass
      criteria still rest on deterministic controls wherever
      possible.
    - ``residual``: what this case does NOT establish.
    """

    attacker_objective: str
    entry_point: str
    expected_behavior: str
    defense_layer: str  # "deterministic" | "model_behavior"
    severity: str  # "high" | "medium" | "low"
    residual: str


@dataclass(frozen=True)
class ResearchStep:
    """One scripted model response inside a research pass."""

    kind: str  # "tool_call" | "text"
    name: str | None = None  # tool name for "tool_call"
    args: dict[str, Any] | None = None
    text: str | None = None  # content for "text"


@dataclass(frozen=True)
class TopicRubric:
    """One topic's fixture-scoring criteria.

    ``expected_markers`` are substantive phrases a real
    explanation of the topic contains (authored by the
    eval designer, not derived from the answer). A topic
    scores ``adequate`` only when the draft contains at
    least one marker OUTSIDE known unavailability
    disclaimers — naming the topic alone is not adequate.
    """

    topic: str
    expected_markers: tuple[str, ...]


@dataclass(frozen=True)
class CaseExpectations:
    """Deterministic pass/fail expectations for a case."""

    review_status: str = "approved"
    markers: tuple[str, ...] = ()  # substrings of final_answer
    missing_topics: tuple[str, ...] = ()
    official_status: str | None = None
    unverified_marker: bool = False
    #: when True, the case asserts that plan/research
    #: prompts steered an official-documentation question
    #: toward the CURRENT documentation domains.
    official_domain_steering: bool = False
    # --- Phase 4C security expectations (all content-free) ---
    #: minimum injection-signal categories recorded in
    #: metrics for this run (defense-in-depth SIGNAL — its
    #: absence is never treated as proof of safety).
    min_injection_signals: int = 0
    #: minimum count per deterministic refusal category
    #: (validation / policy / dlp / ssrf) in the metrics
    #: security block.
    min_refusals: dict[str, int] = field(default_factory=dict)
    #: minimum credential redactions applied to the answer.
    min_answer_redactions: int = 0
    #: substrings that must NOT appear in the final answer
    #: (e.g. a synthetic canary credential).
    forbidden_substrings: tuple[str, ...] = ()
    #: when True, deterministic citation validation must
    #: pass for the final answer.
    require_citations_ok: bool = False
    #: when set, at least this many citations must resolve
    #: to unknown (non-existent) sources — asserts that the
    #: citation gate DETECTED fabricated references.
    min_unknown_citations: int | None = None
    #: when set, at least this many cited sources must show
    #: LOW lexical claim-to-source overlap (the advisory
    #: grounding check found no shared vocabulary between
    #: the citing sentence and the cited source's text).
    min_low_overlap_citations: int | None = None
    #: when set, at least this many distinct source PAGES
    #: must have been fetched (retrieved evidence).
    min_retrieved_sources: int | None = None
    #: tool names that must NOT appear among executed calls
    #: (spoofed names in structured or text form).
    forbidden_tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvalCase:
    """One benchmark case — fixtures + expectations + rubric."""

    case_id: str
    category: str
    question: str
    # --- fixture world ---
    search_results: tuple[dict[str, str], ...] = ()
    pages: dict[str, str] = field(default_factory=dict)
    fail_fetch_urls: frozenset[str] = frozenset()
    fail_queries: frozenset[str] = frozenset()
    # --- scripted model behavior ---
    plan_queries: tuple[str, ...] = ("q",)
    research_steps: tuple[ResearchStep, ...] = ()
    drafts: tuple[str, ...] = ("Draft answer.",)
    review_verdicts: tuple[str, ...] = ("approved",)
    #: call index -> error kind ("rate_limit") raised by
    #: the scripted model, to exercise retry behavior.
    raise_on_call: dict[int, str] = field(default_factory=dict)
    #: per-case Settings overrides (e.g. a higher
    #: MIN_RETRIEVED_SOURCES).
    settings_overrides: dict[str, Any] = field(
        default_factory=dict
    )
    #: per-run research depth ("quick" | "standard" | "deep",
    #: Phase 4E.1 WS-G) — resolved by agent.depth against the
    #: case's settings.
    research_depth: str = "standard"
    #: per-run evidence policy ("balanced" | "verified",
    #: Phase 4E.1 WS-A).
    evidence_policy: str = "balanced"
    # --- evaluation ---
    expected: CaseExpectations = field(
        default_factory=CaseExpectations
    )
    rubric: tuple[TopicRubric, ...] = ()
    #: security-scenario metadata when this is a Phase 4C
    #: adversarial case (None for ordinary benchmark cases).
    security: CaseSecurity | None = None

    def __post_init__(self) -> None:
        """Normalize bare values to tuples.

        Case definitions may write ``drafts="..."`` or a
        single ``ResearchStep`` instead of 1-tuples;
        without this, iterating a string would yield
        individual characters as scripted responses.
        """
        for name in (
            "plan_queries",
            "drafts",
            "review_verdicts",
        ):
            value = getattr(self, name)
            if isinstance(value, str):
                object.__setattr__(self, name, (value,))
        steps = self.research_steps
        if isinstance(steps, ResearchStep):
            object.__setattr__(self, "research_steps", (steps,))


def _result(title: str, url: str, snippet: str = "snippet") -> dict:
    return {"title": title, "snippet": snippet, "url": url}


#: The Phase 4A live-failure reproduction (Workstream 4):
#: state management explained, conditional routing merely
#: NAMED. Originally EVERY deterministic gate passed and the
#: scripted reviewer approved (only the semantic rubric
#: flagged it); since Phase 4E.1 (WS-B) the deterministic
#: explanation-depth check catches the named-only topic and
#: the gate downgrades the approval — the rubric still
#: scores conditional routing INADEQUATE, and the two now
#: agree.
_WS4_STEPS = (
    ResearchStep(
        kind="tool_call",
        name="web_search",
        args={"query": "q"},
    ),
    ResearchStep(kind="text", text="done"),
)
_WS4_DRAFT = (
    "LangGraph state management persists state through "
    "checkpointers that snapshot the state at each node. "
    "Conditional routing is a core LangGraph capability for "
    "directing flow. See [S1] and [S2]."
)


def benchmark_cases() -> tuple[EvalCase, ...]:
    """The ~23-case benchmark (deterministic, offline)."""
    official = (
        "https://docs.langchain.com/oss/python/langgraph/overview"
    )
    reference = "https://reference.langchain.com/langgraph"
    blog = "https://blog.example.com/langgraph-tutorial"
    deepwiki = "https://deepwiki.example.com/langgraph-guide"
    wiki_a = "https://wiki.example.org/state-management"
    wiki_b = "https://wiki.example.org/conditional-routing"

    cases: list[EvalCase] = []

    # 1 — general technical research, multi-source happy path.
    cases.append(
        EvalCase(
            case_id="general-multi-source",
            category="general technical research",
            question="How do Python decorators work?",
            search_results=(
                _result("Decorators", "https://example.com/decorators"),
                _result("PEP 318", "https://example.com/pep318"),
            ),
            pages={
                "https://example.com/decorators": (
                    "A decorator wraps a function, returning a "
                    "new callable that adds behavior."
                ),
                "https://example.com/pep318": (
                    "PEP 318 introduced the @decorator syntax."
                ),
            },
            research_steps=(
                ResearchStep(
                    kind="tool_call",
                    name="web_search",
                    args={"query": "python decorators"},
                ),
                ResearchStep(
                    kind="tool_call",
                    name="fetch_page",
                    args={"url": "https://example.com/decorators"},
                ),
            ),
            drafts=(
                "Python decorators wrap functions to add "
                "behavior, per [S1]; the syntax comes from "
                "PEP 318 [S2]."
            ),
            expected=CaseExpectations(
                review_status="approved",
                markers=["decorator", "[S1]"],
            ),
            rubric=(
                TopicRubric(
                    "python decorators",
                    ("wrap", "callable", "behavior"),
                ),
            ),
        )
    )

    # 2 — official documentation, success path.
    cases.append(
        EvalCase(
            case_id="official-docs-success",
            category="official technical documentation",
            question=(
                "Explain LangGraph state management using "
                "official documentation."
            ),
            search_results=(
                _result("Official overview", official),
                _result("Tutorial", blog),
            ),
            pages={
                official: (
                    "LangGraph state is managed through a "
                    "checkpointer that persists state at each "
                    "superstep."
                ),
                blog: "Third-party tutorial text.",
            },
            research_steps=(
                ResearchStep(
                    kind="tool_call",
                    name="web_search",
                    args={"query": "q"},
                ),
                ResearchStep(kind="text", text="done"),
            ),
            drafts=(
                "LangGraph state management persists state "
                "through checkpointers, per the official docs "
                "[S1]."
            ),
            expected=CaseExpectations(
                review_status="approved",
                markers=["checkpointer", "[S1]"],
                official_status="verified_official",
            ),
            rubric=(
                TopicRubric(
                    "state management",
                    ("checkpointer", "persist", "superstep"),
                ),
            ),
        )
    )

    # 3 — official page fails TLS; honest incomplete result.
    cases.append(
        EvalCase(
            case_id="official-tls-failure",
            category="official technical documentation",
            question=(
                "Explain LangGraph state management using "
                "official documentation."
            ),
            search_results=(
                _result("Official overview", official),
                _result("DeepWiki", deepwiki),
            ),
            pages={deepwiki: "Third-party overview text."},
            fail_fetch_urls=frozenset({official}),
            research_steps=(
                ResearchStep(
                    kind="tool_call",
                    name="web_search",
                    args={"query": "q"},
                ),
                ResearchStep(kind="text", text="done"),
            ),
            drafts=(
                "LangGraph state management flows through "
                "checkpointer pages like [S2]."
            ),
            expected=CaseExpectations(
                review_status="needs_more_research",
                markers=["OFFICIAL SOURCES: NOT RETRIEVED"],
                official_status="official_discovered_only",
            ),
            rubric=(
                TopicRubric(
                    "state management",
                    ("checkpointer", "persist"),
                ),
            ),
        )
    )

    # 4 — no first-party source discoverable at all.
    cases.append(
        EvalCase(
            case_id="official-none-found",
            category="official technical documentation",
            question=(
                "LangGraph official documentation for state "
                "management"
            ),
            search_results=(_result("Blog", blog),),
            pages={blog: "Blog text."},
            research_steps=(
                ResearchStep(
                    kind="tool_call",
                    name="web_search",
                    args={"query": "q"},
                ),
                ResearchStep(kind="text", text="done"),
            ),
            drafts=(
                "LangGraph state management uses checkpointers, "
                "per [S1]."
            ),
            expected=CaseExpectations(
                review_status="needs_more_research",
                markers=["OFFICIAL SOURCES: NOT RETRIEVED"],
                official_status="no_official_source",
            ),
        )
    )

    # 5 — state management AND conditional routing, both explained.
    cases.append(
        EvalCase(
            case_id="state-and-routing-complete",
            category="state management and conditional routing",
            question=(
                "Explain LangGraph state management and "
                "conditional routing."
            ),
            search_results=(
                _result("State docs", wiki_a),
                _result("Routing docs", wiki_b),
            ),
            pages={
                wiki_a: (
                    "State is persisted by checkpointers at "
                    "every superstep."
                ),
                wiki_b: (
                    "Conditional routing branches execution "
                    "along edges guarded by predicate functions "
                    "that read the state."
                ),
            },
            research_steps=(
                ResearchStep(
                    kind="tool_call",
                    name="web_search",
                    args={"query": "q"},
                ),
                ResearchStep(kind="text", text="done"),
            ),
            drafts=(
                "LangGraph state management persists state via "
                "checkpointers [S1]. Conditional routing "
                "branches execution along edges guarded by "
                "predicate functions that read the state [S2]."
            ),
            expected=CaseExpectations(
                review_status="approved",
                markers=["checkpointer", "predicate"],
            ),
            rubric=(
                TopicRubric(
                    "state management", ("checkpointer", "persist")
                ),
                TopicRubric(
                    "conditional routing",
                    ("predicate", "edge", "branch"),
                ),
            ),
        )
    )

    # 6 — routing only disclaimed with 'not present' (lexical
    # regression from the October 11 live run).
    cases.append(
        EvalCase(
            case_id="state-routing-disclaimed",
            category="state management and conditional routing",
            question=(
                "Explain LangGraph state management and "
                "conditional routing syntax."
            ),
            search_results=(_result("Docs", wiki_a),),
            pages={
                wiki_a: "State is persisted by checkpointers."
            },
            research_steps=_WS4_STEPS,
            drafts=(
                "Specific details regarding conditional routing "
                "syntax were not present in the retrieved text "
                "excerpts. LangGraph state management "
                "persists state via checkpointers, per [S1]."
            ),
            expected=CaseExpectations(
                review_status="needs_more_research",
                markers=["TOPIC COVERAGE: INCOMPLETE"],
                missing_topics=("conditional", "routing"),
            ),
            rubric=(
                TopicRubric(
                    "conditional routing",
                    ("predicate", "edge", "branch"),
                ),
            ),
        )
    )

    # 7 — Workstream 4: routing merely NAMED. Phase 4E.1
    # (WS-B): the deterministic explanation-depth gate now
    # downgrades the scripted approval to
    # needs_more_research (named-only topic while a page was
    # retrieved), and the semantic rubric independently
    # scores conditional routing inadequate. This case's
    # expectation changed FROM "approved" WITH the fix — it
    # documents that superficial keyword answers are no
    # longer approved, not a weakened assertion.
    cases.append(
        EvalCase(
            case_id="ws4-routing-superficial",
            category="state management and conditional routing",
            question=(
                "Explain LangGraph state management and "
                "conditional routing using official documentation."
            ),
            search_results=(
                _result("Official overview", official),
                _result("Routing guide", wiki_b),
            ),
            pages={
                official: (
                    "LangGraph state is managed through a "
                    "checkpointer that persists state at each "
                    "superstep."
                ),
                wiki_b: (
                    "Conditional routing branches execution "
                    "along edges guarded by predicate functions "
                    "that read the state."
                ),
            },
            research_steps=_WS4_STEPS,
            drafts=(_WS4_DRAFT,),
            expected=CaseExpectations(
                review_status="needs_more_research",
                markers=[
                    "checkpointer",
                    "EXPLANATION DEPTH: SUPERFICIAL",
                ],
                official_status="verified_official",
            ),
            rubric=(
                TopicRubric(
                    "state management",
                    ("checkpointer", "persist"),
                ),
                TopicRubric(
                    "conditional routing",
                    ("predicate", "edge", "branch"),
                ),
            ),
        )
    )

    # 8 — multi-topic question, one topic entirely missing.
    cases.append(
        EvalCase(
            case_id="multi-topic-missing-one",
            category="multi-topic research questions",
            question=(
                "Explain LangGraph streaming, checkpointing "
                "and memory sharing."
            ),
            search_results=(_result("Streaming", wiki_a),),
            pages={
                wiki_a: "Streaming tokens are emitted per superstep."
            },
            research_steps=_WS4_STEPS,
            drafts=(
                "LangGraph streaming emits tokens per "
                "superstep, per [S1]."
            ),
            expected=CaseExpectations(
                review_status="needs_more_research",
                markers=["TOPIC COVERAGE: INCOMPLETE"],
                missing_topics=("checkpointing", "memory"),
            ),
            rubric=(
                TopicRubric(
                    "streaming", ("emit", "token", "stream")
                ),
                TopicRubric(
                    "checkpointing", ("checkpoint", "persist")
                ),
                TopicRubric(
                    "memory sharing", ("share", "memory")
                ),
            ),
        )
    )

    # 9 — ambiguous wording: a pronoun question with no
    # distinctive terms (nothing to require lexically).
    cases.append(
        EvalCase(
            case_id="ambiguous-wording",
            category="requests with ambiguous wording",
            question="Explain how it works.",
            search_results=(_result("General", "https://example.com/general"),),
            pages={"https://example.com/general": "General text."},
            research_steps=_WS4_STEPS,
            drafts="It works as described in [S1].",
            expected=CaseExpectations(
                review_status="approved",
                missing_topics=(),
            ),
        )
    )

    # 10 — contradictory evidence: two pages disagree.
    cases.append(
        EvalCase(
            case_id="contradictory-evidence",
            category="unavailable or contradictory evidence",
            question="What is LangGraph's default retry limit?",
            search_results=(
                _result("Page A", wiki_a),
                _result("Page B", wiki_b),
            ),
            pages={
                wiki_a: "The default retry limit is 3.",
                wiki_b: "The default retry limit is 5.",
            },
            research_steps=_WS4_STEPS,
            drafts=(
                "Sources disagree on LangGraph's default "
                "retry limit: [S1] says 3 while [S2] says "
                "5, so the answer is uncertain."
            ),
            expected=CaseExpectations(
                review_status="approved",
                markers=["disagree"],
            ),
            rubric=(
                TopicRubric(
                    "retry limit", ("disagree", "uncertain", "conflict")
                ),
            ),
        )
    )

    # 11 — incomplete answer is the correct outcome.
    cases.append(
        EvalCase(
            case_id="incomplete-is-correct-outcome",
            category="questions where an incomplete answer is the correct outcome",
            question="Explain LangGraph subgraph nesting depth limits.",
            search_results=(_result("Outline", wiki_a),),
            pages={
                wiki_a: (
                    "Subgraphs nest, but no documented depth "
                    "limit was found in this page."
                )
            },
            research_steps=_WS4_STEPS,
            drafts=(
                "Subgraphs nest; the documented depth limit "
                "was not available in the retrieved sources, "
                "per [S1]."
            ),
            expected=CaseExpectations(
                review_status="needs_more_research",
                markers=["TOPIC COVERAGE: INCOMPLETE"],
                missing_topics=("depth",),
            ),
            rubric=(
                TopicRubric(
                    "nesting depth limits", ("depth", "limit", "nest")
                ),
            ),
        )
    )

    # 12 — all searches fail: controlled no-findings path.
    cases.append(
        EvalCase(
            case_id="search-tool-failure",
            category="tool failure cases",
            question="Explain LangGraph.",
            search_results=(),
            fail_queries=frozenset({"q"}),
            research_steps=_WS4_STEPS,
            drafts="Draft answer.",
            expected=CaseExpectations(
                review_status="needs_more_research",
                markers=["No findings were collected"],
            ),
        )
    )

    # 13 — every fetch fails: unverified banner.
    cases.append(
        EvalCase(
            case_id="fetch-all-fail",
            category="source-retrieval failure cases",
            question="Explain LangGraph.",
            search_results=(_result("Page", wiki_a),),
            fail_fetch_urls=frozenset({wiki_a}),
            research_steps=_WS4_STEPS,
            drafts="LangGraph is a framework, per [S1].",
            expected=CaseExpectations(
                review_status="approved",
                markers=["UNVERIFIED / INCOMPLETE"],
                unverified_marker=True,
            ),
        )
    )

    # 14 — malformed citation (label glued into the URL).
    cases.append(
        EvalCase(
            case_id="malformed-citation",
            category="citation provenance cases",
            question="Explain LangGraph.",
            search_results=(_result("Page", wiki_a),),
            pages={wiki_a: "LangGraph builds graphs."},
            research_steps=_WS4_STEPS,
            drafts=(
                "LangGraph builds graphs, per "
                "https://wiki.example.org/state-management "
                "[snippet]."
            ),
            expected=CaseExpectations(
                review_status="needs_more_research",
                markers=["CITATION QUALITY"],
            ),
        )
    )

    # 15 — fabricated citation identifier.
    cases.append(
        EvalCase(
            case_id="fabricated-sid",
            category="citation provenance cases",
            question="Explain LangGraph.",
            search_results=(_result("Page", wiki_a),),
            pages={wiki_a: "LangGraph builds graphs."},
            research_steps=_WS4_STEPS,
            drafts="LangGraph builds graphs, per [S9].",
            expected=CaseExpectations(
                review_status="needs_more_research",
                markers=["CITATION QUALITY"],
            ),
        )
    )

    # 16 — unknown URL citation.
    cases.append(
        EvalCase(
            case_id="unknown-url-citation",
            category="citation provenance cases",
            question="Explain LangGraph.",
            search_results=(_result("Page", wiki_a),),
            pages={wiki_a: "LangGraph builds graphs."},
            research_steps=_WS4_STEPS,
            drafts=(
                "LangGraph builds graphs, per "
                "https://unknown.example.org/page."
            ),
            expected=CaseExpectations(
                review_status="needs_more_research",
                markers=["CITATION QUALITY"],
            ),
        )
    )

    # 17 — snippets only, below a two-page retrieval
    # threshold: the deterministic force-verify fetches
    # the only candidate, but one page cannot satisfy
    # the two-page requirement — the answer is honestly
    # labeled UNVERIFIED.
    cases.append(
        EvalCase(
            case_id="snippet-only-below-threshold",
            category="source-retrieval failure cases",
            question="Explain LangGraph.",
            search_results=(_result("Page", wiki_a),),
            pages={wiki_a: "LangGraph builds graphs."},
            research_steps=(
                ResearchStep(
                    kind="tool_call",
                    name="web_search",
                    args={"query": "q"},
                ),
                ResearchStep(kind="text", text="done"),
            ),
            settings_overrides={"min_retrieved_sources": 2},
            drafts="LangGraph builds graphs, per [S1].",
            expected=CaseExpectations(
                review_status="approved",
                markers=["UNVERIFIED / INCOMPLETE"],
                unverified_marker=True,
            ),
        )
    )

    # 18 — budget exhaustion: the model keeps searching
    # until the iteration cap; the run stays bounded.
    cases.append(
        EvalCase(
            case_id="budget-exhaustion",
            category="tool failure cases",
            question="Explain LangGraph.",
            search_results=(_result("Page", wiki_a),),
            pages={wiki_a: "LangGraph builds graphs."},
            research_steps=(
                ResearchStep(
                    kind="tool_call",
                    name="web_search",
                    args={"query": "q"},
                ),
            ),
            drafts=("Draft answer.",),
            review_verdicts=("needs_more_research",),
            expected=CaseExpectations(
                review_status="needs_more_research",
            ),
        )
    )

    # 19 — legacy-domain steering: an official question
    # whose planned query targets the legacy domain. The
    # legacy host is still curated first-party, so the
    # page is fetched and the run succeeds — the
    # deterministic assertion is that plan/research
    # prompts steer to the CURRENT domains instead.
    cases.append(
        EvalCase(
            case_id="legacy-domain-steering",
            category="official technical documentation",
            question="LangGraph official documentation.",
            search_results=(
                _result("Legacy", "https://langchain-ai.github.io/langgraph/"),
            ),
            pages={
                "https://langchain-ai.github.io/langgraph/": (
                    "Legacy docs text."
                )
            },
            plan_queries=(
                "site:langchain-ai.github.io langgraph",
            ),
            research_steps=_WS4_STEPS,
            drafts="LangGraph docs, per [S1].",
            expected=CaseExpectations(
                review_status="approved",
                markers=["LangGraph docs"],
                official_status="verified_official",
                official_domain_steering=True,
            ),
        )
    )

    # 20 — rate-limit retry: one rate-limit error, then
    # success; the run recovers.
    cases.append(
        EvalCase(
            case_id="rate-limit-retry",
            category="general technical research",
            question="Explain LangGraph.",
            search_results=(_result("Page", wiki_a),),
            pages={wiki_a: "LangGraph builds graphs."},
            research_steps=_WS4_STEPS,
            drafts="LangGraph builds graphs, per [S1].",
            raise_on_call={1: "rate_limit"},
            expected=CaseExpectations(
                review_status="approved",
            ),
        )
    )

    # 21 — multi-iteration improvement: review sends the
    # run back once; the second pass addresses the gap.
    cases.append(
        EvalCase(
            case_id="multi-iteration-retry",
            category="multi-topic research questions",
            question=(
                "Explain LangGraph state management and "
                "conditional routing."
            ),
            search_results=(
                _result("State", wiki_a),
                _result("Routing", wiki_b),
            ),
            pages={
                wiki_a: "State is persisted by checkpointers.",
                wiki_b: (
                    "Conditional routing branches execution "
                    "along predicate-guarded edges."
                ),
            },
            research_steps=(
                ResearchStep(
                    kind="tool_call",
                    name="web_search",
                    args={"query": "q"},
                ),
                ResearchStep(kind="text", text="done"),
            ),
            drafts=(
                "LangGraph state management persists state "
                "via checkpointers, per [S1].",
                "LangGraph state management persists state "
                "via checkpointers, per [S1]. Conditional "
                "routing branches execution along "
                "predicate-guarded edges, per [S2].",
            ),
            review_verdicts=("needs_more_research", "approved"),
            expected=CaseExpectations(
                review_status="approved",
                markers=["checkpointer", "predicate"],
            ),
            rubric=(
                TopicRubric(
                    "state management", ("checkpointer", "persist")
                ),
                TopicRubric(
                    "conditional routing",
                    ("predicate", "edge", "branch"),
                ),
            ),
        )
    )

    # 22 — WS-A: the model issues a legacy alias argument
    # (``queries`` instead of the canonical ``query``). The
    # executor's narrow, allow-listed normalization renames it
    # (value untouched) and the search executes — if the alias
    # normalization ever regresses, the call is refused as an
    # invalid-arguments schema error, no findings are collected
    # and this case fails.
    cases.append(
        EvalCase(
            case_id="invalid-tool-args-recovery",
            category="tool failure cases",
            question="Explain LangGraph.",
            search_results=(_result("Page", wiki_a),),
            pages={wiki_a: "LangGraph builds graphs."},
            research_steps=(
                ResearchStep(
                    kind="tool_call",
                    name="web_search",
                    args={"queries": "q"},
                ),
                ResearchStep(kind="text", text="done"),
            ),
            drafts="LangGraph builds graphs, per [S1].",
            expected=CaseExpectations(
                review_status="approved",
                markers=["LangGraph builds graphs"],
            ),
        )
    )

    # 23 — WS-B: the first query returns no results (classified
    # ``empty_results`` — the run must not crash or hang), and the
    # model recovers by issuing a different query that succeeds.
    # A regression in search-error classification/recovery breaks
    # the run path and fails the case.
    cases.append(
        EvalCase(
            case_id="search-failure-recovery",
            category="tool failure cases",
            question="Explain LangGraph.",
            search_results=(_result("Page", wiki_a),),
            pages={wiki_a: "LangGraph builds graphs."},
            fail_queries=frozenset({"q"}),
            research_steps=(
                ResearchStep(
                    kind="tool_call",
                    name="web_search",
                    args={"query": "q"},
                ),
                ResearchStep(
                    kind="tool_call",
                    name="web_search",
                    args={"query": "langgraph"},
                ),
                ResearchStep(kind="text", text="done"),
            ),
            # two searches are needed (one fails, one succeeds):
            # lift the per-pass search quota so the second call
            # is in-phase rather than refused by the budget
            settings_overrides={"max_tool_calls_per_research": 4},
            drafts="LangGraph builds graphs, per [S1].",
            expected=CaseExpectations(
                review_status="approved",
                markers=["LangGraph builds graphs"],
            ),
        )
    )

    # --- Phase 4E.1 fixtures (WS-A / WS-B / WS-G) ---
    secops = "https://secops.example.com/overview"
    cicd = "https://cicd.example.com/pipeline-guide"
    earnings = "https://odd.example.com/unrelated-earnings"

    # 24 — WS-B, generalizable second instance of the WS4
    # failure mode: a LONG, keyword-complete, confidently
    # worded answer that only classifies the subject ("is a
    # core capability / a fundamental approach / a standard
    # feature") never explains the mechanism. Verbosity and
    # correct keywords must not be rewarded: the deterministic
    # explanation-depth gate fails it while the rubric scores
    # it inadequate.
    cases.append(
        EvalCase(
            case_id="superficial-keyword-answer",
            category="general technical research",
            question="How does a CI/CD pipeline work?",
            search_results=(_result("Pipeline guide", cicd),),
            pages={
                cicd: (
                    "The pipeline builds, tests and deploys "
                    "every commit through staged environments."
                )
            },
            research_steps=_WS4_STEPS,
            drafts=(
                "A CI/CD pipeline is a core capability for "
                "delivering software reliably. A CI/CD pipeline "
                "is a fundamental approach in modern software "
                "practice. A CI/CD pipeline is a standard "
                "feature of mature engineering organizations. "
                "Teams everywhere rely on this important part "
                "of their day-to-day operations. See [S1].",
            ),
            expected=CaseExpectations(
                review_status="needs_more_research",
                markers=["EXPLANATION DEPTH: SUPERFICIAL"],
            ),
            rubric=(
                TopicRubric(
                    "CI/CD pipeline", ("build", "test", "deploy")
                ),
            ),
        )
    )

    # 25 — WS-A: VERIFIED evidence policy. Every fetch failed,
    # the only support is a search snippet, and the draft cites
    # it: under the verified policy a snippet can NEVER
    # independently support a claim, so deterministic citation
    # validation fails and the run is routed back / labeled.
    cases.append(
        EvalCase(
            case_id="verified-snippet-citation",
            category="citation provenance cases",
            question="Explain LangGraph.",
            search_results=(_result("Page", wiki_a),),
            fail_fetch_urls=frozenset({wiki_a}),
            research_steps=_WS4_STEPS,
            evidence_policy="verified",
            drafts="LangGraph builds graphs, per [S1].",
            expected=CaseExpectations(
                review_status="needs_more_research",
                markers=["CITATION QUALITY"],
            ),
        )
    )

    # 26 — WS-A: the VERIFIED policy succeeds when the cited
    # page was actually retrieved — strictness must not mean
    # suppression of good answers.
    cases.append(
        EvalCase(
            case_id="verified-retrieved-citation",
            category="citation provenance cases",
            question="Explain LangGraph.",
            search_results=(_result("Page", wiki_a),),
            pages={wiki_a: "LangGraph builds graphs."},
            research_steps=_WS4_STEPS,
            evidence_policy="verified",
            drafts="LangGraph builds graphs, per [S1].",
            expected=CaseExpectations(
                review_status="approved",
                require_citations_ok=True,
            ),
        )
    )

    # 27 — WS-A: the BALANCED (default) policy still permits
    # clearly-labeled snippet-only support: discovered sources
    # are useful for discovery/context and are NOT suppressed —
    # the honest UNVERIFIED label carries the weakness.
    cases.append(
        EvalCase(
            case_id="balanced-snippet-allowed",
            category="citation provenance cases",
            question="Explain LangGraph.",
            search_results=(_result("Page", wiki_a),),
            fail_fetch_urls=frozenset({wiki_a}),
            research_steps=_WS4_STEPS,
            evidence_policy="balanced",
            drafts="LangGraph builds graphs, per [S1].",
            expected=CaseExpectations(
                review_status="approved",
                unverified_marker=True,
            ),
        )
    )

    # 28 — WS-G: QUICK depth on a simple definition question.
    # Guards BOTH directions: the run stays approved/cheap, and
    # a legitimate short substantive answer to "What is X?" is
    # never forced into extra research or verbosity by the new
    # explanation-depth gate.
    cases.append(
        EvalCase(
            case_id="quick-mode-secops",
            category="general technical research",
            question="What is SecOps?",
            search_results=(_result("SecOps overview", secops),),
            pages={
                secops: (
                    "SecOps integrates security and IT "
                    "operations teams and workflows to improve "
                    "threat detection and response."
                )
            },
            research_steps=_WS4_STEPS,
            research_depth="quick",
            drafts=(
                "SecOps (security operations) integrates "
                "security and IT operations teams so threats "
                "are detected and responded to faster, per [S1].",
            ),
            expected=CaseExpectations(
                review_status="approved",
                require_citations_ok=True,
            ),
            rubric=(
                TopicRubric(
                    "SecOps", ("integrat", "security", "operations")
                ),
            ),
        )
    )

    # 29 — WS-G: DEEP depth applies its own EXPLICIT budgets
    # (more passes, higher fetch quota, 3 required retrieved
    # pages instead of 1) — the run gathers more independently
    # retrieved evidence per pass than any standard case, while
    # staying inside the unchanged global deadline.
    cases.append(
        EvalCase(
            case_id="deep-mode-budget",
            category="multi-topic research questions",
            question=(
                "Explain LangGraph streaming and checkpointing."
            ),
            search_results=(
                _result("Streaming docs", wiki_a),
                _result("Checkpoint docs", wiki_b),
                _result("Persistence guide", deepwiki),
            ),
            pages={
                wiki_a: (
                    "Streaming emits response tokens as each "
                    "superstep completes."
                ),
                wiki_b: (
                    "Checkpoints persist state at every "
                    "superstep."
                ),
                deepwiki: (
                    "Checkpointers store execution state so an "
                    "interrupted run can resume."
                ),
            },
            research_steps=_WS4_STEPS,
            research_depth="deep",
            drafts=(
                "LangGraph streaming emits response tokens as "
                "each superstep completes, so applications can "
                "display output before the run finishes [S1]. "
                "Checkpointing persists state at every "
                "superstep [S2], which lets an interrupted run "
                "resume from its last saved state [S3].",
            ),
            expected=CaseExpectations(
                review_status="approved",
                markers=["streaming", "Checkpointing"],
                require_citations_ok=True,
                min_retrieved_sources=3,
            ),
            rubric=(
                TopicRubric(
                    "streaming", ("emit", "token", "superstep")
                ),
                TopicRubric(
                    "checkpointing", ("persist", "resume")
                ),
            ),
        )
    )

    # 30 — WS-A: advisory claim-to-source grounding. The cited
    # page was retrieved and the citation is valid, but the
    # claiming sentence shares NO vocabulary with that page —
    # the deterministic lexical check reports low overlap for
    # the reviewer (it never auto-fails: paraphrase is honest
    # sometimes, so semantic judgement stays with the model).
    cases.append(
        EvalCase(
            case_id="citation-mismatch-grounding",
            category="citation provenance cases",
            question="What is SecOps?",
            search_results=(_result("Unrelated page", earnings),),
            pages={
                earnings: (
                    "The quarterly earnings report showed "
                    "revenue growth across all segments this "
                    "year."
                )
            },
            research_steps=_WS4_STEPS,
            drafts=(
                "SecOps integrates security and operations "
                "teams to improve incident response times, "
                "per [S1].",
            ),
            expected=CaseExpectations(
                review_status="approved",
                require_citations_ok=True,
                min_low_overlap_citations=1,
            ),
        )
    )

    return tuple(cases)
