"""Research depth and evidence policy — two INDEPENDENT per-run axes.

Phase 4E.1 (WS-G depth, WS-A evidence). Both dimensions travel in
graph state (JSON-safe strings), so they are checkpointed with the
run and survive resume; defaults live here so the CLI, the API and
the evals all resolve an absent/unknown value identically to
``standard`` / ``balanced``.

Research depth (WS-G) changes HOW MUCH research is done and how the
answer is structured — budgets are resolved from *explicit,
configurable* settings (``quick_*`` / ``deep_*`` in
:class:`agent.config.Settings`) and are always bounded by the
unchanged global execution deadline. It never weakens a quality
gate: a longer answer is not a more trustworthy answer.

Evidence policy (WS-A) changes WHAT may support a factual citation —
``balanced`` (default) permits clearly-labeled snippet-only support;
``verified`` requires retrieved-page grounding and fails citation
validation otherwise (deterministically, in
:func:`agent.citations.validate_citations`).

Neither axis is expressed as a numerical confidence score, and
neither is inferred from the answer: both are explicit run inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from agent.config import Settings

__all__ = [
    "DEFAULT_EVIDENCE_POLICY",
    "DEFAULT_RESEARCH_DEPTH",
    "DEPTH_WORD_TARGETS",
    "EVIDENCE_POLICIES",
    "RESEARCH_DEPTHS",
    "DepthPolicy",
    "EvidencePolicy",
    "ResearchDepth",
    "depth_policy",
    "resolve_depth",
    "resolve_evidence_policy",
    "synthesize_depth_guidance",
    "review_depth_expectation",
]

ResearchDepth = Literal["quick", "standard", "deep"]
EvidencePolicy = Literal["balanced", "verified"]

RESEARCH_DEPTHS: tuple[str, ...] = get_args(ResearchDepth)
EVIDENCE_POLICIES: tuple[str, ...] = get_args(EvidencePolicy)

DEFAULT_RESEARCH_DEPTH: ResearchDepth = "standard"
DEFAULT_EVIDENCE_POLICY: EvidencePolicy = "balanced"

#: Target answer length ranges in WORDS (guidance for synthesis, and
#: an expectation handed to the reviewer — never a hard gate in
#: either direction: padding to hit a range is as wrong as truncating
#: a substantively complete short answer).
DEPTH_WORD_TARGETS: dict[str, tuple[int, int]] = {
    "quick": (150, 300),
    "standard": (500, 900),
    "deep": (1200, 2500),
}


def resolve_depth(value: object) -> ResearchDepth:
    """Coerce a state/CLI/API value to a known depth (default: standard)."""
    if isinstance(value, str) and value.strip().lower() in RESEARCH_DEPTHS:
        return value.strip().lower()  # type: ignore[return-value]
    return DEFAULT_RESEARCH_DEPTH


def resolve_evidence_policy(value: object) -> EvidencePolicy:
    """Coerce a state/CLI/API value to a known policy (default: balanced)."""
    if isinstance(value, str) and value.strip().lower() in EVIDENCE_POLICIES:
        return value.strip().lower()  # type: ignore[return-value]
    return DEFAULT_EVIDENCE_POLICY


@dataclass(frozen=True)
class DepthPolicy:
    """Resolved per-run budget and output expectations for one depth.

    Every numeric field is a run-level ceiling derived from
    configuration — nothing here can exceed the global execution
    deadline (``max_total_execution_seconds``), which is unaffected
    by depth and remains the hard safety ceiling.
    """

    depth: ResearchDepth
    plan_query_cap: int
    max_iterations: int
    max_tool_calls: int
    min_findings: int
    min_retrieved_sources: int
    max_evidence_chars: int
    word_target: tuple[int, int]

    @property
    def label(self) -> str:
        return self.depth.upper()


def depth_policy(
    settings: Settings, depth: object = DEFAULT_RESEARCH_DEPTH
) -> DepthPolicy:
    """Resolve the effective policy for a run's depth.

    ``standard`` uses the historical settings as-is (backward
    compatible: existing deployments and evals behave exactly as
    before). ``quick`` and ``deep`` read their own explicit,
    configurable budget fields.
    """
    resolved = resolve_depth(depth)
    if resolved == "quick":
        values = dict(
            plan_query_cap=settings.quick_max_plan_queries,
            max_iterations=settings.quick_max_research_iterations,
            max_tool_calls=settings.quick_max_tool_calls_per_research,
            min_findings=settings.quick_min_findings,
            min_retrieved_sources=settings.quick_min_retrieved_sources,
            max_evidence_chars=settings.max_evidence_chars,
        )
    elif resolved == "deep":
        values = dict(
            plan_query_cap=settings.deep_max_plan_queries,
            max_iterations=settings.deep_max_research_iterations,
            max_tool_calls=settings.deep_max_tool_calls_per_research,
            min_findings=settings.deep_min_findings,
            min_retrieved_sources=settings.deep_min_retrieved_sources,
            max_evidence_chars=settings.deep_max_evidence_chars,
        )
    else:
        values = dict(
            plan_query_cap=settings.max_plan_queries,
            max_iterations=settings.max_research_iterations,
            max_tool_calls=settings.max_tool_calls_per_research,
            min_findings=settings.min_findings,
            min_retrieved_sources=settings.min_retrieved_sources,
            max_evidence_chars=settings.max_evidence_chars,
        )
    return DepthPolicy(
        depth=resolved,
        word_target=DEPTH_WORD_TARGETS[resolved],
        **values,
    )


def synthesize_depth_guidance(policy: DepthPolicy) -> str:
    """Structure/length guidance appended to the synthesis prompt.

    Question-appropriate structure only: sections are included when
    the question and evidence justify them; padding to reach the
    target range, or inventing content to fill a section, is
    explicitly forbidden. When evidence is insufficient the answer
    must get SHORTER and say why — depth never overrides honesty.
    """
    low, high = policy.word_target
    common = (
        f"Target length: roughly {low}-{high} words when the question "
        "and evidence justify it — a range, not a quota. Never pad to "
        "reach it; if the evidence is thin, write less and say what is "
        "missing."
    )
    if policy.depth == "quick":
        return (
            "Research depth: QUICK — optimize for speed and cost. Give "
            "a direct answer with the key facts and essential citations "
            f"only. {common} Structure: a short direct explanation; no "
            "long report sections unless the question demands them."
        )
    if policy.depth == "deep":
        return (
            "Research depth: DEEP — the question warrants thorough "
            "treatment. Structure the answer with only the sections the "
            "subject justifies, for example: executive summary; "
            "definitions and fundamentals; how it works (mechanisms / "
            "architecture); practical use cases; worked examples; "
            "comparisons; benefits, limitations and risks; "
            "implementation or operational considerations; key "
            "takeaways. For technical subjects emphasize mechanisms and "
            "examples; for business subjects emphasize use cases, "
            "stakeholders, economics and outcomes. Where relevant, note "
            "alternative viewpoints, assumptions and uncertainties. "
            f"{common} Draw on multiple independently retrieved sources "
            "so claims rest on evidence rather than a single page."
        )
    return (
        "Research depth: STANDARD (default) — produce a well-developed, "
        "structured answer. Use only the sections the question justifies, "
        "for example: a short executive summary; core concepts and "
        "mechanisms; relevant subtopics; practical examples; "
        "comparisons where useful; benefits/limitations where "
        "applicable; evidence-grounded conclusions with citations. "
        f"{common}"
    )


def review_depth_expectation(policy: DepthPolicy) -> str:
    """Depth expectation handed to the reviewer — advisory, not a gate.

    The reviewer must never approve BECAUSE of length, and never
    reject a concise quick-mode answer that actually addresses the
    question; it judges substance, coverage and evidentiary support
    against the requested depth.
    """
    low, high = policy.word_target
    return (
        f"Requested research depth: {policy.label} (target roughly "
        f"{low}-{high} words when justified). Judge whether the depth "
        "MATCHES the request: a quick-mode answer is complete when it "
        "directly answers the question with the essential facts — do "
        "not demand a long report. For standard/deep, check that the "
        "topic is developed sufficiently (mechanisms, examples, "
        "limitations where relevant) rather than merely listed. Never "
        "approve because an answer merely reaches a target length or "
        "contains expected headings, and never reject a fully "
        "substantive answer merely because it is short."
    )
