"""Deterministic conditional routing.

Every routing function is PURE: the same state always produces the same
route. They never call LLMs or the network, which makes routing decisions
trivial to unit-test and to audit in production.
"""
from __future__ import annotations

from typing import Literal

from agent.state import GraphState, ReviewStatus

Route = Literal["research", "synthesize", "error", "finalize"]

ROUTE_RESEARCH = "research"
ROUTE_SYNTHESIZE = "synthesize"
ROUTE_ERROR = "error"
ROUTE_FINALIZE = "finalize"


def has_fatal_error(state: GraphState) -> bool:
    """True when research produced errors and nothing usable (no findings)."""
    return bool(state.get("errors")) and not state.get("findings")


def has_sufficient_evidence(state: GraphState, *, min_findings: int) -> bool:
    """True when enough findings were collected to synthesize an answer."""
    return len(state.get("findings") or []) >= min_findings


def research_budget_exhausted(state: GraphState, *, max_iterations: int) -> bool:
    """True when the research loop has spent its allowed iterations."""
    return state.get("iteration", 0) >= max_iterations


def route_after_research(
    state: GraphState,
    *,
    min_findings: int = 3,
    max_iterations: int = 3,
) -> Route:
    """Decide what happens after a research step. First match wins:

    1. ``error``      – every research attempt failed, nothing to work with
    2. ``synthesize`` – enough evidence collected, OR the iteration budget is
                        spent (the synthesize node flags partial evidence)
    3. ``research``   – more evidence needed and budget remains
    """
    if has_fatal_error(state):
        return ROUTE_ERROR
    if has_sufficient_evidence(
        state, min_findings=min_findings
    ) or research_budget_exhausted(state, max_iterations=max_iterations):
        return ROUTE_SYNTHESIZE
    return ROUTE_RESEARCH


def route_after_review(
    state: GraphState,
    *,
    max_iterations: int = 3,
) -> Route:
    """Decide what happens after the quality-review gate.

    Every status is handled explicitly:

    - ``APPROVED``            -> ``finalize`` (answer is good enough)
    - ``FAILED``              -> ``finalize`` (do not retry a failed review)
    - ``NEEDS_MORE_RESEARCH`` -> ``research`` while budget remains,
                                 else ``finalize``
    - anything else (``PENDING``, missing, or unexpected) -> ``finalize``

    Fail-safe rule: an unrecognised status must NEVER silently restart
    research — an unknown state is a bug to surface, not a reason to
    spend more iterations.
    """
    status = state.get("review_status")
    if status in (ReviewStatus.APPROVED, ReviewStatus.FAILED):
        return ROUTE_FINALIZE
    if status == ReviewStatus.NEEDS_MORE_RESEARCH:
        if not research_budget_exhausted(state, max_iterations=max_iterations):
            return ROUTE_RESEARCH
        return ROUTE_FINALIZE
    return ROUTE_FINALIZE
