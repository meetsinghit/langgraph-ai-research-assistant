"""Deterministic demo graph with mock nodes — no LLM, no network.

This module exists to make LangGraph mechanics observable: state
transitions, reducers, conditional edges and bounded loops. Real LLM nodes
and research tools arrive in Phase 3; the routing logic wired in here is
the production logic, unchanged.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agent.config import Settings, get_settings
from agent.routing import (
    ROUTE_ERROR,
    ROUTE_FINALIZE,
    ROUTE_RESEARCH,
    ROUTE_SYNTHESIZE,
    route_after_research,
    route_after_review,
)
from agent.state import Finding, GraphState, ReviewStatus

Node = Callable[[GraphState], dict[str, Any]]

# Deterministic pool of fake findings. Phase 3+ replaces this with real tool
# output; source URLs must always come from actual tool results.
CANNED_FINDINGS: tuple[Finding, ...] = (
    {
        "text": "LangGraph models agents as stateful graphs of nodes and edges.",
        "source_url": "https://langchain-ai.github.io/langgraph/",
    },
    {
        "text": "Nodes are plain functions that return partial state updates.",
        "source_url": "https://langchain-ai.github.io/langgraph/concepts/",
    },
    {
        "text": "Conditional edges choose the next node from the current state.",
        "source_url": "https://langchain-ai.github.io/langgraph/concepts/",
    },
    {
        "text": "Reducers merge updates when several nodes touch one field.",
        "source_url": None,
    },
    {
        "text": "Checkpointers persist graph state across sessions.",
        "source_url": None,
    },
)


def make_plan_node() -> Node:
    """Mock 'plan' node: split the question into deterministic sub-queries."""

    def plan_node(state: GraphState) -> dict[str, Any]:
        question = state["question"]
        return {
            "research_plan": [
                f"search: {question}",
                f"fetch top results for: {question}",
            ]
        }

    return plan_node


def make_research_node(
    *,
    findings_per_call: int = 2,
    fail_first_call: bool = False,
) -> Node:
    """Mock 'research' node: append canned findings to the state.

    ``fail_first_call`` simulates a tool failure so the error route can be
    exercised without any real network access.
    """

    def research_node(state: GraphState) -> dict[str, Any]:
        iteration = state.get("iteration", 0) + 1
        update: dict[str, Any] = {"iteration": iteration}

        if fail_first_call and iteration == 1:
            update["errors"] = ["Search tool failed: simulated tool error"]
            return update

        pool = list(CANNED_FINDINGS)
        start = (iteration - 1) * findings_per_call
        batch = [pool[(start + i) % len(pool)] for i in range(findings_per_call)]
        update["findings"] = batch
        update["sources"] = [f["source_url"] for f in batch if f["source_url"]]
        return update

    return research_node


def make_synthesize_node(*, min_findings: int = 3) -> Node:
    """Mock 'synthesize' node: draft an answer from the collected findings."""

    def synthesize_node(state: GraphState) -> dict[str, Any]:
        findings = state.get("findings") or []
        limited = len(findings) < min_findings
        caveat = " (limited evidence - iteration budget exhausted)" if limited else ""
        answer = f"Draft answer built from {len(findings)} finding(s){caveat}."
        return {"final_answer": answer}

    return synthesize_node


def make_review_node(*, approve_after_iterations: int = 1) -> Node:
    """Mock 'review' gate: deterministic stand-in for an LLM quality judge.

    Approves once the research loop has run at least
    ``approve_after_iterations`` times.
    """

    def review_node(state: GraphState) -> dict[str, Any]:
        approved = state.get("iteration", 0) >= approve_after_iterations
        return {
            "review_status": (
                ReviewStatus.APPROVED
                if approved
                else ReviewStatus.NEEDS_MORE_RESEARCH
            )
        }

    return review_node


def make_error_node() -> Node:
    """Mock error handler: turn failures into a controlled, readable answer."""

    def error_node(state: GraphState) -> dict[str, Any]:
        errors = state.get("errors") or []
        detail = "; ".join(errors) if errors else "unknown error"
        return {
            "final_answer": f"Research could not be completed: {detail}",
            "review_status": ReviewStatus.FAILED,
        }

    return error_node


def make_finalize_node() -> Node:
    """Mock 'finalize' node: terminal pass-through."""

    def finalize_node(state: GraphState) -> dict[str, Any]:
        return {}  # final_answer is already set by synthesize or handle_error

    return finalize_node


_RESEARCH_ROUTES = {
    ROUTE_RESEARCH: "research",
    ROUTE_SYNTHESIZE: "synthesize",
    ROUTE_ERROR: "handle_error",
}

_REVIEW_ROUTES = {
    ROUTE_RESEARCH: "research",
    ROUTE_FINALIZE: "finalize",
}


def build_demo_graph(
    settings: Settings,
    *,
    research_node: Node | None = None,
    review_node: Node | None = None,
) -> CompiledStateGraph:
    """Wire the demo graph. Conditional edges use the REAL routing functions."""

    builder = StateGraph(GraphState)
    builder.add_node("plan", make_plan_node())
    builder.add_node("research", research_node or make_research_node())
    builder.add_node(
        "synthesize", make_synthesize_node(min_findings=settings.min_findings)
    )
    builder.add_node("review", review_node or make_review_node())
    builder.add_node("finalize", make_finalize_node())
    builder.add_node("handle_error", make_error_node())

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "research")
    builder.add_edge("synthesize", "review")
    builder.add_edge("finalize", END)
    builder.add_edge("handle_error", END)

    builder.add_conditional_edges(
        "research",
        lambda state: route_after_research(
            state,
            min_findings=settings.min_findings,
            max_iterations=settings.max_research_iterations,
        ),
        _RESEARCH_ROUTES,
    )
    builder.add_conditional_edges(
        "review",
        lambda state: route_after_review(
            state, max_iterations=settings.max_research_iterations
        ),
        _REVIEW_ROUTES,
    )

    return builder.compile()


def run_demo(
    question: str,
    settings: Settings | None = None,
    **node_overrides: Any,
) -> GraphState:
    """Build the demo graph and run it once — handy for manual exploration."""
    graph = build_demo_graph(settings or get_settings(), **node_overrides)
    return graph.invoke({"question": question})
