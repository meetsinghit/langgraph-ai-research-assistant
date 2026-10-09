"""Tests for deterministic conditional routing and loop termination."""
from agent.routing import (
    has_fatal_error,
    has_sufficient_evidence,
    research_budget_exhausted,
    route_after_research,
    route_after_review,
)
from agent.state import Finding, GraphState, ReviewStatus


def finding(text: str = "evidence") -> Finding:
    return {"text": text, "source_url": "https://example.com"}


def test_sufficient_evidence_routes_to_synthesize():
    state: GraphState = {"findings": [finding() for _ in range(3)], "iteration": 1}
    assert route_after_research(state, min_findings=3, max_iterations=3) == "synthesize"


def test_insufficient_evidence_with_budget_loops_to_research():
    state: GraphState = {"findings": [finding()], "iteration": 1}
    assert route_after_research(state, min_findings=3, max_iterations=3) == "research"


def test_exhausted_budget_routes_to_synthesize_not_research():
    state: GraphState = {"findings": [finding()], "iteration": 3}
    assert route_after_research(state, min_findings=3, max_iterations=3) == "synthesize"


def test_total_failure_routes_to_error():
    state: GraphState = {"errors": ["search failed"], "iteration": 1}
    assert route_after_research(state, min_findings=3, max_iterations=3) == "error"


def test_partial_failure_with_findings_is_not_fatal():
    state: GraphState = {
        "findings": [finding() for _ in range(3)],
        "errors": ["one search failed"],
        "iteration": 1,
    }
    assert route_after_research(state, min_findings=3, max_iterations=3) == "synthesize"


def test_approved_review_terminates():
    state: GraphState = {"review_status": ReviewStatus.APPROVED, "iteration": 1}
    assert route_after_review(state, max_iterations=3) == "finalize"


def test_review_asking_for_more_evidence_loops_back():
    state: GraphState = {"review_status": ReviewStatus.NEEDS_MORE_RESEARCH, "iteration": 1}
    assert route_after_review(state, max_iterations=3) == "research"


def test_review_loop_is_bounded_by_budget():
    state: GraphState = {"review_status": ReviewStatus.NEEDS_MORE_RESEARCH, "iteration": 3}
    assert route_after_review(state, max_iterations=3) == "finalize"


def test_routing_is_deterministic():
    state: GraphState = {"findings": [finding()], "iteration": 1}
    first = route_after_research(state, min_findings=3, max_iterations=3)
    second = route_after_research(state, min_findings=3, max_iterations=3)
    assert first == second == "research"


def test_research_loop_terminates_within_budget():
    """Simulate a greedy loop: research always under-delivers; the router must stop it."""
    state: GraphState = {"findings": [], "iteration": 0}
    route = None
    for _ in range(100):  # generous bound: the router must exit long before this
        state["iteration"] += 1
        state["findings"].append(finding())
        route = route_after_research(state, min_findings=3, max_iterations=2)
        if route != "research":
            break
    assert route == "synthesize"
    assert state["iteration"] == 2  # never exceeded the budget


def test_helpers():
    assert has_fatal_error({"errors": ["x"]}) is True
    assert has_fatal_error({"errors": ["x"], "findings": [finding()]}) is False
    assert has_sufficient_evidence({"findings": [finding()]}, min_findings=1) is True
    assert research_budget_exhausted({"iteration": 3}, max_iterations=3) is True
