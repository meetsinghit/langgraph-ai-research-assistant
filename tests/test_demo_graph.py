"""End-to-end demo-graph tests: state transitions with mock nodes (no LLM/API)."""
from agent.config import Settings
from agent.demo_graph import (
    build_demo_graph,
    make_research_node,
    make_review_node,
)
from agent.state import ReviewStatus


def make_settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_happy_path_research_then_synthesize_then_finalize():
    """Route taken: plan -> research -> research -> synthesize -> review -> finalize."""
    settings = make_settings(min_findings=3, max_research_iterations=3)
    graph = build_demo_graph(
        settings, research_node=make_research_node(findings_per_call=2)
    )

    result = graph.invoke({"question": "What is LangGraph?"})

    assert result["iteration"] == 2
    assert len(result["findings"]) == 4
    assert len(result["sources"]) == 3  # first four canned findings: three have URLs
    assert result["review_status"] == ReviewStatus.APPROVED
    assert result["final_answer"] == "Draft answer built from 4 finding(s)."
    assert not result.get("errors")  # reducer channels start as empty lists


def test_tool_failure_routes_to_error_handler():
    settings = make_settings(min_findings=3, max_research_iterations=3)
    graph = build_demo_graph(
        settings, research_node=make_research_node(fail_first_call=True)
    )

    result = graph.invoke({"question": "What is LangGraph?"})

    assert result["errors"] == ["Search tool failed: simulated tool error"]
    assert result["review_status"] == ReviewStatus.FAILED
    assert result["final_answer"].startswith("Research could not be completed")


def test_budget_exhaustion_synthesizes_with_caveat():
    settings = make_settings(min_findings=3, max_research_iterations=2)
    graph = build_demo_graph(
        settings, research_node=make_research_node(findings_per_call=1)
    )

    result = graph.invoke({"question": "What is LangGraph?"})

    assert result["iteration"] == 2  # loop stopped exactly at the budget
    assert len(result["findings"]) == 2  # below min_findings=3
    assert "limited evidence" in result["final_answer"]
    assert result["review_status"] == ReviewStatus.APPROVED


def test_review_gate_can_send_work_back_to_research():
    settings = make_settings(min_findings=1, max_research_iterations=5)
    graph = build_demo_graph(
        settings,
        research_node=make_research_node(findings_per_call=2),
        review_node=make_review_node(approve_after_iterations=2),
    )

    result = graph.invoke({"question": "What is LangGraph?"})

    assert result["review_status"] == ReviewStatus.APPROVED
    assert result["iteration"] == 2  # first draft was rejected and re-researched
    assert len(result["findings"]) == 4


def test_pathological_research_node_still_terminates():
    """A research node that never produces findings must not loop forever."""
    settings = make_settings(min_findings=3, max_research_iterations=2)
    graph = build_demo_graph(
        settings, research_node=make_research_node(findings_per_call=0)
    )

    result = graph.invoke({"question": "What is LangGraph?"})

    assert result["iteration"] == 2
    assert "limited evidence" in result["final_answer"]
