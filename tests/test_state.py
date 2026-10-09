"""Tests for GraphState: reducers, state updates, and LangGraph merging."""
from langgraph.graph import END, START, StateGraph

from agent.state import (
    Finding,
    GraphState,
    ReviewStatus,
    append_items,
)


def test_append_items_first_update_on_empty_channel():
    finding: Finding = {"text": "first", "source_url": None}
    assert append_items(None, [finding]) == [finding]


def test_append_items_concatenates_instead_of_overwriting():
    current: list[Finding] = [{"text": "a", "source_url": None}]
    new: list[Finding] = [{"text": "b", "source_url": "https://example.com"}]
    assert append_items(current, new) == current + new


def test_append_items_never_mutates_its_inputs():
    current: list[Finding] = [{"text": "a", "source_url": None}]
    new: list[Finding] = [{"text": "b", "source_url": None}]
    merged = append_items(current, new)
    assert len(current) == 1
    assert len(new) == 1
    assert len(merged) == 2


def test_review_status_enum_values():
    assert ReviewStatus.APPROVED.value == "approved"
    assert ReviewStatus.NEEDS_MORE_RESEARCH.value == "needs_more_research"


def test_two_nodes_appending_to_one_field_are_merged():
    """The core reducer guarantee: parallel contributors don't clobber each other."""

    def node_a(state: GraphState) -> dict:
        return {"findings": [{"text": "from A", "source_url": None}]}

    def node_b(state: GraphState) -> dict:
        return {"findings": [{"text": "from B", "source_url": None}]}

    builder = StateGraph(GraphState)
    builder.add_node("a", node_a)
    builder.add_node("b", node_b)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", END)

    result = builder.compile().invoke({})
    assert [f["text"] for f in result["findings"]] == ["from A", "from B"]


def test_parallel_branches_merge_into_shared_state():
    """Genuine fan-out: two branches run in parallel, then converge.

    Both branches append to the same reducer-backed field; the
    convergence node must see the MERGED state.
    """

    def node_a(state: GraphState) -> dict:
        return {"findings": [{"text": "from A", "source_url": None}]}

    def node_b(state: GraphState) -> dict:
        return {"findings": [{"text": "from B", "source_url": None}]}

    def converge(state: GraphState) -> dict:
        return {"iteration": len(state.get("findings") or [])}

    builder = StateGraph(GraphState)
    builder.add_node("a", node_a)
    builder.add_node("b", node_b)
    builder.add_node("converge", converge)
    builder.add_edge(START, "a")
    builder.add_edge(START, "b")
    builder.add_edge("a", "converge")
    builder.add_edge("b", "converge")
    builder.add_edge("converge", END)

    result = builder.compile().invoke({})
    # Parallel execution order is nondeterministic, so compare as sets.
    assert sorted(f["text"] for f in result["findings"]) == ["from A", "from B"]
    assert result["iteration"] == 2  # convergence node saw BOTH updates


def test_plain_fields_are_last_write_wins():
    def node_a(state: GraphState) -> dict:
        return {"iteration": 1}

    def node_b(state: GraphState) -> dict:
        return {"iteration": 2}

    builder = StateGraph(GraphState)
    builder.add_node("a", node_a)
    builder.add_node("b", node_b)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", END)

    result = builder.compile().invoke({})
    assert result["iteration"] == 2
