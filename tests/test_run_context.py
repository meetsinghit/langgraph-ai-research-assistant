"""Phase 4D engine adaptation: runtime-only RunContext.

Covers the separation of LIVE per-invocation objects (monotonic
deadline/cancel flag, progress sink) from durable graph state:

- the deadline object never enters state (state keeps only the
  serializable ``deadline_total_seconds`` primitive)
- cooperative cancellation reuses the deadline machinery and is
  reported as CANCELLED (never mislabeled TIMED_OUT)
- progress events are emitted content-free through the sink
- follow-up ``conversation_context`` reaches prompts as advisory,
  clearly-fenced context
- the error-node path (which bypasses finalize) redacts
  credential-shaped text (the Phase 4C exit-backstop fix)

Everything runs offline against the scripted fakes.
"""
from __future__ import annotations

from typing import Any

from agent.config import Settings
from agent.deadline import RunDeadline
from agent.graph import (
    ResearchPlan,
    ReviewVerdict,
    build_research_graph,
    make_error_node,
)
from agent.metrics import MetricsCollector
from agent.runtime import RunContext, get_run_context
from tests.fakes import FakeClock, FakeTools, ScriptedChatModel
from tests.test_tool_schema import _tool_call_response

from langchain_core.messages import AIMessage

FAKE_KEY = "AIza" + "Sy0123456789abcdefghijklmnopqrstuvw"


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "min_findings": 1,
        "max_research_iterations": 2,
        "max_tool_calls_per_research": 2,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


# -- RunContext unit behaviour ------------------------------------------


def test_get_run_context_is_none_outside_a_graph_run():
    assert get_run_context() is None


def test_emit_swallows_sink_failures():
    def broken_sink(event: dict[str, Any]) -> None:
        raise RuntimeError("transport down")

    context = RunContext(sink=broken_sink)
    context.emit({"type": "stage.started", "node": "plan"})  # must not raise


def test_cancel_makes_the_deadline_expire_without_waiting():
    clock = FakeClock()
    deadline = RunDeadline(180.0, clock=clock)
    assert not deadline.expired()
    deadline.cancel()
    deadline.cancel()  # idempotent
    assert deadline.cancelled is True
    assert deadline.expired() is True
    # no new operation may start once cancelled
    assert deadline.pause_budget(5.0) == 0.0
    assert deadline.timeout_budget(30.0) == 0.0


# -- graph-level behaviour ------------------------------------------------


def test_state_carries_only_the_serializable_budget_primitive():
    settings = _settings()
    model = ScriptedChatModel(
        responses=[AIMessage(content="Done.", usage_metadata=None)],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="ok"),
        ],
    )
    graph = build_research_graph(settings, model, FakeTools())

    result = graph.invoke({"question": "question"})

    assert result["deadline_total_seconds"] == (
        settings.max_total_execution_seconds
    )
    # the LIVE RunDeadline must never appear in state — a checkpoint
    # of this state has to be JSON-serializable
    assert "deadline" not in result
    for value in result.values():
        assert not isinstance(value, RunDeadline)


def test_cancelled_run_terminates_as_cancelled_with_honest_wording():
    """A cooperative cancel stops the run before research and is
    reported CANCELLED — never TIMED_OUT, never 'deadline' wording."""
    clock = FakeClock()
    settings = _settings()
    model = ScriptedChatModel(
        responses=[],
        structured_outputs=[ResearchPlan(queries=["q"])],
    )
    metrics = MetricsCollector()
    graph = build_research_graph(settings, model, FakeTools(), metrics=metrics)
    context = RunContext(deadline=RunDeadline(180.0, clock=clock))
    context.deadline.cancel()

    result = graph.invoke({"question": "question"}, context=context)

    assert result.get("deadline_exceeded") is True
    answer = result["final_answer"]
    assert answer.startswith("Research could not be completed:")
    assert "cancelled" in answer.lower()
    assert "deadline expired" not in answer.lower()
    # no model call ever ran
    assert model.seen_messages == []
    snapshot = metrics.snapshot()
    assert snapshot["outcome"]["status"] == "CANCELLED"


def test_happy_path_emits_content_free_stage_and_tool_events():
    settings = _settings()
    url = "https://example.com/doc"
    tools = FakeTools(
        search_results=[{"title": "Doc", "snippet": "S", "url": url}],
        pages={url: "Fetched page content."},
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("call_1", "web_search", {"query": "q"}),
            AIMessage(content="done"),
        ],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="ok"),
        ],
    )
    events: list[dict[str, Any]] = []
    graph = build_research_graph(settings, model, tools)
    context = RunContext(
        deadline=RunDeadline(180.0), sink=events.append
    )

    graph.invoke({"question": "question"}, context=context)

    types = [event["type"] for event in events]
    # every wired node reports its stage boundary
    for node in ("start", "plan", "research", "synthesize", "review", "finalize"):
        assert f"stage.started:{node}" in [
            f"{e['type']}:{e.get('node')}" for e in events
        ]
        assert f"stage.finished:{node}" in [
            f"{e['type']}:{e.get('node')}" for e in events
        ]
    # tool + source lifecycle inside research
    assert types.count("tool.started") == types.count("tool.finished")
    tools_seen = [e["tool"] for e in events if e["type"] == "tool.finished"]
    assert "web_search" in tools_seen
    assert "fetch_page" in tools_seen
    discovered = [e for e in events if e["type"] == "source.discovered"]
    assert [e["url"] for e in discovered] == [url]
    assert discovered[0]["title"] == "Doc"
    retrieved = [e for e in events if e["type"] == "source.retrieved"]
    assert [e["url"] for e in retrieved] == [url]
    # ordering: research's tools happen between its stage boundaries
    def index_where(predicate: Any) -> int:
        for i, event in enumerate(events):
            if predicate(event):
                return i
        raise AssertionError("expected event not found")

    research_started = index_where(
        lambda e: e["type"] == "stage.started" and e.get("node") == "research"
    )
    first_tool = index_where(lambda e: e["type"] == "tool.started")
    research_finished = index_where(
        lambda e: e["type"] == "stage.finished" and e.get("node") == "research"
    )
    assert research_started < first_tool < research_finished
    # content-free guarantee: no event carries page text
    for event in events:
        assert "Fetched page content." not in str(event)


def test_followup_context_reaches_prompts_but_not_the_question():
    settings = _settings()
    model = ScriptedChatModel(
        responses=[AIMessage(content="Done.", usage_metadata=None)],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="ok"),
        ],
    )
    graph = build_research_graph(settings, model, FakeTools())

    result = graph.invoke(
        {
            "question": "follow-up question",
            "conversation_context": "Earlier you asked about X.",
        }
    )

    plan_prompt = model.seen_structured_messages[0][1].content
    assert "Earlier you asked about X." in plan_prompt
    assert "NOT evidence" in plan_prompt
    research_prompt = model.seen_messages[0][1].content
    assert "Earlier you asked about X." in research_prompt
    assert result["conversation_context"] == "Earlier you asked about X."


def test_oversized_followup_context_is_truncated_in_prompts():
    settings = _settings()
    model = ScriptedChatModel(
        responses=[AIMessage(content="Done.", usage_metadata=None)],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="ok"),
        ],
    )
    graph = build_research_graph(settings, model, FakeTools())

    graph.invoke(
        {
            "question": "follow-up question",
            "conversation_context": "A" * 50_000,
        }
    )

    plan_prompt = model.seen_structured_messages[0][1].content
    assert len(plan_prompt) < 10_000


def test_error_node_redacts_credentials_bypassing_finalize():
    """Two-layer guarantee on the handle_error path (which never
    reaches finalize):

    1. the provider exception text no longer enters state at all —
       only the exception TYPE does — so the answer carries neither
       the credential nor a redaction marker for it;
    2. if a credential-shaped value DOES arrive in state errors (any
       future/other path), the error node's own exit backstop
       redacts it before it can become the answer.
    """
    settings = _settings()
    model = ScriptedChatModel(
        responses=[AIMessage(content="never used")],
        structured_outputs=[ResearchPlan(queries=["q"])],
        raise_on_call={0: RuntimeError(f"upstream exploded: {FAKE_KEY}")},
    )
    metrics = MetricsCollector()
    graph = build_research_graph(settings, model, FakeTools(), metrics=metrics)

    result = graph.invoke({"question": "question"})

    answer = result["final_answer"]
    # layer 1: raw exception text never entered state
    assert FAKE_KEY not in answer
    assert answer.startswith("Research could not be completed:")
    assert answer == (
        "Research could not be completed: LLM provider failed: RuntimeError"
    )

    # layer 2: the backstop still redacts a credential that arrives
    # in state errors directly (bypassing the type-only construction)
    backstop = make_error_node(metrics)
    state_result = backstop(
        {"errors": [f"LLM provider failed: {FAKE_KEY}"]}  # type: ignore[arg-type]
    )
    backstop_answer = state_result["final_answer"]
    assert FAKE_KEY not in backstop_answer
    assert "[REDACTED-CREDENTIAL]" in backstop_answer
    assert metrics.snapshot()["security"]["answer_redactions"] >= 1
