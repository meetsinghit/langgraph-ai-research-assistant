"""WS-C: the global run deadline (``MAX_TOTAL_EXECUTION_SECONDS``).

A single monotonic deadline is stamped once per graph invocation and
every node/retry/tool consults it. The tests use FakeClock /
TickingClock so timeout paths are deterministic and never sleep.

Caveat under test: the deadline is COOPERATIVE — checked at operation
boundaries, not a pre-emptive kill switch (in-flight blocking calls
run to completion; see agent/deadline.py docstring).
"""
from __future__ import annotations

from typing import Any

import pytest

from agent.config import Settings
from agent.deadline import RunDeadline
from agent.graph import build_research_graph
from agent.metrics import MetricsCollector
from agent.state import ReviewStatus
from agent.tools import ResearchTools
from tests.fakes import FakeClock, FakeTools, ScriptedChatModel
from tests.test_tool_schema import _tool_call_response

from langchain_core.messages import AIMessage

from agent.graph import ResearchPlan, ReviewVerdict


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "min_findings": 1,
        "max_research_iterations": 2,
        "max_tool_calls_per_research": 2,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


class _TickingClock:
    """A clock that advances ``step`` seconds on EVERY read — used to
    expire a short deadline between two adjacent graph nodes."""

    def __init__(self, step: float, start: float = 1_000.0) -> None:
        self._now = float(start)
        self._step = float(step)

    def __call__(self) -> float:
        self._now += self._step
        return self._now


class _AdvancingTools(FakeTools):
    """FakeTools whose search pushes the fake clock far forward —
    simulating the run's budget being consumed during research."""

    def __init__(self, clock: FakeClock, jump: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._clock = clock
        self._jump = jump

    def web_search(self, query: str) -> list[dict[str, str]]:
        result = super().web_search(query)
        self._clock.advance(self._jump)
        return result


class _BudgetRecordingTools(FakeTools):
    """Records with_budget() propagation from the graph."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.budget_calls: list[dict[str, Any]] = []

    def with_budget(
        self,
        *,
        timeout_seconds: float | None = None,
        deadline: Any = None,
    ) -> "_BudgetRecordingTools":
        self.budget_calls.append(
            {"timeout_seconds": timeout_seconds, "deadline": deadline}
        )
        return self


# -- RunDeadline unit behaviour -----------------------------------------


def test_deadline_expires_exactly_at_total_seconds():
    clock = FakeClock()
    deadline = RunDeadline(5.0, clock=clock)
    assert not deadline.expired()
    clock.advance(4.99)
    assert not deadline.expired()
    clock.advance(0.01)
    assert deadline.expired()
    assert deadline.remaining() == pytest.approx(0.0)


def test_pause_budget_is_clamped_to_remaining_time():
    clock = FakeClock()
    deadline = RunDeadline(2.0, clock=clock)
    assert deadline.pause_budget(1.0) == 1.0
    clock.advance(1.5)  # 0.5s left
    assert deadline.pause_budget(1.0) == pytest.approx(0.5)
    clock.advance(1.0)  # expired
    assert deadline.pause_budget(1.0) == 0.0
    assert deadline.pause_budget(-3) == 0.0


def test_timeout_budget_caps_with_a_usable_floor():
    clock = FakeClock()
    deadline = RunDeadline(10.0, clock=clock)
    # requested below the remaining budget: unchanged
    assert deadline.timeout_budget(3.0) == 3.0
    clock.advance(8.0)  # 2s left
    assert deadline.timeout_budget(3.0) == pytest.approx(2.0)
    clock.advance(1.95)  # 0.05s left: floor keeps the request usable
    assert deadline.timeout_budget(3.0) == pytest.approx(0.1)
    clock.advance(1.0)  # expired: no request may start
    assert deadline.timeout_budget(3.0) == 0.0


def test_non_positive_total_is_rejected():
    with pytest.raises(ValueError):
        RunDeadline(0)
    with pytest.raises(ValueError):
        RunDeadline(-1.0)


# -- with_budget contract ------------------------------------------------


def test_with_budget_returns_fresh_copy_without_mutating_shared_tools():
    settings = Settings(_env_file=None, request_timeout_seconds=30.0)
    shared = ResearchTools(settings)
    clock = FakeClock()
    deadline = RunDeadline(180.0, clock=clock)

    clone = shared.with_budget(timeout_seconds=7.0, deadline=deadline)

    assert clone is not shared
    # the clone caps the request timeout...
    assert clone._settings.request_timeout_seconds == 7.0
    assert clone._deadline is deadline
    # ...while the graph-level toolset is untouched for the next run
    assert shared._settings.request_timeout_seconds == 30.0
    assert shared._deadline is None


# -- graph-level behaviour ------------------------------------------------


def test_happy_path_deadline_is_stamped_and_not_exceeded():
    settings = _settings()
    model = ScriptedChatModel(
        responses=[AIMessage(content="Done.", usage_metadata=None)],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="ok"),
        ],
    )
    metrics = MetricsCollector()
    graph = build_research_graph(settings, model, FakeTools(), metrics=metrics)

    result = graph.invoke({"question": "question"})

    assert result["deadline_total_seconds"] == (
        settings.max_total_execution_seconds
    )
    assert result.get("deadline_exceeded") is False
    snapshot = metrics.snapshot()
    assert snapshot["deadline"]["configured_seconds"] == pytest.approx(
        settings.max_total_execution_seconds
    )
    assert snapshot["deadline"]["exceeded"] is False
    assert snapshot["outcome"]["status"] in (
        "COMPLETED",
        "COMPLETED_WITH_WARNINGS",
    )


def test_deadline_expiring_mid_run_yields_timed_out_partial_sources():
    """The budget runs out during research: synthesis and review must
    NOT run, and the answer must preserve provenance without any
    synthesized claim — reported as TIMED_OUT."""
    clock = FakeClock()
    settings = _settings(max_total_execution_seconds=180.0)
    url = "https://example.com/p"
    tools = _AdvancingTools(
        clock, 1_000.0,  # one search blows far past the 180s budget
        search_results=[{"title": "T", "snippet": "S", "url": url}],
        pages={url: "text"},
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("call_1", "web_search", {"query": "q"}),
            AIMessage(content="done"),
        ],
        structured_outputs=[ResearchPlan(queries=["q"])],
    )
    metrics = MetricsCollector()
    graph = build_research_graph(
        settings, model, tools, metrics=metrics, clock=clock
    )

    result = graph.invoke({"question": "question"})

    assert result.get("deadline_exceeded") is True
    # no reviewer ran after expiry, so no approval was recorded
    assert result.get("review_status") != ReviewStatus.APPROVED
    answer = result["final_answer"]
    assert "Sources gathered before the deadline" in answer
    assert url in answer
    assert "no claims were generated" in answer
    snapshot = metrics.snapshot()
    assert snapshot["deadline"]["exceeded"] is True
    assert snapshot["outcome"]["status"] == "TIMED_OUT"


def test_deadline_expiring_before_research_is_a_controlled_short_circuit():
    """Expiry between plan and research: no research model call is
    ever made and the run terminates with a controlled, readable
    TIMED_OUT error — no crash, no hang."""
    clock = _TickingClock(step=1.0)  # expires the 0.05s budget instantly
    settings = _settings(max_total_execution_seconds=0.05)
    model = ScriptedChatModel(
        responses=[],
        structured_outputs=[ResearchPlan(queries=["q"])],
    )
    metrics = MetricsCollector()
    graph = build_research_graph(
        settings, model, FakeTools(), metrics=metrics, clock=clock
    )

    result = graph.invoke({"question": "question"})

    # plan and research were both skipped before any invocation
    assert model.seen_messages == []
    assert "deadline" in result["final_answer"].lower()
    assert result["final_answer"].startswith(
        "Research could not be completed:"
    )
    snapshot = metrics.snapshot()
    assert snapshot["outcome"]["status"] == "TIMED_OUT"
    assert snapshot["deadline"]["exceeded"] is True


def test_research_pass_receives_a_budgeted_tool_copy():
    """Each research pass caps owned request timeouts by the
    remaining budget via a fresh tool copy (never mutating the
    shared graph-level toolset)."""
    settings = _settings(request_timeout_seconds=7.0)
    tools = _BudgetRecordingTools()
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
    graph = build_research_graph(settings, model, tools)

    graph.invoke({"question": "question"})

    assert tools.budget_calls, "research never budgeted its tools"
    call = tools.budget_calls[0]
    # ~180s of budget remain, so the configured timeout is untouched
    assert call["timeout_seconds"] == pytest.approx(7.0)
    assert isinstance(call["deadline"], RunDeadline)
