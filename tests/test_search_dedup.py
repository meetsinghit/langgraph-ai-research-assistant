"""Phase 4E.2 (WS-C): per-run identical-query search dedup.

A follow-up research pass (or a second call within one pass) that
repeats an ALREADY SUCCESSFULLY searched query is served from the
run's live query cache — no search-backend hit, identical results,
and the saved work is visible in metrics (``counts.search_cache_hits``)
instead of silently inflating "searches". Failures are never cached
(the query-recovery ladder must stay able to retry), the cache is
run-scoped (a fresh invocation gets a fresh cache), and without a
run context the wrapper behaves exactly like the pre-4E.2 code.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage

from agent.config import Settings
from agent.graph import (
    _DiscoveredSources,
    _policy_guarded_functions,
    _search_query_key,
)
from agent.graph import ResearchPlan, ReviewVerdict, build_research_graph
from agent.metrics import MetricsCollector
from agent.tools import ToolError
from tests.fakes import FakeTools, ScriptedChatModel


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "min_findings": 1,
        "max_research_iterations": 3,
        "max_tool_calls_per_research": 2,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _search_result(title: str, url: str) -> dict[str, str]:
    return {"title": title, "snippet": "snippet", "url": url}


def _tool_call_response(
    call_id: str, name: str = "web_search", args: dict | None = None
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args or {"query": "q"}, "id": call_id}
        ],
    )


def _guarded_search(
    search, *, cache=None, metrics=None
):
    guarded = _policy_guarded_functions(
        {"web_search": search},
        _DiscoveredSources(),
        search_cache=cache,
        metrics=metrics,
    )
    return guarded["web_search"]


def test_search_query_key_is_conservative():
    # casefold + whitespace collapse only
    assert _search_query_key(" LangGraph   DOCS ") == "langgraph docs"
    # operators/site filters change what the backend returns and are
    # PRESERVED — never normalized away
    assert _search_query_key("site:example.com LangGraph") != _search_query_key(
        "LangGraph"
    )


def test_cache_serves_isolated_copies():
    calls: list[str] = []

    def search(query: str):
        calls.append(query)
        return [{"title": "t", "snippet": "s", "url": "https://example.com/x"}]

    guarded = _guarded_search(search, cache={})
    first = guarded("q")
    # a downstream consumer mutating its result must not corrupt the cache
    first[0]["title"] = "MUTATED"
    second = guarded("q")
    assert second[0]["title"] == "t"
    assert calls == ["q"], "second identical call must not hit the backend"


def test_failures_are_never_cached():
    calls: list[str] = []

    def search(query: str):
        calls.append(query)
        if len(calls) == 1:
            raise ToolError("backend down")
        return [{"title": "t", "snippet": "s", "url": "https://example.com/x"}]

    guarded = _guarded_search(search, cache={})
    with pytest.raises(ToolError):
        guarded("q")
    result = guarded("q")  # retry must reach the backend again
    assert calls == ["q", "q"]
    assert result[0]["title"] == "t"


def test_empty_results_are_never_cached():
    calls: list[str] = []

    def search(query: str):
        calls.append(query)
        return []

    guarded = _guarded_search(search, cache={})
    guarded("q")
    guarded("q")
    assert calls == ["q", "q"], "empty results must stay retryable"


def test_without_a_cache_the_wrapper_is_transparent():
    calls: list[str] = []

    def search(query: str):
        calls.append(query)
        return [{"title": "t", "snippet": "s", "url": "https://example.com/x"}]

    guarded = _guarded_search(search, cache=None)
    guarded("q")
    guarded("q")
    assert calls == ["q", "q"], "no run context = pre-4E.2 behavior"


def test_second_pass_repeating_a_query_is_served_from_cache():
    settings = _settings(min_findings=5, max_research_iterations=2)
    fake_tools = FakeTools(
        search_results=[_search_result("r", "https://example.com/r")]
    )
    # both passes spend 1 search + 1 fetch on the SAME query — the
    # classic follow-up-pass pattern observed in live runs
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1"),
            _tool_call_response(
                "c2", name="fetch_page", args={"url": "https://example.com/r"}
            ),
            _tool_call_response("c3"),
            _tool_call_response(
                "c4", name="fetch_page", args={"url": "https://example.com/r"}
            ),
            AIMessage(content="Draft answer."),
        ],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="ok"),
        ],
    )
    metrics = MetricsCollector()

    graph = build_research_graph(settings, model, fake_tools, metrics=metrics)
    result = graph.invoke({"question": "question"})

    assert result["iteration"] == 2
    # the backend was searched ONCE; the second pass repeat came from cache
    assert fake_tools.search_calls == ["q"]
    assert len(fake_tools.fetch_calls) == 2, "fetches are never cached"
    assert len(result["findings"]) == 4, "cached results still yield findings"
    counts = metrics.snapshot()["counts"]
    assert counts["searches"] == 2, "both executed calls are counted"
    assert counts["search_cache_hits"] == 1
    assert counts["failed_searches"] == 0


def test_repeat_query_within_one_pass_is_served_from_cache():
    settings = _settings(
        max_tool_calls_per_research=4, max_research_iterations=1
    )
    fake_tools = FakeTools(
        search_results=[_search_result("r", "https://example.com/r")]
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1"),
            _tool_call_response("c2"),  # identical query, same pass
            _tool_call_response(
                "c3", name="fetch_page", args={"url": "https://example.com/r"}
            ),
            AIMessage(content="Draft answer."),
        ],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="ok"),
        ],
    )
    metrics = MetricsCollector()

    graph = build_research_graph(settings, model, fake_tools, metrics=metrics)
    graph.invoke({"question": "question"})

    assert fake_tools.search_calls == ["q"]
    counts = metrics.snapshot()["counts"]
    assert counts["searches"] == 2
    assert counts["search_cache_hits"] == 1


def test_cache_normalizes_case_and_whitespace_only():
    settings = _settings(max_research_iterations=2, min_findings=5)
    fake_tools = FakeTools(
        search_results=[_search_result("r", "https://example.com/r")]
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1", args={"query": "LangGraph  Docs"}),
            _tool_call_response(
                "c2", name="fetch_page", args={"url": "https://example.com/r"}
            ),
            # same query, different case/spacing — a cache hit
            _tool_call_response("c3", args={"query": "langgraph docs"}),
            _tool_call_response(
                "c4", name="fetch_page", args={"url": "https://example.com/r"}
            ),
            AIMessage(content="Draft answer."),
        ],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="ok"),
        ],
    )
    metrics = MetricsCollector()

    graph = build_research_graph(settings, model, fake_tools, metrics=metrics)
    graph.invoke({"question": "question"})

    assert fake_tools.search_calls == ["LangGraph  Docs"]
    assert metrics.snapshot()["counts"]["search_cache_hits"] == 1


def test_distinct_queries_both_reach_the_backend():
    settings = _settings(max_research_iterations=2, min_findings=5)
    fake_tools = FakeTools(
        search_results=[_search_result("r", "https://example.com/r")]
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1", args={"query": "alpha"}),
            _tool_call_response(
                "c2", name="fetch_page", args={"url": "https://example.com/r"}
            ),
            _tool_call_response("c3", args={"query": "beta"}),
            _tool_call_response(
                "c4", name="fetch_page", args={"url": "https://example.com/r"}
            ),
            AIMessage(content="Draft answer."),
        ],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="ok"),
        ],
    )
    metrics = MetricsCollector()

    graph = build_research_graph(settings, model, fake_tools, metrics=metrics)
    graph.invoke({"question": "question"})

    assert fake_tools.search_calls == ["alpha", "beta"]
    assert metrics.snapshot()["counts"]["search_cache_hits"] == 0


def test_failed_searches_are_retried_not_served_from_cache():
    settings = _settings(max_research_iterations=2)
    fake_tools = FakeTools(
        search_results=[_search_result("r", "https://example.com/r")],
        fail_queries={"q"},
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1"),
            AIMessage(content="pass one done"),
            _tool_call_response("c2"),
            AIMessage(content="pass two done"),
        ],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="ok"),
        ],
    )
    metrics = MetricsCollector()

    graph = build_research_graph(settings, model, fake_tools, metrics=metrics)
    graph.invoke({"question": "question"})

    # the failed query reached the backend BOTH times — a failure is
    # never cached, so the recovery ladder can retry it
    assert fake_tools.search_calls == ["q", "q"]
    counts = metrics.snapshot()["counts"]
    assert counts["searches"] == 2
    assert counts["failed_searches"] == 2
    assert counts["search_cache_hits"] == 0


def test_cache_is_per_invocation_not_per_graph():
    """Two invocations of the SAME compiled graph must not share a
    cache — the run context (and therefore the cache) is fresh per
    invocation, so one run can never serve another's queries."""
    settings = _settings(max_research_iterations=1)
    fake_tools = FakeTools(
        search_results=[_search_result("r", "https://example.com/r")]
    )
    # the scripted model replays sequentially across ALL plain
    # invokes (research rounds AND synthesis), so each invocation
    # gets its own search + stop + draft triple
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1"),
            AIMessage(content="Draft answer about LangGraph."),
            AIMessage(content="Answer one."),
            _tool_call_response("c2"),
            AIMessage(content="Draft answer about LangGraph."),
            AIMessage(content="Answer two."),
        ],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="ok"),
        ],
        structured_replay="cycle",
    )
    metrics = MetricsCollector()

    graph = build_research_graph(settings, model, fake_tools, metrics=metrics)
    graph.invoke({"question": "What is LangGraph?"})
    graph.invoke({"question": "What is LangGraph?"})

    assert fake_tools.search_calls == ["q", "q"]
    assert metrics.snapshot()["counts"]["search_cache_hits"] == 0
