"""WS-A: tool-schema validation and recovery.

The live incident: Gemini called ``web_search(queries=...)`` while
the callable expected ``query``. These tests pin the contract:

- the schema is enforced BEFORE execution (no partial side effects);
- allow-listed, rename-only aliases are normalized with the value
  untouched — nothing else is repaired;
- schema violations become ONE structured, actionable error with a
  ``validation`` metrics category (distinct from network/provider
  failures);
- the provider transcript stays valid: the ToolMessage answering a
  refused call carries the ORIGINAL tool-call id.
"""
from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from agent.config import Settings
from agent.graph import ResearchPlan, ReviewVerdict, build_research_graph
from agent.metrics import MetricsCollector
from agent.tools import ToolExecutor
from tests.fakes import FakeTools, ScriptedChatModel


def _spy_tool(calls: list[dict[str, Any]]) -> Any:
    def tool(**kwargs: Any) -> str:
        calls.append(kwargs)
        return "ok"

    return tool


def _executor(
    functions: dict[str, Any], metrics: MetricsCollector | None = None
) -> ToolExecutor:
    return ToolExecutor(functions, metrics=metrics)


# -- schema enforcement -------------------------------------------------


def test_missing_required_argument_fails_before_execution():
    """A malformed call must never reach the tool body — the schema
    rejects it and the error names the expected argument contract."""
    calls: list[dict[str, Any]] = []
    executor = _executor({"web_search": _spy_tool(calls)})

    result = executor.execute(
        {"name": "web_search", "args": {"search_term": "langgraph"}}
    )

    assert result["ok"] is False
    assert result["error"] is not None
    assert "invalid arguments for web_search" in result["error"]
    assert "query" in result["error"]  # the contract, named
    assert "Call web_search again" in result["error"]
    assert calls == []  # nothing executed


def test_unexpected_keyword_is_rejected():
    calls: list[dict[str, Any]] = []
    executor = _executor({"web_search": _spy_tool(calls)})

    result = executor.execute(
        {"name": "web_search", "args": {"query": "q", "count": 9}}
    )

    assert result["ok"] is False
    assert "invalid arguments" in (result["error"] or "")
    assert calls == []


def test_empty_query_is_rejected_by_min_length():
    executor = _executor({"web_search": _spy_tool([])})

    result = executor.execute({"name": "web_search", "args": {"query": ""}})

    assert result["ok"] is False
    assert "invalid arguments" in (result["error"] or "")


def test_schema_error_is_recorded_as_validation_category():
    """Schema failures must be distinguishable from network/provider/
    search failures in metrics (fixed vocabulary, no error text)."""
    metrics = MetricsCollector()
    executor = _executor(
        {"web_search": _spy_tool([])}, metrics=metrics
    )

    executor.execute(
        {"name": "web_search", "args": {"bogus": 1}}
    )

    counts = metrics.snapshot()["counts"]["tool_error_categories"]
    assert counts.get("validation") == 1


# -- allow-listed alias normalization -----------------------------------


def test_queries_alias_is_normalized_and_value_untouched():
    calls: list[dict[str, Any]] = []
    executor = _executor({"web_search": _spy_tool(calls)})

    result = executor.execute(
        {"name": "web_search", "args": {"queries": "langgraph state"}}
    )

    assert result["ok"] is True, result["error"]
    # renamed only — the value passes through byte-for-byte
    assert calls == [{"query": "langgraph state"}]


def test_search_query_alias_is_normalized():
    calls: list[dict[str, Any]] = []
    executor = _executor({"web_search": _spy_tool(calls)})

    executor.execute(
        {"name": "web_search", "args": {"search_query": "q"}}
    )

    assert calls == [{"query": "q"}]


def test_fetch_page_url_aliases_are_normalized():
    calls: list[dict[str, Any]] = []
    executor = _executor({"fetch_page": _spy_tool(calls)})

    executor.execute(
        {"name": "fetch_page", "args": {"page_url": "https://example.com/a"}}
    )
    executor.execute(
        {"name": "fetch_page", "args": {"target_url": "https://example.com/b"}}
    )

    assert calls == [
        {"url": "https://example.com/a"},
        {"url": "https://example.com/b"},
    ]


def test_alias_is_not_applied_when_canonical_present():
    """Alias normalization applies ONLY when the canonical field is
    absent — with both present the call is genuinely ambiguous and
    stays a visible schema error (extra='forbid')."""
    calls: list[dict[str, Any]] = []
    executor = _executor({"web_search": _spy_tool(calls)})

    result = executor.execute(
        {"name": "web_search", "args": {"query": "keep", "queries": "drop"}}
    )

    assert result["ok"] is False
    assert "invalid arguments" in (result["error"] or "")
    assert calls == []


def test_unlisted_alias_is_not_repaired():
    """Only the documented alias table is normalized; anything else
    (here a plausible-looking 'keywords') fails visibly instead of
    being silently reinterpreted."""
    calls: list[dict[str, Any]] = []
    executor = _executor({"web_search": _spy_tool(calls)})

    result = executor.execute(
        {"name": "web_search", "args": {"keywords": "langgraph"}}
    )

    assert result["ok"] is False
    assert calls == []


# -- controlled dispatch errors (existing contracts, now categorized) --


def test_unknown_tool_and_non_object_args_stay_controlled():
    executor = _executor({"web_search": _spy_tool([])})

    unknown = executor.execute({"name": "nope", "args": {}})
    assert unknown["ok"] is False
    assert "unknown tool" in (unknown["error"] or "")

    non_object = executor.execute(
        {"name": "web_search", "args": ["not", "a", "dict"]}
    )
    assert non_object["ok"] is False
    assert "must be an object" in (non_object["error"] or "")


# -- graph-level: transcript integrity ----------------------------------


def _tool_call_response(call_id: str, name: str, args: dict[str, Any]):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id}],
    )


def test_refused_call_is_answered_with_original_tool_call_id():
    """A schema-refused call still gets a ToolMessage with the
    provider's original id, so the transcript stays valid and the
    model can see exactly what went wrong."""
    settings = Settings(
        _env_file=None,
        min_findings=1,
        max_research_iterations=1,
        max_tool_calls_per_research=2,
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response(
                "call_bad", "web_search", {"wrong_name": "q"}
            ),
            AIMessage(content="done"),
        ],
        structured_outputs=[ResearchPlan(queries=["q"])],
    )
    graph = build_research_graph(settings, model, FakeTools())

    result = graph.invoke({"question": "question"})

    # the run stayed controlled (no crash) and completed a pass
    assert result["iteration"] == 1
    # the SECOND research invoke saw the tool message answering the
    # refused call, addressed to the original id
    follow_up = model.seen_messages[1]
    tool_messages = [
        m for m in follow_up if isinstance(m, ToolMessage)
    ]
    assert tool_messages, "no ToolMessage produced for the refused call"
    assert tool_messages[0].tool_call_id == "call_bad"
    assert "invalid arguments" in tool_messages[0].content


def test_normalized_alias_call_executes_end_to_end():
    """The exact live mis-call (``queries=``) runs to completion
    through the real graph — search executes, findings appear."""
    settings = Settings(
        _env_file=None,
        min_findings=1,
        max_research_iterations=1,
        max_tool_calls_per_research=2,
    )
    fake_tools = FakeTools(
        search_results=[
            {
                "title": "T",
                "snippet": "S",
                "url": "https://example.com/p",
            }
        ],
        pages={"https://example.com/p": "LangGraph text."},
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response(
                "call_1", "web_search", {"queries": "langgraph"}
            ),
            AIMessage(content="done"),
        ],
        structured_outputs=[
            ResearchPlan(queries=["langgraph"]),
            ReviewVerdict(status="approved", reason="fixture"),
        ],
    )
    graph = build_research_graph(settings, model, fake_tools)

    result = graph.invoke({"question": "question"})

    assert fake_tools.search_calls == ["langgraph"]
    assert result["findings"]
