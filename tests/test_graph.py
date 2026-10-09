"""End-to-end graph tests with a scripted LLM (no network/API)."""
import pytest
from langchain_core.messages import AIMessage

from agent.config import Settings
from agent.graph import ResearchPlan, ReviewVerdict, build_research_graph
from agent.state import ReviewStatus
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
            {
                "name": name,
                "args": args or {"query": "q"},
                "id": call_id,
            }
        ],
    )


def test_happy_path_plan_research_synthesize_review_finalize():
    settings = _settings()
    fake_tools = FakeTools(
        search_results=[
            _search_result("LangGraph docs", "https://langchain-ai.github.io/langgraph/")
        ]
    )
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": "q"},
                        "id": "call_1",
                    }
                ],
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
            ),
            AIMessage(
                content="Research complete.",
                usage_metadata={
                    "input_tokens": 12, "output_tokens": 3, "total_tokens": 15
                },
            ),
            AIMessage(
                content="Draft answer about LangGraph.",
                usage_metadata={
                    "input_tokens": 20, "output_tokens": 8, "total_tokens": 28
                },
            ),
        ],
        structured_outputs=[
            ResearchPlan(queries=["query one", "query two"]),
            ReviewVerdict(status="approved", reason="sufficient evidence"),
        ],
    )

    graph = build_research_graph(settings, model, fake_tools)
    result = graph.invoke({"question": "What is LangGraph?"})

    assert result["research_plan"] == ["query one", "query two"]
    assert result["iteration"] == 1
    assert len(result["findings"]) == 1
    assert (
        result["findings"][0]["source_url"]
        == "https://langchain-ai.github.io/langgraph/"
    )
    assert result["sources"] == ["https://langchain-ai.github.io/langgraph/"]
    assert result["review_status"] == ReviewStatus.APPROVED
    assert result["final_answer"] == "Draft answer about LangGraph."
    assert result["tokens_in"] == 10 + 12 + 20
    assert result["tokens_out"] == 5 + 3 + 8
    assert result["estimated_cost_usd"] == 0.0  # free-tier model
    assert not result.get("errors")


def test_provider_failure_ends_with_clear_error():
    settings = _settings()
    fake_tools = FakeTools()
    model = ScriptedChatModel(
        responses=[],
        structured_outputs=[ResearchPlan(queries=["q"])],
        raise_on_call={0: RuntimeError("provider down")},
    )

    graph = build_research_graph(settings, model, fake_tools)
    result = graph.invoke({"question": "What is LangGraph?"})

    assert result["final_answer"].startswith("Research could not be completed")
    assert any("provider failed" in error for error in result["errors"])
    assert result["review_status"] == ReviewStatus.FAILED


def test_tool_call_budget_and_iteration_budget():
    settings = _settings(min_findings=5, max_research_iterations=2)
    fake_tools = FakeTools(
        search_results=[_search_result("r", "https://example.com/r")]
    )
    model = ScriptedChatModel(responses=[_tool_call_response("call")])

    graph = build_research_graph(settings, model, fake_tools)
    result = graph.invoke({"question": "question"})

    assert result["iteration"] == 2  # iteration budget hit
    assert len(result["findings"]) == 4  # 2 tool calls x 2 passes
    assert len(fake_tools.search_calls) == 4
    assert "Partial evidence" in result["final_answer"]


def test_unknown_tool_call_is_handled_gracefully():
    settings = _settings(min_findings=3, max_research_iterations=2)
    fake_tools = FakeTools()
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("call_1", name="unknown_tool", args={}),
            AIMessage(content="done"),
        ],
        structured_outputs=[ResearchPlan(queries=["q"])],
    )

    graph = build_research_graph(settings, model, fake_tools)
    result = graph.invoke({"question": "question"})

    # unknown tool produced a controlled error result, not a crash;
    # the loop continued and terminated within budget
    assert result["iteration"] == 2
    assert result["final_answer"] == "No findings were collected during research."


def test_review_gate_sends_work_back_to_research():
    settings = _settings(min_findings=1, max_research_iterations=3)
    fake_tools = FakeTools(
        search_results=[_search_result("r", "https://example.com/r")]
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1"),
            AIMessage(content="done"),
            AIMessage(content="Draft answer v1."),
            _tool_call_response("c2"),
            AIMessage(content="done"),
            AIMessage(content="Draft answer v2."),
        ],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="needs_more_research", reason="thin evidence"),
            ReviewVerdict(status="approved", reason="good"),
        ],
    )

    graph = build_research_graph(settings, model, fake_tools)
    result = graph.invoke({"question": "question"})

    assert result["iteration"] == 2  # first draft was sent back
    assert len(result["findings"]) == 2
    assert result["review_status"] == ReviewStatus.APPROVED
    assert result["final_answer"] == "Draft answer v2."


def test_rate_limit_is_retried_once():
    settings = _settings()
    fake_tools = FakeTools(
        search_results=[_search_result("r", "https://example.com/r")]
    )
    model = ScriptedChatModel(
        responses=[
            AIMessage(content="ignored"),
            _tool_call_response("c1"),
            AIMessage(content="done"),
            AIMessage(content="Draft answer."),
        ],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="ok"),
        ],
        raise_on_call={0: RuntimeError("429 Too Many Requests")},
    )

    graph = build_research_graph(
        settings, model, fake_tools, rate_limit_retry_delay=0.0
    )
    result = graph.invoke({"question": "question"})

    assert not result.get("errors")
    assert len(result["findings"]) == 1
    assert result["final_answer"] == "Draft answer."
    assert model.call_index == 4  # one failed + three successful invokes


def test_plan_falls_back_to_raw_question():
    settings = _settings(min_findings=3, max_research_iterations=2)
    fake_tools = FakeTools()
    model = ScriptedChatModel(
        responses=[AIMessage(content="done")],
        # wrong object type -> plan parsing fails -> fallback
        structured_outputs=[ReviewVerdict(status="approved", reason="x")],
    )

    graph = build_research_graph(settings, model, fake_tools)
    result = graph.invoke({"question": "What is LangGraph?"})

    assert result["research_plan"] == ["What is LangGraph?"]
    assert not result.get("errors")


def test_failed_review_finalizes_without_restarting():
    settings = _settings(min_findings=1, max_research_iterations=3)
    fake_tools = FakeTools(
        search_results=[_search_result("r", "https://example.com/r")]
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1"),
            AIMessage(content="done"),
            AIMessage(content="Draft answer."),
        ],
        structured_outputs=[ResearchPlan(queries=["q"]), "not-a-verdict-object"],
    )

    graph = build_research_graph(settings, model, fake_tools)
    result = graph.invoke({"question": "question"})

    assert result["review_status"] == ReviewStatus.FAILED
    assert any("review failed" in error for error in result["errors"])
    assert result["iteration"] == 1  # no silent restart
