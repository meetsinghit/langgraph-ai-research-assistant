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


def _multi_tool_call_response(
    calls: list[tuple[str, str]],
) -> AIMessage:
    """One LLM response requesting SEVERAL tool calls at once
    (Gemini frequently batches parallel web_search calls)."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "web_search", "args": {"query": query}, "id": call_id}
            for call_id, query in calls
        ],
    )


def test_tool_call_budget_limits_batched_calls_in_one_response():
    """One response requested 6 calls with a budget of 4:
    exactly 4 invocations are executed, never 6."""
    settings = _settings(
        min_findings=10,
        max_research_iterations=2,
        max_tool_calls_per_research=4,
    )
    fake_tools = FakeTools(
        search_results=[_search_result("r", "https://example.com/r")]
    )
    model = ScriptedChatModel(
        responses=[
            _multi_tool_call_response(
                [
                    ("c1", "q1"),
                    ("c2", "q2"),
                    ("c3", "q3"),
                    ("c4", "q4"),
                    ("c5", "q5"),
                    ("c6", "q6"),
                ]
            ),
            AIMessage(content="done"),
            AIMessage(content="Draft answer."),
        ],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="ok"),
        ],
    )

    graph = build_research_graph(settings, model, fake_tools)
    result = graph.invoke({"question": "question"})

    assert len(fake_tools.search_calls) == 4  # not 6
    assert len(result["findings"]) == 4
    assert result["iteration"] == 2
    assert result["review_status"] == ReviewStatus.APPROVED
    assert result["final_answer"].startswith("Draft answer.")
    # only 4 of the 10 required findings -> partial-evidence flag
    assert "Partial evidence" in result["final_answer"]


def test_tool_call_budget_limits_batches_across_responses():
    """Batches of 2 calls across responses: the budget counts
    executed invocations, so a third batch is never reached."""
    settings = _settings(
        min_findings=10,
        max_research_iterations=1,
        max_tool_calls_per_research=4,
    )
    fake_tools = FakeTools(
        search_results=[_search_result("r", "https://example.com/r")]
    )
    model = ScriptedChatModel(
        responses=[
            _multi_tool_call_response([("c1", "q1"), ("c2", "q2")]),
            _multi_tool_call_response([("c3", "q3"), ("c4", "q4")]),
            _multi_tool_call_response([("c5", "q5"), ("c6", "q6")]),
            AIMessage(content="Draft answer."),
        ],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="ok"),
        ],
    )

    graph = build_research_graph(settings, model, fake_tools)
    result = graph.invoke({"question": "question"})

    assert len(fake_tools.search_calls) == 4  # 2 + 2; batch 3 never runs
    assert model.call_index == 3  # 2 research invokes + 1 synthesis
    assert len(result["findings"]) == 4
    assert result["iteration"] == 1
    assert result["review_status"] == ReviewStatus.APPROVED


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


def _block_answer(text: str, signature: str = "sig") -> AIMessage:
    """A Gemini-style answer carrying structured content blocks:
    visible text plus a thought-signature block that must be
    preserved on internal messages but never rendered as prose."""
    return AIMessage(
        content=[
            {"type": "text", "text": text},
            {"type": "thought_signature", "thought_signature": signature},
        ]
    )


def test_answer_renders_gemini_content_blocks_as_clean_text():
    """Structured content blocks (Gemini 3) render as clean,
    human-readable Markdown — no thought-signature metadata in
    the final answer."""
    settings = _settings()
    fake_tools = FakeTools(
        search_results=[_search_result("r", "https://example.com/r")]
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1"),
            AIMessage(content="done"),
            _block_answer("**Answer** in Markdown."),
        ],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="ok"),
        ],
    )

    graph = build_research_graph(settings, model, fake_tools)
    result = graph.invoke({"question": "question"})

    answer = result["final_answer"]
    assert answer == "**Answer** in Markdown."
    assert "sig" not in answer          # signature stripped
    assert "thought_signature" not in answer


def test_internal_messages_preserve_thought_signature_metadata():
    """The research loop must NOT strip Gemini's structured
    content blocks / thought signatures from messages it feeds
    back to the model — those are required model-side metadata."""
    settings = _settings()
    fake_tools = FakeTools(
        search_results=[_search_result("r", "https://example.com/r")]
    )
    signed = _block_answer("thinking out loud", signature="must-survive")
    # a real tool-calling turn that ALSO carries signed content —
    # only this gets echoed back to the model on the next step
    signed.tool_calls = [
        {"name": "web_search", "args": {"query": "q"}, "id": "c1", "type": "tool_call"}
    ]
    model = ScriptedChatModel(
        responses=[
            signed,
            AIMessage(content="done"),
            AIMessage(content="Draft answer."),
        ],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="ok"),
        ],
    )

    graph = build_research_graph(settings, model, fake_tools)
    graph.invoke({"question": "question"})

    # the assistant turn echoed back to the model on the second
    # research step retains its original structured content
    assistant_turns = [
        m
        for batch in model.seen_messages
        for m in batch
        if getattr(m, "type", "") == "ai" and m is signed
    ]
    assert assistant_turns, "signed assistant message never re-fed to model"
    for m in assistant_turns:
        assert m.content == signed.content


def test_fetch_page_retrieves_and_inspects_primary_source():
    """The workflow can retrieve a primary-source page with
    fetch_page and turn its text into a retrieved finding."""
    settings = _settings()
    fake_tools = FakeTools(page_text="Primary source full text.")
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1", name="fetch_page", args={"url": "https://docs.example.com/"}),
            AIMessage(content="done"),
            AIMessage(content="Answer grounded in the fetched page."),
        ],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="ok"),
        ],
    )

    graph = build_research_graph(settings, model, fake_tools)
    result = graph.invoke({"question": "question"})

    assert fake_tools.fetch_calls == ["https://docs.example.com/"]
    assert len(result["findings"]) == 1
    assert result["findings"][0]["source_url"] == "https://docs.example.com/"
    # fetch_page evidence is inspected content, not a discovered snippet
    assert result["findings"][0].get("evidence") == "retrieved"
    assert result["sources"] == ["https://docs.example.com/"]


def test_search_evidence_is_distinguished_from_retrieved_content():
    """Findings carry provenance so a snippet (discovered URL) is
    never conflated with a retrieved/inspected page."""
    settings = _settings()
    fake_tools = FakeTools(
        search_results=[_search_result("snippet", "https://example.com/s")]
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1"),
            AIMessage(content="done"),
            AIMessage(content="Draft answer."),
        ],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="ok"),
        ],
    )

    graph = build_research_graph(settings, model, fake_tools)
    result = graph.invoke({"question": "question"})

    assert result["findings"][0].get("evidence") == "snippet"


def test_failed_search_query_recovers_via_requery():
    """A query that yields no results surfaces as a meaningful
    tool error; the workflow recovers by issuing a different
    query rather than crashing."""
    settings = _settings()
    fake_tools = FakeTools(
        search_results=[_search_result("ok", "https://example.com/ok")],
        fail_queries={"bad site:example.com over-specific"},
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1", args={"query": "bad site:example.com over-specific"}),
            _tool_call_response("c2", args={"query": "broad query"}),
            AIMessage(content="done"),
            AIMessage(content="Recovered answer."),
        ],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="ok"),
        ],
    )

    graph = build_research_graph(settings, model, fake_tools)
    result = graph.invoke({"question": "question"})

    assert fake_tools.search_calls == [
        "bad site:example.com over-specific",
        "broad query",
    ]
    # the failed query yielded no findings; the retry did
    assert result["findings"][0]["source_url"] == "https://example.com/ok"
    assert result["review_status"] == ReviewStatus.APPROVED
    assert not result.get("errors")


def test_synthesis_prompt_carries_every_finding_with_its_source():
    """Claim-to-source consistency: the synthesis prompt lists
    every finding text next to its source URL, so the drafted
    answer can only be grounded in cited evidence."""
    settings = _settings(min_findings=1)
    fake_tools = FakeTools(
        search_results=[
            _search_result("Fact One", "https://example.com/one"),
            _search_result("Fact Two", "https://example.com/two"),
        ]
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1"),
            AIMessage(content="done"),
            AIMessage(content="Draft answer."),
        ],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="ok"),
        ],
    )

    graph = build_research_graph(settings, model, fake_tools)
    graph.invoke({"question": "question"})

    # the synthesis node is the third model call; its prompt is
    # the last message
    synthesis_prompt = model.seen_messages[2][-1].content
    for url in ("https://example.com/one", "https://example.com/two"):
        assert url in synthesis_prompt
    assert "Fact One" in synthesis_prompt
    assert "Fact Two" in synthesis_prompt
