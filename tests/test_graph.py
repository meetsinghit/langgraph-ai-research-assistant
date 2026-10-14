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
    # 1 snippet + 1 deterministic verification fetch of the same URL
    assert len(result["findings"]) == 2
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
    # each pass spends its 2-call budget as 1 search + 1 fetch
    # (the budget is partitioned between discovery and verification)
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1"),
            _tool_call_response("c2", name="fetch_page", args={"url": "https://example.com/r"}),
            _tool_call_response("c3"),
            _tool_call_response("c4", name="fetch_page", args={"url": "https://example.com/r"}),
            AIMessage(content="Draft answer."),
        ]
    )

    graph = build_research_graph(settings, model, fake_tools)
    result = graph.invoke({"question": "question"})

    assert result["iteration"] == 2  # iteration budget hit
    assert len(result["findings"]) == 4  # 2 tool calls x 2 passes
    # WS-C (Phase 4E.2): the second pass repeats the SAME query, so
    # it is served from the per-run cache — the backend is searched
    # once, both executed calls still count, and the saved work is
    # visible as a cache hit.
    assert len(fake_tools.search_calls) == 1  # 1 backend search total
    assert len(fake_tools.fetch_calls) == 2  # 1 fetch per pass (never cached)
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

    # the search quota halves the budget: only 2 of the 4 slots can
    # be searches, the model's extra search requests were refused
    assert len(fake_tools.search_calls) == 2
    assert len(result["findings"]) == 3  # 2 snippets + 1 forced fetch
    assert result["iteration"] == 2
    assert result["review_status"] == ReviewStatus.APPROVED
    assert result["final_answer"].startswith("Draft answer.")
    # fewer findings than the 10 required -> partial-evidence flag
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

    # 2 searches executed (quota = half of the 4-call budget); the
    # second batch's search requests were refused in the fetch phase
    assert len(fake_tools.search_calls) == 2  # 2 + refused; batch 3 never runs
    assert model.call_index == 3  # 2 research invokes + 1 synthesis
    assert len(result["findings"]) == 2
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
    assert len(result["findings"]) == 3  # 2 from pass 1 + 1 from pass 2
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
    assert len(result["findings"]) == 2  # snippet + forced fetch
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
    fetch_page and turn its text into a retrieved finding.

    The URL is first surfaced by web_search (Phase 4C policy: fetch
    only accepts search-discovered sources).
    """
    settings = _settings()
    fake_tools = FakeTools(
        page_text="Primary source full text.",
        search_results=[
            {
                "title": "Primary source",
                "snippet": "snippet",
                "url": "https://docs.example.com/",
            }
        ],
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c0", name="web_search", args={"query": "q"}),
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
    # search surfaced the URL (snippet finding) and the fetch added
    # the inspected copy: two findings, one deduplicated source.
    assert len(result["findings"]) == 2
    assert result["findings"][0].get("evidence") != "retrieved"
    assert result["findings"][1]["source_url"] == "https://docs.example.com/"
    # fetch_page evidence is inspected content, not a discovered snippet
    assert result["findings"][1].get("evidence") == "retrieved"
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


# ---------- deterministic search-to-fetch workflow ----------


def test_search_to_fetch_phase_transition_and_budget_split():
    """With the default 4-call budget the workflow is deterministic:
    up to 2 searches, then fetch-only. The toolset handed to the
    model changes with the phase, and the model is told why."""
    settings = _settings(max_tool_calls_per_research=4)
    fake_tools = FakeTools(
        search_results=[
            _search_result("Fact One", "https://example.com/one"),
            _search_result("Fact Two", "https://example.com/two"),
        ],
        page_text="Inspected page text.",
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1", args={"query": "q1"}),
            _tool_call_response("c2", args={"query": "q2"}),
            _tool_call_response(
                "c3", name="fetch_page", args={"url": "https://example.com/one"}
            ),
            _tool_call_response(
                "c4", name="fetch_page", args={"url": "https://example.com/two"}
            ),
            AIMessage(content="Final answer."),
        ],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="ok"),
        ],
    )

    graph = build_research_graph(settings, model, fake_tools)
    result = graph.invoke({"question": "question"})

    # exactly 2 searches then 2 fetches — the budget was partitioned
    assert fake_tools.search_calls == ["q1", "q2"]
    assert fake_tools.fetch_calls == [
        "https://example.com/one",
        "https://example.com/two",
    ]
    # the model saw both tools first, then only fetch_page
    assert model.tool_bind_history[0] == ["web_search", "fetch_page"]
    assert model.tool_bind_history[2] == ["fetch_page"]
    assert model.tool_bind_history[3] == ["fetch_page"]
    # the phase transition was explained to the model
    assert any(
        "web_search is no longer available" in str(getattr(m, "content", ""))
        for m in model.seen_messages[2]
    )
    # 4 snippet findings (2 results x 2 searches) + 2 retrieved
    # findings from the fetched pages
    assert len(result["findings"]) == 6
    retrieved = [f for f in result["findings"] if f.get("evidence") == "retrieved"]
    assert len(retrieved) == 2
    # verified (retrieved >= MIN_RETRIEVED_SOURCES) -> no label
    assert result["final_answer"] == "Final answer."


def test_tool_budget_never_spends_every_slot_on_searches():
    """A model that keeps requesting searches is refused after the
    search quota — the rest of the budget cannot be consumed by
    searches while source URLs are available for verification."""
    settings = _settings(max_tool_calls_per_research=4)
    fake_tools = FakeTools(
        search_results=[_search_result("r", "https://example.com/r")]
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1", args={"query": "q1"}),
            _tool_call_response("c2", args={"query": "q2"}),
            _tool_call_response("c3", args={"query": "q3"}),
            _tool_call_response("c4", args={"query": "q4"}),
            AIMessage(content="Draft answer."),
        ],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="ok"),
        ],
    )

    graph = build_research_graph(settings, model, fake_tools)
    result = graph.invoke({"question": "question"})

    # never more than 2 searches with a 4-call budget
    assert len(fake_tools.search_calls) == 2
    assert fake_tools.fetch_calls == []  # the model never verified
    # the refused calls were answered with a controlled tool message
    tool_texts = [
        str(getattr(m, "content", ""))
        for batch in model.seen_messages
        for m in batch
        if getattr(m, "type", "") == "tool"
    ]
    assert any("web_search is not available" in text for text in tool_texts)
    # 15 snippets would not satisfy the threshold either:
    # zero retrieved pages -> the answer is labeled UNVERIFIED
    assert "UNVERIFIED" in result["final_answer"]


def test_failed_page_retrieval_leaves_answer_unverified():
    """When page verification fails, the workflow attempts it
    deterministically, then returns a labeled unverified answer
    instead of claiming full evidence verification."""
    settings = _settings(max_tool_calls_per_research=4)
    fake_tools = FakeTools(
        search_results=[
            _search_result("A", "https://example.com/a"),
            _search_result("B", "https://example.com/b"),
        ],
        fail_fetch=True,
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1", args={"query": "q1"}),
            _tool_call_response("c2", args={"query": "q2"}),
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

    # verification was ATTEMPTED despite every fetch failing
    assert len(fake_tools.fetch_calls) >= 1
    assert all(f.get("evidence") == "snippet" for f in result["findings"])
    # no crash; an honest label rather than a false "verified"
    assert "UNVERIFIED" in result["final_answer"]
    assert result["review_status"] == ReviewStatus.APPROVED


def test_min_retrieved_sources_threshold_is_configurable():
    """MIN_RETRIEVED_SOURCES=3: fetching two pages is still not
    enough, so the answer carries the unverified label with counts."""
    settings = _settings(
        max_tool_calls_per_research=4, min_retrieved_sources=3
    )
    fake_tools = FakeTools(
        search_results=[
            _search_result("A", "https://example.com/a"),
            _search_result("B", "https://example.com/b"),
        ],
        page_text="Inspected.",
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1", args={"query": "q1"}),
            _tool_call_response(
                "c2", name="fetch_page", args={"url": "https://example.com/a"}
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

    retrieved = [
        f for f in result["findings"] if f.get("evidence") == "retrieved"
    ]
    assert len(retrieved) == 2  # a (model) + b (forced); still < 3
    assert "UNVERIFIED" in result["final_answer"]
    assert "2 of 3" in result["final_answer"]


def test_retrieved_threshold_met_means_no_label():
    """Default threshold of 1: once a page is fetched, the answer
    is verified and carries no warning label."""
    settings = _settings(max_tool_calls_per_research=4)
    fake_tools = FakeTools(
        search_results=[
            _search_result("A", "https://example.com/a"),
            _search_result("B", "https://example.com/b"),
        ],
        page_text="Inspected.",
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1", args={"query": "q1"}),
            _tool_call_response(
                "c2", name="fetch_page", args={"url": "https://example.com/a"}
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

    assert "UNVERIFIED" not in result["final_answer"]
    assert result["final_answer"] == "Draft answer."


def test_official_source_is_verified_first_when_requested():
    """When the user explicitly asks for official documentation,
    first-party pages are verified before third-party ones."""
    question = "LangGraph official documentation"
    third_party = "https://blog.example.com/langgraph-tutorial"
    official = "https://langchain-ai.github.io/langgraph/docs/"
    settings = _settings(max_tool_calls_per_research=4)
    fake_tools = FakeTools(
        search_results=[
            _search_result("Tutorial", third_party),
            _search_result("Official docs", official),
        ],
        page_text="Official docs text.",
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1"),
            AIMessage(content="done"),
            AIMessage(content=f"Answer citing {official}."),
        ],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="ok"),
        ],
    )

    graph = build_research_graph(settings, model, fake_tools)
    result = graph.invoke({"question": question})

    # the deterministic verification step fetched the first-party
    # page first, even though the third-party one was discovered first
    assert fake_tools.fetch_calls == [official]

    # synthesis was explicitly told to prioritize the retrieved
    # official source, and an answer citing it passes the citation
    # gate without being marked
    human_contents = [
        str(message.content)
        for invocation in model.seen_messages
        for message in invocation
    ]
    assert any("Official/first-party sources gathered" in c for c in human_contents)
    assert result["review_status"] == ReviewStatus.APPROVED
    assert "CITATION QUALITY" not in result["final_answer"]


def test_official_docs_citing_only_third_party_snippets_is_marked_and_sent_back():
    """Live-run regression: the official page was retrieved, but the
    answer cited only a snippet-based third-party URL. The
    citation-quality check fails deterministically, the answer is
    marked, and even an LLM 'approved' verdict is overridden.
    (URL provenance only — this is not a factual-correctness claim.)"""
    question = "LangGraph official documentation"
    third_party = "https://blog.example.com/langgraph-tutorial"
    official = "https://langchain-ai.github.io/langgraph/docs/"
    settings = _settings(max_tool_calls_per_research=4, max_research_iterations=1)
    fake_tools = FakeTools(
        search_results=[
            _search_result("Tutorial", third_party),
            _search_result("Official docs", official),
        ],
        page_text="Official docs text.",
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1"),
            AIMessage(content="done"),
            AIMessage(content=f"Use this tutorial instead: {third_party}"),
        ],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="looks fine"),
        ],
    )

    graph = build_research_graph(settings, model, fake_tools)
    result = graph.invoke({"question": question})

    # the official page was still force-retrieved by the budgeted step
    assert fake_tools.fetch_calls == [official]
    # the answer is clearly marked as insufficiently verified...
    assert "CITATION QUALITY" in result["final_answer"]
    assert third_party in result["final_answer"]
    # ...and the deterministic gate overrode the scripted approval;
    # with the iteration budget spent, routing finalizes instead of
    # restarting research (budgeting unchanged)
    assert result["review_status"] == ReviewStatus.NEEDS_MORE_RESEARCH


def test_unknown_cited_url_fails_citation_quality_and_marks_answer():
    """A URL the research never gathered must not pass as a citation."""
    settings = _settings(max_tool_calls_per_research=4, max_research_iterations=1)
    fake_tools = FakeTools(
        search_results=[_search_result("Page", "https://example.com/a")]
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1"),
            AIMessage(content="done"),
            AIMessage(
                content="See https://not-gathered.example/page for the answer."
            ),
        ],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="looks fine"),
        ],
    )

    graph = build_research_graph(settings, model, fake_tools)
    result = graph.invoke({"question": "question"})

    assert fake_tools.fetch_calls == ["https://example.com/a"]
    assert "CITATION QUALITY" in result["final_answer"]
    # unknown citations are not silently waved through
    assert result["review_status"] == ReviewStatus.NEEDS_MORE_RESEARCH


def test_review_prompt_carries_citation_report_and_semantic_disclaimer():
    """The reviewer receives the deterministic citation report AND is
    told it only proves provenance, so semantic support checking
    stays the reviewer's job."""
    settings = _settings(max_tool_calls_per_research=2)
    fake_tools = FakeTools(
        search_results=[_search_result("Page", "https://example.com/a")]
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1"),
            AIMessage(content="done"),
            AIMessage(content="See https://example.com/a."),
        ],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="ok"),
        ],
    )

    graph = build_research_graph(settings, model, fake_tools)
    graph.invoke({"question": "question"})

    contents = [
        str(message.content)
        for invocation in model.seen_structured_messages
        for message in invocation
    ]
    review_prompts = [c for c in contents if "Draft answer:" in c]
    assert review_prompts, "review node was never invoked"
    assert "Citation validation" in review_prompts[-1]
    assert "does NOT verify" in review_prompts[-1]
    # system prompt: reviewer must judge semantic support itself
    assert any(
        "assess that support semantically yourself" in c for c in contents
    )


# ---------- canonical citations & topic completeness ----------


def test_synthesis_prompt_uses_source_identifiers_not_labelled_urls():
    """Root-cause regression: the findings display used to be
    'url [evidence]', which the model imitated inside citations
    ('https://... [snippet]'). Findings now carry [S#] identifiers
    and provenance labels live in a separately rendered table."""
    settings = _settings(min_findings=1)
    fake_tools = FakeTools(
        search_results=[_search_result("Fact", "https://example.com/a")]
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

    synthesis_prompt = model.seen_messages[2][-1].content
    assert "- [S1] " in synthesis_prompt
    assert "Fact" in synthesis_prompt
    assert "Sources (cite findings by their identifier" in synthesis_prompt
    assert "- [S1] https://example.com/a [retrieved]" in synthesis_prompt
    # the system prompt forbids model-side provenance labels
    system_prompt = str(model.seen_messages[2][0].content)
    assert "Never append provenance labels" in system_prompt


def test_malformed_citation_answer_is_marked_and_sent_back():
    """A citation with a provenance label glued into the URL fails
    the deterministic gate even when the LLM reviewer approves."""
    settings = _settings(max_tool_calls_per_research=4, max_research_iterations=1)
    fake_tools = FakeTools(
        search_results=[_search_result("Page", "https://example.com/a")]
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1"),
            AIMessage(content="done"),
            AIMessage(content="Details at https://example.com/a [snippet]."),
        ],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="ok"),
        ],
    )

    graph = build_research_graph(settings, model, fake_tools)
    result = graph.invoke({"question": "question"})

    assert "CITATION QUALITY" in result["final_answer"]
    assert result["review_status"] == ReviewStatus.NEEDS_MORE_RESEARCH


def test_missing_requested_topic_is_marked_and_drives_followup_research():
    """Live regression: the question explicitly asked for conditional
    routing, the draft never mentioned it, and the reviewer approved
    it anyway. The deterministic coverage check marks the answer and
    the next research pass is told to target the missing topics."""
    settings = _settings(max_tool_calls_per_research=2, max_research_iterations=2)
    fake_tools = FakeTools(
        search_results=[_search_result("LangGraph", "https://example.com/graphs")]
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1"),
            AIMessage(content="done"),
            AIMessage(content="LangGraph builds graphs from nodes."),
        ],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="ok"),
        ],
    )

    graph = build_research_graph(settings, model, fake_tools)
    result = graph.invoke({"question": "Explain LangGraph conditional routing"})

    # deterministic marker + state carries the missing topics
    assert "TOPIC COVERAGE" in result["final_answer"]
    assert result["missing_topics"] == ["conditional", "routing"]
    # the follow-up research pass was told to target them
    followup_prompts = [
        str(batch[-1].content)
        for batch in model.seen_messages
        if batch and "Follow-up focus" in str(batch[-1].content)
    ]
    assert followup_prompts, "follow-up research prompt never sent"
    assert "conditional" in followup_prompts[0]
    assert "routing" in followup_prompts[0]
    # coverage failure overrode the scripted approval
    assert result["review_status"] == ReviewStatus.NEEDS_MORE_RESEARCH


def test_official_source_cited_by_identifier_is_approved_unmarked():
    question = "LangGraph official documentation"
    third_party = "https://blog.example.com/langgraph-tutorial"
    official = "https://langchain-ai.github.io/langgraph/docs/"
    settings = _settings(max_tool_calls_per_research=4)
    fake_tools = FakeTools(
        search_results=[
            _search_result("Tutorial", third_party),
            _search_result("Official docs", official),
        ],
        page_text="Official docs text.",
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1"),
            AIMessage(content="done"),
            AIMessage(content="See [S2] for LangGraph details."),
        ],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="ok"),
        ],
    )

    graph = build_research_graph(settings, model, fake_tools)
    result = graph.invoke({"question": question})

    # [S1]=tutorial snippet, [S2]=official page (upgraded to
    # retrieved by the deterministic fetch)
    assert result["review_status"] == ReviewStatus.APPROVED
    assert "CITATION QUALITY" not in result["final_answer"]
    assert "TOPIC COVERAGE" not in result["final_answer"]


def test_live_repro_official_not_cited_and_disclaimer_only_topics_fail():
    """Exact live-run reproduction (both failures, final iteration):

    1. Question asks for official documentation; S2 is the current
       official docs.langchain.com overview and was fetched
       [retrieved], yet the draft cites only [S1]/[S3]/[S4]
       (discovered). The deterministic official-citation rule must
       fail the check and override the scripted approval — even at
       the final allowed research iteration.
    2. The draft "explains" conditional routing only by disclaiming
       it ("details were not available") — that must not count as
       coverage."""
    question = (
        "Explain LangGraph state management and conditional routing "
        "using official documentation."
    )
    s1 = "https://blog.example.com/langgraph-state"
    s2 = "https://docs.langchain.com/oss/python/langgraph/overview"
    s3 = "https://dev.example.org/graphs-checkpoints"
    s4 = "https://notes.example.net/langgraph-notes"
    settings = _settings(max_tool_calls_per_research=4, max_research_iterations=1)
    fake_tools = FakeTools(
        search_results=[
            _search_result("State blog", s1),
            _search_result("Official overview", s2),
            _search_result("Checkpoints", s3),
            _search_result("Notes", s4),
        ],
        page_text="Official docs text.",
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1"),
            AIMessage(content="done"),
            AIMessage(
                content=(
                    "LangGraph state management uses a checkpointer [S1]. "
                    "More state management background is in [S3] and [S4]. "
                    "Conditional routing details were not available in the "
                    "retrieved sources."
                )
            ),
        ],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="looks complete"),
        ],
    )

    graph = build_research_graph(settings, model, fake_tools)
    result = graph.invoke({"question": question})

    # the official overview was force-fetched first (priority works)
    assert fake_tools.fetch_calls == [s2]
    # citation gate: retrieved official source never cited -> the
    # scripted approval is overridden even at the final iteration
    assert "CITATION QUALITY" in result["final_answer"]
    # completeness gate: disclaimer-only mention is not coverage
    assert "TOPIC COVERAGE" in result["final_answer"]
    assert result["missing_topics"] == ["conditional", "routing"]
    assert result["review_status"] == ReviewStatus.NEEDS_MORE_RESEARCH


def test_source_urls_are_deduplicated_across_searches():
    """The same URL surfaced by several results/searches is listed
    once in state['sources']."""
    settings = _settings(max_tool_calls_per_research=4)
    fake_tools = FakeTools(
        search_results=[
            _search_result("A", "https://example.com/a"),
            _search_result("A again", "https://example.com/a"),
            _search_result("B", "https://example.com/b"),
        ]
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1", args={"query": "q1"}),
            _tool_call_response("c2", args={"query": "q2"}),
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

    assert result["sources"] == ["https://example.com/a", "https://example.com/b"]


# --- official-source retrieval gate & disclaimer coverage (live runs) ---


def test_official_fetch_tls_failure_blocks_unqualified_approval():
    """Live run: the official docs.langchain.com page failed TLS
    verification ('self-signed certificate in certificate chain'),
    a third-party page fetched fine, and NO official page was
    retrieved. Even with a scripted 'approved' verdict the run must
    not end unqualified: it carries the official-source marker and
    is downgraded (bounded by the iteration budget)."""
    question = (
        "Explain LangGraph state management using official documentation."
    )
    official = "https://docs.langchain.com/oss/python/langgraph/overview"
    third = "https://deepwiki.example.com/langgraph-guide"
    settings = _settings(
        max_tool_calls_per_research=4, max_research_iterations=1
    )
    fake_tools = FakeTools(
        search_results=[
            _search_result("Official overview", official),
            _search_result("DeepWiki", third),
        ],
        page_text="Third-party overview.",
        fail_fetch_urls={official},
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1"),
            AIMessage(content="done"),
            AIMessage(
                content="LangGraph state management flows through "
                "checkpointer pages like [S2]."
            ),
        ],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="looks complete"),
        ],
    )

    graph = build_research_graph(settings, model, fake_tools)
    result = graph.invoke({"question": question})

    # the official URL is attempted FIRST (first-party priority) and
    # fails; the third-party page is then fetched successfully
    assert fake_tools.fetch_calls == [official, third]
    # honest incomplete label, never an unqualified result
    assert "OFFICIAL SOURCES: NOT RETRIEVED" in result["final_answer"]
    # citation + coverage gates pass, yet the official gate holds
    assert "CITATION QUALITY" not in result["final_answer"]
    assert "TOPIC COVERAGE" not in result["final_answer"]
    assert result["review_status"] == ReviewStatus.NEEDS_MORE_RESEARCH
    # the reviewer was shown the deterministic official status
    review_prompt = str(model.seen_structured_messages[-1])
    assert "Official-source status: OFFICIAL_DISCOVERED_ONLY" in (
        review_prompt
    )


def test_no_official_source_downgrades_and_steers_to_current_domains():
    """Official question where only third-party sources exist: no
    unqualified APPROVED, an honest marker, and plan/research
    prompts steering queries to the CURRENT documentation domains."""
    question = "LangGraph official documentation for state management"
    settings = _settings(
        max_tool_calls_per_research=4, max_research_iterations=1
    )
    fake_tools = FakeTools(
        search_results=[
            _search_result(
                "Blog", "https://blog.example.com/langgraph-state"
            )
        ],
        page_text="Blog text.",
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1"),
            AIMessage(content="done"),
            AIMessage(
                content="LangGraph state management uses checkpointers, "
                "per [S1]."
            ),
        ],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="complete"),
        ],
    )

    graph = build_research_graph(settings, model, fake_tools)
    result = graph.invoke({"question": question})

    assert "OFFICIAL SOURCES: NOT RETRIEVED" in result["final_answer"]
    assert result["review_status"] == ReviewStatus.NEEDS_MORE_RESEARCH
    review_prompt = str(model.seen_structured_messages[-1])
    assert "Official-source status: NO_OFFICIAL_SOURCE" in review_prompt
    # Task 4: the planner is steered to current domains
    plan_prompt = str(model.seen_structured_messages[0])
    assert "docs.langchain.com" in plan_prompt
    assert "reference.langchain.com" in plan_prompt
    # ... and so is every research round (legacy domains named only
    # as the thing to avoid)
    research_prompt = str(model.seen_messages[0])
    assert "Official-source guidance" in research_prompt
    assert "docs.langchain.com" in research_prompt
    assert "reference.langchain.com" in research_prompt
    assert "langchain-ai.github.io" in research_prompt


def test_not_present_disclaimer_is_marked_and_bounded():
    """Live phrase: conditional routing only mentioned via '...
    were not present in the retrieved text excerpts.' must be
    flagged as missing coverage; the bounded retry ends with the
    marker rather than an approved answer."""
    question = "Explain LangGraph conditional routing syntax"
    settings = _settings(
        max_tool_calls_per_research=4, max_research_iterations=1
    )
    fake_tools = FakeTools(
        search_results=[_search_result("Docs", "https://example.com/a")]
    )
    model = ScriptedChatModel(
        responses=[
            _tool_call_response("c1"),
            AIMessage(content="done"),
            AIMessage(
                content="Specific details regarding conditional routing "
                "syntax were not present in the retrieved text "
                "excerpts. LangGraph syntax uses nodes and edges, "
                "per [S1]."
            ),
        ],
        structured_outputs=[
            ResearchPlan(queries=["q"]),
            ReviewVerdict(status="approved", reason="complete"),
        ],
    )

    graph = build_research_graph(settings, model, fake_tools)
    result = graph.invoke({"question": question})

    assert result["missing_topics"] == ["conditional", "routing"]
    assert "TOPIC COVERAGE" in result["final_answer"]
    assert result["review_status"] == ReviewStatus.NEEDS_MORE_RESEARCH
