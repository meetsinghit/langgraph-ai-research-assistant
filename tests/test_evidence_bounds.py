"""Regression tests for tool-response token amplification.

The instrumented live run showed one 183,164-character fetched
page consuming ~90% of a run's input tokens: the full fetch_page
result was serialized into the transcript ToolMessage and re-sent
on the next research invocation (189,546 prompt chars). These
tests reproduce that world offline with a ~183K mocked page and
assert the transcript stays bounded, relevant evidence survives,
provenance/citations are unchanged, and the transcript structure
remains valid.

Documented budgets for this fixture (research question of ~60
chars, one search result, one fetched page, default settings):

- any single ToolMessage payload <= MAX_TOOL_RESULT_CHARS (8000)
- any research invocation prompt < 20,000 characters
"""
from __future__ import annotations

import json

from langchain_core.messages import (
    AIMessage,
    ToolMessage,
)

from agent.citations import validate_citations
from agent.config import Settings
from agent.graph import (
    ResearchPlan,
    ReviewStatus,
    ReviewVerdict,
    build_research_graph,
)
from agent.metrics import MetricsCollector
from agent.tools import shape_evidence, to_findings
from tests.fakes import FakeTools, ScriptedChatModel

URL = "https://docs.example.com/langgraph"
QUESTION = (
    "Explain LangGraph state management and conditional routing."
)

#: Planted near the MIDDLE of the page — a naive first-N-character
#: truncation could never keep it; only query-relevance can.
RELEVANT_SENTENCE = (
    "LangGraph state management persists state through "
    "checkpointers at every superstep, and conditional routing "
    "reads that state to choose the next edge."
)
#: Navigation boilerplate with no topic terms — must be dropped.
BOILERPLATE = (
    "Skip to content Home Products Pricing Enterprise Careers "
    "Contact us Terms of service Privacy policy Cookie "
    "preferences Newsletter signup Follow us on social media. "
)
DRAFT = (
    "LangGraph state management persists state via "
    "checkpointers [S1]; conditional routing reads that state "
    "to choose the next edge [S1]."
)


def _huge_page(total: int = 183_000) -> str:
    lead = "Acme documentation portal. "
    filler = BOILERPLATE * (total // len(BOILERPLATE) + 2)
    # place the relevant sentence between COMPLETE boilerplate
    # sentences (as in real pages) — no mid-sentence merging
    half = (len(filler) // 2 // len(BOILERPLATE)) * len(BOILERPLATE)
    page = lead + filler[:half] + RELEVANT_SENTENCE + " " + filler[half:]
    return page[:total]


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "min_findings": 1,
        "max_research_iterations": 3,
        "max_tool_calls_per_research": 4,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _run(page: str, *, settings: Settings | None = None):
    """One research run: model searches, fetches, then stops.

    Returns (graph state, scripted model, metrics collector).
    """
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "web_search",
                    "args": {"query": "langgraph state management"},
                    "id": "call_search",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "fetch_page",
                    "args": {"url": URL},
                    "id": "call_fetch",
                }
            ],
        ),
        AIMessage(content=DRAFT),
    ]
    model = ScriptedChatModel(
        responses=responses,
        structured_outputs=[
            ResearchPlan(queries=["langgraph state management"]),
            ReviewVerdict(status="approved", reason="fixture"),
        ],
        synthesize_usage=True,
    )
    tools = FakeTools(
        search_results=[
            {
                "title": "LangGraph docs",
                "snippet": "State management overview",
                "url": URL,
            }
        ],
        pages={URL: page},
    )
    metrics = MetricsCollector()
    graph = build_research_graph(
        settings or _settings(),
        model,
        tools,
        rate_limit_retry_delay=0.0,
        metrics=metrics,
    )
    metrics.begin_run()  # stamp the wall-clock start of the run
    result = graph.invoke({"question": QUESTION})
    metrics.finish()
    return result, model, metrics


def _fetch_tool_message(model: ScriptedChatModel) -> ToolMessage:
    """The fetch ToolMessage as seen by the post-fetch invocation."""
    for message in model.seen_messages[2]:
        if isinstance(message, ToolMessage) and "fetch_page" in message.content:
            return message
    raise AssertionError("fetch ToolMessage not found in transcript")


class TestTranscriptBounding:
    def test_fetch_tool_message_is_bounded(self) -> None:
        _, model, _ = _run(_huge_page())
        message = _fetch_tool_message(model)
        assert (
            len(message.content)
            <= _settings().max_tool_result_chars
        ), len(message.content)

    def test_post_fetch_invocation_prompt_is_bounded(self) -> None:
        # documented budget for this fixture: < 20,000 chars
        _, _, metrics = _run(_huge_page())
        calls = metrics.snapshot()["model_calls"]
        research_prompts = [
            call["prompt_chars"]
            for call in calls
            if call["node"] == "research"
        ]
        assert max(research_prompts) < 20_000, max(research_prompts)

    def test_relevant_sections_kept_boilerplate_dropped(self) -> None:
        _, model, _ = _run(_huge_page())
        payload = json.loads(_fetch_tool_message(model).content)
        assert payload["ok"] is True
        assert payload["args"]["url"] == URL
        assert RELEVANT_SENTENCE[:100] in payload["data"]
        assert "Skip to content" not in payload["data"]

    def test_state_holds_bounded_relevant_excerpt_only(
        self,
    ) -> None:
        result, _, _ = _run(_huge_page())
        retrieved = [
            finding
            for finding in result["findings"]
            if finding["evidence"] == "retrieved"
        ]
        assert len(retrieved) == 1
        finding = retrieved[0]
        assert finding["source_url"] == URL
        # relevant content accessible for grounded synthesis
        assert RELEVANT_SENTENCE[:100] in finding["text"]
        # full raw document not retained
        assert (
            len(finding["text"])
            <= _settings().max_evidence_chars
        )


class TestTranscriptValidity:
    def test_tool_messages_pair_with_their_tool_calls(self) -> None:
        _, model, _ = _run(_huge_page())
        messages = model.seen_messages[2]
        kinds = [type(message).__name__ for message in messages]
        assert kinds == [
            "SystemMessage",
            "HumanMessage",
            "AIMessage",
            "ToolMessage",
            "AIMessage",
            "ToolMessage",
        ]
        tool_calls = [
            call["id"]
            for message in messages
            if isinstance(message, AIMessage)
            for call in (message.tool_calls or [])
        ]
        tool_ids = [
            message.tool_call_id
            for message in messages
            if isinstance(message, ToolMessage)
        ]
        assert tool_ids == tool_calls
        assert all(tool_ids)

    def test_search_tool_message_unchanged(self) -> None:
        # small payloads must pass through verbatim (no shaping)
        _, model, _ = _run(_huge_page())
        search_message = next(
            message
            for message in model.seen_messages[2]
            if isinstance(message, ToolMessage)
            and "web_search" in message.content
        )
        payload = json.loads(search_message.content)
        assert payload["data"] == [
            {
                "title": "LangGraph docs",
                "snippet": "State management overview",
                "url": URL,
            }
        ]


class TestGroundingAndCitations:
    def test_synthesis_still_grounds_and_cites(self) -> None:
        result, model, _ = _run(_huge_page())
        # synthesis prompt carried the relevant evidence
        synthesis_prompt = str(model.seen_messages[3][-1].content)
        assert RELEVANT_SENTENCE[:100] in synthesis_prompt
        assert URL in synthesis_prompt
        # final answer cites the source and validates
        assert "[S1]" in result["final_answer"]
        report = validate_citations(
            result["final_answer"], result["findings"], QUESTION
        )
        assert report.ok, report.unknown
        assert result["review_status"] == ReviewStatus.APPROVED

    def test_relevant_excerpt_beats_naive_head(self) -> None:
        # to_findings keeps query-relevant sections, not [:N]
        page = _huge_page()
        findings = to_findings(
            {
                "tool": "fetch_page",
                "ok": True,
                "data": page,
                "error": None,
                "args": {"url": URL},
            },
            query=QUESTION,
            max_chars=6000,
        )
        excerpt = findings[0]["text"]
        assert len(excerpt) <= 6000
        assert RELEVANT_SENTENCE[:100] in excerpt
        assert excerpt != page[: len(excerpt)]  # not a naive head cut
        assert "Skip to content" not in excerpt


class TestFailedFetchIntact:
    def test_failed_fetch_still_yields_unverified_answer(
        self,
    ) -> None:
        responses = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": "q"},
                        "id": "call_search",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "fetch_page",
                        "args": {"url": URL},
                        "id": "call_fetch",
                    }
                ],
            ),
            AIMessage(content=DRAFT),
        ]
        model = ScriptedChatModel(
            responses=responses,
            structured_outputs=[
                ResearchPlan(queries=["q"]),
                ReviewVerdict(status="approved", reason="fixture"),
            ],
        )
        tools = FakeTools(
            search_results=[
                {"title": "Docs", "snippet": "Overview", "url": URL}
            ],
            fail_fetch_urls={URL},
        )
        graph = build_research_graph(
            _settings(),
            model,
            tools,
            rate_limit_retry_delay=0.0,
            metrics=MetricsCollector(),
        )
        result = graph.invoke({"question": QUESTION})
        # the failed fetch is answered with its error (bounded)
        fetch_message = next(
            message
            for seen in model.seen_messages
            for message in seen
            if isinstance(message, ToolMessage)
            and "fetch_page" in message.content
        )
        payload = json.loads(fetch_message.content)
        assert payload["ok"] is False
        assert "failed" in payload["error"]
        assert len(fetch_message.content) <= _settings().max_tool_result_chars
        # and the answer stays honestly labeled
        assert "UNVERIFIED" in result["final_answer"]
        assert not any(
            finding["evidence"] == "retrieved"
            for finding in result["findings"]
        )


class TestBeforeAfter:
    """Same oversized-page fixture with the cap effectively off vs
    on — quantifies the amplification this fix removes."""

    def test_unbounded_transcript_reproduces_amplification(
        self,
    ) -> None:
        _, _, metrics = _run(
            _huge_page(),
            settings=_settings(max_tool_result_chars=10_000_000),
        )
        calls = metrics.snapshot()["model_calls"]
        biggest = max(
            call["prompt_chars"]
            for call in calls
            if call["node"] == "research"
        )
        # the ~183K page reached the next invocation, as live
        assert biggest > 150_000, biggest

    def test_default_cap_keeps_prompts_small(self) -> None:
        _, _, metrics = _run(_huge_page())
        calls = metrics.snapshot()["model_calls"]
        biggest = max(
            call["prompt_chars"]
            for call in calls
            if call["node"] == "research"
        )
        assert biggest < 20_000, biggest


class TestMetricsFinalization:
    def test_run_report_records_completed_timing(self) -> None:
        _, _, metrics = _run(_huge_page())
        snapshot = metrics.snapshot()
        assert snapshot["finished_at"] is not None
        assert snapshot["total_execution_seconds"] is not None
        assert snapshot["total_execution_seconds"] >= 0


class TestShapeEvidence:
    def test_passthrough_within_budget(self) -> None:
        text = "Short page text."
        assert shape_evidence(text, "query", 500) == text

    def test_output_always_bounded_with_marker(self) -> None:
        text = ("Sentence about nothing in particular. " * 5000)
        shaped = shape_evidence(text, "langgraph state", 1000)
        assert len(shaped) <= 1000
        assert "evidence bounded" in shaped

    def test_single_line_document_keeps_middle_relevance(self) -> None:
        page = _huge_page()
        shaped = shape_evidence(page, QUESTION, 4000)
        assert len(shaped) <= 4000
        assert RELEVANT_SENTENCE[:100] in shaped
        assert "Skip to content" not in shaped
        # located past the halfway point of the original page
        assert page.index(RELEVANT_SENTENCE) > len(page) // 2

    def test_empty_query_still_bounded(self) -> None:
        page = _huge_page()
        shaped = shape_evidence(page, "", 1000)
        assert len(shaped) <= 1000
        assert "evidence bounded" in shaped

    def test_terminator_free_text_bounded(self) -> None:
        page = "x" * 50_000
        shaped = shape_evidence(page, "query", 1000)
        assert len(shaped) <= 1000

    def test_marker_reports_original_size(self) -> None:
        page = _huge_page()
        shaped = shape_evidence(page, QUESTION, 2000)
        assert f"{len(page)} characters" in shaped
