"""Offline evaluation framework tests.

Covers: the benchmark catalog, fully-offline execution (no
network, no provider), deterministic pass/fail against
fixtures, the Workstream 4 false-approval flag, semantic
scoring separation, bounded prompt sizes (evidence caps) and
per-call token/latency instrumentation in reports.
"""
from __future__ import annotations

import json

import pytest

from evals.cases import (
    CaseExpectations,
    EvalCase,
    ResearchStep,
    TopicRubric,
    benchmark_cases,
)
from evals.harness import _case_settings, run_benchmark, run_case
from evals.semantic import (
    RubricJudge,
    evaluate_semantics,
    rubric_judge_enabled,
)


class TestBenchmarkCatalog:
    def test_has_about_twenty_cases(self) -> None:
        cases = benchmark_cases()
        assert 18 <= len(cases) <= 30

    def test_case_ids_unique(self) -> None:
        ids = [case.case_id for case in benchmark_cases()]
        assert len(ids) == len(set(ids))

    @pytest.mark.parametrize(
        "category",
        [
            "official technical documentation",
            "state management and conditional routing",
            "multi-topic research questions",
            "general technical research",
            "requests with ambiguous wording",
            "unavailable or contradictory evidence",
            "questions where an incomplete answer is the correct outcome",
            "tool failure cases",
            "source-retrieval failure cases",
        ],
    )
    def test_required_categories_present(self, category: str) -> None:
        assert any(
            case.category == category
            for case in benchmark_cases()
        ), f"no case in category {category!r}"

    def test_every_case_defines_expectations_and_rubric_where_needed(
        self,
    ) -> None:
        for case in benchmark_cases():
            assert case.question
            assert case.expected.review_status in (
                "approved",
                "needs_more_research",
            )
            # multi-topic cases carry rubrics for semantic scoring
            if case.case_id in (
                "ws4-routing-superficial",
                "state-and-routing-complete",
            ):
                assert len(case.rubric) >= 2


class TestOfflineGuarantee:
    def test_no_network_access_during_benchmark(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Any httpx client construction during eval = failure."""
        import httpx

        def boom(*args: object, **kwargs: object) -> object:
            raise AssertionError(
                "network access attempted during offline eval"
            )

        monkeypatch.setattr(httpx, "Client", boom)
        monkeypatch.setattr(httpx, "AsyncClient", boom)
        report = run_benchmark()
        assert report.failed == 0, [
            (c.case_id, c.deterministic_failures)
            for c in report.cases
            if not c.passed
        ]
        assert report.metadata["live_provider_calls"] is False

    def test_llm_judge_disabled_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("EVAL_LLM_JUDGE", raising=False)
        assert rubric_judge_enabled() is False
        with pytest.raises(RuntimeError, match="EVAL_LLM_JUDGE"):
            RubricJudge(model=None)

    def test_llm_judge_is_opt_in(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EVAL_LLM_JUDGE", "1")
        assert rubric_judge_enabled() is True
        RubricJudge(model=None)  # constructs when explicitly enabled


class TestDeterministicScoring:
    def test_all_cases_meet_fixture_expectations(self) -> None:
        report = run_benchmark()
        failures = [
            (case.case_id, case.deterministic_failures)
            for case in report.cases
            if not case.passed
        ]
        assert failures == []

    @pytest.mark.parametrize(
        "case_id",
        [
            "malformed-citation",
            "fabricated-sid",
            "unknown-url-citation",
        ],
    )
    def test_citation_failures_downgrade_to_research(
        self, case_id: str
    ) -> None:
        case = next(
            c for c in benchmark_cases() if c.case_id == case_id
        )
        result = run_case(case)
        assert result.metrics["review_status"] == (
            "needs_more_research"
        )
        assert result.metrics["citations_ok"] is False

    def test_official_retrieval_gate_downgrades(self) -> None:
        case = next(
            c
            for c in benchmark_cases()
            if c.case_id == "official-tls-failure"
        )
        result = run_case(case)
        assert result.metrics["review_status"] == (
            "needs_more_research"
        )
        assert result.metrics["official_source_status"] == (
            "official_discovered_only"
        )
        assert result.metrics["markers_present"][
            "OFFICIAL SOURCES: NOT RETRIEVED"
        ] is True

    def test_budget_respected_in_every_case(self) -> None:
        for case in benchmark_cases():
            result = run_case(case)
            # each pass must stay within THAT case's configured
            # tool-call budget (a fixture may lift its own quota —
            # e.g. the two-search recovery scenario — but never
            # silently exceed the shared default of 2)
            limit = _case_settings(case).max_tool_calls_per_research
            assert result.metrics["budget_compliant"], case.case_id
            assert (
                result.metrics["max_tool_calls_per_pass"] <= limit
            ), case.case_id


class TestWorkstream4:
    """The live failure reproduction: a superficial answer that
    every deterministic gate and the reviewer accept, but the
    semantic rubric rejects."""

    @pytest.fixture()
    def ws4_result(self):
        case = next(
            c
            for c in benchmark_cases()
            if c.case_id == "ws4-routing-superficial"
        )
        return run_case(case)

    def test_deterministically_passes(self, ws4_result) -> None:
        assert ws4_result.deterministic_pass, (
            ws4_result.deterministic_failures
        )
        assert ws4_result.metrics["review_status"] == "approved"
        assert ws4_result.metrics["citations_ok"] is True
        assert (
            ws4_result.metrics["official_source_status"]
            == "verified_official"
        )

    def test_semantic_rubric_flags_conditional_routing(
        self, ws4_result
    ) -> None:
        semantic = ws4_result.semantic
        topic = next(
            t
            for t in semantic["topics"]
            if t["topic"] == "conditional routing"
        )
        assert topic["adequate"] is False
        assert semantic["overall_adequate"] is False
        assert semantic["inadequate_topics"] == [
            "conditional routing"
        ]
        # while state management IS explained
        stated = next(
            t
            for t in semantic["topics"]
            if t["topic"] == "state management"
        )
        assert stated["adequate"] is True

    def test_reviewer_false_approval_recorded(
        self, ws4_result
    ) -> None:
        semantic = ws4_result.semantic
        assert semantic["reviewer_approved"] is True
        assert semantic["reviewer_false_approval"] is True
        assert semantic["notes"] == [
            "reviewer APPROVED a semantically inadequate answer"
        ]

    def test_report_surfaces_the_false_approval(self) -> None:
        report = run_benchmark()
        assert "ws4-routing-superficial" in report.false_approvals
        data = report.to_dict()
        assert data["summary"]["reviewer_false_approvals"] == [
            "ws4-routing-superficial"
        ]


class TestSemanticSeparation:
    def test_disclaimer_only_mention_is_not_adequate(
        self,
    ) -> None:
        from types import SimpleNamespace

        case = SimpleNamespace(
            case_id="t",
            rubric=(TopicRubric("conditional routing", ("edge",)),),
        )

        disclaimed = (
            "Details regarding conditional routing edge syntax "
            "were not present in the retrieved excerpts."
        )
        result = evaluate_semantics(case, disclaimed)
        assert result.topics[0].adequate is False
        assert result.topics[0].matched_in_disclaimer == ("edge",)

        substantive = (
            "Conditional routing selects the next edge with a "
            "predicate over the state."
        )
        result = evaluate_semantics(case, substantive)
        assert result.topics[0].adequate is True

    def test_empty_rubric_stays_adequate(self) -> None:
        from types import SimpleNamespace

        case = SimpleNamespace(case_id="t", rubric=())
        result = evaluate_semantics(case, "anything")
        assert result.overall_adequate is True
        assert result.reviewer_false_approval is False


class TestReportStructure:
    def test_report_is_json_serializable(self) -> None:
        report = run_benchmark()
        dumped = json.dumps(report.to_dict())
        data = json.loads(dumped)
        assert data["summary"]["total"] == len(benchmark_cases())
        assert data["summary"]["deterministic_failed"] == 0
        assert data["metadata"]["live_provider_calls"] is False
        assert "token_source" in data["metadata"]
        assert data["metadata"]["judge_backend"].startswith(
            "fixture"
        )
        for case in data["cases"]:
            assert set(case) >= {
                "case_id",
                "category",
                "deterministic_pass",
                "deterministic_failures",
                "deterministic_metrics",
                "semantic",
                "run_metrics",
            }


class TestTokenAccounting:
    """Workstream 2: per-call token and latency tracking."""

    @pytest.fixture()
    def result(self):
        case = next(
            c
            for c in benchmark_cases()
            if c.case_id == "state-and-routing-complete"
        )
        return run_case(case)

    def test_state_accumulates_tokens(self, result) -> None:
        assert result.metrics["tokens_in"] > 0
        assert result.metrics["tokens_out"] > 0

    def test_tokens_attributed_by_node(self, result) -> None:
        by_node = result.metrics["tokens_by_node"]
        assert by_node["research"]["input_tokens"] > 0
        assert by_node["synthesize"]["input_tokens"] > 0
        assert by_node["research"]["latency_ms"] >= 0
        assert by_node["synthesize"]["prompt_chars"] > 0

    def test_structured_nodes_measure_prompt_but_not_tokens(
        self, result
    ) -> None:
        # plan and review use with_structured_output: the offline
        # scripted model reports no usage metadata for structured
        # calls, so WS-E records honestly null (never a fake zero)
        # and counts those calls as usage-unavailable. Prompt size
        # and latency are still captured.
        by_node = result.metrics["tokens_by_node"]
        assert by_node["plan"]["prompt_chars"] > 0
        assert by_node["plan"]["input_tokens"] == 0
        assert (
            by_node["plan"]["usage_unavailable_calls"]
            == by_node["plan"]["calls"]
        )
        assert by_node["review"]["prompt_chars"] > 0
        assert by_node["review"]["input_tokens"] == 0
        assert (
            by_node["review"]["usage_unavailable_calls"]
            == by_node["review"]["calls"]
        )

    def test_per_call_entries_have_tokens_and_latency(
        self, result
    ) -> None:
        calls = result.run_metrics["model_calls"]
        assert len(calls) >= 4
        for call in calls:
            assert "latency_ms" in call
            assert "prompt_chars" in call
            assert "input_tokens" in call
            assert "usage_source" in call
            assert call["usage_source"] in (
                "provider",
                "estimated",
                "unavailable",
            )
            # honest measurement state: a value or null — null means
            # "not measured", never a fabricated zero
            assert call["input_tokens"] is None or call["input_tokens"] >= 0
            assert "error_category" in call
        measured = [
            c for c in calls if c["input_tokens"] is not None
        ]
        assert any(c["input_tokens"] > 0 for c in measured)
        assert any(c["input_tokens"] is None for c in calls)
        assert any(c["prompt_chars"] > 0 for c in calls)
        coverage = result.run_metrics["token_usage"]
        assert (
            coverage["provider_reported_calls"]
            + coverage["estimated_calls"]
            + coverage["unavailable_calls"]
            == len(calls)
        )

    def test_rate_limit_retry_is_counted(self) -> None:
        case = next(
            c for c in benchmark_cases() if c.case_id == "rate-limit-retry"
        )
        result = run_case(case)
        assert result.metrics["retries"] == 1
        errored = [
            c
            for c in result.run_metrics["model_calls"]
            if c["error_category"] == "rate_limit"
        ]
        assert len(errored) == 1
        # the retry itself succeeded and is recorded
        assert result.metrics["review_status"] == "approved"


class TestBoundedPrompts:
    """Workstream 2 evidence caps: a huge fetched page must not
    translate into a huge prompt on later model calls."""

    def _huge_page_case(self, page_chars: int) -> EvalCase:
        url = "https://example.com/huge"
        page = "HUGEPAGEMARKER " * (page_chars // 15)
        return EvalCase(
            case_id="huge-page",
            category="general technical research",
            question="Explain LangGraph.",
            search_results=(
                {"title": "Huge", "snippet": "snippet", "url": url},
            ),
            pages={url: page},
            # search, fetch, THEN one more model invocation — the
            # post-fetch invoke is where an unbounded ToolMessage
            # would re-enter the transcript (the live 61K/43K
            # amplification path)
            research_steps=(
                ResearchStep(
                    kind="tool_call",
                    name="web_search",
                    args={"query": "q"},
                ),
                ResearchStep(
                    kind="tool_call",
                    name="fetch_page",
                    args={"url": url},
                ),
                ResearchStep(kind="text", text="done"),
            ),
            settings_overrides={"max_tool_calls_per_research": 4},
            drafts=("LangGraph builds graphs, per [S1].",),
            expected=CaseExpectations(review_status="approved"),
        )

    def test_prompt_stays_bounded_despite_huge_page(
        self,
    ) -> None:
        case = self._huge_page_case(100_000)
        result = run_case(case)
        assert result.deterministic_pass, (
            result.deterministic_failures
        )
        calls = result.run_metrics["model_calls"]
        max_prompt = max(c["prompt_chars"] for c in calls)
        # page is ~100k chars; every prompt must stay far below it
        assert max_prompt < 20_000, max_prompt

    def test_page_content_never_reaches_metrics(
        self,
    ) -> None:
        case = self._huge_page_case(50_000)
        result = run_case(case)
        dumped = json.dumps(result.run_metrics)
        assert "HUGEPAGEMARKER" not in dumped
