"""Metrics collector: recording, aggregation, JSON-safety and
the content-redaction guarantee (sizes/counts only, never
content, URLs or raw error text)."""
from __future__ import annotations

import json

import pytest

from agent.metrics import MetricsCollector, categorize_error
from agent.tools import ToolError, ToolExecutor
from tests.fakes import FakeTools


class TestCategorizeError:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("429 too many requests", "rate_limit"),
            ("Rate limit exceeded", "rate_limit"),
            ("resource exhausted", "rate_limit"),
            ("certificate verify failed: self-signed", "tls"),
            ("SSLError: handshake failed", "tls"),
            ("url points to a non-public range", "ssrf"),
            ("target is a private address", "ssrf"),
            ("unknown tool: 'bing'", "validation"),
            ("tool arguments must be an object", "validation"),
            ("llm provider failed", "provider"),
            ("provider failed with 500", "provider"),
            ("execution deadline of 180s exhausted", "deadline"),
            ("global execution deadline has expired", "deadline"),
            ("something inexplicable", "unknown"),
        ],
    )
    def test_categories(self, text: str, expected: str) -> None:
        assert categorize_error(text) == expected

    def test_exception_instance(self) -> None:
        assert categorize_error(ToolError("429 rate limit")) == "rate_limit"

    def test_no_error_text_leaks_from_category(self) -> None:
        # the category is a fixed vocabulary — no URL or message
        category = categorize_error(
            "fetch failed for https://internal.example.com/secret"
        )
        assert category == "unknown"
        assert "internal.example.com" not in category


class TestRecording:
    def test_model_call_records_sizes_not_content(self) -> None:
        collector = MetricsCollector()
        question = "PRIVATE roadmap of Project Falcon Q4"
        collector.set_iteration(2)
        collector.record_model_call(
            "research",
            latency_ms=12.5,
            input_tokens=100,
            output_tokens=10,
            prompt_chars=len(question),
        )
        call = collector.snapshot()["model_calls"][0]
        assert call["node"] == "research"
        assert call["iteration"] == 2
        assert call["input_tokens"] == 100
        assert call["prompt_chars"] == len(question)

    def test_snapshot_totals_and_by_node(self) -> None:
        collector = MetricsCollector()
        collector.record_model_call(
            "research", latency_ms=10.0, input_tokens=100,
            prompt_chars=400,
        )
        collector.record_model_call(
            "synthesize", latency_ms=5.0, input_tokens=50,
            prompt_chars=200,
        )
        collector.record_model_call(
            "research", latency_ms=1.0, error_category="rate_limit",
        )
        collector.record_tool_call(
            "research", "web_search", ok=True, latency_ms=3.0,
            result_chars=120,
        )
        collector.record_tool_call(
            "research", "fetch_page", ok=False, latency_ms=7.0,
            error_category="tls",
        )
        collector.record_node("research", 20.0)
        collector.record_retry("research", "rate_limit")

        snap = collector.snapshot()
        assert snap["totals"]["input_tokens"] == 150
        assert snap["totals"]["prompt_chars"] == 600
        assert snap["totals"]["model_calls"] == 3
        assert snap["totals"]["retries"] == 1
        assert snap["tokens_by_node"]["research"]["input_tokens"] == 100
        assert snap["tokens_by_node"]["research"]["prompt_chars"] == 400
        assert snap["counts"]["searches"] == 1
        assert snap["counts"]["fetches"] == 1
        assert snap["counts"]["failed_tool_calls"] == 1
        assert snap["counts"]["tool_error_categories"] == {"tls": 1}
        assert snap["retries"][0]["reason_category"] == "rate_limit"
        assert len(snap["nodes"]) == 1

    def test_snapshot_is_json_serializable(self) -> None:
        collector = MetricsCollector()
        collector.record_model_call(
            "plan", latency_ms=1.234567, prompt_chars=10,
        )
        collector.record_node("plan", 1.234567)
        collector.begin_run()  # stamps a fresh run id
        text = json.dumps(collector.snapshot(), indent=2)
        assert '"run_id"' in text
        assert json.loads(text)["run_id"] == collector.run_id

    def test_begin_run_resets_accumulated_events(self) -> None:
        collector = MetricsCollector()
        collector.record_node("research", 5.0)
        old_run = collector.run_id
        collector.begin_run()
        snap = collector.snapshot()
        assert snap["nodes"] == []
        assert snap["totals"]["model_calls"] == 0
        assert collector.run_id != old_run

    def test_finish_stamps_completed_timing(self) -> None:
        # finish() must guarantee a completed, non-negative span
        # even when started_at was never accessed during the run
        collector = MetricsCollector()
        collector.record_node("research", 5.0)
        collector.finish()
        snap = collector.snapshot()
        assert snap["finished_at"] is not None
        assert snap["total_execution_seconds"] is not None
        assert snap["total_execution_seconds"] >= 0


class TestRedaction:
    """The collector must never store content, URLs or raw
    error text — by construction, not by scrubbing."""

    def test_question_text_never_reaches_snapshot(self) -> None:
        secret = "SECRET acquisition of Initech by Umbrella Corp"
        collector = MetricsCollector()
        collector.record_model_call(
            "research", latency_ms=2.0, prompt_chars=len(secret)
        )
        dumped = json.dumps(collector.snapshot())
        for word in ("SECRET", "Initech", "Umbrella"):
            assert word not in dumped

    def test_tool_executor_omits_args_results_and_error_text(
        self,
    ) -> None:
        tools = FakeTools(
            search_results=[
                {
                    "title": "Secret title",
                    "snippet": "Secret snippet body",
                    "url": "https://secret.example.com/page",
                }
            ],
            fail_fetch_urls={"https://secret.example.com/page"},
        )
        collector = MetricsCollector()
        executor = ToolExecutor(
            tools.tool_functions(), metrics=collector, node="research"
        )

        ok = executor.execute(
            {"name": "web_search", "args": {"query": "secret query"}}
        )
        failed = executor.execute(
            {"name": "fetch_page", "args": {"url": "https://secret.example.com/page"}}
        )

        assert ok["ok"] is True
        assert failed["ok"] is False

        snap = collector.snapshot()
        dumped = json.dumps(snap)
        # no args, no results, no URL, no error text
        for leak in (
            "secret query",
            "Secret snippet",
            "secret.example.com",
            "self-signed",
            "certificate",
        ):
            assert leak not in dumped, leak
        # but the sizes and categories ARE recorded
        search = next(
            c for c in snap["tool_calls"] if c["tool"] == "web_search"
        )
        assert search["result_chars"] > 0
        assert search["ok"] is True
        fetch = next(
            c for c in snap["tool_calls"] if c["tool"] == "fetch_page"
        )
        assert fetch["ok"] is False
        assert fetch["error_category"] == "tls"
        assert fetch["result_chars"] == 0

    def test_error_category_recorded_without_message(self) -> None:
        collector = MetricsCollector()
        collector.record_model_call(
            "research",
            latency_ms=1.0,
            error_category=categorize_error(
                RuntimeError("429 from https://api.example.com?key=abc123")
            ),
        )
        dumped = json.dumps(collector.snapshot())
        assert "rate_limit" in dumped
        assert "api.example.com" not in dumped
        assert "abc123" not in dumped


class TestTokenHonestyAndOutcome:
    """WS-E / WS-F: unmeasured usage stays null (never a fake zero),
    coverage is reported, and the terminal outcome follows the
    COMPLETED / COMPLETED_WITH_WARNINGS / INCOMPLETE / TIMED_OUT /
    FAILED taxonomy with deterministic warning merging."""

    def test_unmeasured_usage_is_null_with_unavailable_source(self) -> None:
        collector = MetricsCollector()
        collector.record_model_call("plan", latency_ms=1.0, prompt_chars=90)

        snap = collector.snapshot()
        call = snap["model_calls"][0]
        assert call["input_tokens"] is None
        assert call["output_tokens"] is None
        assert call["usage_source"] == "unavailable"
        usage = snap["token_usage"]
        assert usage["provider_reported_calls"] == 0
        assert usage["unavailable_calls"] == 1
        assert usage["coverage_ratio"] == 0.0
        # totals sum only measured values — they never fabricate zeros
        assert usage["input_tokens"] == 0

    def test_provider_reported_and_estimated_sources_are_distinguished(
        self,
    ) -> None:
        collector = MetricsCollector()
        collector.record_model_call(
            "research", latency_ms=1.0, input_tokens=100,
            output_tokens=20,
        )
        collector.record_model_call(
            "synthesize", latency_ms=1.0, input_tokens=50,
            output_tokens=10, usage_source="estimated",
        )

        usage = collector.snapshot()["token_usage"]
        # the triad partitions every call — estimated usage is never
        # passed off as provider-reported
        assert usage["provider_reported_calls"] == 1
        assert usage["estimated_calls"] == 1
        assert usage["unavailable_calls"] == 0
        assert usage["coverage_ratio"] == 1.0
        assert usage["sources"] == {"provider": 1, "estimated": 1}
        # totals include measured values of both kinds
        assert usage["input_tokens"] == 150

    def test_clean_run_stays_completed_without_warnings(self) -> None:
        collector = MetricsCollector()
        collector.record_model_call(
            "research", latency_ms=1.0, input_tokens=10,
            output_tokens=5,
        )
        collector.set_outcome("COMPLETED")

        snap = collector.snapshot()
        assert snap["outcome"]["status"] == "COMPLETED"
        assert snap["outcome"]["warnings"] == []

    def test_completed_with_observations_upgrades_to_with_warnings(
        self,
    ) -> None:
        collector = MetricsCollector()
        collector.record_model_call(
            "research", latency_ms=1.0, input_tokens=10,
            output_tokens=5,
        )
        collector.record_tool_call(
            "research", "web_search", ok=False, latency_ms=2.0,
            error_category="timeout",
        )
        collector.record_retry("research", "unavailable")
        collector.set_outcome("COMPLETED")

        outcome = collector.snapshot()["outcome"]
        assert outcome["status"] == "COMPLETED_WITH_WARNINGS"
        assert "tool_failures" in outcome["warnings"]
        assert "provider_retries" in outcome["warnings"]

    def test_mixed_token_coverage_is_a_warning_not_a_silent_gap(
        self,
    ) -> None:
        collector = MetricsCollector()
        collector.record_model_call(
            "research", latency_ms=1.0, input_tokens=10,
        )
        collector.record_model_call("plan", latency_ms=1.0)  # unmeasured
        collector.set_outcome("COMPLETED")

        outcome = collector.snapshot()["outcome"]
        assert outcome["status"] == "COMPLETED_WITH_WARNINGS"
        assert "partial_token_coverage" in outcome["warnings"]

    def test_worse_statuses_keep_warnings_but_do_not_upgrade(
        self,
    ) -> None:
        collector = MetricsCollector()
        collector.set_outcome("TIMED_OUT", warnings=["state_errors"])

        outcome = collector.snapshot()["outcome"]
        assert outcome["status"] == "TIMED_OUT"
        assert "state_errors" in outcome["warnings"]

    def test_unknown_outcome_status_is_rejected(self) -> None:
        collector = MetricsCollector()
        with pytest.raises(ValueError, match="unknown run outcome"):
            collector.set_outcome("WORKED_SORT_OF")

    def test_deadline_and_quality_surface_in_snapshot(self) -> None:
        collector = MetricsCollector()
        collector.record_deadline(180.0)
        collector.mark_deadline_exceeded()
        collector.set_quality(
            citations_ok=True,
            coverage_ok=False,
            review_status="needs_more_research",
            retrieved_sources=2,
            snippet_sources=1,
        )

        snap = collector.snapshot()
        assert snap["deadline"] == {
            "configured_seconds": 180.0,
            "exceeded": True,
        }
        assert snap["quality"]["citations_ok"] is True
        assert snap["quality"]["coverage_ok"] is False
        assert snap["quality"]["retrieved_sources"] == 2
        # booleans/counts/enum strings only — no text, no URLs
        assert isinstance(snap["quality"]["review_status"], str)

    def test_begin_run_resets_outcome_deadline_and_quality(self) -> None:
        collector = MetricsCollector()
        collector.record_deadline(180.0)
        collector.mark_deadline_exceeded()
        collector.set_outcome("FAILED", warnings=["state_errors"])
        collector.set_quality(citations_ok=False)

        collector.begin_run()
        snap = collector.snapshot()
        assert snap["deadline"] == {
            "configured_seconds": None,
            "exceeded": False,
        }
        assert snap["outcome"]["status"] == "UNSPECIFIED"
        assert snap["outcome"]["warnings"] == []
        assert snap["quality"] is None
