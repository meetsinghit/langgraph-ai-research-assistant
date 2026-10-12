"""Offline evaluation harness — drives the real graph
with deterministic fixtures and scores each run.

No network, no API keys, no live provider: the graph is
built with :class:`tests.fakes.ScriptedChatModel` and
:class:`tests.fakes.FakeTools` from each case's fixtures.
Deterministic metrics are computed from the resulting
state, the collected run metrics and a re-validation of
the final answer; semantic scores come from the
fixture rubric (``evals.semantic``).

The harness deliberately keeps two separate verdicts:

- ``deterministic_pass`` — the run met its fixture
  expectations (workflow, budgets, gates, markers);
- ``semantic`` — whether the draft actually explains
  each requested topic. A run can pass deterministically
  yet be semantically inadequate (that is exactly the
  Workstream 4 failure mode), and the report keeps
  both visible.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage

from agent.citations import (
    official_source_status,
    requests_official_sources,
    validate_citations,
)
from agent.completeness import check_completeness
from agent.config import Settings
from agent.graph import (
    ResearchPlan,
    ReviewStatus,
    ReviewVerdict,
    build_research_graph,
    strip_marked_notes,
)
from agent.metrics import MetricsCollector
from evals.cases import EvalCase, ResearchStep
from evals.semantic import evaluate_semantics
from tests.fakes import FakeTools, ScriptedChatModel

__all__ = ["CaseResult", "BenchmarkReport", "run_case", "run_benchmark", "summarize"]


def _error_for(kind: str) -> Exception:
    if kind == "rate_limit":
        return RuntimeError("429 resource exhausted")
    return RuntimeError(kind)


def _scripted_model(case: EvalCase) -> ScriptedChatModel:
    """Build the scripted model for a case."""
    responses: list[AIMessage] = []
    call_id = 0
    for step in case.research_steps:
        if step.kind == "tool_call":
            call_id += 1
            responses.append(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": step.name,
                            "args": step.args or {},
                            "id": f"call_{call_id}",
                        }
                    ],
                )
            )
        else:
            responses.append(
                AIMessage(content=step.text or "")
            )
    responses.extend(
        AIMessage(content=draft) for draft in case.drafts
    )
    structured: list[Any] = [
        ResearchPlan(queries=list(case.plan_queries))
    ]
    structured.extend(
        ReviewVerdict(status=status, reason="eval fixture")
        for status in case.review_verdicts
    )
    return ScriptedChatModel(
        responses=responses,
        structured_outputs=structured,
        raise_on_call={
            index: _error_for(kind)
            for index, kind in case.raise_on_call.items()
        },
        structured_replay=(
            "cycle" if len(case.review_verdicts) > 1 else "last"
        ),
        # deterministic synthetic usage so the report shows the
        # token-accounting pipeline end to end (estimated, not
        # provider usage — see report metadata)
        synthesize_usage=True,
    )


def _case_settings(case: EvalCase) -> Settings:
    base: dict[str, Any] = {
        "_env_file": None,
        "min_findings": 1,
        "max_research_iterations": 3,
        "max_tool_calls_per_research": 2,
    }
    base.update(case.settings_overrides)
    return Settings(**base)  # type: ignore[arg-type]


@dataclass
class CaseResult:
    """One case's evaluation result."""

    case_id: str
    category: str
    question: str
    deterministic_pass: bool
    deterministic_failures: tuple[str, ...]
    metrics: dict[str, Any]
    semantic: dict[str, Any]
    #: full metrics snapshot for the run — per-call tokens,
    #: prompt sizes and latencies (content-free by design)
    run_metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.deterministic_pass

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "question": self.question,
            "deterministic_pass": self.deterministic_pass,
            "deterministic_failures": list(
                self.deterministic_failures
            ),
            "deterministic_metrics": self.metrics,
            "semantic": self.semantic,
            "run_metrics": self.run_metrics,
        }


@dataclass
class BenchmarkReport:
    """The full benchmark report (JSON-serializable)."""

    cases: tuple[CaseResult, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> int:
        return sum(1 for case in self.cases if case.passed)

    @property
    def failed(self) -> int:
        return len(self.cases) - self.passed

    @property
    def false_approvals(self) -> list[str]:
        return [
            case.case_id
            for case in self.cases
            if case.semantic.get("reviewer_false_approval")
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata,
            "summary": {
                "total": len(self.cases),
                "deterministic_passed": self.passed,
                "deterministic_failed": self.failed,
                "reviewer_false_approvals": (
                    self.false_approvals
                ),
            },
            "cases": [case.to_dict() for case in self.cases],
        }


def _deterministic_metrics(
    case: EvalCase,
    result: dict[str, Any],
    fake_tools: FakeTools,
    metrics: MetricsCollector,
    model: ScriptedChatModel,
) -> dict[str, Any]:
    """Compute the deterministic metric set for a run."""
    findings = result.get("findings") or []
    answer = result.get("final_answer") or ""
    draft = strip_marked_notes(answer)
    citation_report = validate_citations(
        draft, findings, case.question
    )
    completeness = check_completeness(case.question, draft)
    official = official_source_status(findings, case.question)
    snapshot = metrics.snapshot()

    # budget compliance: tool calls per research pass
    per_iteration: dict[int, int] = {}
    for call in snapshot["tool_calls"]:
        iteration = call.get("iteration") or 0
        per_iteration[iteration] = (
            per_iteration.get(iteration, 0) + 1
        )
    max_calls_per_pass = (
        max(per_iteration.values()) if per_iteration else 0
    )

    # official-domain steering (official questions only):
    # did the planner and the research rounds see
    # current-domain guidance? Only the presence of
    # the guidance TEXT is checked, never prompt
    # contents beyond that.
    plan_steered = False
    research_steered = False
    if requests_official_sources(case.question):
        if model.seen_structured_messages:
            plan_steered = "docs.langchain.com" in str(
                model.seen_structured_messages[0]
            )
        research_steered = any(
            "Official-source guidance" in str(messages)
            for messages in model.seen_messages
        )
    return {
        "workflow_completed": (
            bool(answer)
            and result.get("review_status")
            != ReviewStatus.FAILED
        ),
        "review_status": (
            result.get("review_status").value
            if hasattr(result.get("review_status"), "value")
            else result.get("review_status")
        ),
        "iterations": result.get("iteration", 0),
        "searches": len(fake_tools.search_calls),
        "fetches": len(fake_tools.fetch_calls),
        "findings": len(findings),
        "retrieved_sources": len(
            {
                f.get("source_url")
                for f in findings
                if f.get("evidence") == "retrieved"
            }
        ),
        "official_source_status": official.value,
        "citations_ok": citation_report.ok,
        "unknown_citations": len(citation_report.unknown),
        "malformed_citations": len(citation_report.malformed),
        "missing_topics": list(completeness.missing),
        "markers_present": {
            marker: marker in answer
            for marker in case.expected.markers
        },
        "plan_steered_to_current_domains": plan_steered,
        "research_steered_to_current_domains": (
            research_steered
        ),
        "max_tool_calls_per_pass": max_calls_per_pass,
        "budget_compliant": (
            max_calls_per_pass
            <= _case_settings(case).max_tool_calls_per_research
        ),
        "tokens_in": result.get("tokens_in", 0),
        "tokens_out": result.get("tokens_out", 0),
        "prompt_chars_total": snapshot["totals"]["prompt_chars"],
        "prompt_chars_by_node": {
            node: entry["prompt_chars"]
            for node, entry in snapshot["tokens_by_node"].items()
        },
        "model_calls": snapshot["totals"]["model_calls"],
        "tool_calls": snapshot["totals"]["tool_calls"],
        "retries": snapshot["totals"]["retries"],
        "total_latency_ms": sum(
            node["latency_ms"] for node in snapshot["nodes"]
        ),
        "tokens_by_node": snapshot["tokens_by_node"],
        "tool_error_categories": snapshot["counts"][
            "tool_error_categories"
        ],
        "run_id": snapshot["run_id"],
    }


def run_case(
    case: EvalCase,
    *,
    settings: Settings | None = None,
) -> CaseResult:
    """Run one case offline and evaluate it."""
    case_settings = settings or _case_settings(case)
    fake_tools = FakeTools(
        search_results=list(case.search_results),
        pages=dict(case.pages),
        fail_fetch_urls=set(case.fail_fetch_urls),
        fail_queries=set(case.fail_queries),
    )
    model = _scripted_model(case)
    metrics = MetricsCollector()
    graph = build_research_graph(
        case_settings,
        model,
        fake_tools,
        # zero backoff: retry delay never changes outcomes and
        # the offline benchmark should not sleep 2s per retry
        rate_limit_retry_delay=0.0,
        metrics=metrics,
    )
    result = graph.invoke({"question": case.question})

    deterministic = _deterministic_metrics(
        case, result, fake_tools, metrics, model
    )
    failures: list[str] = []
    expected = case.expected

    if deterministic["review_status"] != expected.review_status:
        failures.append(
            f"review_status: expected {expected.review_status}, "
            f"got {deterministic['review_status']}"
        )
    for marker, present in deterministic[
        "markers_present"
    ].items():
        if not present:
            failures.append(
                f"marker missing from answer: {marker!r}"
            )
    actual_missing = set(deterministic["missing_topics"])
    if not set(expected.missing_topics) <= actual_missing:
        failures.append(
            "missing_topics: expected at least "
            f"{sorted(expected.missing_topics)}, "
            f"got {sorted(actual_missing)}"
        )
    if (
        expected.official_status is not None
        and deterministic["official_source_status"]
        != expected.official_status
    ):
        failures.append(
            "official_source_status: expected "
            f"{expected.official_status}, got "
            f"{deterministic['official_source_status']}"
        )
    if expected.unverified_marker and (
        "UNVERIFIED / INCOMPLETE"
        not in (result.get("final_answer") or "")
    ):
        failures.append("expected UNVERIFIED marker in answer")
    if expected.official_domain_steering and not (
        deterministic["plan_steered_to_current_domains"]
        and deterministic["research_steered_to_current_domains"]
    ):
        failures.append(
            "expected current-domain steering in plan and "
            "research prompts"
        )
    if not deterministic["workflow_completed"]:
        failures.append("workflow did not complete")
    if not deterministic["budget_compliant"]:
        failures.append(
            "tool-call budget exceeded in a research pass"
        )

    semantic = evaluate_semantics(
        case,
        strip_marked_notes(result.get("final_answer") or ""),
        reviewer_approved=(
            deterministic["review_status"] == "approved"
        ),
    )

    return CaseResult(
        case_id=case.case_id,
        category=case.category,
        question=case.question,
        deterministic_pass=not failures,
        deterministic_failures=tuple(failures),
        metrics=deterministic,
        semantic=semantic.to_dict(),
        run_metrics=metrics.snapshot(),
    )


def _git_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
        )
    except Exception:
        return "unknown"


def run_benchmark(
    cases: tuple[EvalCase, ...] | None = None,
) -> BenchmarkReport:
    """Run every case and build the JSON report."""
    from evals.cases import benchmark_cases

    evaluated = cases if cases is not None else benchmark_cases()
    results = tuple(run_case(case) for case in evaluated)
    return BenchmarkReport(
        cases=results,
        metadata={
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "python": os.environ.get("PYTHON_VERSION", "n/a"),
            "judge_backend": "fixture (LLM judge disabled)",
            "live_provider_calls": False,
            "token_source": (
                "synthetic (fake model estimates "
                "input_tokens ~= prompt chars / 4; "
                "structured plan/review calls report 0 "
                "tokens by design)"
            ),
            "case_count": len(results),
        },
    )


def summarize(report: BenchmarkReport) -> str:
    """Human-readable summary of a benchmark report."""
    lines = [
        f"Benchmark: {report.metadata.get('case_count')} cases "
        f"(commit {report.metadata.get('git_commit')})",
        (
            f"deterministic: {report.passed} passed, "
            f"{report.failed} failed"
        ),
    ]
    false_approvals = report.false_approvals
    if false_approvals:
        lines.append(
            "reviewer FALSE APPROVALS (semantically "
            f"inadequate but approved): "
            + ", ".join(false_approvals)
        )
    else:
        lines.append("reviewer false approvals: none")
    for case in report.cases:
        status = "PASS" if case.passed else "FAIL"
        lines.append(
            f"- [{status}] {case.case_id} ({case.category})"
        )
        for failure in case.deterministic_failures:
            lines.append(f"    ! {failure}")
        if case.semantic.get("inadequate_topics"):
            lines.append(
                "    ~ semantically inadequate topics: "
                + ", ".join(case.semantic["inadequate_topics"])
            )
        metrics = case.metrics
        lines.append(
            "    tokens: {tokens_in} in / {tokens_out} out, "
            "prompt {prompt_chars_total} chars, "
            "{model_calls} model calls, {retries} retries, "
            "review={review_status}".format(**metrics)
        )
    return "\n".join(lines)
