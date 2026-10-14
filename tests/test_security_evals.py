"""The Phase 4C adversarial evaluation suite itself (WS-G).

Guards the suite's integrity and honesty rules:
- every case is offline (scripted model, fixture sources, no keys);
- every case carries security metadata (objective, entry point,
  defense layer, severity, residual limitation);
- case count is >= 20 and pass criteria lean on deterministic
  controls wherever observable;
- the machine-readable security summary is structure-complete and
  content-free (no fixture credential ever appears in the report);
- runs are deterministic: the same case passes twice in a row.
"""
from __future__ import annotations

import json

from evals.cases import benchmark_cases
from evals.harness import run_benchmark, run_case
from evals.security_cases import (
    FAKE_KEY,
    security_cases,
    security_summary,
    summarize_security,
)

_DEFENSE_LAYERS = {"deterministic", "model_behavior"}
_SEVERITIES = {"high", "medium", "low"}
_ENTRY_POINTS = {
    "search-snippet",
    "search-title",
    "fetched-page",
    "tool-error",
    "model-tool-call",
    "model-response-text",
    "draft-answer",
    "fetch-failure",
    "page-instruction",
}


def test_suite_size_meets_the_20_case_minimum():
    cases = security_cases()
    assert len(cases) >= 20


def test_every_case_has_complete_security_metadata():
    for case in security_cases():
        meta = case.security
        assert meta is not None, f"{case.case_id} lacks security metadata"
        assert meta.attacker_objective
        assert meta.entry_point in _ENTRY_POINTS, case.case_id
        assert meta.expected_behavior
        assert meta.defense_layer in _DEFENSE_LAYERS, case.case_id
        assert meta.severity in _SEVERITIES, case.case_id
        assert meta.residual, f"{case.case_id} lacks a residual-limits note"


def test_security_cases_are_not_benchmark_cases():
    """Same EvalCase shape, disjoint ids — the benchmark catalog
    (23 as of Phase 4D; 30 since the Phase 4E.1 WS-A/WS-B/WS-G
    fixtures) never contains an adversarial case, and vice versa."""
    security_ids = {case.case_id for case in security_cases()}
    benchmark_ids = {case.case_id for case in benchmark_cases()}
    assert not (security_ids & benchmark_ids)
    assert len(benchmark_ids) >= 23
    assert len(benchmark_ids) == 30


def test_deterministic_defense_cases_outnumber_behavioral_ones():
    """Most pass criteria must rest on code-enforced controls —
    model-behavior cases exist to keep the limitation visible, not
    to carry the suite."""
    cases = security_cases()
    deterministic = [
        case for case in cases if case.security.defense_layer == "deterministic"
    ]
    assert len(deterministic) >= 15


def test_all_security_cases_pass_offline():
    report = run_benchmark(security_cases(), suite="security")
    failures = [
        (case.case_id, case.deterministic_failures)
        for case in report.cases
        if not case.deterministic_pass
    ]
    assert failures == []


def test_report_metadata_labels_the_suite():
    report = run_benchmark(security_cases(), suite="security")
    payload = report.to_dict()
    assert payload["metadata"]["suite"] == "security"
    assert payload["metadata"]["live_provider_calls"] is False
    default_report = run_benchmark().to_dict()
    assert default_report["metadata"]["suite"] == "benchmark"


def test_security_summary_shape_and_totals():
    report = run_benchmark(security_cases(), suite="security")
    summary = security_summary(report)
    assert summary["total_cases"] == len(security_cases())
    assert summary["passed"] == summary["total_cases"]
    assert summary["failed"] == 0
    assert sum(summary["by_severity"].values()) == summary["total_cases"]
    assert (
        sum(summary["by_defense_layer"].values()) == summary["total_cases"]
    )
    assert summary["by_entry_point"]
    # deterministic refusals actually fired during the suite
    assert summary["refusals_by_category"].get("policy", 0) >= 1
    assert summary["refusals_by_category"].get("validation", 0) >= 1
    assert summary["refusals_by_category"].get("dlp", 0) >= 1
    assert summary["injection_signals_total"] >= 10
    assert summary["answer_redactions_total"] >= 2
    # per-case entries carry metadata for downstream tooling
    for entry in summary["cases"]:
        assert entry["case_id"]
        assert entry["severity"] in _SEVERITIES
        assert entry["defense_layer"] in _DEFENSE_LAYERS
        assert entry["residual"]


def test_security_report_is_content_free():
    """The machine-readable report must never carry the fixture
    credential (or any credential-shaped value) — the DLP/redaction
    cases use a synthetic key, and it stays in fixtures."""
    report = run_benchmark(security_cases(), suite="security")
    payload = json.dumps(report.to_dict(), default=str)
    assert FAKE_KEY not in payload
    summary = json.dumps(security_summary(report))
    assert FAKE_KEY not in summary


def test_security_summary_console_lines_stay_count_only():
    report = run_benchmark(security_cases(), suite="security")
    text = "\n".join(summarize_security(report))
    assert FAKE_KEY not in text
    assert "passed" in text
    assert "deterministic" in text
    # the honesty note about mocked fixtures is always printed
    assert "cannot prove real-model" in text


def test_case_runs_are_deterministic():
    """The same adversarial case produces the same verdict on a
    second run (scripted model, fixed fixtures, clock-faked)."""
    case = security_cases()[0]
    first = run_case(case)
    second = run_case(case)
    assert first.deterministic_pass == second.deterministic_pass
    assert first.deterministic_failures == second.deterministic_failures
