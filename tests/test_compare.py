"""WS-F: ``python -m evals.compare`` — the run/benchmark comparison
utility.

Covers both report shapes (MetricsCollector run reports and eval
benchmark reports), regression detection (outcome severity, quality
gates, case flips, false approvals, failure-category increases) and
the CLI contract (table / --json / --fail-on-regression exit codes).
No databases, dashboards or network — stdlib in, stdlib out.
"""
from __future__ import annotations

import json

import pytest

from evals.compare import (
    compare_reports,
    detect_kind,
    format_comparison,
    load_report,
    main,
)


def _run_report(**overrides) -> dict:
    base: dict = {
        "run_id": "baseline001",
        "total_execution_seconds": 42.0,
        "totals": {
            "input_tokens": 1000,
            "output_tokens": 200,
            "model_calls": 6,
            "tool_calls": 4,
            "retries": 0,
        },
        "counts": {
            "failed_tool_calls": 1,
            "search_success_rate": 1.0,
            "fetch_success_rate": 0.5,
            "tool_error_categories": {"empty_results": 1},
        },
        "token_usage": {"coverage_ratio": 1.0},
        "outcome": {"status": "COMPLETED", "warnings": []},
        "deadline": {"configured_seconds": 180.0, "exceeded": False},
        "quality": {
            "citations_ok": True,
            "coverage_ok": True,
            "review_status": "approved",
            "retrieved_sources": 2,
        },
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return base


def _eval_report(**overrides) -> dict:
    base: dict = {
        "metadata": {"git_commit": "abc1234"},
        "summary": {
            "total": 3,
            "deterministic_passed": 3,
            "deterministic_failed": 0,
            "reviewer_false_approvals": [],
        },
        "cases": [
            {
                "case_id": f"case-{n}",
                "deterministic_pass": True,
                "deterministic_metrics": {
                    "tokens_in": 100,
                    "tokens_out": 20,
                    "total_latency_ms": 50,
                    "retries": 0,
                    "retrieved_sources": 1,
                    "tool_error_categories": {},
                },
            }
            for n in range(3)
        ],
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = {**base[key], **value}
        else:
            base[key] = value
    return base


# -- kind detection -------------------------------------------------------


def test_detect_kind_for_both_report_shapes():
    assert detect_kind(_run_report()) == "run"
    assert detect_kind(_eval_report()) == "eval"
    with pytest.raises(ValueError):
        detect_kind({"something": "else"})


def test_kind_mismatch_is_rejected():
    with pytest.raises(ValueError, match="kinds differ"):
        compare_reports(_run_report(), _eval_report())


# -- run-report comparison ------------------------------------------------


def test_identical_run_reports_have_no_regressions():
    diff = compare_reports(_run_report(), _run_report())
    assert diff["kind"] == "run"
    assert diff["regressions"] == []
    assert diff["improvements"] == []
    assert diff["metrics"]["input_tokens"]["delta"] == 0


def test_run_report_token_and_latency_deltas():
    candidate = _run_report(
        total_execution_seconds=31.5,
        totals={"input_tokens": 800, "output_tokens": 240},
    )
    diff = compare_reports(_run_report(), candidate)

    tokens = diff["metrics"]["input_tokens"]
    assert tokens["baseline"] == 1000
    assert tokens["candidate"] == 800
    assert tokens["delta"] == -200
    assert tokens["delta_pct"] == -20.0
    seconds = diff["metrics"]["total_execution_seconds"]
    assert seconds["delta"] == pytest.approx(-10.5)


def test_run_report_outcome_severity_regression():
    diff = compare_reports(
        _run_report(),
        _run_report(outcome={"status": "TIMED_OUT", "warnings": []}),
    )
    assert "outcome: COMPLETED -> TIMED_OUT" in diff["regressions"]


def test_run_report_outcome_improvement_direction():
    diff = compare_reports(
        _run_report(outcome={"status": "FAILED", "warnings": []}),
        _run_report(outcome={"status": "COMPLETED", "warnings": []}),
    )
    assert diff["regressions"] == []
    assert any("FAILED -> COMPLETED" in item for item in diff["improvements"])


def test_run_report_tool_failure_increase_is_a_regression():
    diff = compare_reports(
        _run_report(),
        _run_report(
            counts={"tool_error_categories": {"empty_results": 4}}
        ),
    )
    entry = diff["tool_error_categories"]["empty_results"]
    assert entry["delta"] == 3
    assert any(
        "empty_results" in item for item in diff["regressions"]
    )


def test_run_report_quality_and_deadline_regressions():
    diff = compare_reports(
        _run_report(),
        _run_report(
            quality={
                "citations_ok": False,
                "review_status": "needs_more_research",
            },
            deadline={"exceeded": True},
        ),
    )
    assert "quality [citations_ok]: True -> False" in diff["regressions"]
    assert any(
        "review_status" in item for item in diff["regressions"]
    )
    assert any(
        "deadline" in item for item in diff["regressions"]
    )


def test_missing_fields_in_old_reports_stay_none():
    # a pre-Phase-4B report: no outcome/deadline/token_usage, and no
    # failure categories recorded yet
    old = {
        "run_id": "old",
        "totals": {"input_tokens": 100},
        "counts": {"tool_error_categories": {}},
    }
    diff = compare_reports(old, _run_report())
    assert diff["metrics"]["input_tokens"]["delta"] == 900
    assert diff["metrics"]["token_coverage_ratio"]["delta"] is None
    # candidate has one empty_results failure the old report never
    # tracked — surfaced, not silently dropped
    assert any("empty_results" in r for r in diff["regressions"])


# -- eval-report comparison ------------------------------------------------


def test_eval_case_flip_is_a_regression_and_fix_is_an_improvement():
    baseline = _eval_report()
    regressed = _eval_report(
        summary={
            "deterministic_passed": 2,
            "deterministic_failed": 1,
        },
    )
    regressed["cases"][1]["deterministic_pass"] = False

    diff = compare_reports(baseline, regressed)
    assert diff["case_regressions"] == ["case-1"]
    assert any("case regressed: case-1" in r for r in diff["regressions"])

    fixed = compare_reports(regressed, baseline)
    assert fixed["case_fixes"] == ["case-1"]
    assert fixed["regressions"] == []


def test_eval_false_approval_added_is_a_regression():
    baseline = _eval_report()
    candidate = _eval_report(
        summary={"reviewer_false_approvals": ["case-2"]}
    )

    diff = compare_reports(baseline, candidate)

    assert diff["reviewer_false_approvals"]["added"] == ["case-2"]
    assert any(
        "false approval added: case-2" in r
        for r in diff["regressions"]
    )


def test_eval_aggregate_token_totals():
    baseline = _eval_report()
    candidate = _eval_report()
    for case in candidate["cases"]:
        case["deterministic_metrics"]["tokens_in"] = 150

    diff = compare_reports(baseline, candidate)

    tokens = diff["metrics"]["tokens_in_total"]
    assert tokens["baseline"] == 300
    assert tokens["candidate"] == 450
    assert tokens["delta"] == 150


# -- presentation and CLI ---------------------------------------------------


def test_format_comparison_renders_table():
    diff = compare_reports(
        _run_report(),
        _run_report(outcome={"status": "FAILED", "warnings": []}),
        baseline_path="a.json",
        candidate_path="b.json",
    )
    text = format_comparison(diff)

    assert "report comparison (run)" in text
    assert "input_tokens" in text
    assert "tool failure categories" in text
    assert "outcome: COMPLETED -> FAILED" in text
    assert "improvements:" in text
    # never prints report content — only metric names/values
    assert "http" not in text


def test_main_json_output_and_exit_codes(tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps(_run_report()), encoding="utf-8")
    candidate.write_text(
        json.dumps(_run_report(outcome={"status": "FAILED"})),
        encoding="utf-8",
    )

    code = main(["--json", str(baseline), str(candidate)])
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["kind"] == "run"
    assert payload["regressions"]

    code = main(
        ["--fail-on-regression", str(baseline), str(candidate)]
    )
    assert code == 1


def test_main_reports_unusable_inputs(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"nope": 1}), encoding="utf-8")

    code = main([str(bad), str(bad)])
    captured = capsys.readouterr()
    assert code == 2
    assert "compare failed" in captured.err


def test_load_report_rejects_non_object_root(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError):
        load_report(str(path))
