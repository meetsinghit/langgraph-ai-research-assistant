"""``python -m evals.compare BASELINE CANDIDATE`` — compare two saved
JSON reports and highlight regressions.

Two report shapes are recognised (both must be the same kind):

- **run reports** — ``MetricsCollector`` snapshots written by
  ``python main.py --metrics PATH``: token usage and measurement
  coverage, latency, tool success rates and failure categories,
  retries, run outcome, deadline exhaustion and quality-gate status;
- **eval reports** — ``python -m evals --out PATH``: deterministic
  pass counts per case (including pass -> fail flips), aggregate
  tokens/latency/retries, tool-error categories and reviewer false
  approvals.

Output is a console table by default or ``--json`` for tooling.
``--fail-on-regression`` exits 1 when any regression was detected
(handy in CI). Deliberately stdlib-only: no databases, no dashboards,
no network, no paid services. Reports are compared as data — nothing
from prompts, pages or answers is ever printed or stored.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

__all__ = [
    "compare_reports",
    "detect_kind",
    "format_comparison",
    "load_report",
    "main",
]

#: Worse outcomes rank higher; a candidate outranking the baseline
#: outcome is a regression (e.g. COMPLETED -> TIMED_OUT).
_OUTCOME_RANK: dict[str, int] = {
    "COMPLETED": 0,
    "COMPLETED_WITH_WARNINGS": 1,
    "INCOMPLETE": 2,
    "TIMED_OUT": 3,
    "FAILED": 4,
    "UNSPECIFIED": 2,
}


def load_report(path: str) -> dict[str, Any]:
    """Load a JSON report; raises ValueError on a non-object root."""
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    if not isinstance(report, dict):
        raise ValueError(f"{path}: report root must be a JSON object")
    return report


def detect_kind(report: dict[str, Any]) -> str:
    """``"eval"`` for benchmark reports, ``"run"`` for run reports."""
    if "cases" in report and "summary" in report:
        return "eval"
    if "totals" in report or "token_usage" in report:
        return "run"
    raise ValueError(
        "unrecognised report shape: expected an eval report "
        "(cases + summary) or a run report (totals)"
    )


def _delta(
    baseline: Any, candidate: Any
) -> dict[str, Any]:
    """Baseline/candidate/delta/delta_pct for one scalar metric.

    Non-numeric or missing values stay ``None`` (an old report
    lacking a field is skipped, never guessed).
    """
    if not isinstance(baseline, (int, float)) or not isinstance(
        candidate, (int, float)
    ):
        return {
            "baseline": baseline,
            "candidate": candidate,
            "delta": None,
            "delta_pct": None,
        }
    delta = candidate - baseline
    pct = round(100.0 * delta / baseline, 2) if baseline else None
    return {
        "baseline": baseline,
        "candidate": candidate,
        "delta": round(delta, 3),
        "delta_pct": pct,
    }


def _get(report: dict[str, Any], *path: str) -> Any:
    """Nested dict lookup; None when any level is missing."""
    node: Any = report
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _sum_cases(
    report: dict[str, Any], key: str
) -> int:
    total = 0
    for case in report.get("cases") or []:
        value = _get(case, "deterministic_metrics", key)
        if isinstance(value, (int, float)):
            total += value
    return total


def _merge_error_categories(
    report: dict[str, Any],
) -> dict[str, int]:
    merged: dict[str, int] = {}
    for case in report.get("cases") or []:
        categories = _get(case, "deterministic_metrics",
                          "tool_error_categories") or {}
        if isinstance(categories, dict):
            for name, count in categories.items():
                if isinstance(count, (int, float)):
                    merged[str(name)] = merged.get(str(name), 0) + int(count)
    return merged


# ---------------------------------------------------------------------------
# run-report comparison
# ---------------------------------------------------------------------------


def _compare_categories(
    baseline: dict[str, Any], candidate: dict[str, Any],
    regressions: list[str], improvements: list[str],
) -> dict[str, dict[str, Any]]:
    names = sorted(
        set(baseline) | set(candidate),
        key=str,
    )
    result: dict[str, dict[str, Any]] = {}
    for name in names:
        entry = _delta(int(baseline.get(name, 0)),
                       int(candidate.get(name, 0)))
        result[name] = entry
        delta = entry["delta"]
        if isinstance(delta, int) and delta != 0:
            label = (
                f"tool failures [{name}]: "
                f"{entry['baseline']} -> {entry['candidate']}"
            )
            (regressions if delta > 0 else improvements).append(label)
    return result


def compare_run_reports(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Diff two run reports (MetricsCollector snapshots)."""
    regressions: list[str] = []
    improvements: list[str] = []

    metrics = {
        "total_execution_seconds": _delta(
            baseline.get("total_execution_seconds"),
            candidate.get("total_execution_seconds"),
        ),
        "input_tokens": _delta(
            _get(baseline, "totals", "input_tokens"),
            _get(candidate, "totals", "input_tokens"),
        ),
        "output_tokens": _delta(
            _get(baseline, "totals", "output_tokens"),
            _get(candidate, "totals", "output_tokens"),
        ),
        "model_calls": _delta(
            _get(baseline, "totals", "model_calls"),
            _get(candidate, "totals", "model_calls"),
        ),
        "tool_calls": _delta(
            _get(baseline, "totals", "tool_calls"),
            _get(candidate, "totals", "tool_calls"),
        ),
        "retries": _delta(
            _get(baseline, "totals", "retries"),
            _get(candidate, "totals", "retries"),
        ),
        "failed_tool_calls": _delta(
            _get(baseline, "counts", "failed_tool_calls"),
            _get(candidate, "counts", "failed_tool_calls"),
        ),
        "search_success_rate": _delta(
            _get(baseline, "counts", "search_success_rate"),
            _get(candidate, "counts", "search_success_rate"),
        ),
        "fetch_success_rate": _delta(
            _get(baseline, "counts", "fetch_success_rate"),
            _get(candidate, "counts", "fetch_success_rate"),
        ),
        "token_coverage_ratio": _delta(
            _get(baseline, "token_usage", "coverage_ratio"),
            _get(candidate, "token_usage", "coverage_ratio"),
        ),
        "retrieved_sources": _delta(
            _get(baseline, "quality", "retrieved_sources"),
            _get(candidate, "quality", "retrieved_sources"),
        ),
    }

    tool_error_categories = _compare_categories(
        _get(baseline, "counts", "tool_error_categories") or {},
        _get(candidate, "counts", "tool_error_categories") or {},
        regressions,
        improvements,
    )

    # outcome severity (a worse terminal status is a regression)
    b_outcome = _get(baseline, "outcome", "status")
    c_outcome = _get(candidate, "outcome", "status")
    if b_outcome and c_outcome and b_outcome != c_outcome:
        label = f"outcome: {b_outcome} -> {c_outcome}"
        b_rank = _OUTCOME_RANK.get(str(b_outcome), 2)
        c_rank = _OUTCOME_RANK.get(str(c_outcome), 2)
        (regressions if c_rank > b_rank else improvements).append(label)

    # deadline exhaustion is always worth surfacing
    if _get(candidate, "deadline", "exceeded") and not _get(
        baseline, "deadline", "exceeded"
    ):
        regressions.append("global deadline newly exhausted in candidate")

    # quality-gate status flips (True -> False is a regression)
    for field in ("citations_ok", "coverage_ok"):
        b_val = _get(baseline, "quality", field)
        c_val = _get(candidate, "quality", field)
        if b_val is True and c_val is False:
            regressions.append(f"quality [{field}]: True -> False")
        elif b_val is False and c_val is True:
            improvements.append(f"quality [{field}]: False -> True")
    b_review = _get(baseline, "quality", "review_status")
    c_review = _get(candidate, "quality", "review_status")
    if b_review and c_review and b_review != c_review:
        label = f"quality [review_status]: {b_review} -> {c_review}"
        if b_review == "approved":
            regressions.append(label)
        elif c_review == "approved":
            improvements.append(label)

    return {
        "kind": "run",
        "metrics": metrics,
        "tool_error_categories": tool_error_categories,
        "regressions": regressions,
        "improvements": improvements,
    }


# ---------------------------------------------------------------------------
# eval-report comparison
# ---------------------------------------------------------------------------


def compare_eval_reports(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Diff two benchmark reports, case by case."""
    regressions: list[str] = []
    improvements: list[str] = []

    metrics = {
        "deterministic_passed": _delta(
            _get(baseline, "summary", "deterministic_passed"),
            _get(candidate, "summary", "deterministic_passed"),
        ),
        "deterministic_failed": _delta(
            _get(baseline, "summary", "deterministic_failed"),
            _get(candidate, "summary", "deterministic_failed"),
        ),
        "tokens_in_total": _delta(
            _sum_cases(baseline, "tokens_in"),
            _sum_cases(candidate, "tokens_in"),
        ),
        "tokens_out_total": _delta(
            _sum_cases(baseline, "tokens_out"),
            _sum_cases(candidate, "tokens_out"),
        ),
        "total_latency_ms": _delta(
            _sum_cases(baseline, "total_latency_ms"),
            _sum_cases(candidate, "total_latency_ms"),
        ),
        "retries_total": _delta(
            _sum_cases(baseline, "retries"),
            _sum_cases(candidate, "retries"),
        ),
        "retrieved_sources_total": _delta(
            _sum_cases(baseline, "retrieved_sources"),
            _sum_cases(candidate, "retrieved_sources"),
        ),
    }

    tool_error_categories = _compare_categories(
        _merge_error_categories(baseline),
        _merge_error_categories(candidate),
        regressions,
        improvements,
    )

    # per-case deterministic flips
    b_cases = {
        case.get("case_id"): case
        for case in baseline.get("cases") or []
    }
    c_cases = {
        case.get("case_id"): case
        for case in candidate.get("cases") or []
    }
    regressed: list[str] = []
    fixed: list[str] = []
    for case_id in sorted(set(b_cases) & set(c_cases), key=str):
        b_pass = b_cases[case_id].get("deterministic_pass")
        c_pass = c_cases[case_id].get("deterministic_pass")
        if b_pass and c_pass is False:
            regressed.append(str(case_id))
        elif b_pass is False and c_pass:
            fixed.append(str(case_id))
    for case_id in sorted(set(c_cases) - set(b_cases), key=str):
        if c_cases[case_id].get("deterministic_pass") is False:
            regressed.append(f"{case_id} (new)")
    for case_id in sorted(set(b_cases) - set(c_cases), key=str):
        improvements.append(f"{case_id} (removed from benchmark)")

    regressions = [
        f"case regressed: {case_id}" for case_id in regressed
    ] + regressions
    improvements = [
        f"case fixed: {case_id}" for case_id in fixed
    ] + improvements

    # reviewer false approvals: additions are regressions
    b_false = list(
        _get(baseline, "summary", "reviewer_false_approvals") or []
    )
    c_false = list(
        _get(candidate, "summary", "reviewer_false_approvals") or []
    )
    added = [x for x in c_false if x not in b_false]
    removed = [x for x in b_false if x not in c_false]
    for case_id in added:
        regressions.append(f"reviewer false approval added: {case_id}")
    for case_id in removed:
        improvements.append(f"reviewer false approval removed: {case_id}")

    return {
        "kind": "eval",
        "metrics": metrics,
        "tool_error_categories": tool_error_categories,
        "case_regressions": regressed,
        "case_fixes": fixed,
        "reviewer_false_approvals": {
            "baseline": b_false,
            "candidate": c_false,
            "added": added,
            "removed": removed,
        },
        "regressions": regressions,
        "improvements": improvements,
    }


def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    baseline_path: str = "",
    candidate_path: str = "",
) -> dict[str, Any]:
    """Compare two reports of the same kind (kind auto-detected)."""
    baseline_kind = detect_kind(baseline)
    candidate_kind = detect_kind(candidate)
    if baseline_kind != candidate_kind:
        raise ValueError(
            f"report kinds differ: baseline is {baseline_kind!r}, "
            f"candidate is {candidate_kind!r}"
        )
    if baseline_kind == "eval":
        diff = compare_eval_reports(baseline, candidate)
    else:
        diff = compare_run_reports(baseline, candidate)
    diff["baseline"] = {
        "path": baseline_path,
        "run_id": baseline.get("run_id")
        or _get(baseline, "metadata", "git_commit"),
    }
    diff["candidate"] = {
        "path": candidate_path,
        "run_id": candidate.get("run_id")
        or _get(candidate, "metadata", "git_commit"),
    }
    return diff


# ---------------------------------------------------------------------------
# presentation
# ---------------------------------------------------------------------------


def _fmt(entry: dict[str, Any]) -> str:
    base = _short(entry.get("baseline"))
    cand = _short(entry.get("candidate"))
    if entry.get("delta") is None:
        return f"{base:>14} -> {cand:<14}"
    delta = entry["delta"]
    pct = entry.get("delta_pct")
    pct_text = f" ({pct:+.2f}%)" if isinstance(pct, (int, float)) else ""
    return f"{base:>14} -> {cand:<14} {delta:+g}{pct_text}"


def _short(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return str(value)


def format_comparison(diff: dict[str, Any]) -> str:
    """Human-readable console table (stdlib string formatting only)."""
    kind = diff.get("kind", "?")
    lines = [
        f"report comparison ({kind})",
        f"baseline : {diff['baseline']['path'] or '(memory)'}"
        + (f"  [{diff['baseline']['run_id']}]" if diff["baseline"].get("run_id") else ""),
        f"candidate: {diff['candidate']['path'] or '(memory)'}"
        + (f"  [{diff['candidate']['run_id']}]" if diff["candidate"].get("run_id") else ""),
        "",
        f"{'metric':<32} {'baseline':>14} {'candidate':<14}  delta",
        "-" * 84,
    ]
    for name, entry in (diff.get("metrics") or {}).items():
        lines.append(f"{name:<32} {_fmt(entry)}")
    categories = diff.get("tool_error_categories") or {}
    if categories:
        lines.append("")
        lines.append("tool failure categories:")
        for name, entry in categories.items():
            lines.append(f"  {name:<30} {_fmt(entry)}")
    if diff.get("case_regressions") is not None:
        lines.append("")
        lines.append(
            "case regressions: "
            + (", ".join(diff["case_regressions"]) or "(none)")
        )
        lines.append(
            "case fixes      : "
            + (", ".join(diff.get("case_fixes") or []) or "(none)")
        )
        false = diff.get("reviewer_false_approvals") or {}
        lines.append(
            "false approvals : "
            + f"baseline {false.get('baseline')}, "
            + f"candidate {false.get('candidate')}"
        )
    lines.append("")
    lines.append(
        "regressions : " + ("\n  - " if diff["regressions"] else "(none)")
        + "\n  - ".join(diff["regressions"])
    )
    lines.append(
        "improvements: " + ("\n  - " if diff["improvements"] else "(none)")
        + "\n  - ".join(diff["improvements"])
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evals.compare",
        description=(
            "Compare two saved run or evaluation reports and "
            "highlight regressions."
        ),
    )
    parser.add_argument("baseline", help="baseline JSON report")
    parser.add_argument("candidate", help="candidate JSON report")
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="emit machine-readable JSON instead of the table",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="exit 1 when any regression is detected",
    )
    args = parser.parse_args(argv)

    baseline = load_report(args.baseline)
    candidate = load_report(args.candidate)
    try:
        diff = compare_reports(
            baseline,
            candidate,
            baseline_path=args.baseline,
            candidate_path=args.candidate,
        )
    except ValueError as exc:
        print(f"compare failed: {exc}", file=sys.stderr)
        return 2
    if args.as_json:
        print(json.dumps(diff, indent=2))
    else:
        print(format_comparison(diff))
    if args.fail_on_regression and diff["regressions"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
