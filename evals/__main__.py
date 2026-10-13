"""``python -m evals`` — run the offline benchmark.

Options:
    --out PATH   also write the machine-readable JSON
                 report to PATH (default: evals/report.json)
    --no-out     do not write a report file
    --security   run the Phase 4C adversarial security suite
                 (25 offline cases) instead of the quality
                 benchmark; report default: evals/security_report.json

The benchmark never makes live provider calls: every
case runs on scripted model responses and fixture
search/page data. The LLM-as-judge rubric scorer is
disabled unless EVAL_LLM_JUDGE=1 is set explicitly.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from evals.harness import run_benchmark, summarize

DEFAULT_REPORT = os.path.join(
    os.path.dirname(__file__), "report.json"
)
DEFAULT_SECURITY_REPORT = os.path.join(
    os.path.dirname(__file__), "security_report.json"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Offline evaluation benchmark "
            "(no live provider calls)"
        )
    )
    parser.add_argument(
        "--out",
        default=None,
        help="write the JSON report to this path",
    )
    parser.add_argument(
        "--no-out",
        action="store_true",
        help="do not write a report file",
    )
    parser.add_argument(
        "--security",
        action="store_true",
        help=(
            "run the Phase 4C adversarial security suite "
            "instead of the quality benchmark"
        ),
    )
    args = parser.parse_args(argv)

    if args.security:
        from evals.security_cases import (
            security_cases,
            security_summary,
            summarize_security,
        )

        report = run_benchmark(security_cases(), suite="security")
        print(summarize(report))
        print()
        for line in summarize_security(report):
            print(line)
        out_path = args.out or DEFAULT_SECURITY_REPORT
        if not args.no_out:
            payload = report.to_dict()
            payload["security_summary"] = security_summary(report)
            with open(out_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            print(f"\nJSON report written to {out_path}")
    else:
        report = run_benchmark()
        print(summarize(report))

        out_path = args.out or DEFAULT_REPORT
        if not args.no_out:
            with open(out_path, "w", encoding="utf-8") as handle:
                json.dump(report.to_dict(), handle, indent=2)
            print(f"\nJSON report written to {out_path}")

    if report.failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
