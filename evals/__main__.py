"""``python -m evals`` — run the offline benchmark.

Options:
    --out PATH   also write the machine-readable JSON
                 report to PATH (default: evals/report.json)
    --no-out     do not write a report file

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Offline evaluation benchmark "
            "(no live provider calls)"
        )
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_REPORT,
        help="write the JSON report to this path",
    )
    parser.add_argument(
        "--no-out",
        action="store_true",
        help="do not write a report file",
    )
    args = parser.parse_args(argv)

    report = run_benchmark()
    print(summarize(report))

    if not args.no_out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report.to_dict(), handle, indent=2)
        print(f"\nJSON report written to {args.out}")

    if report.failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
