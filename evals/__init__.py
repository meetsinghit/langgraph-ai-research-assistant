"""Offline evaluation framework (Phase 4A).

Run the benchmark without any live provider calls:

    python -m evals                # console summary
    python -m evals --out PATH   # also write JSON

The benchmark is fully deterministic: scripted model
responses and canned search/page fixtures drive the
real graph. See ``evals.cases`` for the case catalog
and ``evals.harness`` for the runner.
"""
from evals.cases import benchmark_cases
from evals.harness import run_benchmark, summarize

__all__ = ["benchmark_cases", "run_benchmark", "summarize"]
