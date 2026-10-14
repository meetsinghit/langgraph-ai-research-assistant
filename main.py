"""CLI entry point for the AI Research Assistant.

Usage:
    python main.py "your research question"
    python main.py --verify-connection
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from agent.citations import assign_source_ids
from agent.config import get_settings
from agent.graph import build_research_graph
from agent.llm import build_chat_model, verify_connection
from agent.logging_config import setup_logging
from agent.metrics import MetricsCollector, categorize_error
from agent.tools import ResearchTools


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="AI Research Assistant (LangGraph POC)"
    )
    parser.add_argument("question", nargs="?", help="the question to research")
    parser.add_argument(
        "--verify-connection",
        action="store_true",
        help=(
            "opt-in live smoke test: one minimal request to the "
            "configured provider"
        ),
    )
    parser.add_argument(
        "--metrics",
        metavar="PATH",
        help=(
            "write a machine-readable JSON metrics report "
            "(run id, per-node/per-call latency, token "
            "counts, tool outcomes) to PATH"
        ),
    )
    parser.add_argument(
        "--depth",
        choices=("quick", "standard", "deep"),
        default="standard",
        help=(
            "research depth for this run (default: standard) — "
            "quick = minimal sufficient retrieval, deep = broader "
            "explicit budgets; see docs/OPERATIONS.md"
        ),
    )
    parser.add_argument(
        "--evidence-policy",
        choices=("balanced", "verified"),
        default="balanced",
        help=(
            "citation evidence policy (default: balanced) — "
            "verified requires every factual claim to rest on a "
            "retrieved page and fails citation validation otherwise"
        ),
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    logger = setup_logging(settings)

    if args.verify_connection:
        result = verify_connection(settings)
        if result["ok"]:
            print(
                f"Connection OK: {result['provider']} / {result['model']} "
                f"({result['input_tokens']} in / "
                f"{result['output_tokens']} out)"
            )
            return 0
        print(f"Connection FAILED: {result.get('error')}", file=sys.stderr)
        return 1

    if not args.question:
        parser.error("a question is required (or use --verify-connection)")

    logger.info("starting research run: %s", args.question)
    model = build_chat_model(settings)
    tools = ResearchTools(settings)
    metrics = MetricsCollector()
    graph = build_research_graph(settings, model, tools, metrics=metrics)

    state: dict[str, Any] = {
        "question": args.question,
        "research_depth": args.depth,
        "evidence_policy": args.evidence_policy,
    }
    run_input = dict(state)
    # stamp the wall-clock start BEFORE the run so the metrics
    # report measures the actual execution, not snapshot latency
    metrics.begin_run()
    try:
        for step in graph.stream(run_input):
            for node, updates in step.items():
                logger.info("node %s completed", node)
                for key, value in (updates or {}).items():
                    if isinstance(value, list) and isinstance(
                        state.get(key), list
                    ):
                        state[key] = state[key] + value  # reducer semantics
                    else:
                        state[key] = value
                print(f"[{node}] {_summarize(updates)}")
    except Exception as exc:
        # Honest failure accounting: even an unhandled exception
        # produces a terminal FAILED report — category only, never
        # the exception text (it can echo the question, a URL or a
        # page fragment).
        logger.error(
            "run failed with %s (category: %s)",
            type(exc).__name__,
            categorize_error(exc),
        )
        metrics.set_outcome(
            "FAILED", warnings=["unhandled_exception"]
        )
        metrics.finish()
        if args.metrics:
            with open(args.metrics, "w", encoding="utf-8") as handle:
                json.dump(metrics.snapshot(), handle, indent=2)
            print(f"metrics report: {args.metrics}", file=sys.stderr)
        raise

    print()
    print("=== ANSWER ===")
    print(state.get("final_answer", "(no answer produced)"))
    print()
    print("=== SOURCES ===")
    # [S#] identifiers match the citations in the answer; provenance
    # labels are rendered here by application code, never by the
    # model inside citations. [retrieved] = page fetched and
    # inspected; [discovered] = URL found via search, content never
    # verified.
    refs = assign_source_ids(state.get("findings") or [])
    if refs:
        for ref in refs:
            tag = "retrieved" if ref.evidence == "retrieved" else "discovered"
            print(f"- [{ref.sid}] {ref.url} [{tag}]")
    else:
        for url in dict.fromkeys(state.get("sources") or []):
            print(f"- {url} [discovered]")
    print()
    print(f"iterations: {state.get('iteration', 0)}")
    print(
        f"research depth: {args.depth} | evidence policy: "
        f"{args.evidence_policy}"
    )
    print(
        f"tokens: {state.get('tokens_in', 0)} in / "
        f"{state.get('tokens_out', 0)} out"
    )
    cost = state.get("estimated_cost_usd")
    if cost is None:
        print("estimated cost: n/a for this model")
    else:
        print(f"estimated cost: ${cost:.4f}")
    # terminal accounting (WS-F): stamp timing, then report the run
    # outcome — COMPLETED / COMPLETED_WITH_WARNINGS / INCOMPLETE /
    # TIMED_OUT / FAILED with content-free warning labels
    metrics.finish()
    report = metrics.snapshot()
    outcome = report.get("outcome") or {}
    status = outcome.get("status", "UNSPECIFIED")
    warnings = outcome.get("warnings") or []
    suffix = f" [{', '.join(warnings)}]" if warnings else ""
    print(f"run outcome: {status}{suffix}")
    total = report.get("total_execution_seconds")
    if total is not None:
        print(f"execution time: {total:.2f}s")
    if args.metrics:
        with open(args.metrics, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"metrics report: {args.metrics}")
    return 0


def _summarize(updates: dict[str, Any] | None) -> str:
    if not updates:
        return "(no state change)"
    parts = []
    for key, value in updates.items():
        if isinstance(value, list):
            parts.append(f"{key}: +{len(value)}")
        else:
            parts.append(f"{key}: {str(value)[:60]}")
    return ", ".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
