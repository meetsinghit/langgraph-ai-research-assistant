"""CLI entry point for the AI Research Assistant.

Usage:
    python main.py "your research question"
    python main.py --verify-connection
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

from agent.config import get_settings
from agent.graph import build_research_graph
from agent.llm import build_chat_model, verify_connection
from agent.logging_config import setup_logging
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
    graph = build_research_graph(settings, model, tools)

    state: dict[str, Any] = {"question": args.question}
    for step in graph.stream({"question": args.question}):
        for node, updates in step.items():
            logger.info("node %s completed", node)
            for key, value in (updates or {}).items():
                if isinstance(value, list) and isinstance(state.get(key), list):
                    state[key] = state[key] + value  # reducer semantics
                else:
                    state[key] = value
            print(f"[{node}] {_summarize(updates)}")

    print()
    print("=== ANSWER ===")
    print(state.get("final_answer", "(no answer produced)"))
    print()
    print("=== SOURCES ===")
    for url in dict.fromkeys(state.get("sources") or []):
        print(f"- {url}")
    print()
    print(f"iterations: {state.get('iteration', 0)}")
    print(
        f"tokens: {state.get('tokens_in', 0)} in / "
        f"{state.get('tokens_out', 0)} out"
    )
    cost = state.get("estimated_cost_usd")
    if cost is None:
        print("estimated cost: n/a for this model")
    else:
        print(f"estimated cost: ${cost:.4f}")
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
