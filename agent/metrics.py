"""Lightweight run metrics — sizes, counts and latencies only.

Design rules (these are the redaction guarantee, by
construction):

- the collector NEVER stores message content, tool results,
  prompts, webpage text, URLs or exception strings — only
  their SIZES (character counts), counts, categories and
  timings;
- error information is reduced to a coarse category
  (``rate_limit`` / ``tls`` / ``ssrf`` / ``provider`` /
  ``tool`` / ``validation`` / ``unknown``), never the raw
  error text (which can echo a URL, a question or a snippet);
- the snapshot is JSON-serializable so it can be written to
  a per-run report without further transformation.

Deliberately built on plain dataclasses + ``time.monotonic``
— no external observability service.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

__all__ = ["MetricsCollector", "categorize_error"]


def categorize_error(exc: BaseException | str) -> str:
    """Coarse error category — never the error text itself."""
    text = (
        f"{type(exc).__name__}: {exc}"
        if isinstance(exc, BaseException)
        else str(exc)
    ).lower()
    if "429" in text or "rate limit" in text or "resource exhausted" in text:
        return "rate_limit"
    if any(marker in text for marker in ("certificate", "ssl", "tls")):
        return "tls"
    if "non-public" in text or "private" in text or "unsafe" in text:
        return "ssrf"
    if "unknown tool" in text or "invalid arguments" in text or "must be an object" in text:
        return "validation"
    if "llm provider" in text or "provider failed" in text:
        return "provider"
    if "tool" in text:
        return "tool"
    return "unknown"


@dataclass
class ModelCallMetrics:
    """One model invocation (size, not content)."""

    node: str
    iteration: int | None
    invocation: int
    latency_ms: float
    input_tokens: int
    output_tokens: int
    prompt_chars: int
    error_category: str | None = None


@dataclass
class ToolCallMetrics:
    """One tool execution (name/outcome/size, not results)."""

    node: str
    iteration: int | None
    tool: str
    ok: bool
    latency_ms: float
    result_chars: int
    error_category: str | None = None


@dataclass
class NodeMetrics:
    """One graph-node execution."""

    node: str
    iteration: int | None
    latency_ms: float


@dataclass
class RetryMetrics:
    """One retry (rate-limit backoff)."""

    node: str
    iteration: int | None
    reason_category: str


@dataclass
class MetricsCollector:
    """Accumulates a single run's metrics.

    ``run_id`` is stamped lazily on the first recorded event
    so one compiled graph (and one collector) can serve many
    invocations without stale identifiers. Call
    :meth:`begin_run` to reset explicitly.
    """

    _run_id: str | None = field(default=None, init=False)
    _started_at: float | None = field(default=None, init=False)
    _finished_at: float | None = field(default=None, init=False)
    _model_calls: list[ModelCallMetrics] = field(
        default_factory=list, init=False
    )
    _tool_calls: list[ToolCallMetrics] = field(
        default_factory=list, init=False
    )
    _nodes: list[NodeMetrics] = field(default_factory=list, init=False)
    _retries: list[RetryMetrics] = field(
        default_factory=list, init=False
    )
    _iteration: int | None = field(default=None, init=False)

    # -- lifecycle ------------------------------------------------

    @property
    def run_id(self) -> str:
        if self._run_id is None:
            self._run_id = uuid.uuid4().hex[:12]
        return self._run_id

    @property
    def started_at(self) -> float:
        if self._started_at is None:
            self._started_at = time.time()
        return self._started_at

    def begin_run(self) -> None:
        """Reset for a fresh invocation (new run id + timers)."""
        self._run_id = uuid.uuid4().hex[:12]
        self._started_at = time.time()
        self._finished_at = None
        self._model_calls.clear()
        self._tool_calls.clear()
        self._nodes.clear()
        self._retries.clear()
        self._iteration = None

    def set_iteration(self, iteration: int | None) -> None:
        """Current research iteration (context for later events)."""
        self._iteration = iteration

    # -- recording ------------------------------------------------

    def record_model_call(
        self,
        node: str,
        *,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        prompt_chars: int = 0,
        error_category: str | None = None,
    ) -> None:
        self._model_calls.append(
            ModelCallMetrics(
                node=node,
                iteration=self._iteration,
                invocation=len(self._model_calls) + 1,
                latency_ms=round(latency_ms, 3),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                prompt_chars=prompt_chars,
                error_category=error_category,
            )
        )

    def record_tool_call(
        self,
        node: str,
        tool: str,
        *,
        ok: bool,
        latency_ms: float,
        result_chars: int = 0,
        error_category: str | None = None,
    ) -> None:
        self._tool_calls.append(
            ToolCallMetrics(
                node=node,
                iteration=self._iteration,
                tool=tool,
                ok=ok,
                latency_ms=round(latency_ms, 3),
                result_chars=result_chars,
                error_category=error_category,
            )
        )

    def record_node(self, node: str, latency_ms: float) -> None:
        self._nodes.append(
            NodeMetrics(
                node=node,
                iteration=self._iteration,
                latency_ms=round(latency_ms, 3),
            )
        )

    def record_retry(
        self, node: str, reason_category: str
    ) -> None:
        self._retries.append(
            RetryMetrics(
                node=node,
                iteration=self._iteration,
                reason_category=reason_category,
            )
        )

    def finish(self) -> None:
        # ensure the clock has started — a snapshot taken after
        # finish() must never compute a negative span
        _ = self.started_at
        self._finished_at = time.time()

    # -- aggregation ------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """JSON-safe metrics report (sizes and counts only)."""
        total_in = sum(call.input_tokens for call in self._model_calls)
        total_out = sum(call.output_tokens for call in self._model_calls)
        by_node: dict[str, dict[str, Any]] = {}
        for call in self._model_calls:
            entry = by_node.setdefault(
                call.node,
                {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "prompt_chars": 0,
                    "latency_ms": 0.0,
                },
            )
            entry["calls"] += 1
            entry["input_tokens"] += call.input_tokens
            entry["output_tokens"] += call.output_tokens
            entry["prompt_chars"] += call.prompt_chars
            entry["latency_ms"] += call.latency_ms
        searches = [c for c in self._tool_calls if c.tool == "web_search"]
        fetches = [c for c in self._tool_calls if c.tool == "fetch_page"]
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self._finished_at,
            "total_execution_seconds": (
                round(self._finished_at - self.started_at, 3)
                if self._finished_at
                else None
            ),
            "totals": {
                "input_tokens": total_in,
                "output_tokens": total_out,
                "model_calls": len(self._model_calls),
                "tool_calls": len(self._tool_calls),
                "retries": len(self._retries),
                "prompt_chars": sum(
                    call.prompt_chars for call in self._model_calls
                ),
            },
            "tokens_by_node": by_node,
            "model_calls": [
                {
                    "node": c.node,
                    "iteration": c.iteration,
                    "invocation": c.invocation,
                    "latency_ms": c.latency_ms,
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                    "prompt_chars": c.prompt_chars,
                    "error_category": c.error_category,
                }
                for c in self._model_calls
            ],
            "tool_calls": [
                {
                    "node": c.node,
                    "iteration": c.iteration,
                    "tool": c.tool,
                    "ok": c.ok,
                    "latency_ms": c.latency_ms,
                    "result_chars": c.result_chars,
                    "error_category": c.error_category,
                }
                for c in self._tool_calls
            ],
            "nodes": [
                {
                    "node": n.node,
                    "iteration": n.iteration,
                    "latency_ms": n.latency_ms,
                }
                for n in self._nodes
            ],
            "counts": {
                "searches": len(searches),
                "fetches": len(fetches),
                "failed_tool_calls": sum(
                    1 for c in self._tool_calls if not c.ok
                ),
                "tool_error_categories": _count_by(
                    c.error_category for c in self._tool_calls if not c.ok
                ),
            },
            "retries": [
                {
                    "node": r.node,
                    "iteration": r.iteration,
                    "reason_category": r.reason_category,
                }
                for r in self._retries
            ],
        }


def _count_by(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts
