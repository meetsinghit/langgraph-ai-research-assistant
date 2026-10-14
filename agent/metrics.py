"""Lightweight run metrics — sizes, counts and latencies only.

Design rules (these are the redaction guarantee, by
construction):

- the collector NEVER stores message content, tool results,
  prompts, webpage text, URLs or exception strings — only
  their SIZES (character counts), counts, categories and
  timings;
- error information is reduced to a coarse category
  (``rate_limit`` / ``tls`` / ``ssrf`` / ``provider`` /
  ``tool`` / ``validation`` / ``deadline`` / ``unknown``),
  never the raw error text (which can echo a URL, a question
  or a snippet); search failures additionally carry their
  taxonomy category (``empty_results`` / ``timeout`` / ...);
- token counts distinguish WHAT WAS MEASURED: a model call
  that reported usage contributes its provider-reported
  numbers; a call whose usage was not observable records
  ``null`` tokens with ``usage_source: "unavailable"`` —
  unmeasured usage is NEVER reported as a genuine zero, and
  ``token_usage.coverage`` states how much of the run is
  actually measured;
- the snapshot is JSON-serializable so it can be written to
  a per-run report without further transformation.

Deliberately built on plain dataclasses + ``time.monotonic``
— no external observability service.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable

__all__ = [
    "MetricsCollector",
    "RunOutcome",
    "categorize_error",
]

#: Outcome taxonomy for a completed run (WS-F). Computed by the
#: graph's terminal nodes, never guessed from partial data:
#: COMPLETED, COMPLETED_WITH_WARNINGS, INCOMPLETE, TIMED_OUT,
#: CANCELLED (cooperative cancel, Phase 4D), FAILED
#: (+ UNSPECIFIED when no terminal node ran — e.g. the caller never
#: let the graph finish).
RunOutcome = (
    "COMPLETED",
    "COMPLETED_WITH_WARNINGS",
    "INCOMPLETE",
    "TIMED_OUT",
    "CANCELLED",
    "FAILED",
    "UNSPECIFIED",
)
_OUTCOMES = frozenset(RunOutcome)


def categorize_error(exc: BaseException | str) -> str:
    """Coarse error category — never the error text itself."""
    text = (
        f"{type(exc).__name__}: {exc}"
        if isinstance(exc, BaseException)
        else str(exc)
    ).lower()
    if "429" in text or "rate limit" in text or "resource exhausted" in text:
        return "rate_limit"
    # 4E.3: a run whose FROZEN model configuration is no longer usable
    # (retired model) — checked before generic markers so the message
    # never degrades to "unknown", and never implies a fallback.
    if "model unavailable" in text or "model_unavailable" in text:
        return "model_unavailable"
    # 4E.3 review (item 4): the provider refused access to the
    # SELECTED model/credentials (e.g. Google's 2.5-family restriction
    # for new projects). Fixed vocabulary only — no raw provider text,
    # and the run never retries on a different model.
    if "model access failed" in text:
        return "model_access"
    # 4E.3: invalid/contradictory run configuration (bad thinking
    # level for the model, missing provider configuration, ...).
    if "configurationerror" in text or "modelselectionerror" in text:
        return "configuration"
    if any(marker in text for marker in ("certificate", "ssl", "tls")):
        return "tls"
    if "non-public" in text or "private" in text or "unsafe" in text:
        return "ssrf"
    if "execution deadline" in text or "deadline has expired" in text:
        return "deadline"
    if "unknown tool" in text or "invalid arguments" in text or "must be an object" in text:
        return "validation"
    # Live-acceptance (Problem A): the graph now stamps the CLASSIFIED
    # provider category into the state error as a fixed-vocabulary
    # phrase (see ``provider_failure_suffix``). A transient, already-
    # retried provider fault must NOT read like a permanent failure —
    # these are checked before the generic "provider" bucket below,
    # which would otherwise swallow every "LLM provider failed (...)"
    # string. No provider message text is ever involved: only the
    # fixed phrases the classifier itself produced.
    if any(
        phrase in text
        for phrase in (
            "service temporarily unavailable",
            "provider internal error",
            "request timed out",
            "network error",
        )
    ):
        return "provider_transient"
    if "llm provider" in text or "provider failed" in text:
        return "provider"
    if "tool" in text:
        return "tool"
    return "unknown"


#: Error categories produced by DETERMINISTIC security controls (as
#: opposed to operational failures like timeouts). The snapshot's
#: ``security.tool_refusals`` block counts these, so "the guard
#: fired" is observable without ever recording what was refused or
#: why in text form.
SECURITY_REFUSAL_CATEGORIES = frozenset(
    {"validation", "policy", "dlp", "ssrf"}
)


@dataclass
class ModelCallMetrics:
    """One model invocation (size, not content).

    ``input_tokens``/``output_tokens`` are ``None`` when the
    provider did not report usage for this call (structured-output
    calls without raw-message access, failed attempts). ``None`` is
    JSON ``null`` — deliberately distinct from a measured zero.
    ``usage_source`` says where any number came from:
    ``provider`` (actually reported), ``estimated`` (only when
    estimation is explicitly enabled), ``unavailable``.
    """

    node: str
    iteration: int | None
    invocation: int
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    #: Provider-reported thinking/reasoning tokens for this call
    #: (Gemini ``thoughts_token_count`` via ``output_token_details``
    #: ``reasoning``). ``None`` = the provider did not report a
    #: thinking breakdown — never fabricated as 0.
    thinking_tokens: int | None = None
    prompt_chars: int = 0
    usage_source: str = "unavailable"
    error_category: str | None = None


@dataclass
class ToolCallMetrics:
    """One tool execution (name/outcome/size, not results).

    ``result_chars`` is ``None`` for calls RECONSTRUCTED from the
    durable event log after a crash+resume (Phase 4E.2): their
    outcome is known (the event recorded it) but their result size
    never survived the crash — ``null`` in the report, deliberately
    distinct from a measured zero.
    """

    node: str
    iteration: int | None
    tool: str
    ok: bool
    latency_ms: float
    result_chars: int | None = 0
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
    # -- WS-C / WS-F run-level reporting ----------------------------
    _deadline_seconds: float | None = field(default=None, init=False)
    _deadline_exceeded: bool = field(default=False, init=False)
    _outcome: str | None = field(default=None, init=False)
    _outcome_warnings: tuple[str, ...] = field(
        default_factory=tuple, init=False
    )
    _quality: dict[str, Any] | None = field(default=None, init=False)
    # -- WS-4C security reporting ------------------------------------
    #: (source_kind, categories) per untrusted-text scan. Categories
    #: are fixed-vocabulary; matched text is never stored.
    _injection_signals: list[tuple[str, tuple[str, ...]]] = field(
        default_factory=list, init=False
    )
    _answer_redactions: int = field(default=0, init=False)
    # -- WS-C / WS-D (Phase 4E.2) -----------------------------------
    #: Successful web_search calls served from the per-run query
    #: cache instead of re-hitting the search backend.
    _search_cache_hits: int = field(default=0, init=False)
    #: Execution segments this run had (1 = no interruption; >1 after
    #: a crash+resume — the tool counters are reconciled across them
    #: from the durable event log, but token usage only covers the
    #: final segment, which the snapshot labels explicitly).
    _segments: int = field(default=1, init=False)
    # -- WS-G (Phase 4E.3) --------------------------------------------
    #: The run's frozen configuration (model, thinking, depth,
    #: evidence) as PERSISTED — ``None`` fields for pre-4E.3 runs,
    #: never back-filled with a guess. Set by the run manager per
    #: execution segment (same values on resume).
    _run_config: dict[str, Any] | None = field(default=None, init=False)

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
        self._deadline_seconds = None
        self._deadline_exceeded = False
        self._outcome = None
        self._outcome_warnings = ()
        self._quality = None
        self._injection_signals.clear()
        self._answer_redactions = 0
        self._search_cache_hits = 0
        self._segments = 1
        self._run_config = None

    def set_run_config(
        self,
        *,
        model_id: str | None,
        thinking_level: str | None,
        research_depth: str | None,
        evidence_policy: str | None,
    ) -> None:
        """Stamp the run's frozen configuration into diagnostics (WS-G).

        Values are what the run row PERSISTED: ``None`` for legacy
        pre-4E.3 runs (displayed as "not recorded"), never
        back-filled from current settings after the fact.
        """
        self._run_config = {
            "model_id": model_id,
            "thinking_level": thinking_level,
            "research_depth": research_depth,
            "evidence_policy": evidence_policy,
        }

    def set_iteration(self, iteration: int | None) -> None:
        """Current research iteration (context for later events)."""
        self._iteration = iteration

    # -- run-level reporting (WS-C / WS-F) ---------------------------

    def record_deadline(self, total_seconds: float) -> None:
        """Stamp the run's configured global execution budget."""
        self._deadline_seconds = float(total_seconds)

    def mark_deadline_exceeded(self) -> None:
        """A node observed the global deadline as expired."""
        self._deadline_exceeded = True

    def set_outcome(
        self, status: str, warnings: Iterable[str] = ()
    ) -> None:
        """Terminal run status (see :data:`RunOutcome`).

        Warnings are content-free labels: the caller's (state-derived)
        ones first, then what the collector itself observed
        (``tool_failures``, ``provider_retries``,
        ``partial_token_coverage`` — mixed measured/unmeasured calls).
        Deterministic upgrade rule: COMPLETED carrying any warning
        becomes COMPLETED_WITH_WARNINGS, so a run with failed tools or
        retries can never be reported as a clean COMPLETED.
        """
        if status not in _OUTCOMES:
            raise ValueError(f"unknown run outcome: {status!r}")
        merged: list[str] = list(dict.fromkeys(warnings))
        if any(not c.ok for c in self._tool_calls):
            merged.append("tool_failures")
        if self._retries:
            merged.append("provider_retries")
        reported = sum(
            1
            for call in self._model_calls
            if call.input_tokens is not None
            or call.output_tokens is not None
        )
        if reported and reported < len(self._model_calls):
            merged.append("partial_token_coverage")
        if status == "COMPLETED" and merged:
            status = "COMPLETED_WITH_WARNINGS"
        self._outcome = status
        self._outcome_warnings = tuple(dict.fromkeys(merged))

    def set_quality(self, **fields: Any) -> None:
        """Content-free quality summary (booleans, counts, enum
        strings — never text, URLs or topic names)."""
        summary = dict(self._quality or {})
        summary.update(fields)
        self._quality = summary

    # -- security reporting (Phase 4C) -------------------------------

    def record_injection_signals(
        self, source: str, categories: tuple[str, ...]
    ) -> None:
        """Record instruction-like PATTERN signals in untrusted text.

        Content-free by construction: only the source kind (e.g.
        "search_result", "page", "tool_error", "draft") and
        fixed-vocabulary category names are stored — never the
        matched text. A signal is a defense-in-depth indicator, NOT
        proof that an attack occurred or succeeded.
        """
        if not categories:
            return
        self._injection_signals.append((str(source), tuple(categories)))

    def record_answer_redactions(self, count: int) -> None:
        """Record credential-pattern redactions applied to the
        outgoing answer (a count only — never the redacted value)."""
        if count > 0:
            self._answer_redactions += int(count)

    # -- recording ------------------------------------------------

    def record_model_call(
        self,
        node: str,
        *,
        latency_ms: float,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        thinking_tokens: int | None = None,
        prompt_chars: int = 0,
        error_category: str | None = None,
        usage_source: str | None = None,
    ) -> None:
        """Record one model invocation.

        Omitting ``input_tokens``/``output_tokens`` means the usage
        was NOT measured — the call records ``null`` with
        ``usage_source: "unavailable"`` instead of a fake zero.
        Passing numbers without an explicit source means the provider
        reported them (``provider``); estimation must pass
        ``usage_source="estimated"`` so reports never confuse the
        two. ``thinking_tokens`` stays ``None`` unless the provider
        genuinely reported a thinking breakdown.
        """
        measured = input_tokens is not None or output_tokens is not None
        self._model_calls.append(
            ModelCallMetrics(
                node=node,
                iteration=self._iteration,
                invocation=len(self._model_calls) + 1,
                latency_ms=round(latency_ms, 3),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                thinking_tokens=thinking_tokens,
                prompt_chars=prompt_chars,
                usage_source=usage_source
                or ("provider" if measured else "unavailable"),
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

    def record_search_cache_hit(self) -> None:
        """One web_search call was served from the per-run query
        cache (Phase 4E.2 / WS-C) — no search-backend hit occurred.

        The call itself is still recorded as a normal tool call (the
        model DID invoke the tool, and the events/diagnostics parity
        must hold); this counter is what makes the saved work
        visible instead of silently inflating "searches".
        """
        self._search_cache_hits += 1

    def reconcile_prior_segments(
        self,
        prior_tool_calls: list[dict[str, Any]],
        *,
        segments: int,
    ) -> None:
        """Re-attach tool calls recorded by earlier, crashed segments.

        Phase 4E.2 (WS-D): after a crash+resume the collector only
        covers the RESUMED segment, while the durable event channel
        (and therefore the progress timeline) spans the whole run.
        The application layer reads the earlier segments' tool
        ``tool.finished`` events — the only record that survived the
        crash — and hands their content-free fields here, so the
        final snapshot's counts agree with what the progress view
        showed. Reconstructed calls carry ``result_chars=None``:
        their result size is genuinely unknown, never a fake zero.

        Token usage of the crashed segments is NOT recoverable (the
        collector lived in the dead process), so token totals stay
        scoped to the final segment — the snapshot labels that
        explicitly via ``token_usage.scope``.
        """
        for call in prior_tool_calls:
            self._tool_calls.append(
                ToolCallMetrics(
                    node="research",
                    iteration=None,
                    tool=str(call.get("tool") or "unknown"),
                    ok=bool(call.get("ok")),
                    latency_ms=round(float(call.get("latency_ms") or 0.0), 3),
                    result_chars=None,
                    error_category=call.get("error_category"),
                )
            )
        self._segments = max(1, int(segments))

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
        # The usage-source triad partitions every call: provider
        # (actually reported), estimated (explicitly labelled) and
        # unavailable (never measured — null, not zero).
        provider_calls = [
            call
            for call in self._model_calls
            if call.usage_source == "provider"
        ]
        estimated_calls = [
            call
            for call in self._model_calls
            if call.usage_source == "estimated"
        ]
        measured = provider_calls + estimated_calls
        unavailable_calls = len(self._model_calls) - len(measured)
        total_in = sum(call.input_tokens or 0 for call in measured)
        total_out = sum(call.output_tokens or 0 for call in measured)
        by_node: dict[str, dict[str, Any]] = {}
        for call in self._model_calls:
            entry = by_node.setdefault(
                call.node,
                {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "usage_unavailable_calls": 0,
                    "prompt_chars": 0,
                    "latency_ms": 0.0,
                },
            )
            entry["calls"] += 1
            if call.input_tokens is None and call.output_tokens is None:
                entry["usage_unavailable_calls"] += 1
            entry["input_tokens"] += call.input_tokens or 0
            entry["output_tokens"] += call.output_tokens or 0
            entry["prompt_chars"] += call.prompt_chars
            entry["latency_ms"] += call.latency_ms
        searches = [c for c in self._tool_calls if c.tool == "web_search"]
        fetches = [c for c in self._tool_calls if c.tool == "fetch_page"]
        failed_searches = [c for c in searches if not c.ok]
        failed_fetches = [c for c in fetches if not c.ok]
        # Thinking tokens: summed ONLY from calls where the provider
        # reported a thinking breakdown; null when none did (never a
        # fabricated zero).
        thinking_reported = [
            c for c in self._model_calls if c.thinking_tokens is not None
        ]

        # Outcome: status + fully merged warning labels (explicit
        # state-derived ones and collector-observed ones were merged
        # deterministically by set_outcome).
        warnings = list(self._outcome_warnings)

        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self._finished_at,
            "total_execution_seconds": (
                round(self._finished_at - self.started_at, 3)
                if self._finished_at
                else None
            ),
            "outcome": {
                "status": self._outcome or "UNSPECIFIED",
                "warnings": warnings,
            },
            "deadline": {
                "configured_seconds": self._deadline_seconds,
                "exceeded": self._deadline_exceeded,
            },
            # Phase 4E.2 (WS-D): how many execution segments the run
            # had (1 = uninterrupted). >1 means a crash+resume: tool
            # counters were reconciled from the durable event log and
            # cover the WHOLE run; token usage covers only the final
            # segment (see token_usage.scope).
            "segments": self._segments,
            # Phase 4E.3 (WS-G): the configuration this run was
            # created with — persisted values, so a legacy pre-4E.3
            # run honestly reports nulls instead of today's defaults.
            "run_config": self._run_config,
            "totals": {
                "input_tokens": total_in,
                "output_tokens": total_out,
                "model_calls": len(self._model_calls),
                "tool_calls": len(self._tool_calls),
                "retries": len(self._retries),
                "prompt_chars": sum(
                    call.prompt_chars for call in self._model_calls
                ),
                # Sum of per-call tool latency (WS-C): how much of
                # the run was spent in search/fetch, separately from
                # model thinking and orchestration overhead.
                "tool_latency_ms": round(
                    sum(call.latency_ms for call in self._tool_calls), 1
                ),
            },
            "token_usage": {
                "input_tokens": total_in,
                "output_tokens": total_out,
                # WS-D: after a crash+resume the crashed segments'
                # token usage died with their collector — the totals
                # here are honestly scoped, never back-filled.
                "scope": (
                    "whole_run" if self._segments <= 1 else "since_resume"
                ),
                "provider_reported_calls": len(provider_calls),
                "estimated_calls": len(estimated_calls),
                "unavailable_calls": unavailable_calls,
                "coverage_ratio": (
                    round(len(measured) / len(self._model_calls), 3)
                    if self._model_calls
                    else 0.0
                ),
                # Thinking tokens, honestly scoped (4E.3 WS-G): a sum
                # over calls where the provider reported a thinking
                # breakdown; null when NO call reported one.
                "thinking_tokens": (
                    sum(c.thinking_tokens or 0 for c in thinking_reported)
                    if thinking_reported
                    else None
                ),
                "thinking_reported_calls": len(thinking_reported),
                "sources": _count_by(
                    call.usage_source for call in self._model_calls
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
                    "thinking_tokens": c.thinking_tokens,
                    "prompt_chars": c.prompt_chars,
                    "usage_source": c.usage_source,
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
                "failed_searches": len(failed_searches),
                "failed_fetches": len(failed_fetches),
                "search_success_rate": (
                    round((len(searches) - len(failed_searches)) / len(searches), 3)
                    if searches
                    else None
                ),
                "fetch_success_rate": (
                    round((len(fetches) - len(failed_fetches)) / len(fetches), 3)
                    if fetches
                    else None
                ),
                "failed_tool_calls": sum(
                    1 for c in self._tool_calls if not c.ok
                ),
                # WS-C: web_search calls served from the per-run query
                # cache (a repeat of an already-searched query) —
                # counted as executed calls above, broken out here so
                # "searches" is never read as backend hits.
                "search_cache_hits": self._search_cache_hits,
                "tool_error_categories": _count_by(
                    c.error_category for c in self._tool_calls if not c.ok
                ),
            },
            # Phase 4C: deterministic security posture of THIS run —
            # refusals that fired, injection-signal counts and answer
            # redactions. Counts and fixed-vocabulary categories
            # only: never matched text, URLs or credentials.
            "security": {
                "injection_signals": {
                    "total": sum(
                        len(cats) for _, cats in self._injection_signals
                    ),
                    "by_category": _count_by(
                        cat
                        for _, cats in self._injection_signals
                        for cat in cats
                    ),
                    "by_source": _count_by(
                        source
                        for source, cats in self._injection_signals
                        if cats
                    ),
                },
                "tool_refusals": _count_by(
                    c.error_category
                    for c in self._tool_calls
                    if not c.ok
                    and c.error_category in SECURITY_REFUSAL_CATEGORIES
                ),
                "answer_redactions": self._answer_redactions,
            },
            "quality": self._quality,
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
