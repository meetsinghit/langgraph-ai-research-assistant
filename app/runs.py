"""Run lifecycle: bounded execution, events, cancel, resume (4D.1).

The RunManager owns a bounded thread pool (one worker per run, at
most ``max_concurrent_runs`` in flight), the per-run event channels
and the capability-gated resume path.

Lifecycle (single-user, in-process — no broker):

    queued ──claim(CAS)──▶ running ──▶ completed | failed | cancelled
       │                      │
       │ cancel (CAS)         │ cancel: cooperative flag on the live
       ▼                      ▼          RunDeadline; graph stops at
    cancelled             cancelling ──▶ cancelled   the next boundary

    interrupted  = startup recovery found a non-terminal row with no
                   live worker (a previous process died). Resume is
                   the ONLY path back to running, and only after the
                   checkpoint-existence + capability checks pass.

Security invariants owned here:

- progress events are WHITELISTED and sanitized before they are
  persisted or streamed (never arguments, page text, snippets,
  exception strings or credentials);
- worker failures are logged and persisted as CATEGORY only;
- the graph runs with the same engine settings as the CLI — all
  Phase 4C controls (SSRF, fetch policy, DLP, redaction, budgets,
  citation validation) are unchanged inside the engine;
- checkpoints are pruned when a run reaches a terminal state so
  bounded webpage excerpts exist only while a run might resume.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agent.citations import assign_source_ids
from agent.config import Settings
from agent.deadline import RunDeadline
from agent.depth import resolve_depth, resolve_evidence_policy
from agent.graph import build_research_graph
from agent.llm import build_chat_model
from agent.metrics import MetricsCollector, categorize_error
from agent.runtime import RunContext
from agent.tools import ResearchTools
from app.db import Database
from app.events import RunEventChannel
from app.repository import (
    RUN_STATUS_CANCELLED,
    RUN_STATUS_CANCELLING,
    RUN_STATUS_COMPLETED,
    RUN_STATUS_FAILED,
    RUN_STATUS_INTERRUPTED,
    RUN_STATUS_QUEUED,
    RUN_STATUS_RUNNING,
    TERMINAL_STATUSES,
    Repository,
)

logger = logging.getLogger(__name__)

#: Node names that may appear in a stage event.
_STAGE_NODES = frozenset(
    {"start", "plan", "research", "synthesize", "review", "finalize",
     "handle_error"}
)

#: Event-type → whitelisted payload fields. Anything not listed here
#: is DROPPED before an event is persisted or streamed.
_EVENT_FIELDS: dict[str, tuple[str, ...]] = {
    "stage.started": ("node",),
    "stage.finished": ("node",),
    "tool.started": ("tool",),
    "tool.finished": ("tool", "ok", "error_category", "latency_ms"),
    "source.discovered": ("url", "title"),
    "source.retrieved": ("url",),
    "run.started": (),
    "run.finished": ("outcome", "review_status"),
    "status.changed": ("status",),
    "run.failed": ("error_category",),
}

_ALLOWED_STATUS_EVENTS = frozenset(
    {
        RUN_STATUS_QUEUED,
        RUN_STATUS_RUNNING,
        RUN_STATUS_CANCELLING,
        RUN_STATUS_COMPLETED,
        RUN_STATUS_FAILED,
        RUN_STATUS_CANCELLED,
        RUN_STATUS_INTERRUPTED,
    }
)


class NotFoundError(Exception):
    """The referenced session/run does not exist."""


class ResumeUnavailable(Exception):
    """Resume was refused; ``reason`` is the machine-readable cause."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _strip_controls(value: str) -> str:
    return "".join(
        char for char in value if char.isprintable() or char in " \t"
    )


class RunManager:
    """Owns run execution, event channels and lifecycle transitions."""

    def __init__(
        self,
        settings: Settings,
        db: Database,
        *,
        checkpointer: Any | None = None,
        model_factory: Callable[[Settings], Any] | None = None,
        tools_factory: Callable[[Settings], Any] | None = None,
        graph_factory: Callable[..., Any] | None = None,
        resume_enabled: bool = True,
    ) -> None:
        self._settings = settings
        self._db = db
        self._repo = Repository(db)
        self._checkpointer = checkpointer
        self._resume_enabled = resume_enabled
        self._model_factory = model_factory or build_chat_model
        self._tools_factory = tools_factory or ResearchTools
        if graph_factory is not None:
            self._graph_factory = graph_factory
        else:
            self._graph_factory = self._build_graph
        self._executor = ThreadPoolExecutor(
            max_workers=settings.max_concurrent_runs,
            thread_name_prefix="research-run",
        )
        self._channels: dict[str, RunEventChannel] = {}
        self._contexts: dict[str, RunContext] = {}
        self._registry_lock = threading.Lock()

    # -- graph/model wiring ---------------------------------------------

    def _build_graph(
        self,
        settings: Settings,
        model: Any,
        tools: Any,
        metrics: MetricsCollector,
    ) -> Any:
        return build_research_graph(
            settings,
            model,
            tools,
            metrics=metrics,
            checkpointer=self._checkpointer,
        )

    @property
    def resume_capability(self) -> bool:
        """True when the checkpoint-gated resume path is available
        (built with a checkpointer and not explicitly disabled)."""
        return self._resume_enabled and self._checkpointer is not None

    # -- event channels -------------------------------------------------

    def channel(self, run_id: str) -> RunEventChannel:
        with self._registry_lock:
            channel = self._channels.get(run_id)
            if channel is None:
                channel = RunEventChannel(self._db, run_id)
                self._channels[run_id] = channel
            return channel

    def _reset_channel(self, run_id: str) -> RunEventChannel:
        """Fresh channel after resume (sequence continues from the
        durable history; the old one is closed)."""
        with self._registry_lock:
            old = self._channels.pop(run_id, None)
            if old is not None:
                old.close()
            channel = RunEventChannel(self._db, run_id)
            self._channels[run_id] = channel
            return channel

    def _publish(
        self, run_id: str, event_type: str, payload: dict[str, Any]
    ) -> None:
        """Sanitize through the whitelist, then publish. Never raises."""
        fields = _EVENT_FIELDS.get(event_type)
        if fields is None:
            logger.warning("dropping unknown event type %s", event_type)
            return
        clean: dict[str, Any] = {}
        for key in fields:
            value = payload.get(key)
            if value is None:
                continue
            if key in ("url",):
                text = str(value)
                if not (
                    text.startswith("http://") or text.startswith("https://")
                ):
                    continue
                clean[key] = text[:2048]
            elif key == "title":
                text = _strip_controls(str(value))[:200].strip()
                if text:
                    clean[key] = text
            elif key == "node":
                text = str(value)
                if text in _STAGE_NODES:
                    clean[key] = text
            elif key == "tool":
                text = str(value)
                if text in ("web_search", "fetch_page"):
                    clean[key] = text
            elif key == "error_category":
                text = str(value)
                if text.replace("_", "").isalnum():
                    clean[key] = text[:40]
            elif key == "status":
                text = str(value)
                if text in _ALLOWED_STATUS_EVENTS:
                    clean[key] = text
            elif key == "outcome":
                text = str(value)
                if text.replace("_", "").isalnum():
                    clean[key] = text[:40]
            elif key == "review_status":
                text = str(value)
                if text.replace("_", "").isalnum():
                    clean[key] = text[:40]
            elif key == "ok":
                clean[key] = bool(value)
            elif key == "latency_ms":
                try:
                    clean[key] = max(0, int(value))
                except (TypeError, ValueError):
                    continue
        try:
            self.channel(run_id).publish(event_type, clean)
        except Exception:  # noqa: BLE001 — event transport never fails a run
            logger.warning("event publish failed for run %s", run_id)

    # -- follow-up context ----------------------------------------------

    def derive_context(self, session_id: str, run_id: str | None = None) -> str:
        """Server-derived follow-up context from prior ANSWERED runs.

        Client input never reaches this function: only earlier
        questions and their already-redacted answers of the SAME
        session, bounded in turns and characters.
        """
        max_turns = self._settings.followup_context_max_turns
        if max_turns <= 0:
            return ""
        turns = self._repo.recent_turns(
            session_id,
            exclude_run_id=run_id or "",
            limit=max_turns,
        )
        turns.reverse()  # oldest → newest
        parts: list[str] = []
        for turn in turns:
            question = str(turn["question"])[:300]
            answer = str(turn["answer"] or "")[:1200]
            parts.append(f"User asked: {question}\nAssistant answered: {answer}")
        return "\n\n".join(parts)[
            : self._settings.followup_context_max_chars
        ]

    # -- run creation ----------------------------------------------------

    def start_run(
        self,
        *,
        session_id: str,
        question: str,
        idempotency_key: str | None = None,
        research_depth: str = "standard",
        evidence_policy: str = "balanced",
    ) -> dict[str, Any]:
        """Create a run (idempotently) and schedule its execution.

        ``research_depth`` / ``evidence_policy`` (Phase 4E.1) are
        normalized here as defense in depth (the API already
        validates them as Literals) and persisted as run metadata so
        history keeps the settings each run actually used.
        """
        if self._repo.get_session(session_id) is None:
            raise NotFoundError(f"session {session_id} not found")
        if idempotency_key:
            existing = self._repo.find_run_by_idempotency_key(
                session_id, idempotency_key
            )
            if existing is not None:
                return existing
        context = self.derive_context(session_id)
        try:
            run = self._repo.create_run(
                session_id=session_id,
                question=question,
                conversation_context=context,
                idempotency_key=idempotency_key or None,
                research_depth=resolve_depth(research_depth),
                evidence_policy=resolve_evidence_policy(
                    evidence_policy
                ),
            )
        except sqlite3.IntegrityError:
            # Concurrent double-POST with the same key: the unique
            # index decided the winner; return that run.
            if idempotency_key:
                existing = self._repo.find_run_by_idempotency_key(
                    session_id, idempotency_key
                )
                if existing is not None:
                    return existing
            raise
        self._repo.touch_session(session_id)
        self._publish(run["id"], "status.changed", {"status": RUN_STATUS_QUEUED})
        self._executor.submit(self._execute, run["id"])
        return run

    # -- execution -------------------------------------------------------

    def _execute(self, run_id: str, *, resume: bool = False) -> None:
        if not self._repo.claim_for_execution(run_id):
            # Cancelled (or recovered) while queued: no worker runs.
            return
        row = self._repo.get_run(run_id)
        if row is None:
            return
        channel = self.channel(run_id)
        metrics = MetricsCollector()
        deadline = RunDeadline(self._settings.max_total_execution_seconds)
        context = RunContext(deadline=deadline, sink=self._sink(run_id))
        with self._registry_lock:
            self._contexts[run_id] = context
        self._publish(run_id, "status.changed", {"status": RUN_STATUS_RUNNING})
        self._publish(run_id, "run.started", {})
        #: Terminal status once the run is HANDLED (success or a
        #: caught Exception). ``None`` while unknown — and it stays
        #: None if something escapes as BaseException (a simulated
        #: process death): the finally-block then performs NO
        #: terminal cleanup, so the row stays ``running``, the event
        #: channel stays open and — critically — the checkpoint is
        #: NOT pruned. Startup recovery turns exactly that state
        #: into ``interrupted`` + a resumable checkpoint.
        terminal_status: str | None = None
        try:
            metrics.begin_run()
            model = self._model_factory(self._settings)
            tools = self._tools_factory(self._settings)
            graph = self._graph_factory(
                self._settings, model, tools, metrics=metrics
            )
            # A RESUME passes input=None — LangGraph's contract for
            # "continue this thread from its checkpoint". Passing the
            # question again would RESET the thread and replay every
            # node, defeating the recovery boundary.
            payload: dict[str, Any] | None = None
            if not resume:
                payload = {
                    "question": row["question"],
                    "conversation_context": row["conversation_context"] or "",
                    # Phase 4E.1 run metadata travels in the input so
                    # it lands in graph state (and therefore in the
                    # checkpoint); a RESUME re-reads it from state.
                    "research_depth": row.get("research_depth")
                    or "standard",
                    "evidence_policy": row.get("evidence_policy")
                    or "balanced",
                }
            state: dict[str, Any] = graph.invoke(
                payload,
                {"configurable": {"thread_id": run_id}},
                context=context,
            )
            # WS-D (Phase 4E.2): after a crash+resume this collector
            # only saw the RESUMED segment while the durable event
            # channel spans the whole run — reconcile the earlier
            # segments' tool calls from it so the persisted
            # diagnostics agree with the progress timeline.
            self._reconcile_prior_segments(channel, metrics)
            metrics.finish()
            snapshot = metrics.snapshot()
            outcome = str(snapshot["outcome"]["status"])
            terminal_status = self._status_for(outcome)
            # Graph-internal controlled failures (the error node ran
            # and the graph returned normally) still get a coarse
            # category — from the STATE error strings, which since
            # the Phase 4D redaction fix are type-only text, never a
            # raw exception.
            error_category: str | None = None
            if terminal_status == RUN_STATUS_FAILED:
                state_errors = state.get("errors") or []
                if state_errors:
                    error_category = categorize_error(
                        "; ".join(str(item) for item in state_errors)
                    )
            review = state.get("review_status")
            review_value = getattr(review, "value", None) or (
                str(review) if review else None
            )
            refs = assign_source_ids(state.get("findings") or [])
            self._repo.finish_run(
                run_id,
                status=terminal_status,
                answer=state.get("final_answer") or None,
                review_status=review_value,
                outcome_status=outcome,
                outcome_warnings=list(snapshot["outcome"].get("warnings") or []),
                error_category=error_category,
                iterations=int(state.get("iteration", 0) or 0),
                tokens_in=int(snapshot["totals"]["input_tokens"]),
                tokens_out=int(snapshot["totals"]["output_tokens"]),
                estimated_cost_usd=float(state.get("estimated_cost_usd") or 0.0),
                total_execution_seconds=float(
                    snapshot.get("total_execution_seconds") or 0.0
                ),
                metrics=snapshot,
            )
            self._repo.replace_sources(
                run_id,
                [
                    {"sid": ref.sid, "url": ref.url, "evidence": ref.evidence}
                    for ref in refs
                ],
            )
            self._publish(
                run_id,
                "run.finished",
                {
                    "outcome": outcome,
                    "review_status": review_value or "",
                },
            )
        except Exception as exc:  # noqa: BLE001 — boundary: category only
            category = categorize_error(exc)
            logger.error(
                "run %s failed with %s (category: %s)",
                run_id,
                type(exc).__name__,
                category,
            )
            terminal_status = RUN_STATUS_FAILED
            self._repo.finish_run(
                run_id, status=terminal_status, error_category=category
            )
            self._publish(
                run_id, "run.failed", {"error_category": category}
            )
        finally:
            with self._registry_lock:
                self._contexts.pop(run_id, None)
            if terminal_status is not None:
                self._publish(
                    run_id, "status.changed", {"status": terminal_status}
                )
                channel.close()
                self._prune_checkpoints(run_id, terminal_status)
            # terminal_status is None (unhandled BaseException): the
            # run is treated exactly like a dead process — no terminal
            # event, no close, no checkpoint prune; recovery decides.

    def _sink(self, run_id: str) -> Callable[[dict[str, Any]], None]:
        def sink(event: dict[str, Any]) -> None:
            event_type = str(event.get("type") or "")
            self._publish(run_id, event_type, event)

        return sink

    def _reconcile_prior_segments(
        self, channel: RunEventChannel, metrics: MetricsCollector
    ) -> None:
        """Re-attach earlier segments' tool calls after a crash+resume.

        WS-D (Phase 4E.2): every execution segment publishes its own
        ``run.started``; the durable event channel is NOT cleared on
        resume, so the events before the LAST ``run.started`` belong
        to earlier (crashed) segments whose in-process collector died
        with them. Their ``tool.finished`` events — the only record
        that survived — carry exactly the content-free fields the
        snapshot needs (tool name, ok, error category, latency), so
        the final counts match what the progress timeline showed
        for the whole run. Runs without a resume are untouched
        (single ``run.started`` = no prior segments).
        """
        events = channel.replay(0)
        started_seqs = [
            event.seq for event in events if event.type == "run.started"
        ]
        if len(started_seqs) <= 1:
            return
        # The current invocation published its run.started before the
        # graph ran; everything durable before it is a prior segment.
        cutoff = started_seqs[-1]
        prior = [
            event
            for event in events
            if event.seq < cutoff and event.type == "tool.finished"
        ]
        metrics.reconcile_prior_segments(
            [
                {
                    "tool": event.payload.get("tool"),
                    "ok": event.payload.get("ok"),
                    "error_category": event.payload.get("error_category"),
                    "latency_ms": event.payload.get("latency_ms"),
                }
                for event in prior
            ],
            segments=len(started_seqs),
        )

    @staticmethod
    def _status_for(outcome: str) -> str:
        if outcome == "CANCELLED":
            return RUN_STATUS_CANCELLED
        if outcome in ("COMPLETED", "COMPLETED_WITH_WARNINGS", "INCOMPLETE"):
            return RUN_STATUS_COMPLETED
        return RUN_STATUS_FAILED

    # -- cancellation ----------------------------------------------------

    def cancel(self, run_id: str) -> str | None:
        """Cooperative cancel. Returns the resulting status label:
        ``cancelled`` (stopped before it started),
        ``cancelling`` (flag delivered; the worker will finish it),
        or ``None`` when the run was already terminal — nothing to
        cancel."""
        row = self._repo.get_run(run_id)
        if row is None:
            raise NotFoundError(f"run {run_id} not found")
        status = str(row["status"])
        if status == RUN_STATUS_QUEUED:
            if self._repo.cas_status(
                run_id,
                from_status=RUN_STATUS_QUEUED,
                to_status=RUN_STATUS_CANCELLED,
            ):
                self._repo.finish_run(run_id, status=RUN_STATUS_CANCELLED)
                self._publish(
                    run_id,
                    "status.changed",
                    {"status": RUN_STATUS_CANCELLED},
                )
                self.channel(run_id).close()
                return RUN_STATUS_CANCELLED
            status = RUN_STATUS_RUNNING  # a worker claimed it meanwhile
        if status in (RUN_STATUS_RUNNING, RUN_STATUS_CANCELLING):
            with self._registry_lock:
                context = self._contexts.get(run_id)
            if context is not None and context.deadline is not None:
                context.deadline.cancel()
            if self._repo.mark_cancelling(run_id):
                self._publish(
                    run_id,
                    "status.changed",
                    {"status": RUN_STATUS_CANCELLING},
                )
            return RUN_STATUS_CANCELLING
        return None  # already terminal: nothing to cancel

    # -- startup recovery + resume ---------------------------------------

    def recover_interrupted(self) -> list[str]:
        """Mark non-terminal rows from a dead process as interrupted."""
        run_ids = self._repo.recover_interrupted_run_ids()
        self._repo.mark_interrupted(run_ids)
        for run_id in run_ids:
            self._publish(
                run_id, "status.changed", {"status": RUN_STATUS_INTERRUPTED}
            )
            self.channel(run_id).close()
        if run_ids:
            logger.warning(
                "recovered %d interrupted run(s)", len(run_ids)
            )
        return run_ids

    def _has_checkpoint(self, run_id: str) -> bool:
        if self._checkpointer is None:
            return False
        try:
            config = {"configurable": {"thread_id": run_id}}
            for _ in self._checkpointer.list(config, limit=1):
                return True
        except Exception:  # noqa: BLE001 — treat as "cannot verify"
            return False
        return False

    def resume(self, run_id: str) -> None:
        """Resume an interrupted run from its checkpoint.

        Capability-gated: refused with ``resume_unverified`` when the
        app was built without checkpointing, ``not_interrupted`` for
        any non-interrupted status, ``no_checkpoint`` when no
        durable state exists to resume from. The interrupted→queued
        CAS guarantees a concurrent resume cannot create a second
        worker.
        """
        if not self._resume_enabled or self._checkpointer is None:
            # capability gate AFTER existence so a 404 stays a 404
            row = self._repo.get_run(run_id)
            if row is None:
                raise NotFoundError(f"run {run_id} not found")
            raise ResumeUnavailable("resume_unverified")
        row = self._repo.get_run(run_id)
        if row is None:
            raise NotFoundError(f"run {run_id} not found")
        if str(row["status"]) != RUN_STATUS_INTERRUPTED:
            raise ResumeUnavailable("not_interrupted")
        if not self._has_checkpoint(run_id):
            raise ResumeUnavailable("no_checkpoint")
        if not self._repo.cas_status(
            run_id,
            from_status=RUN_STATUS_INTERRUPTED,
            to_status=RUN_STATUS_QUEUED,
        ):
            raise ResumeUnavailable("not_interrupted")
        # Fresh runtime context: a resumed run gets a FRESH deadline
        # budget (documented semantics — wall-clock time spent
        # crashed is not counted against any deadline).
        self._reset_channel(run_id)
        self._publish(run_id, "status.changed", {"status": RUN_STATUS_QUEUED})
        self._executor.submit(self._execute, run_id, resume=True)

    # -- teardown --------------------------------------------------------

    def _prune_checkpoints(self, run_id: str, status: str) -> None:
        """Delete a terminal run's checkpoints (bounded retention of
        webpage excerpts; interrupted runs keep theirs for resume)."""
        if status not in TERMINAL_STATUSES or self._checkpointer is None:
            return
        try:
            self._checkpointer.delete_thread(run_id)
        except Exception:  # noqa: BLE001 — pruning is best-effort
            logger.warning("checkpoint prune failed for run %s", run_id)

    def delete_session(self, session_id: str) -> list[str]:
        run_ids = self._repo.delete_session(session_id)
        if self._checkpointer is not None:
            for run_id in run_ids:
                try:
                    self._checkpointer.delete_thread(run_id)
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "checkpoint prune failed for run %s", run_id
                    )
        return run_ids

    def wait_idle(self) -> None:
        """Testing/ops helper: block until queued work drained."""
        self._executor.shutdown(wait=True)
        # ThreadPoolExecutor cannot restart after shutdown; a fresh
        # pool keeps the manager usable in long-lived tests.
        self._executor = ThreadPoolExecutor(
            max_workers=self._settings.max_concurrent_runs,
            thread_name_prefix="research-run",
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
