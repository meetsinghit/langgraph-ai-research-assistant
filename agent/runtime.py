"""Runtime-only per-invocation context (Phase 4D).

A graph invocation carries LIVE objects — the run's monotonic
deadline/cancel flag and an optional progress-event sink. These must
never be persisted: a monotonic expiry is meaningless across a
process restart, and LangGraph's checkpoint serializer raises on
arbitrary objects. So they travel OUTSIDE graph state, in LangGraph's
runtime context (``graph.invoke(..., context=RunContext(...))``),
which the checkpointer never serializes. Durable state keeps only
serializable primitives (e.g. ``deadline_total_seconds``).

Nodes and their helpers read the context through
:func:`get_run_context`, which is safe to call anywhere (returns
``None`` outside a graph run — direct unit calls of node helpers
therefore see "no deadline", exactly like the pre-4D code path when
no deadline was stamped).

Progress events (``sink``) are a Phase 4D transport concern: the
graph emits small, content-free dicts (stage/tool/source lifecycle);
the application layer assigns sequence numbers, sanitizes, persists
and streams them to the UI. The sink is invoked from graph worker
threads and MUST NEVER break a run — :meth:`RunContext.emit`
swallows sink failures.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from langgraph.runtime import get_runtime

from agent.deadline import RunDeadline

logger = logging.getLogger(__name__)

#: Receives small content-free event dicts; owned by the application.
ProgressSink = Callable[[dict[str, Any]], None]


@dataclass
class RunContext:
    """Live, runtime-only objects for ONE graph invocation.

    Never placed in graph state and never serialized: the deadline is
    a monotonic-clock object, the sink is a callable.
    """

    deadline: RunDeadline | None = None
    sink: ProgressSink | None = None
    #: Per-run search-query cache (Phase 4E.2 / WS-C): maps a
    #: normalized query to the last SUCCESSFUL result list, so a
    #: follow-up pass that repeats an identical query is served
    #: without re-hitting the search backend. Live-only like the
    #: deadline: a resume starts a fresh context, so a resumed run
    #: may legitimately search again. Created lazily by the research
    #: node (``None`` = dedup disabled for this invocation).
    search_cache: dict[str, Any] | None = None

    def emit(self, event: dict[str, Any]) -> None:
        """Publish a progress event; a broken sink can never fail a run."""
        if self.sink is None:
            return
        try:
            self.sink(event)
        except Exception:  # noqa: BLE001 — transport errors are not run errors
            logger.debug("progress sink failed for %s", event.get("type"))


def get_run_context() -> RunContext | None:
    """The active :class:`RunContext`, or ``None`` outside a graph run.

    Reads LangGraph's runtime context (a contextvar set during node
    execution); ``None`` when no context was supplied to the
    invocation or the current context is not a ``RunContext``.
    """
    try:
        runtime = get_runtime()
    except RuntimeError:
        # Called outside any runnable context (direct helper/node
        # invocation in tests, import-time code, ...).
        return None
    context = runtime.context
    return context if isinstance(context, RunContext) else None


def current_deadline() -> RunDeadline | None:
    """The active run's deadline/cancel object, or ``None``."""
    context = get_run_context()
    return context.deadline if context is not None else None
