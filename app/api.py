"""HTTP + SSE endpoints (Phase 4D.1).

All routes except ``/api/health`` require the Bearer token (enforced
by the middleware in ``app.main``). Error bodies carry a fixed
category + fixed message — never raw exception text.

SSE contract: ``GET /api/runs/{id}/events`` streams ``text/event-stream``
with monotonically sequenced events (``id: <seq>``). Reconnect with
the ``Last-Event-ID`` header to replay everything after that
sequence; replay is read-only and subscribing can never create or
affect a research run. ``: heartbeat`` comments keep idle streams
alive.
"""
from __future__ import annotations

import json
import logging
import queue
from typing import Any

from fastapi import APIRouter, Header, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from app.events import RunEvent, _CLOSED
from app.runs import IdempotencyConflict, NotFoundError, ResumeUnavailable
from app.schemas import (
    CancelOut,
    ConfigOut,
    HealthOut,
    ModelsOut,
    ResumeOut,
    RunAcceptedOut,
    RunCreate,
    RunDetailOut,
    RunSummaryOut,
    SessionCreate,
    SessionDetailOut,
    SessionOut,
    SessionRename,
    SourceOut,
)
from agent.gemini_models import ModelSelectionError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


def _error(category: str, message: str, status: int) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"category": category, "message": message}},
    )


def _run_summary(row: dict[str, Any]) -> RunSummaryOut:
    return RunSummaryOut(
        id=row["id"],
        session_id=row["session_id"],
        question=row["question"],
        status=row["status"],
        has_answer=bool(row.get("answer")),
        created_at=row["created_at"],
        started_at=row.get("started_at"),
        finished_at=row.get("finished_at"),
        research_depth=row.get("research_depth") or "standard",
        evidence_policy=row.get("evidence_policy") or "balanced",
        model_id=row.get("model_id"),
        thinking_level=row.get("thinking_level"),
    )


def _run_detail(request: Request, row: dict[str, Any]) -> RunDetailOut:
    repo = request.app.state.repo
    sources = [
        SourceOut(
            sid=source["sid"],
            url=source["url"],
            host=source.get("host") or "",
            evidence=source["evidence"],
            position=source["position"],
        )
        for source in repo.list_sources(row["id"])
    ]
    diagnostics = None
    raw = row.get("metrics_json")
    if raw:
        try:
            diagnostics = json.loads(raw)
        except (TypeError, ValueError):
            diagnostics = None
    warnings: list[str] = []
    try:
        warnings = list(json.loads(row.get("outcome_warnings") or "[]"))
    except (TypeError, ValueError):
        warnings = []
    return RunDetailOut(
        id=row["id"],
        session_id=row["session_id"],
        question=row["question"],
        status=row["status"],
        answer=row.get("answer"),
        review_status=row.get("review_status"),
        outcome_status=row.get("outcome_status"),
        outcome_warnings=warnings,
        error_category=row.get("error_category"),
        iterations=int(row.get("iterations") or 0),
        tokens_in=int(row.get("tokens_in") or 0),
        tokens_out=int(row.get("tokens_out") or 0),
        estimated_cost_usd=float(row.get("estimated_cost_usd") or 0.0),
        total_execution_seconds=float(
            row.get("total_execution_seconds") or 0.0
        ),
        sources=sources,
        diagnostics=diagnostics,
        research_depth=row.get("research_depth") or "standard",
        evidence_policy=row.get("evidence_policy") or "balanced",
        model_id=row.get("model_id"),
        thinking_level=row.get("thinking_level"),
        created_at=row["created_at"],
        started_at=row.get("started_at"),
        finished_at=row.get("finished_at"),
        events_url=f"/api/runs/{row['id']}/events",
    )


def _sse_lines(event: RunEvent) -> str:
    data = json.dumps(event.as_dict())
    return f"id: {event.seq}\nevent: {event.type}\ndata: {data}\n\n"


def _event_stream(
    request: Request, run_id: str, last_event_id: int, heartbeat: float
):
    manager = request.app.state.manager
    channel = manager.channel(run_id)
    subscription = channel.subscribe()
    try:
        # 1) Replay durable history up to the attach cursor — covers
        #    everything published before we subscribed, exactly once.
        for event in channel.replay(last_event_id, upto=subscription.cursor):
            yield _sse_lines(event)
        if subscription.closed:
            # Terminal before we arrived: the cursor snapshot is the
            # final sequence, so the replay above is complete.
            return
        # 2) Live tail: only events published AFTER the attach cursor
        #    can be on the queue — no duplicates, no gaps.
        while True:
            try:
                item = subscription.events.get(timeout=heartbeat)
            except queue.Empty:
                yield ": heartbeat\n\n"
                continue
            if item is _CLOSED:
                break
            yield _sse_lines(item)
        # No DB drain after the sentinel: publish persists an event
        # BEFORE fanning it out and close() enqueues the sentinel last
        # (both under the channel lock), so the queue already carried
        # every event published after the attach cursor — replaying
        # them again would duplicate the stream.
    finally:
        channel.detach(subscription)


# -- health / config ------------------------------------------------------


@router.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    """Liveness only: no configuration, no counts, no secrets."""
    return HealthOut()


@router.get("/config", response_model=ConfigOut)
def config(request: Request) -> ConfigOut:
    settings = request.app.state.settings
    return ConfigOut(
        provider=settings.llm_provider,
        model=settings.llm_model,
        thinking_level=settings.llm_thinking_level,
        max_total_execution_seconds=settings.max_total_execution_seconds,
        max_research_iterations=settings.max_research_iterations,
        max_tool_calls_per_research=settings.max_tool_calls_per_research,
        min_findings=settings.min_findings,
        min_retrieved_sources=settings.min_retrieved_sources,
        max_concurrent_runs=settings.max_concurrent_runs,
        resume_enabled=request.app.state.manager.resume_capability,
    )


@router.get("/models", response_model=ModelsOut)
def list_models(request: Request) -> ModelsOut:
    """Backend-managed model catalogue (4E.3 WS-A).

    Served from the maintained capability/free-tier registry — the
    ONLY authority for ``selectable``. Live ``models.list`` discovery
    (opt-in via ``MODEL_DISCOVERY_ENABLED``, TTL-cached, single-flight, never
    called by the automated test suite) only annotates ``availability``
    and can never make an unvetted model selectable. No secrets, no
    raw provider error text.
    """
    catalog = request.app.state.model_catalog
    return catalog.snapshot()


# -- sessions --------------------------------------------------------------


@router.post("/sessions", response_model=SessionOut, status_code=201)
def create_session(body: SessionCreate, request: Request) -> SessionOut:
    repo = request.app.state.repo
    return SessionOut(**repo.create_session(body.title))


@router.get("/sessions", response_model=list[SessionOut])
def list_sessions(request: Request) -> list[SessionOut]:
    repo = request.app.state.repo
    return [SessionOut(**row) for row in repo.list_sessions()]


@router.get("/sessions/{session_id}", response_model=SessionDetailOut)
def get_session(session_id: str, request: Request) -> Any:
    repo = request.app.state.repo
    session = repo.get_session(session_id)
    if session is None:
        return _error("not_found", "session not found", 404)
    runs = repo.list_runs(session_id)
    return SessionDetailOut(
        session=SessionOut(**session),
        runs=[_run_summary(row) for row in runs],
    )


@router.patch("/sessions/{session_id}", response_model=SessionOut)
def rename_session(
    session_id: str, body: SessionRename, request: Request
) -> Any:
    """Rename a session (4E.3 WS-F).

    Title validation is identical to creation (1-120 chars, control
    characters stripped) so a renamed title can never be
    un-displayable. ``updated_at`` is deliberately untouched: renaming
    is metadata editing, not research activity, so list order and
    history are unaffected. Requires the Bearer token like every
    other mutating route.
    """
    repo = request.app.state.repo
    if repo.get_session(session_id) is None:
        return _error("not_found", "session not found", 404)
    renamed = repo.rename_session(session_id, body.title)
    if renamed is None:
        return _error("not_found", "session not found", 404)
    return SessionOut(**renamed)


@router.delete(
    "/sessions/{session_id}", status_code=204, response_class=Response
)
def delete_session(session_id: str, request: Request) -> Response:
    manager = request.app.state.manager
    repo = request.app.state.repo
    if repo.get_session(session_id) is None:
        return _error("not_found", "session not found", 404)
    manager.delete_session(session_id)
    return Response(status_code=204)


# -- runs -------------------------------------------------------------------


@router.post(
    "/sessions/{session_id}/runs", response_model=RunAcceptedOut, status_code=202
)
def create_run(
    session_id: str,
    body: RunCreate,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
) -> Any:
    manager = request.app.state.manager
    try:
        run = manager.start_run(
            session_id=session_id,
            question=body.question,
            idempotency_key=idempotency_key,
            research_depth=body.research_depth,
            evidence_policy=body.evidence_policy,
            model_id=body.model_id,
            thinking_level=body.thinking_level,
        )
    except NotFoundError:
        return _error("not_found", "session not found", 404)
    except ModelSelectionError as exc:
        # Fixed categories (never provider text): unknown_model |
        # model_not_selectable | unsupported_thinking_level.
        return _error(exc.category, str(exc), 400)
    except IdempotencyConflict:
        return _error(
            "idempotency_conflict",
            "idempotency key already used with different question "
            "or research settings",
            409,
        )
    return RunAcceptedOut(
        run_id=run["id"],
        status=run["status"],
        events_url=f"/api/runs/{run['id']}/events",
    )


@router.get("/runs/{run_id}", response_model=RunDetailOut)
def get_run(run_id: str, request: Request) -> Any:
    repo = request.app.state.repo
    row = repo.get_run(run_id)
    if row is None:
        return _error("not_found", "run not found", 404)
    return _run_detail(request, row)


@router.get("/runs/{run_id}/events")
def stream_run_events(
    run_id: str,
    request: Request,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> Any:
    repo = request.app.state.repo
    if repo.get_run(run_id) is None:
        return _error("not_found", "run not found", 404)
    try:
        after_seq = int(last_event_id) if last_event_id else -1
    except ValueError:
        after_seq = -1
    heartbeat = request.app.state.settings.sse_heartbeat_seconds
    return StreamingResponse(
        _event_stream(request, run_id, max(after_seq, -1), heartbeat),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/runs/{run_id}/cancel", response_model=CancelOut)
def cancel_run(run_id: str, request: Request) -> Any:
    manager = request.app.state.manager
    repo = request.app.state.repo
    try:
        status = manager.cancel(run_id)
    except NotFoundError:
        return _error("not_found", "run not found", 404)
    if status is None:
        row = repo.get_run(run_id)
        current = row["status"] if row else "unknown"
        return _error(
            "conflict",
            f"run is already {current}; only queued or running runs can be cancelled",
            409,
        )
    return CancelOut(run_id=run_id, status=status)


@router.post("/runs/{run_id}/resume", response_model=ResumeOut, status_code=202)
def resume_run(run_id: str, request: Request) -> Any:
    manager = request.app.state.manager
    try:
        manager.resume(run_id)
    except NotFoundError:
        return _error("not_found", "run not found", 404)
    except ResumeUnavailable as exc:
        return _error("conflict", f"resume unavailable: {exc.reason}", 409)
    return ResumeOut(run_id=run_id)
