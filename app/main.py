"""FastAPI application factory (Phase 4D.1).

Security posture (single-user, LOCAL-ONLY):

- Bearer-token auth on every /api route (including SSE); only
  ``/api/health`` is exempt and it reveals nothing. The token comes
  from ``APP_TOKEN`` or is bootstrapped once into
  ``<data_dir>/app_token.txt`` (git-ignored; treat like a credential).
- Binds ``127.0.0.1`` by default. Do not publish this service.
- CORS allowlist for the Vite dev server only; no credentials mode
  (the token travels in an explicit Authorization header).
- API docs/OpenAPI are disabled (least exposure for a local tool).
- Request bodies are capped (64 KiB) — a research question never
  legitimately needs more.
- Credentials and system prompts never reach a response: the config
  endpoint returns an explicit field list, run details expose the
  content-free metrics snapshot, and events are whitelisted by the
  RunManager.

Dependency injection for tests: pass ``settings``, ``db_path``,
``token``, ``model_factory``/``tools_factory``/``graph_factory`` and
``checkpointer`` (``None`` disables checkpointing → resume reports
``resume_unverified``, as designed).
"""
from __future__ import annotations

import logging
import os
import secrets
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent.config import Settings, get_settings
from agent.logging_config import setup_logging
from app.api import router
from app.db import Database
from app.repository import Repository
from app.runs import RunManager

logger = logging.getLogger(__name__)

_MAX_BODY_BYTES = 64 * 1024
_UNAUTHORIZED = JSONResponse(
    status_code=401,
    content={
        "error": {
            "category": "unauthorized",
            "message": "missing or invalid bearer token",
        }
    },
)


def resolve_token(settings: Settings, data_dir: Path) -> str:
    """The API token: configured, bootstrapped file, or new."""
    if settings.app_token:
        return settings.app_token
    data_dir.mkdir(parents=True, exist_ok=True)
    token_file = data_dir / "app_token.txt"
    if token_file.exists():
        token = token_file.read_text(encoding="utf-8").strip()
        if token:
            return token
    token = secrets.token_urlsafe(32)
    # Owner-only where the platform supports it (POSIX); on Windows
    # the file inherits the user profile's ACLs.
    descriptor = os.open(
        token_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(token + "\n")
    logger.info("bootstrapped API token file at %s", token_file)
    return token


def make_checkpointer(db_path: Path) -> Any:
    """One shared SqliteSaver for the app.

    The installed implementation documents exactly this usage: a
    single connection with ``check_same_thread=False`` is safe
    because every operation takes the saver's internal lock. WAL is
    enabled by ``setup()``. Bounded run concurrency keeps the
    serialized checkpoint traffic well within budget.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path), check_same_thread=False)
    saver = SqliteSaver(connection)
    saver.setup()
    return saver


def create_app(
    settings: Settings | None = None,
    *,
    db_path: str | Path | None = None,
    token: str | None = None,
    checkpointer: Any | None = "auto",
    model_factory: Any | None = None,
    tools_factory: Any | None = None,
    graph_factory: Any | None = None,
    resume_enabled: bool = True,
    recover: bool = True,
    log: bool = False,
) -> FastAPI:
    """Build the application. ``checkpointer``: ``"auto"`` creates a
    SqliteSaver in the data dir, ``None`` disables checkpointing."""
    settings = settings or get_settings()
    if log:
        setup_logging(settings)
    data_dir = Path(settings.data_dir)
    resolved_db = Path(db_path) if db_path else data_dir / "app.db"
    if checkpointer == "auto":
        resolved_checkpointer: Any | None = make_checkpointer(
            resolved_db.parent / "checkpoints.db"
        )
    else:
        resolved_checkpointer = checkpointer
    resolved_token = token or resolve_token(settings, data_dir)

    db = Database(resolved_db)
    repo = Repository(db)
    manager = RunManager(
        settings,
        db,
        checkpointer=resolved_checkpointer,
        model_factory=model_factory,
        tools_factory=tools_factory,
        graph_factory=graph_factory,
        resume_enabled=resume_enabled,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        manager.shutdown()

    app = FastAPI(
        title="AI Research Assistant",
        version="0.4.1",
        docs_url=None,  # least exposure: no interactive API docs
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.db = db
    app.state.repo = repo
    app.state.manager = manager
    app.state.checkpointer = resolved_checkpointer

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,  # token travels in a header, not cookies
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "Idempotency-Key",
            "Last-Event-ID",
        ],
    )

    expected_header = f"Bearer {resolved_token}"

    @app.middleware("http")
    async def security_middleware(request: Request, call_next: Any) -> Any:
        # Body cap: a question (<= 2000 chars) plus JSON framing never
        # approaches this; anything larger is abuse, not usage.
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > _MAX_BODY_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": {
                                "category": "validation",
                                "message": "request body too large",
                            }
                        },
                    )
            except ValueError:
                pass
        if request.url.path != "/api/health":
            provided = request.headers.get("authorization", "")
            if not secrets.compare_digest(provided, expected_header):
                return _UNAUTHORIZED
        return await call_next(request)

    if recover:
        manager.recover_interrupted()

    app.include_router(router)
    return app


def serve() -> None:
    """Entry point: ``python -m app.main`` (or scripts/dev.*)."""
    import uvicorn

    settings = get_settings()
    setup_logging(settings)
    uvicorn.run(
        create_app(settings, log=False),
        host=settings.api_host,
        port=settings.api_port,
        log_config=None,
    )


if __name__ == "__main__":
    serve()
