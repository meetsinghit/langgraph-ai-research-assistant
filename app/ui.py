"""Static serving for the built browser UI (Phase 4D.3).

Production arrangement: ONE process. FastAPI serves the built
``frontend/dist`` bundle and the ``/api`` routes on the same origin,
so the browser never needs a dev server, a proxy, or relaxed CORS —
the Bearer token travels on our own requests exactly as in 4D.2.

Security posture:

- Static files are NOT Bearer-protected (they contain no secrets —
  the token is never embedded in the bundle; see WS3) — every
  ``/api`` route stays protected by the middleware in ``app.main``.
- Paths are resolved inside the dist directory only (traversal
  attempts 404 — never served, never revealed).
- UI responses carry ``nosniff``, ``Referrer-Policy: no-referrer``
  and a restrictive CSP (``'self'`` only; the production bundle has
  no inline scripts and fetches only same-origin ``/api``).
- Cache headers match the content: hashed ``/assets/*`` are
  immutable for a year; ``index.html`` is always revalidated, so a
  new build is picked up on the next load.
- Unmatched ``/api/...`` paths return the API's JSON 404 contract —
  they never fall through to the UI.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, Response

_IMMUTABLE_PREFIX = "/assets/"
_CACHE_CONTROL_IMMUTABLE = "public, max-age=31536000, immutable"
_CACHE_CONTROL_REVALIDATE = "no-cache"
_CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data:; "
    "font-src 'self'; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "base-uri 'none'; "
    "frame-ancestors 'none'"
)
_NOT_BUILT_MESSAGE = (
    "the browser UI is not built — run `npm install && npm run build` "
    "in frontend/ (or use start-research-assistant-dev.bat), then retry"
)


def _ui_headers(path: str) -> dict[str, str]:
    """Cache + hardening headers for a UI response."""
    cache = (
        _CACHE_CONTROL_IMMUTABLE
        if path.startswith(_IMMUTABLE_PREFIX)
        else _CACHE_CONTROL_REVALIDATE
    )
    return {
        "Cache-Control": cache,
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
        "Content-Security-Policy": _CSP,
    }


def _api_404() -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={
            "error": {"category": "not_found", "message": "unknown endpoint"}
        },
    )


def register_ui(app: FastAPI, dist_dir: Path) -> bool:
    """Serve the built UI when ``dist_dir`` exists.

    Returns True when a UI was registered. Unmatched ``/api`` paths
    still answer with the JSON 404 contract, with or without a UI.
    """
    dist = Path(dist_dir)

    @app.get("/{full_path:path}", include_in_schema=False)
    def ui_index(full_path: str) -> Response:  # pragma: no cover - trivial
        if full_path.startswith("api/") or full_path == "api":
            return _api_404()
        if not (dist.is_dir() and (dist / "index.html").is_file()):
            return JSONResponse(
                status_code=404,
                content={
                    "error": {"category": "not_found", "message": _NOT_BUILT_MESSAGE}
                },
            )
        index = dist / "index.html"
        if not full_path:
            return FileResponse(index, headers=_ui_headers("/index.html"))
        # Resolve strictly inside dist (traversal-safe).
        candidate = (dist / full_path).resolve()
        try:
            candidate.relative_to(dist.resolve())
        except ValueError:
            return _api_404()
        if candidate.is_file():
            return FileResponse(
                candidate,
                headers=_ui_headers("/" + full_path),
            )
        # SPA fallback: unknown non-file paths load the app shell.
        return FileResponse(index, headers=_ui_headers("/index.html"))

    app.state.ui_dist = dist
    return True
