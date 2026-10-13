# syntax=docker/dockerfile:1
# =============================================================================
# AI Research Assistant - local container image (Phase 4D.3).
#
# STATUS: STATICALLY VALIDATED ONLY. Docker was not installed on the review
# machine, so this file has NOT been built or run there. See docs/OPERATIONS.md
# ("Docker launch - UNVERIFIED").
#
# Topology: ONE container. The Python server also serves the built browser UI
# from the same origin (/api + static files), so there is no reverse proxy,
# no second container, and no inter-service networking to get wrong.
#
# Security posture (mirrors the local install):
#   - non-root user (uid 10001);
#   - no secrets in the image: keys/token come from environment at runtime;
#   - the compose file publishes the port on 127.0.0.1 only;
#   - SQLite data lives in a mounted volume, never in the image layer.
# =============================================================================

# ---- Stage 1: build the browser UI (Node only needed here) ------------------
FROM node:20-alpine AS ui-build
WORKDIR /build

# Dependency layer first for cache efficiency.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build
# Result: /build/dist (Vite production bundle, hashed asset names).

# ---- Stage 2: runtime -------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Inside the container the server must bind all interfaces so the
    # published port reaches it; docker-compose publishes it on the
    # host loopback only, so the app stays machine-local.
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    DATA_DIR=/app/data

WORKDIR /app

# Pinned dependencies (requirements.txt is fully pinned) -> reproducible layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application code only (see .dockerignore: no tests, no .env, no data).
COPY app/ ./app/
COPY agent/ ./agent/
# Built UI from stage 1; served same-origin by the API (app/ui.py).
COPY --from=ui-build /build/dist ./frontend/dist

# Non-root user; only /app/data is writable (the SQLite volume).
RUN useradd --system --create-home --uid 10001 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app/data
USER appuser

VOLUME /app/data
EXPOSE 8000

# /api/health needs no token and reveals nothing.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request, sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"

CMD ["python", "-m", "app.main"]
