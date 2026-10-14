"""Model catalogue + bounded live discovery (Phase 4E.3, WS-A).

``GET /api/models`` is served from here. Two layers:

- **Registry** (``agent.gemini_models``): the authority for what is
  SELECTABLE — capabilities and free-tier eligibility are maintained
  by hand from official docs and never derived from a listing.
- **Discovery** (opt-in): one authenticated ``models.list`` call via
  the official ``google-genai`` SDK, annotated onto the registry as
  ``availability`` (confirmed / not_listed), plus an ``unregistered``
  observation list that is NEVER selectable (eligibility unknown).

Bounding rules (no unbounded discovery calls):

- disabled unless ``MODEL_DISCOVERY_ENABLED=true`` — the default makes NO
  external calls, so offline tests, launchers and first runs never
  touch the network;
- at most one call per ``model_discovery_ttl_seconds`` (TTL cache);
- single-flight: concurrent requests wait for one in-progress call;
- failures back off for ``model_discovery_backoff_seconds`` and are
  reported as a fixed category (never raw provider error text);
- a hard cap on listing pages so a hostile/huge response cannot
  balloon memory.

The discovery client is injectable for tests — the real SDK path is
never exercised by the automated suite.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from agent.config import Settings
from agent.gemini_models import (
    GEMINI_MODEL_REGISTRY,
    ModelSpec,
    control_by_family,
    family_of,
    is_selectable,
)
from app.schemas import (
    ModelDiscoveryOut,
    ModelOut,
    ModelThinkingOut,
    ModelsOut,
)

#: Cap on models processed from a live listing (pagination bound).
MAX_DISCOVERY_MODELS = 200
#: Cap on the ``unregistered`` observation list.
MAX_UNREGISTERED = 50

#: id -> {"display_name": str, "actions": tuple[str, ...]}
Listing = dict[str, dict[str, Any]]

#: Injectable fetcher: () -> Listing. Raises on failure.
DiscoveryFetch = Callable[[], Listing]


def _categorize_discovery_error(exc: BaseException) -> str:
    """Fixed category for a failed discovery call — never message text."""
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        if code in (401, 403):
            return "auth"
        if code == 429:
            return "rate_limit"
        if 400 <= code < 500:
            return "invalid_request"
        return "provider"
    name = type(exc).__name__.lower()
    if "timeout" in name or "timedout" in name:
        return "timeout"
    if isinstance(exc, (ConnectionError, OSError)):
        return "network"
    return "provider"


def _default_fetch(settings: Settings) -> Listing:
    """Live ``models.list`` via the official SDK (opt-in only)."""
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=settings.gemini_api_key,
        http_options=types.HttpOptions(
            timeout=int(settings.model_discovery_timeout_seconds * 1000)
        ),
    )
    listing: Listing = {}
    try:
        pages = client.models.list(
            config=types.ListModelsConfig(query_base=True)
        )
        for model in pages:
            name = str(model.name or "").removeprefix("models/")
            if not name:
                continue
            listing[name] = {
                "display_name": str(model.display_name or ""),
                "actions": tuple(model.supported_actions or ()),
            }
            # Hard cap: the pager fetches lazily, so this bounds the
            # number of pages fetched as well as memory.
            if len(listing) >= MAX_DISCOVERY_MODELS:
                break
    finally:
        try:
            client.close()
        except Exception:  # noqa: BLE001 — closing is best-effort
            pass
    return listing


class ModelCatalog:
    """Thread-safe registry + bounded-discovery state for /api/models."""

    def __init__(
        self,
        settings: Settings,
        fetch: DiscoveryFetch | None = None,
    ) -> None:
        self._settings = settings
        self._fetch = fetch
        self._state_lock = threading.Lock()
        self._refresh_lock = threading.Lock()
        # discovery state
        self._listing: Listing | None = None
        self._checked_at: float | None = None
        self._last_failure_at: float | None = None
        self._discovery_status = "disabled"  # disabled|available|unavailable
        self._error_category: str | None = None

    # -- discovery -----------------------------------------------------

    def _ensure_fresh(self) -> None:
        """Refresh the listing when due (TTL + backoff + single-flight)."""
        if not self._settings.model_discovery_enabled:
            return
        with self._state_lock:
            now = time.time()
            fresh = (
                self._checked_at is not None
                and now - self._checked_at
                < self._settings.model_discovery_ttl_seconds
            )
            backing_off = (
                self._error_category is not None
                and self._last_failure_at is not None
                and now - self._last_failure_at
                < self._settings.model_discovery_backoff_seconds
            )
            if fresh or backing_off:
                return
        # Single-flight: only one thread performs the network call.
        if not self._refresh_lock.acquire(blocking=False):
            return  # a refresh is already in progress
        try:
            with self._state_lock:
                now = time.time()
                if self._checked_at is not None and (
                    now - self._checked_at
                    < self._settings.model_discovery_ttl_seconds
                ):
                    return  # won the race; cache already fresh
            fetch = self._fetch
            if fetch is None:
                fetch = lambda: _default_fetch(self._settings)  # noqa: E731
            try:
                listing = fetch()
            except Exception as exc:  # noqa: BLE001 — category only
                with self._state_lock:
                    self._discovery_status = "unavailable"
                    self._error_category = _categorize_discovery_error(exc)
                    self._last_failure_at = time.time()
                return
            with self._state_lock:
                # Defense in depth: trim even an injected/hostile
                # listing so one snapshot stays bounded.
                self._listing = dict(list(listing.items())[:MAX_DISCOVERY_MODELS])
                self._checked_at = time.time()
                self._last_failure_at = None
                self._discovery_status = "available"
                self._error_category = None
        finally:
            self._refresh_lock.release()

    # -- snapshot ------------------------------------------------------

    def snapshot(self) -> ModelsOut:
        """Registry-backed catalogue, optionally availability-annotated."""
        self._ensure_fresh()
        with self._state_lock:
            listing = dict(self._listing) if self._listing else None
            checked_at = self._checked_at
            status = (
                self._discovery_status
                if self._settings.model_discovery_enabled
                else "disabled"
            )
            error_category = self._error_category

        default_model_id = self._settings.llm_model
        models: list[ModelOut] = []
        seen: set[str] = set()

        # The configured default sorts first (display order), even
        # when it is a custom, unregistered model — it is what runs
        # use when no selection is made.
        for spec in self._ordered_specs(default_model_id):
            models.append(self._registry_entry(spec, listing))
            seen.add(spec.model_id)
        if default_model_id not in seen:
            models.insert(0, self._configured_default_entry(default_model_id))

        unregistered: list[ModelOut] = []
        if listing:
            for model_id in sorted(listing):
                if model_id in seen:
                    continue
                info = listing[model_id]
                actions = info.get("actions") or ()
                # Text-generation observations only: image/audio/video
                # models are out of scope for a research chat model.
                if actions and "generateContent" not in actions:
                    continue
                unregistered.append(
                    ModelOut(
                        model_id=model_id,
                        display_name=str(
                            info.get("display_name") or model_id
                        ),
                        family=_family_prefix(model_id),
                        status="unknown",
                        free_tier="unknown",
                        selectable=False,
                        exclusion_reason="eligibility_unknown",
                        thinking=None,
                        availability="confirmed",
                        verified_on="",
                        docs_url="",
                        notes=(
                            "Observed via model discovery. Free-tier "
                            "eligibility and thinking capabilities are "
                            "not verified — never selectable."
                        ),
                        source="discovery",
                    )
                )
                if len(unregistered) >= MAX_UNREGISTERED:
                    break

        return ModelsOut(
            default_model_id=default_model_id,
            default_thinking_level=self._settings.llm_thinking_level or "medium",
            discovery=ModelDiscoveryOut(
                status=status,
                checked_at=(
                    datetime.fromtimestamp(checked_at, tz=UTC).isoformat()
                    if checked_at
                    else None
                ),
                error_category=error_category,
                model_count=len(listing) if listing is not None else None,
            ),
            models=models,
            unregistered=unregistered,
        )

    # -- helpers -------------------------------------------------------

    @staticmethod
    def _ordered_specs(default_model_id: str) -> list[ModelSpec]:
        registry = list(GEMINI_MODEL_REGISTRY.values())

        def sort_key(spec: ModelSpec) -> tuple[int, int, str]:
            return (
                0 if spec.model_id == default_model_id else 1,
                0 if is_selectable(spec) else 1,
                spec.model_id,
            )

        return sorted(registry, key=sort_key)

    @staticmethod
    def _availability(spec_id: str, listing: Listing | None) -> str:
        if listing is None:
            return "not_checked"
        return "confirmed" if spec_id in listing else "not_listed"

    @classmethod
    def _registry_entry(
        cls, spec: Any, listing: Listing | None
    ) -> ModelOut:
        selectable = is_selectable(spec)
        if selectable:
            reason = None
        elif spec.status in ("deprecated", "shutdown"):
            reason = spec.status
        elif spec.free_tier == "paid_only":
            reason = "paid_tier"
        else:
            reason = "eligibility_unknown"
        return ModelOut(
            model_id=spec.model_id,
            display_name=spec.display_name,
            family=spec.family,
            status=spec.status,
            free_tier=spec.free_tier,
            selectable=selectable,
            exclusion_reason=reason,
            thinking=ModelThinkingOut(
                control=spec.thinking_control,
                levels=list(spec.thinking_levels),
                default=spec.default_thinking,
            ),
            availability=cls._availability(spec.model_id, listing),
            verified_on=spec.verified_on,
            docs_url=spec.docs_url,
            notes=spec.notes,
            source="registry",
        )

    @classmethod
    def _configured_default_entry(cls, model_id: str) -> ModelOut:
        """Entry for an admin-configured model outside the registry.

        Shown as the current default (it IS what runs use) but with
        ``free_tier: unknown`` — its eligibility was never verified
        here, and the UI still submits it as ``model_id: null``
        ("configured default") rather than an explicit selection.
        """
        return ModelOut(
            model_id=model_id,
            display_name=f"{model_id} (configured default)",
            family=family_of(model_id),
            status="stable",
            free_tier="unknown",
            selectable=True,
            exclusion_reason=None,
            thinking=ModelThinkingOut(
                control=control_by_family(model_id),
                levels=["auto", "minimal", "low", "medium", "high"],
                default="medium",
            ),
            availability="not_checked",
            verified_on="",
            docs_url="",
            notes=(
                "Configured default model (not in the free-tier "
                "registry). Eligibility unverified — submitted as "
                "'configured default', never as a free-tier claim."
            ),
            source="configured_default",
        )


def _family_prefix(model_id: str) -> str:
    for prefix in ("gemini-4", "gemini-3", "gemini-2.5", "gemini-2.0"):
        if model_id.startswith(prefix):
            return prefix
    return "unknown"
