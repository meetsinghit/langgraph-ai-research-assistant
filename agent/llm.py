"""Modular LLM factory with provider-specific configuration.

Each provider is constructed with its own native integration — we do
NOT assume every backend speaks the same wire protocol:

- ``gemini``     -> official ``langchain-google-genai`` integration
                    (free tier available; key from aistudio.google.com)
- ``openrouter`` -> OpenAI-compatible client pointed at OpenRouter's
                    documented OpenAI-compatible API (free models exist)
- ``openai``     -> OpenAI-compatible client, PAID usage — disabled
                    unless explicitly opted in via ALLOW_PAID_PROVIDERS

No network call is made at construction time. Use ``verify_connection``
for an explicit, opt-in live smoke test.

Retry policy: ``max_retries=1`` on every provider disables SDK-level
retries. For the Google SDK this is the documented way to disable
retries — ``max_retries=0`` is misinterpreted by the SDK as "use the
default", which retries up to 6 attempts with backoff. Unbounded SDK
retries on top of our own retry policy can extend a stuck request to
minutes (the observed verification hang). Retries are controlled in
exactly one place: the graph's ``_invoke_with_retry`` (one retry after
a rate-limit error).

Thinking policy: thinking configuration is MODEL-AWARE and registry
driven (``agent.gemini_models`` — per-model capabilities verified
against the official Gemini thinking docs). Gemini 3+ models use
``thinking_level`` low|medium|high (``minimal`` is rejected with 400
INVALID_ARGUMENT on e.g. gemini-3.8-flash); Gemini 2.5 models use
``thinking_budget`` token counts (minimal/low 1,024; medium 8,192;
high 24,576) and reject ``thinking_level``; ``auto`` sends NO
thinking parameter (model default) — minimal thinking is never
conflated with disabled thinking. The two parameters are never sent
together. Normal research runs default to "medium" reasoning (Gemini
3) unless configured otherwise; connectivity verification always uses
the cheapest supported config ("low" level / budget 0).
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from agent.config import Settings
from agent.gemini_models import GEMINI_MODEL_REGISTRY, thinking_kwargs

logger = logging.getLogger("agent")


class ConfigurationError(RuntimeError):
    """Raised when the configured provider cannot be constructed."""


# Models with a free tier: estimated cost is $0.00 while within quota.
# Beyond the free quota, list prices apply — verify current pricing
# with the provider before relying on cost estimates in production.
# Single source of truth: the documented eligibility registry (4E.3),
# verified against the official pricing page — NOT the model listing.
_FREE_TIER_MODELS = {
    spec.model_id
    for spec in GEMINI_MODEL_REGISTRY.values()
    if spec.free_tier == "eligible"
}

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#: Full thinking selector vocabulary (see agent.gemini_models: the
#: registry decides which values a given model actually accepts).
ThinkingLevel = Literal["auto", "minimal", "low", "medium", "high"]


def content_to_text(content: Any) -> str:
    """Normalize LLM content into clean, renderable text.

    Gemini 3+ models return content as structured blocks::

        [{"type": "text", "text": "..."},
         {"type": "thinking", "thinking": "...",
          "signature": "..."},
         {"type": "text", "text": "...",
          "extras": {"signature": "..."}}]

    Only ``text`` blocks (and plain strings) are rendered;
    thinking blocks and other non-text parts are skipped so
    internal reasoning and signature metadata never leak
    into displayed answers.

    This is a RENDERING helper. Internal model messages
    must keep their raw content blocks: Gemini
    function-calling models REQUIRE thought signatures to
    be preserved across tool-calling round trips.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                parts.append(block["text"])
        return "".join(parts)
    return str(content)


def _thinking_kwargs(
    model: str, *, level: ThinkingLevel, verification: bool = False
) -> dict[str, Any]:
    """Model-aware thinking configuration (registry-driven, 4E.3).

    Delegates to :func:`agent.gemini_models.thinking_kwargs`, the
    single source of truth mapping a selector level to the parameter
    the model actually accepts: ``thinking_level`` for Gemini 3+
    (validated against the per-model supported set — e.g. "minimal"
    is a 400 on gemini-3.8-flash), ``thinking_budget`` token counts
    for Gemini 2.5 (minimal/low 1,024; medium 8,192; high 24,576),
    nothing for ``auto`` or models without thinking controls. The
    verification path keeps thinking cheap (level "low"; budget 0 on
    budget models — a verification-only constraint, never a research
    mapping). ``thinking_level`` and ``thinking_budget`` are never
    returned together.
    """
    return thinking_kwargs(model, level=level, verification=verification)


def build_chat_model(
    settings: Settings,
    *,
    verification: bool = False,
) -> BaseChatModel:
    """Construct the chat model selected by ``settings.llm_provider``.

    Normal research runs use "medium" reasoning depth (Gemini 3
    models) unless ``settings.llm_thinking_level`` overrides it.

    ``verification=True`` builds a connection-check configuration:
    the lowest SUPPORTED thinking level ("low") and a modest
    output-token cap, so a smoke test cannot burn its token
    budget on reasoning tokens (thinking tokens count toward
    ``max_output_tokens``).
    """
    provider = settings.llm_provider

    if provider == "gemini":
        if not settings.gemini_api_key:
            raise ConfigurationError(
                "Gemini selected but GEMINI_API_KEY is not set. "
                "Get a free-tier key at https://aistudio.google.com/apikey "
                "and add it to .env"
            )
        kwargs: dict[str, Any] = {
            "model": settings.llm_model,
            "api_key": settings.gemini_api_key,
            "temperature": 0.2,
            "timeout": settings.request_timeout_seconds,
            "max_retries": 1,  # disables SDK retries (see module docstring)
        }
        if verification:
            # Connectivity check: cheapest supported thinking config
            # (level "low"; budget 0 on budget-controlled models —
            # thinking off, a verification-only constraint) and a
            # modest output budget — thinking tokens count toward
            # max_output_tokens, so 64 was too tight.
            kwargs.update(
                _thinking_kwargs(
                    settings.llm_model, level="low", verification=True
                )
            )
            kwargs["max_output_tokens"] = (
                settings.connection_verify_max_output_tokens
            )
        else:
            # Default research depth: medium reasoning (Gemini 3),
            # unless configured otherwise. "auto" is a valid setting
            # and sends no thinking parameter (model default).
            thinking_level: ThinkingLevel = (
                settings.llm_thinking_level or "medium"
            )
            kwargs.update(
                _thinking_kwargs(settings.llm_model, level=thinking_level)
            )
        return ChatGoogleGenerativeAI(**kwargs)

    if provider == "openrouter":
        if not settings.openrouter_api_key:
            raise ConfigurationError(
                "OpenRouter selected but OPENROUTER_API_KEY is not set. "
                "Get a key at https://openrouter.ai/keys and add it to .env"
            )
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openrouter_api_key,
            base_url=_OPENROUTER_BASE_URL,
            temperature=0.2,
            timeout=settings.request_timeout_seconds,
            max_retries=1,  # retries are owned by the graph layer
            default_headers={
                "HTTP-Referer": "https://github.com/local/ai-research-assistant",
                "X-Title": "AI Research Assistant (POC)",
            },
        )

    if provider == "openai":
        if not settings.allow_paid_providers:
            raise ConfigurationError(
                "OpenAI is a paid provider and is disabled by default. "
                "Set ALLOW_PAID_PROVIDERS=true in .env to enable it "
                "explicitly (this may incur costs)."
            )
        if not settings.openai_api_key:
            raise ConfigurationError(
                "OpenAI selected but OPENAI_API_KEY is not set."
            )
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0.2,
            timeout=settings.request_timeout_seconds,
            max_retries=1,  # retries are owned by the graph layer
        )

    raise ConfigurationError(f"unknown provider: {provider!r}")


def estimate_cost(
    provider: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
) -> float | None:
    """Indicative USD cost estimate, or None when unsupported.

    Free-tier models estimate to $0.00 while within their free
    quota. Returns None for models without a known pricing model —
    the estimate is a learning aid, never a billing statement.
    """
    if model in _FREE_TIER_MODELS:
        return 0.0
    return None


# ---------------------------------------------------------------------------
# Provider failure classification (WS-D)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderFailure:
    """A classified provider error: fixed-vocabulary ``category``,
    whether it is safe to retry, and the provider's ``Retry-After``
    hint in seconds when observable (never the raw error text)."""

    category: str
    retryable: bool
    retry_after: float | None = None


#: Fixed vocabulary for provider failures (WS-D). Metrics record the
#: category, never the message.
PROVIDER_FAILURE_CATEGORIES = (
    "rate_limit",       # 429 / quota exhaustion — retryable
    "unavailable",      # 503 / overloaded — retryable
    "internal",         # 500/502/504 — retryable
    "timeout",          # request deadline — retryable once
    "network",          # transient connection faults — retryable
    "invalid_request",  # 400/malformed request — permanent, no retry
    "auth",             # 401/403/bad key — permanent, no retry
    "unknown",          # unrecognised — NOT retried blindly
)

_RETRY_AFTER_RE = re.compile(
    r"retry[- ]after[^0-9]{0,40}(\d+(?:\.\d+)?)", re.IGNORECASE
)


def _retry_after_seconds(
    exc: BaseException, text: str
) -> float | None:
    """Provider Retry-After hint, best effort: an exception attribute
    first, then the well-known header/phrase in the message."""
    value = getattr(exc, "retry_after", None)
    if value is not None:
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            pass
    match = _RETRY_AFTER_RE.search(text)
    if match:
        return max(0.0, float(match.group(1)))
    return None


def classify_provider_error(exc: BaseException) -> ProviderFailure:
    """Classify a model-invocation failure for retry decisions.

    Deterministic, message-marker based (integrations wrap SDK
    exceptions inconsistently, so class checks alone are unreliable):

    - retryable: ``rate_limit``, ``unavailable``, ``internal``,
      ``timeout``, ``network`` — these get bounded exponential
      backoff with jitter, honouring Retry-After when present;
    - permanent: ``auth``, ``invalid_request`` — NEVER retried (a
      bad key or malformed request fails identically every time);
    - ``unknown`` — not retried: an unrecognised failure could be
      anything, and blind retries on an unknown error multiply load
      without evidence of transience.

    Combined with SDK retries being disabled at construction
    (``max_retries=1``), exactly ONE layer performs retries.
    """
    name = type(exc).__name__.lower()
    text = f"{type(exc).__name__}: {exc}".lower()

    if any(
        marker in text
        for marker in (
            "401",
            "403",
            "api key",
            "api_key",
            "unauthenticated",
            "unauthorized",
            "unauthorised",
            "permission denied",
            "permission_denied",
            "invalid authentication",
        )
    ):
        return ProviderFailure("auth", False)
    if any(
        marker in text
        for marker in (
            "400",
            "404",
            "bad request",
            "badrequest",
            "invalid_argument",
            "invalid argument",
            "invalid value",
            "invalid name",
            "invalid tool",
            "unsupported",
        )
    ):
        return ProviderFailure("invalid_request", False)
    if any(
        marker in text
        for marker in (
            "429",
            "rate limit",
            "ratelimit",
            "too many requests",
            "resource exhausted",
            "quota",
        )
    ):
        return ProviderFailure(
            "rate_limit", True, _retry_after_seconds(exc, text)
        )
    if any(
        marker in text
        for marker in (
            "503",
            "unavailable",
            "overloaded",
            "try again later",
        )
    ):
        return ProviderFailure(
            "unavailable", True, _retry_after_seconds(exc, text)
        )
    if any(
        marker in text
        for marker in ("500", "502", "504", "internal error", "internal server")
    ):
        return ProviderFailure(
            "internal", True, _retry_after_seconds(exc, text)
        )
    if (
        "timeout" in name
        or "timed out" in text
        or "timeout" in text
        or "deadline" in text
    ):
        return ProviderFailure(
            "timeout", True, _retry_after_seconds(exc, text)
        )
    if any(
        marker in text
        for marker in (
            "connection",
            "transport",
            "broken pipe",
            "reset by peer",
            "unreachable",
            "temporarily disconnected",
            "dns",
        )
    ):
        return ProviderFailure(
            "network", True, _retry_after_seconds(exc, text)
        )
    return ProviderFailure("unknown", False)


#: Markers for a model that the provider no longer serves TO THIS
#: PROJECT (404/NOT_FOUND style — e.g. Google's restriction of the
#: 2.5 family to projects that actively used them: those requests
#: fail with 403 PERMISSION_DENIED or 404 NOT_FOUND "no longer
#: available to new users"). Used only to pick the fixed error
#: vocabulary; the raw message never leaves this function.
_MODEL_GONE_MARKERS = (
    "404",
    "no longer available",
    "not found",
    "models/",
)


def is_model_access_failure(exc: BaseException) -> bool:
    """True when the provider refused the MODEL or the credentials.

    Covers the two permanent access failures (4E.3 review, item 4):

    - ``auth`` — 401/403/permission-denied: bad key OR the project
      is not permitted to use the model (Google's 2.5-family
      restriction for new projects);
    - ``invalid_request`` carrying a model-gone marker (404 NOT_FOUND
      "no longer available to new users").

    Callers use this to raise the CLEAR, fixed-vocabulary
    ``Model access failed: <type name>`` state error instead of the
    generic provider one. Retry behaviour is unchanged (both classes
    were already permanent), and nothing here ever selects, switches
    to, or falls back to another model — the run simply fails with
    its frozen configuration intact.
    """
    failure = classify_provider_error(exc)
    if failure.category == "auth":
        return True
    if failure.category != "invalid_request":
        return False
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _MODEL_GONE_MARKERS)


def verify_connection(
    settings: Settings,
    *,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Opt-in live smoke test: exactly ONE minimal request.

    Designed so it CANNOT hang:

    - the request runs under a hard end-to-end deadline
      (``asyncio.wait_for`` around the async client path — a
      timeout cancels the in-flight HTTP request)
    - SDK retries are disabled (``max_retries=1``)
    - thinking is minimized to the lowest SUPPORTED level and
      output tokens are capped, so the budget cannot be
      consumed by reasoning tokens

    ``deadline`` overrides ``settings.connection_verify_seconds``
    (used by tests to keep verification fast).
    """
    effective_deadline = (
        deadline
        if deadline is not None
        else settings.connection_verify_seconds
    )
    try:
        model = build_chat_model(settings, verification=True)
        response = asyncio.run(
            asyncio.wait_for(
                model.ainvoke("Reply with exactly: ok"),
                timeout=effective_deadline,
            )
        )
        usage = getattr(response, "usage_metadata", None) or {}
        return {
            "ok": True,
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "content": content_to_text(response.content)[:50],
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }
    except ConfigurationError:
        raise
    except TimeoutError:
        # asyncio.TimeoutError is an alias of TimeoutError (3.11+)
        return {
            "ok": False,
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "error": (
                f"verification timed out after {effective_deadline:.1f}s"
            ),
        }
    except Exception as exc:
        return {
            "ok": False,
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "error": f"{type(exc).__name__}: {exc}",
        }
