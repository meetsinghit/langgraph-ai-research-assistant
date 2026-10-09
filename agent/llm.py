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
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from agent.config import Settings

logger = logging.getLogger("agent")


class ConfigurationError(RuntimeError):
    """Raised when the configured provider cannot be constructed."""


# Models with a free tier: estimated cost is $0.00 while within quota.
# Beyond the free quota, list prices apply — verify current pricing
# with the provider before relying on cost estimates in production.
_FREE_TIER_MODELS = {
    "gemini-3.8-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
}

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def build_chat_model(
    settings: Settings,
    *,
    verification: bool = False,
) -> BaseChatModel:
    """Construct the chat model selected by ``settings.llm_provider``.

    ``verification=True`` builds a connection-check configuration:
    thinking minimized and output tokens capped, so a smoke test
    cannot burn its token budget on reasoning tokens (Gemini 3
    models think by default).
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
            kwargs["thinking_level"] = "minimal"  # lowest reasoning depth
            kwargs["max_output_tokens"] = (
                settings.connection_verify_max_output_tokens
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
    - thinking is minimized and output tokens are capped, so the
      small budget cannot be consumed by reasoning tokens

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
            "content": (response.content or "")[:50],
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
