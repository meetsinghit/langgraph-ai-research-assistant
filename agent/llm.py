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
"""
from __future__ import annotations

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
_FREE_TIER_MODELS = {"gemini-2.5-flash", "gemini-2.0-flash"}

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Construct the chat model selected by ``settings.llm_provider``."""
    provider = settings.llm_provider

    if provider == "gemini":
        if not settings.gemini_api_key:
            raise ConfigurationError(
                "Gemini selected but GEMINI_API_KEY is not set. "
                "Get a free-tier key at https://aistudio.google.com/apikey "
                "and add it to .env"
            )
        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            api_key=settings.gemini_api_key,
            temperature=0.2,
            timeout=settings.request_timeout_seconds,
        )

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
            raise ConfigurationError("OpenAI selected but OPENAI_API_KEY is not set.")
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0.2,
            timeout=settings.request_timeout_seconds,
        )

    raise ConfigurationError(f"unknown provider: {provider!r}")


def estimate_cost(
    provider: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
) -> float | None:
    """Indicative USD cost estimate, or None when unsupported.

    Free-tier models estimate to $0.00 while within their free quota.
    Returns None for models without a known pricing model — the
    estimate is a learning aid, never a billing statement.
    """
    if model in _FREE_TIER_MODELS:
        return 0.0
    return None


def verify_connection(settings: Settings) -> dict[str, Any]:
    """Opt-in live smoke test: exactly ONE minimal request.

    Never called automatically — only from ``main.py
    --verify-connection`` or the opt-in live integration test.
    """
    try:
        model = build_chat_model(settings)
        response = model.invoke("Reply with exactly: ok")
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
    except Exception as exc:
        return {
            "ok": False,
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "error": f"{type(exc).__name__}: {exc}",
        }
