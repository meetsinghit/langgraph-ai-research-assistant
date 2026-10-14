"""Tests for the LLM factory and connection verification.

All tests are mocked — no network calls, no API keys required.
The native LangChain Gemini client is exercised independently of
the LangGraph application by patching the google-genai SDK's
generate_content methods and capturing the outgoing request.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from google import genai
from google.genai import types
from langchain_core.messages import AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

import agent.llm as llm
from agent.config import Settings
from agent.llm import (
    ConfigurationError,
    build_chat_model,
    content_to_text,
    estimate_cost,
    verify_connection,
)


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {"_env_file": None, "gemini_api_key": "fake-key"}
    base.update(overrides)
    return Settings(**base)


# ---------- provider construction ----------


def test_gemini_uses_native_integration():
    model = build_chat_model(_settings())
    assert isinstance(model, ChatGoogleGenerativeAI)
    assert model.model == "gemini-3.5-flash-lite"  # restored default
    assert model.temperature == 0.2
    assert model.timeout == 30.0
    assert model.max_retries == 1  # SDK retries disabled
    assert model.thinking_level == "medium"  # default research depth


def test_gemini_verification_model_uses_low_thinking():
    """Verification uses "low" — "minimal" is rejected by
    gemini-3.8-flash with 400 INVALID_ARGUMENT."""
    model = build_chat_model(_settings(), verification=True)
    assert isinstance(model, ChatGoogleGenerativeAI)
    assert model.thinking_level == "low"
    # Thinking tokens count toward max_output_tokens, so the
    # verification budget must leave room for the reply.
    assert model.max_output_tokens == 256


def test_configured_thinking_level_is_respected():
    model = build_chat_model(_settings(llm_thinking_level="high"))
    assert model.thinking_level == "high"


def test_gemini_2_5_models_use_thinking_budget():
    """Gemini 2.5 models use thinking_budget (token count),
    not thinking_level.

    4E.3 behavior change (documented): the default research level
    ("medium") now maps to the OFFICIAL medium budget for the
    budget-controlled family (8,192 tokens — the OpenAI-compat
    mapping; 2.5 dynamic thinking defaults to the same ceiling)
    instead of leaving the budget implicit. The assertion became
    more specific (an actual wire value), not weaker.
    """
    model = build_chat_model(_settings(llm_model="gemini-2.5-flash"))
    assert model.thinking_level is None
    assert model.thinking_budget == 8192  # official medium budget

    # "auto" selects the model default: NO thinking parameter.
    auto = build_chat_model(
        _settings(llm_model="gemini-2.5-flash", llm_thinking_level="auto")
    )
    assert auto.thinking_level is None
    assert auto.thinking_budget is None

    verification = build_chat_model(
        _settings(llm_model="gemini-2.5-flash"), verification=True
    )
    assert verification.thinking_level is None
    assert verification.thinking_budget == 0  # disabled for verification


def test_older_gemini_models_get_no_thinking_config():
    """Models without thinking controls get no thinking config."""
    model = build_chat_model(_settings(llm_model="gemini-2.0-flash"))
    assert model.thinking_level is None
    assert model.thinking_budget is None

    verification = build_chat_model(
        _settings(llm_model="gemini-2.0-flash"), verification=True
    )
    assert verification.thinking_level is None
    assert verification.thinking_budget is None
    assert verification.max_output_tokens == 256


def test_gemini_requires_api_key():
    with pytest.raises(ConfigurationError):
        build_chat_model(_settings(gemini_api_key=None))


def test_openrouter_uses_openai_compatible_client():
    model = build_chat_model(
        _settings(
            llm_provider="openrouter",
            llm_model="openai/gpt-4o-mini",
            openrouter_api_key="fake-key",
        )
    )
    assert isinstance(model, ChatOpenAI)
    assert "openrouter.ai" in str(model.openai_api_base)
    assert model.max_retries == 1


def test_openrouter_requires_api_key():
    with pytest.raises(ConfigurationError):
        build_chat_model(_settings(llm_provider="openrouter"))


def test_openai_is_disabled_by_default():
    with pytest.raises(ConfigurationError):
        build_chat_model(
            _settings(llm_provider="openai", openai_api_key="fake-key")
        )


def test_openai_works_with_explicit_opt_in():
    model = build_chat_model(
        _settings(
            llm_provider="openai",
            openai_api_key="fake-key",
            allow_paid_providers=True,
        )
    )
    assert isinstance(model, ChatOpenAI)
    assert model.max_retries == 1


def test_estimate_cost_free_tier_models_are_zero():
    assert estimate_cost("gemini", "gemini-3.8-flash", 1_000, 1_000) == 0.0
    assert estimate_cost("gemini", "gemini-2.5-flash", 1_000, 1_000) == 0.0


def test_estimate_cost_unknown_model_returns_none():
    assert estimate_cost("openrouter", "some/unknown-model", 100, 100) is None


# ---------- native Gemini client (independent of LangGraph) ----------


def _mock_response(text: str) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=text)],
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=3,
            candidates_token_count=2,
            total_token_count=5,
        ),
    )


def _verification_model() -> ChatGoogleGenerativeAI:
    return build_chat_model(
        _settings(request_timeout_seconds=5.0), verification=True
    )


def test_native_gemini_client_receives_timeout_retry_and_thinking_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_generate_content(
        self: Any, **request: Any
    ) -> types.GenerateContentResponse:
        captured.update(request)
        return _mock_response("ok")

    monkeypatch.setattr(
        genai.models.Models, "generate_content", fake_generate_content
    )

    model = _verification_model()
    response = model.invoke("Say ok")

    assert response.content == "ok"
    config = captured["config"]
    assert config.http_options.timeout == 5000  # seconds -> milliseconds
    assert config.http_options.retry_options.attempts == 1
    assert config.thinking_config.thinking_level == (
        types.ThinkingLevel.LOW
    )
    assert config.max_output_tokens == 256


def test_native_gemini_async_client_receives_same_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_generate_content(
        self: Any, **request: Any
    ) -> types.GenerateContentResponse:
        captured.update(request)
        return _mock_response("ok")

    monkeypatch.setattr(
        genai.models.AsyncModels, "generate_content", fake_generate_content
    )

    model = _verification_model()
    response = asyncio.run(model.ainvoke("Say ok"))

    assert response.content == "ok"
    config = captured["config"]
    assert config.http_options.timeout == 5000
    assert config.http_options.retry_options.attempts == 1
    assert config.thinking_config.thinking_level == (
        types.ThinkingLevel.LOW
    )
    assert config.max_output_tokens == 256


def test_native_gemini_client_default_research_uses_medium_thinking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_generate_content(
        self: Any, **request: Any
    ) -> types.GenerateContentResponse:
        captured.update(request)
        return _mock_response("ok")

    monkeypatch.setattr(
        genai.models.Models, "generate_content", fake_generate_content
    )

    model = build_chat_model(_settings(request_timeout_seconds=5.0))
    model.invoke("Say ok")

    config = captured["config"]
    assert config.thinking_config.thinking_level == (
        types.ThinkingLevel.MEDIUM
    )


def test_wire_budget_family_sends_thinking_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """4E.3 review item 6: a selected level must reach the provider
    as the EXACT request configuration — budget-controlled models
    send thinking_budget token counts, not a displayed label."""
    captured: dict[str, Any] = {}

    def fake_generate_content(
        self: Any, **request: Any
    ) -> types.GenerateContentResponse:
        captured.update(request)
        return _mock_response("ok")

    monkeypatch.setattr(
        genai.models.Models, "generate_content", fake_generate_content
    )

    model = build_chat_model(
        _settings(llm_model="gemini-2.5-flash", request_timeout_seconds=5.0)
    )
    model.invoke("Say ok")

    config = captured["config"]
    assert config.thinking_config is not None
    assert config.thinking_config.thinking_budget == 8192
    assert config.thinking_config.thinking_level is None


def test_wire_auto_omits_thinking_config_entirely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """4E.3 review item 6: "auto" means NO thinking parameter on
    the wire — the request carries no thinking_config at all."""
    captured: dict[str, Any] = {}

    def fake_generate_content(
        self: Any, **request: Any
    ) -> types.GenerateContentResponse:
        captured.update(request)
        return _mock_response("ok")

    monkeypatch.setattr(
        genai.models.Models, "generate_content", fake_generate_content
    )

    model = build_chat_model(
        _settings(
            llm_model="gemini-3.6-flash",
            llm_thinking_level="auto",
            request_timeout_seconds=5.0,
        )
    )
    model.invoke("Say ok")

    assert captured["config"].thinking_config is None


def test_wire_minimal_sent_only_to_models_that_support_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """4E.3 review item 6: "minimal" arrives as ThinkingLevel.MINIMAL
    for a model that supports it (the registry rejects it on
    gemini-3.8-flash BEFORE anything reaches the provider)."""
    captured: dict[str, Any] = {}

    def fake_generate_content(
        self: Any, **request: Any
    ) -> types.GenerateContentResponse:
        captured.update(request)
        return _mock_response("ok")

    monkeypatch.setattr(
        genai.models.Models, "generate_content", fake_generate_content
    )

    model = build_chat_model(
        _settings(
            llm_model="gemini-3.6-flash",
            llm_thinking_level="minimal",
            request_timeout_seconds=5.0,
        )
    )
    model.invoke("Say ok")

    config = captured["config"]
    assert config.thinking_config.thinking_level == (
        types.ThinkingLevel.MINIMAL
    )


# ---------- connection verification ----------


class FakeAsyncModel:
    """Deterministic stand-in exposing only ``ainvoke``."""

    def __init__(self, delay: float = 0.0, exc: Exception | None = None):
        self.delay = delay
        self.exc = exc

    async def ainvoke(self, prompt: Any, **kwargs: Any) -> AIMessage:
        if self.exc is not None:
            raise self.exc
        if self.delay:
            await asyncio.sleep(self.delay)
        return AIMessage(
            content="ok",
            usage_metadata={
                "input_tokens": 3,
                "output_tokens": 2,
                "total_tokens": 5,
            },
        )


def test_verify_connection_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        llm, "build_chat_model", lambda settings, **kwargs: FakeAsyncModel()
    )
    result = verify_connection(_settings(), deadline=5.0)
    assert result["ok"] is True
    assert result["model"] == "gemini-3.5-flash-lite"
    assert result["input_tokens"] == 3
    assert result["output_tokens"] == 2


def test_verify_connection_times_out_without_hanging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm,
        "build_chat_model",
        lambda settings, **kwargs: FakeAsyncModel(delay=30.0),
    )
    started = time.monotonic()
    result = verify_connection(_settings(), deadline=0.2)
    elapsed = time.monotonic() - started

    assert result["ok"] is False
    assert "timed out" in result["error"]
    assert elapsed < 5.0  # returned promptly instead of hanging


def test_verify_connection_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        llm,
        "build_chat_model",
        lambda settings, **kwargs: FakeAsyncModel(
            exc=RuntimeError("provider unreachable")
        ),
    )
    result = verify_connection(_settings(), deadline=5.0)
    assert result["ok"] is False
    assert "provider unreachable" in result["error"]


def test_verify_connection_configuration_error_propagates() -> None:
    with pytest.raises(ConfigurationError):
        verify_connection(_settings(gemini_api_key=None), deadline=5.0)


# ---------- content rendering ----------


def test_content_to_text_renders_gemini_blocks_as_clean_markdown():
    """A Gemini 3 answer arrives as a list of blocks; only the
    visible text is rendered, joined as clean Markdown."""
    content = [
        {"type": "text", "text": "# Finding\n"},
        {"type": "text", "text": "A second paragraph."},
        {
            "type": "thinking",
            "thinking": "internal reasoning",
            "signature": "SECRET_SIG",
        },
    ]
    assert content_to_text(content) == "# Finding\nA second paragraph."


def test_content_to_text_passes_plain_strings_through():
    assert content_to_text("already text") == "already text"
    assert content_to_text(None) == ""


def test_content_to_text_drops_thought_signature_metadata():
    """Signature/thinking blocks must never surface as answer
    prose — they are model-internal metadata."""
    content = [
        {
            "type": "thinking",
            "thinking": "chain of thought",
            "signature": "SECRET_SIG",
        },
        {"type": "text", "text": "visible"},
    ]
    rendered = content_to_text(content)
    assert rendered == "visible"
    assert "SECRET_SIG" not in rendered
    assert "chain of thought" not in rendered
