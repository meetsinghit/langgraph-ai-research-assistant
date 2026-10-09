"""Tests for the LLM factory (construction only — no network calls)."""
import pytest
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from agent.config import Settings
from agent.llm import (
    ConfigurationError,
    build_chat_model,
    estimate_cost,
)


def test_gemini_uses_native_integration():
    settings = Settings(_env_file=None, gemini_api_key="fake-key")
    model = build_chat_model(settings)
    assert isinstance(model, ChatGoogleGenerativeAI)


def test_gemini_requires_api_key():
    settings = Settings(_env_file=None, gemini_api_key=None)
    with pytest.raises(ConfigurationError):
        build_chat_model(settings)


def test_openrouter_uses_openai_compatible_client():
    settings = Settings(
        _env_file=None,
        llm_provider="openrouter",
        llm_model="openai/gpt-4o-mini",
        openrouter_api_key="fake-key",
    )
    model = build_chat_model(settings)
    assert isinstance(model, ChatOpenAI)
    assert "openrouter.ai" in str(model.openai_api_base)


def test_openrouter_requires_api_key():
    settings = Settings(_env_file=None, llm_provider="openrouter")
    with pytest.raises(ConfigurationError):
        build_chat_model(settings)


def test_openai_is_disabled_by_default():
    settings = Settings(
        _env_file=None, llm_provider="openai", openai_api_key="fake-key"
    )
    with pytest.raises(ConfigurationError):
        build_chat_model(settings)


def test_openai_works_with_explicit_opt_in():
    settings = Settings(
        _env_file=None,
        llm_provider="openai",
        openai_api_key="fake-key",
        allow_paid_providers=True,
    )
    model = build_chat_model(settings)
    assert isinstance(model, ChatOpenAI)


def test_estimate_cost_free_tier_models_are_zero():
    assert estimate_cost("gemini", "gemini-2.5-flash", 1_000_000, 1_000_000) == 0.0


def test_estimate_cost_unknown_model_returns_none():
    assert estimate_cost("openrouter", "some/unknown-model", 100, 100) is None
