"""Tests for typed configuration loading and secret redaction."""
import pytest
from pydantic import ValidationError

from agent.config import Settings


def test_safe_defaults():
    s = Settings(_env_file=None)
    assert s.llm_provider == "gemini"
    assert s.llm_model == "gemini-2.5-flash"
    assert s.request_timeout_seconds == 30.0
    assert s.max_research_iterations == 3
    assert s.min_findings == 3
    assert s.log_level == "INFO"
    assert s.gemini_api_key is None  # no secret baked into defaults


def test_env_vars_override_defaults(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LLM_MODEL", "llama3.2")
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "45.5")
    monkeypatch.setenv("MAX_RESEARCH_ITERATIONS", "5")
    monkeypatch.setenv("LOG_LEVEL", "debug")

    s = Settings(_env_file=None)
    assert s.llm_provider == "ollama"
    assert s.llm_model == "llama3.2"
    assert s.request_timeout_seconds == 45.5
    assert s.max_research_iterations == 5
    assert s.log_level == "DEBUG"


def test_unknown_provider_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_provider="not-a-provider")


def test_invalid_log_level_is_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, log_level="VERBOSE")


def test_redacted_masks_all_secrets():
    s = Settings(
        _env_file=None,
        gemini_api_key="gem-secret-1",
        openrouter_api_key="or-secret-2",
        openai_api_key="oa-secret-3",
    )
    data = s.redacted()
    assert data["gemini_api_key"] == "***redacted***"
    assert data["openrouter_api_key"] == "***redacted***"
    assert data["openai_api_key"] == "***redacted***"
    for secret in ("gem-secret-1", "or-secret-2", "oa-secret-3"):
        assert secret not in str(data)
