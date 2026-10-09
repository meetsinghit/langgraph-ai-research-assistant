"""Typed application configuration.

Configuration is loaded from environment variables (and an optional ``.env``
file in the project root) and validated by pydantic-settings. Secrets are
held in memory only and are masked by :meth:`Settings.redacted` before they
are ever logged or printed.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["gemini", "openrouter", "openai", "ollama"]

_SECRET_FIELDS = ("gemini_api_key", "openrouter_api_key", "openai_api_key")


class Settings(BaseSettings):
    """Central, typed configuration for the AI Research Assistant."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # GEMINI_API_KEY maps to gemini_api_key
        extra="ignore",  # unrelated env vars are ignored, not fatal
    )

    # --- LLM backend (wired up in Phase 3) ---
    llm_provider: ProviderName = "gemini"
    llm_model: str = "gemini-2.5-flash"
    llm_base_url: str | None = None

    # API keys: read from env/.env only, never logged (see redacted()).
    gemini_api_key: str | None = None
    openrouter_api_key: str | None = None
    openai_api_key: str | None = None

    # --- Agent behaviour ---
    request_timeout_seconds: float = 30.0
    max_research_iterations: int = 3
    search_results_per_query: int = 5
    min_findings: int = 3

    # --- Observability ---
    log_level: str = "INFO"
    log_file: str = "research_agent.log"

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        level = value.upper()
        if not hasattr(logging, level):
            raise ValueError(f"invalid log level: {value!r}")
        return level

    def redacted(self) -> dict[str, Any]:
        """Configuration with every secret masked — the only safe way to log it."""
        data: dict[str, Any] = self.model_dump()
        for field in _SECRET_FIELDS:
            if data.get(field):
                data[field] = "***redacted***"
        return data


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings (env vars are read once per process)."""
    return Settings()
