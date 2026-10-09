"""Opt-in LIVE integration test.

Skipped unless BOTH conditions hold:

- ``RUN_LIVE_TESTS=1`` is set in the environment
- the configured provider has an API key

Makes exactly ONE real request to the configured provider.
Never runs in the default suite.
"""
from __future__ import annotations

import os

import pytest

from agent.config import Settings
from agent.llm import verify_connection

pytestmark = pytest.mark.live


def _enabled() -> bool:
    if os.environ.get("RUN_LIVE_TESTS") != "1":
        return False
    settings = Settings()
    key = {
        "gemini": settings.gemini_api_key,
        "openrouter": settings.openrouter_api_key,
        "openai": settings.openai_api_key,
    }.get(settings.llm_provider)
    return bool(key)


def test_live_connection_to_configured_provider():
    if not _enabled():
        pytest.skip(
            "live test disabled: set RUN_LIVE_TESTS=1 and configure "
            "a provider API key to enable"
        )
    result = verify_connection(Settings())
    assert result["ok"], result.get("error")
    assert result["model"]
