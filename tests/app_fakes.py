"""Shared fakes/fixtures for the application-layer tests (4D.1).

Everything is offline: scripted model, in-memory tools, per-test
temp directories. The token is a fixed synthetic value — never a
real credential.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from agent.config import Settings
from tests.fakes import FakeTools, ScriptedChatModel

TEST_TOKEN = "test-token-not-a-real-credential"


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    """Isolated settings: temp data dir, no env file, small budgets."""
    base: dict[str, object] = {
        "_env_file": None,
        "data_dir": str(tmp_path / "data"),
        "app_token": TEST_TOKEN,
        # Tests never serve the real built UI; packaging tests opt in.
        "frontend_dist": str(tmp_path / "no_ui_here"),
        "min_findings": 1,
        "max_research_iterations": 2,
        "max_tool_calls_per_research": 2,
        "sse_heartbeat_seconds": 0.2,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def model_factory(
    created: list[ScriptedChatModel],
    *,
    responses: list[Any] | None = None,
    structured_outputs: list[Any] | None = None,
    raise_on_call: dict[int, Exception] | None = None,
) -> Callable[[Settings], ScriptedChatModel]:
    """Fresh scripted model per run (state resets between runs),
    recorded on ``created`` so tests can inspect prompts/calls."""

    def factory(_: Settings) -> ScriptedChatModel:
        model = ScriptedChatModel(
            responses=list(responses or []),
            structured_outputs=list(structured_outputs or []),
            raise_on_call=dict(raise_on_call or {}),
        )
        created.append(model)
        return model

    return factory


def tools_factory(
    created: list[FakeTools],
    *,
    search_results: list[dict[str, str]] | None = None,
    pages: dict[str, str] | None = None,
    page_text: str = "Fetched page content.",
) -> Callable[[Settings], FakeTools]:
    def factory(_: Settings) -> FakeTools:
        tools = FakeTools(
            search_results=list(search_results or []),
            pages=dict(pages or {}),
            page_text=page_text,
        )
        created.append(tools)
        return tools

    return factory
