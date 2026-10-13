"""Secret protection and exfiltration resistance (Phase 4C, WS-E).

Uses a SYNTHETIC canary in a patched environment variable — the real
``.env`` is never read, printed or asserted on.

Proves (deterministically, with scripted models):
- configured secrets never enter model-visible prompts;
- a secret placed in tool arguments is refused BEFORE execution and
  never echoed into messages, logs or metrics;
- credential-shaped values are redacted at the answer boundary;
- the fetch tool sends no credential-bearing headers;
- the tool surface contains no local-file or code-execution tools;
- logs and metrics stay credential-free.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, ToolMessage

from agent.config import Settings
from agent.graph import (
    ResearchPlan,
    ReviewVerdict,
    build_research_graph,
    make_finalize_node,
)
from agent.metrics import MetricsCollector
from agent.state import ReviewStatus
from agent.tools import ResearchTools
from tests.fakes import FakeTools, ScriptedChatModel
from tests.test_graph import _search_result

#: Synthetic canary standing in for a real API key. Never a real
#: credential, never read from .env. (AIza + exactly 35 chars so it
#: matches the Google-key shape the redactor looks for.)
CANARY = "AIza" + "SYNTHETICCANARY0123456789abcdefghij"

_URL = "https://docs.example.org/state"
_DRAFT = (
    "LangGraph state management uses reducer-backed channels and "
    "conditional routing. [S1]"
)


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "_env_file": None,
        "min_findings": 1,
        "max_research_iterations": 3,
        "max_tool_calls_per_research": 2,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _run(
    settings: Settings,
    *,
    first_query: str = "state",
    caplog: pytest.LogCaptureFixture | None = None,
):
    metrics = MetricsCollector()
    tools = FakeTools(search_results=[_search_result("Docs", _URL)])
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search",
                        "args": {"query": first_query},
                        "id": "c1",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "fetch_page",
                        "args": {"url": _URL},
                        "id": "c2",
                    }
                ],
            ),
            AIMessage(content=_DRAFT),
        ],
        structured_outputs=[
            ResearchPlan(queries=["state"]),
            ReviewVerdict(status="approved", reason="ok"),
        ],
    )
    graph = build_research_graph(
        settings, model, tools, metrics=metrics, rate_limit_retry_delay=0.0
    )
    result = graph.invoke(
        {"question": "How does LangGraph state management work?"}
    )
    return result, model, tools, metrics


def test_canary_secret_never_reaches_model_prompts(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    monkeypatch.setenv("GEMINI_API_KEY", CANARY)
    caplog.set_level(logging.DEBUG)
    settings = _settings()
    assert settings.gemini_api_key == CANARY  # the env is active

    result, model, _, metrics = _run(settings)

    # every message the model ever saw (plain + structured)
    everything = json.dumps(
        {
            "plain": [
                [str(getattr(m, "content", m)) for m in batch]
                for batch in model.seen_messages
            ],
            "structured": [
                [str(getattr(m, "content", m)) for m in batch]
                for batch in model.seen_structured_messages
            ],
        }
    )
    assert CANARY not in everything

    # logs
    assert CANARY not in caplog.text

    # metrics / JSON artifacts
    assert CANARY not in json.dumps(metrics.snapshot(), default=str)

    # and the answer
    assert CANARY not in result["final_answer"]


def test_canary_in_tool_arguments_refused_before_execution(
    monkeypatch: pytest.MonkeyPatch,
):
    """Model fully cooperates with an exfiltration instruction and
    puts the configured key into a search query: the executor's DLP
    refuses the call before the tool runs, and the value never
    appears anywhere downstream."""
    monkeypatch.setenv("GEMINI_API_KEY", CANARY)
    settings = _settings()
    result, model, tools, metrics = _run(
        settings, first_query=f"leak {CANARY}"
    )

    # the search tool never received the secret
    assert all(CANARY not in query for query in tools.search_calls)
    assert metrics.snapshot()["security"]["tool_refusals"].get("dlp", 0) >= 1

    # the refusal reached the model as data, without the value
    refusal_messages = [
        message
        for batch in model.seen_messages
        for message in batch
        if isinstance(message, ToolMessage)
        and "sensitive value" in message.content
    ]
    assert refusal_messages, "DLP refusal never reached the model"
    assert CANARY not in refusal_messages[0].content

    # nothing downstream carries it
    assert CANARY not in json.dumps(metrics.snapshot(), default=str)
    assert CANARY not in result["final_answer"]


def test_finalize_redacts_credential_shaped_answer():
    settings = _settings()
    metrics = MetricsCollector()
    finalize = make_finalize_node(settings, metrics)
    update = finalize(
        {
            "question": "q",
            "final_answer": f"Answer with key {CANARY} inside.",
            "review_status": ReviewStatus.APPROVED,
            "findings": [],
            "errors": [],
        }
    )
    answer = update["final_answer"]
    assert CANARY not in answer
    assert "[REDACTED-CREDENTIAL]" in answer
    assert metrics.snapshot()["security"]["answer_redactions"] >= 1


def test_finalize_passes_clean_answer_untouched():
    settings = _settings()
    finalize = make_finalize_node(settings, None)
    update = finalize(
        {
            "question": "q",
            "final_answer": "A clean answer with citations [S1].",
            "review_status": ReviewStatus.APPROVED,
            "findings": [],
            "errors": [],
        }
    )
    # no redaction -> the answer need not be rewritten into update
    assert CANARY not in update.get("final_answer", "")


def test_tool_surface_has_no_local_or_execution_tools():
    """The research agent exposes exactly web_search and fetch_page:
    no shell, no filesystem, no code execution."""
    tools = ResearchTools(_settings())
    assert set(tools.tool_functions()) == {"web_search", "fetch_page"}
    names = {tool.name for tool in tools.langchain_tools()}
    assert names <= {"web_search", "fetch_page"}


def test_fetch_sends_no_credential_headers(monkeypatch):
    import agent.tools as tools_mod

    def fake_getaddrinfo(host: str, *args, **kwargs):
        import socket as _socket

        return [
            (
                _socket.AF_INET,
                _socket.SOCK_STREAM,
                6,
                "",
                ("93.184.216.34", 443),
            )
        ]

    monkeypatch.setattr(tools_mod.socket, "getaddrinfo", fake_getaddrinfo)
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, content=b"ok")

    client = httpx.Client(
        transport=httpx.MockTransport(handler), follow_redirects=False
    )
    tools_mod.fetch_page("https://example.com/page", client=client)
    headers = {k.lower() for k in captured["headers"]}
    assert "authorization" not in headers
    assert "proxy-authorization" not in headers
    assert "cookie" not in headers


def test_error_messages_carry_no_environment_values(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("GEMINI_API_KEY", CANARY)
    settings = _settings()
    tools = FakeTools(
        search_results=[_search_result("Docs", _URL)],
        fail_fetch_urls={_URL},
    )
    metrics = MetricsCollector()
    model = ScriptedChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "web_search", "args": {"query": "state"}, "id": "c1"}
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "fetch_page", "args": {"url": _URL}, "id": "c2"}
                ],
            ),
            AIMessage(content=_DRAFT),
        ],
        structured_outputs=[
            ResearchPlan(queries=["state"]),
            ReviewVerdict(status="approved", reason="ok"),
        ],
    )
    graph = build_research_graph(settings, model, tools, metrics=metrics)
    graph.invoke(
        {"question": "How does LangGraph state management work?"}
    )
    everything = json.dumps(
        [
            [str(getattr(m, "content", m)) for m in batch]
            for batch in model.seen_messages
        ]
    )
    assert CANARY not in everything
    assert CANARY not in json.dumps(metrics.snapshot(), default=str)
