"""Tests for research tools: search, retrieval, SSRF protection."""
import socket
from typing import Any

import httpx
import pytest

import agent.tools as tools
from agent.tools import (
    ToolError,
    UnsafeURLError,
    fetch_page,
    to_findings,
    web_search,
)


# ---------- web_search ----------


def make_fake_ddgs(
    results: list[dict[str, Any]] | None = None, fail: bool = False
) -> type:
    class FakeDDGS:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> "FakeDDGS":
            return self

        def __exit__(self, *args: Any) -> bool:
            return False

        def text(self, query: str, max_results: int = 5) -> list[dict]:
            if fail:
                raise RuntimeError("search backend unreachable")
            return list(results or [])

    return FakeDDGS


def test_web_search_maps_results(monkeypatch):
    monkeypatch.setattr(
        tools,
        "DDGS",
        make_fake_ddgs(
            [
                {"title": "T1", "body": "B1", "href": "https://example.com/1"},
                {"title": "T2", "body": "B2", "href": "https://example.com/2"},
            ]
        ),
    )
    results = web_search("query", max_results=5, timeout=5.0)
    assert results == [
        {"title": "T1", "snippet": "B1", "url": "https://example.com/1"},
        {"title": "T2", "snippet": "B2", "url": "https://example.com/2"},
    ]


def test_web_search_maps_legacy_result_keys(monkeypatch):
    """The ddgs package returns title/href/body; the parser
    also accepts the older url/snippet keys."""
    monkeypatch.setattr(
        tools,
        "DDGS",
        make_fake_ddgs(
            [{"title": "T", "snippet": "S", "url": "https://example.com/x"}]
        ),
    )
    results = web_search("query", max_results=5, timeout=5.0)
    assert results == [
        {"title": "T", "snippet": "S", "url": "https://example.com/x"}
    ]


def test_web_search_empty_results_raise_tool_error(monkeypatch):
    """An empty result set is a FAILURE to gather evidence, not
    evidence — it must surface as a meaningful error, never as
    a silent "success" with no data."""
    monkeypatch.setattr(tools, "DDGS", make_fake_ddgs([]))
    with pytest.raises(ToolError, match="no results"):
        web_search("query", timeout=5.0)


def test_web_search_skips_results_without_urls(monkeypatch):
    """Results without a URL have no verifiable provenance and
    are not usable evidence."""
    monkeypatch.setattr(
        tools, "DDGS", make_fake_ddgs([{"title": "T", "body": "B"}])
    )
    with pytest.raises(ToolError, match="no results"):
        web_search("query", timeout=5.0)


def test_web_search_timeout_raises_tool_error(monkeypatch):
    """Backend timeouts surface with their type and message."""

    class TimeoutDDGS:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __enter__(self) -> "TimeoutDDGS":
            return self

        def __exit__(self, *args: Any) -> bool:
            return False

        def text(self, query: str, max_results: int = 5) -> list[dict]:
            raise TimeoutError("search timed out")

    monkeypatch.setattr(tools, "DDGS", TimeoutDDGS)
    with pytest.raises(ToolError, match="timed out"):
        web_search("query", timeout=5.0)


def test_web_search_backend_failure_raises_tool_error(monkeypatch):
    monkeypatch.setattr(tools, "DDGS", make_fake_ddgs(fail=True))
    with pytest.raises(ToolError, match="web_search failed"):
        web_search("query", timeout=5.0)


def test_web_search_rejects_empty_query():
    with pytest.raises(ToolError):
        web_search("   ", timeout=5.0)


# ---------- URL validation (SSRF protection) ----------


def _public_dns(monkeypatch: pytest.MonkeyPatch, address: str = "93.184.216.34") -> None:
    def fake_getaddrinfo(host: str, port: Any, *args: Any, **kwargs: Any):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))]

    monkeypatch.setattr(tools.socket, "getaddrinfo", fake_getaddrinfo)


def test_validate_url_rejects_non_http_schemes():
    for url in ("file:///etc/passwd", "ftp://example.com", "gopher://example.com"):
        with pytest.raises(UnsafeURLError):
            tools._validate_url(url)


def test_validate_url_rejects_localhost():
    with pytest.raises(UnsafeURLError):
        tools._validate_url("https://localhost/admin")


@pytest.mark.parametrize(
    "ip",
    ["127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254", "[::1]"],
)
def test_validate_url_rejects_non_public_literal_ips(ip: str):
    with pytest.raises(UnsafeURLError):
        tools._validate_url(f"http://{ip}/")


def test_validate_url_accepts_public_hostname(monkeypatch):
    _public_dns(monkeypatch)
    tools._validate_url("https://example.com/page")  # no exception


def test_validate_url_rejects_dns_rebinding_to_private_ip(monkeypatch):
    _public_dns(monkeypatch, address="10.0.0.5")
    with pytest.raises(UnsafeURLError, match="non-public"):
        tools._validate_url("https://example.com/page")


# ---------- fetch_page ----------


def _client(handler: Any) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(handler), follow_redirects=False
    )


def test_fetch_page_extracts_public_page_text(monkeypatch):
    _public_dns(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "example.com"
        return httpx.Response(
            200,
            text="<html><head><script>evil()</script>"
            "<style>.x{}</style></head>"
            "<body><p>Hello &amp; welcome</p></body></html>",
        )

    text = fetch_page("https://example.com/page", client=_client(handler))
    assert "Hello & welcome" in text
    assert "evil()" not in text
    assert ".x{}" not in text


def test_fetch_page_blocks_redirect_to_private_ip(monkeypatch):
    _public_dns(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"location": "http://169.254.169.254/latest/meta-data"},
        )

    with pytest.raises(UnsafeURLError):
        fetch_page("https://example.com/start", client=_client(handler))


def test_fetch_page_follows_safe_redirect(monkeypatch):
    _public_dns(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "/target"})
        return httpx.Response(200, text="final destination")

    text = fetch_page("https://example.com/start", client=_client(handler))
    assert text == "final destination"


def test_fetch_page_limits_redirects(monkeypatch):
    _public_dns(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302, headers={"location": f"/loop/{request.url.path}"}
        )

    with pytest.raises(ToolError, match="exceeded"):
        fetch_page(
            "https://example.com/start", client=_client(handler), max_redirects=2
        )


def test_fetch_page_wraps_timeouts(monkeypatch):
    _public_dns(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out")

    with pytest.raises(ToolError, match="request failed"):
        fetch_page("https://example.com/slow", client=_client(handler))


def test_fetch_page_enforces_size_limit(monkeypatch):
    _public_dns(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 10_000)

    text = fetch_page(
        "https://example.com/big", client=_client(handler), max_bytes=1_000
    )
    assert "content truncated" in text
    assert len(text) < 2_000


# ---------- ToolExecutor ----------


def test_executor_runs_known_tool():
    executor = tools.ToolExecutor({"web_search": lambda query: [query]})
    result = executor.execute(
        {"name": "web_search", "args": {"query": "q"}, "id": "1"}
    )
    assert result["ok"] is True
    assert result["data"] == ["q"]


def test_executor_rejects_unknown_tool():
    executor = tools.ToolExecutor({})
    result = executor.execute({"name": "nope", "args": {}, "id": "1"})
    assert result["ok"] is False
    assert "unknown tool" in result["error"]


def test_executor_rejects_non_object_args():
    executor = tools.ToolExecutor({"web_search": lambda query: []})
    result = executor.execute(
        {"name": "web_search", "args": "not-a-dict", "id": "1"}
    )
    assert result["ok"] is False
    assert "arguments must be an object" in result["error"]


def test_executor_converts_tool_exceptions_to_errors():
    def boom(**kwargs: Any) -> Any:
        raise RuntimeError("backend down")

    executor = tools.ToolExecutor({"web_search": boom})
    result = executor.execute(
        {"name": "web_search", "args": {"query": "q"}, "id": "1"}
    )
    assert result["ok"] is False
    assert "backend down" in result["error"]


def test_executor_converts_bad_arguments_to_errors():
    executor = tools.ToolExecutor({"web_search": lambda query: []})
    result = executor.execute(
        {"name": "web_search", "args": {"wrong": 1}, "id": "1"}
    )
    assert result["ok"] is False
    assert "invalid arguments" in result["error"]


# ---------- search -> executor -> findings pipeline ----------


def _executor_with_real_search(monkeypatch) -> tools.ToolExecutor:
    """ToolExecutor wired to the real web_search (with a
        faked ddgs backend — no network)."""
    monkeypatch.setattr(
        tools,
        "DDGS",
        make_fake_ddgs(
            [
                {
                    "title": "LangGraph docs",
                    "body": "build stateful agents",
                    "href": "https://langchain-ai.github.io/langgraph/",
                }
            ]
        ),
    )
    return tools.ToolExecutor(
        {
            "web_search": lambda query: web_search(query, timeout=5.0)
        }
    )


def test_search_results_flow_through_executor_to_findings(monkeypatch):
    """Search results become ToolExecutor outputs, then
    LangGraph findings with verifiable source URLs."""
    executor = _executor_with_real_search(monkeypatch)
    result = executor.execute(
        {"name": "web_search", "args": {"query": "langgraph"}, "id": "1"}
    )
    assert result["ok"] is True
    assert result["tool"] == "web_search"

    findings = to_findings(result)
    assert len(findings) == 1
    assert findings[0]["source_url"] == (
        "https://langchain-ai.github.io/langgraph/"
    )
    assert "LangGraph docs" in findings[0]["text"]


def test_empty_search_becomes_meaningful_tool_result(monkeypatch):
    """Empty results produce a controlled error result the LLM
    can react to — not a silent success with empty data."""
    monkeypatch.setattr(tools, "DDGS", make_fake_ddgs([]))
    executor = tools.ToolExecutor(
        {
            "web_search": lambda query: web_search(query, timeout=5.0)
        }
    )
    result = executor.execute(
        {"name": "web_search", "args": {"query": "q"}, "id": "1"}
    )
    assert result["ok"] is False
    assert "no results" in result["error"]
    assert to_findings(result) == []  # empty is never evidence


def test_failed_search_becomes_meaningful_tool_result(monkeypatch):
    monkeypatch.setattr(tools, "DDGS", make_fake_ddgs(fail=True))
    executor = tools.ToolExecutor(
        {
            "web_search": lambda query: web_search(query, timeout=5.0)
        }
    )
    result = executor.execute(
        {"name": "web_search", "args": {"query": "q"}, "id": "1"}
    )
    assert result["ok"] is False
    assert "web_search failed" in result["error"]
    assert to_findings(result) == []


# ---------- findings extraction ----------


def test_to_findings_from_search_results():
    result = {
        "tool": "web_search",
        "ok": True,
        "data": [{"title": "T", "snippet": "S", "url": "https://example.com"}],
        "error": None,
        "args": {"query": "q"},
    }
    findings = to_findings(result)
    assert len(findings) == 1
    assert findings[0]["source_url"] == "https://example.com"
    assert "T" in findings[0]["text"]


def test_to_findings_from_fetch_result():
    result = {
        "tool": "fetch_page",
        "ok": True,
        "data": "page text " * 200,
        "error": None,
        "args": {"url": "https://example.com/page"},
    }
    findings = to_findings(result)
    assert findings[0]["source_url"] == "https://example.com/page"
    assert len(findings[0]["text"]) <= 500


def test_to_findings_ignores_failed_results():
    result = {
        "tool": "web_search",
        "ok": False,
        "data": None,
        "error": "boom",
        "args": {},
    }
    assert to_findings(result) == []


def test_to_findings_from_empty_search_data():
    """Empty search data yields no findings — defensive: an
    empty result must never count as meaningful evidence."""
    result = {
        "tool": "web_search",
        "ok": True,
        "data": [],
        "error": None,
        "args": {"query": "q"},
    }
    assert to_findings(result) == []


def test_to_findings_ignores_unknown_tools():
    result = {
        "tool": "mystery", "ok": True, "data": [], "error": None, "args": {}
    }
    assert to_findings(result) == []
