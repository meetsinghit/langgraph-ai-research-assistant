"""Research tools: web search and webpage retrieval.

Security rules enforced here:

- HTTP/HTTPS schemes only; every other scheme is rejected
- localhost, private, link-local and reserved IP ranges are blocked —
  for literal IPs AND for hostnames after DNS resolution, which
  blocks DNS-based SSRF bypasses
- redirects are followed manually, and EVERY hop is re-validated,
  which blocks unsafe redirects
- per-request timeout, response-size cap and redirect cap

Note: DNS is resolved for validation and then resolved again by the
HTTP client (a TOCTOU window exists in any such design); every hop
is re-checked to keep that window small.

Retrieved content is DATA for the agent, never instructions.
"""
from __future__ import annotations

import html
import ipaddress
import json
import logging
import re
import socket
import ssl
from typing import Any, Callable, TypedDict
from urllib.parse import urljoin, urlparse

import httpx
from ddgs import DDGS
from langchain_core.tools import BaseTool, StructuredTool

from agent.config import Settings
from agent.state import Finding

logger = logging.getLogger("agent")

try:  # optional: OS trust store (Windows cert store / macOS keychain)
    import truststore  # type: ignore[import-not-found]

    _TRUSTSTORE_AVAILABLE = True
except ImportError:  # pragma: no cover - truststore is a pinned dependency
    truststore = None  # type: ignore[assignment]
    _TRUSTSTORE_AVAILABLE = False


class ToolError(RuntimeError):
    """A research tool failed in a controlled, understandable way."""


class UnsafeURLError(ToolError):
    """The URL (or a redirect target) was rejected by SSRF protection."""


class SearchResult(TypedDict):
    """One web search result with verifiable provenance."""

    title: str
    snippet: str
    url: str


class ToolResult(TypedDict):
    """Uniform tool outcome handed back to the LLM and the graph."""

    tool: str
    ok: bool
    data: Any
    error: str | None
    args: dict[str, Any]


#: Actionable guidance appended to TLS failures. It never suggests
#: disabling verification — the fetch is refused, full stop.
_TLS_FAILURE_HINT = (
    "TLS certificate verification failed: the server's certificate "
    "chain could not be trusted with the configured CA store. "
    "Verification is never disabled. On networks that inspect TLS "
    "with an enterprise CA, set SSL_CA_BUNDLE to that CA bundle (or "
    "rely on the OS trust store via truststore) and retry; "
    "otherwise treat this page as unretrievable and use other "
    "sources."
)


def build_ssl_context(ca_bundle: str | None = None) -> ssl.SSLContext | None:
    """TLS context for page fetches — verification is ALWAYS enabled.

    Trust-store priority (the live-run failure was a corporate
    TLS-inspection CA absent from certifi's bundled store):

    1. ``ca_bundle``: an explicit CA bundle path (enterprise CA) is
       used as the verification store;
    2. ``truststore`` (pinned dependency): the OS trust store —
       on Windows the certificate store where enterprise root CAs
       are installed — merged into a strictly-verifying context;
    3. ``None``: httpx's default (certifi), still verifying.

    There is no ``verify=False`` path and no insecure fallback.
    """
    if ca_bundle:
        return ssl.create_default_context(cafile=ca_bundle)
    if _TRUSTSTORE_AVAILABLE:
        # PROTOCOL_TLS_CLIENT defaults to CERT_REQUIRED + hostname
        # checking; truststore injects the platform trust roots.
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return None


def _describe_http_failure(url: str, exc: Exception) -> str:
    """ToolError text for a failed fetch, TLS-aware and honest."""
    base = f"fetch_page request failed for {url}: {type(exc).__name__}: {exc}"
    text = f"{type(exc).__name__}: {exc}".lower()
    if any(marker in text for marker in ("certificate", "ssl", "tls")):
        return f"{base}. {_TLS_FAILURE_HINT}"
    return base


def web_search(
    query: str,
    *,
    max_results: int = 5,
    timeout: float = 30.0,
    ddgs_factory: Callable[..., Any] | None = None,
) -> list[SearchResult]:
    """Search the web via DuckDuckGo (free, no API key).

    Uses the maintained ``ddgs`` package. An empty result set is
    a FAILURE to gather evidence, not evidence itself: it raises
    ``ToolError`` so the caller (LLM) gets a meaningful message
    and can retry with a different query, instead of silently
    receiving a "successful" response with no data.
    """
    if not query.strip():
        raise ToolError("web_search: query must not be empty")
    factory = ddgs_factory or DDGS
    try:
        with factory(timeout=timeout) as ddgs:
            raw = ddgs.text(query, max_results=max_results)
            if not raw:
                broader = _without_site_filter(query)
                if broader != query:
                    # Site-restricted queries fail when the
                    # index lacks that site: fall back once to
                    # the unrestricted query instead of
                    # returning nothing.
                    logger.info(
                        "web_search: no results for site-restricted "
                        "query %r; retrying as %r",
                        query,
                        broader,
                    )
                    raw = ddgs.text(broader, max_results=max_results)
    except ToolError:
        raise
    except Exception as exc:
        # Provider errors and timeouts surface with their type
        # and message so the LLM can react meaningfully.
        raise ToolError(
            f"web_search failed for query {query!r}: {type(exc).__name__}: {exc}"
        ) from exc

    results: list[SearchResult] = []
    for item in raw or []:
        url = str(item.get("href") or item.get("url") or "").strip()
        if not url:
            # A result without a URL has no verifiable provenance.
            continue
        results.append(
            {
                "title": str(item.get("title") or "").strip(),
                "snippet": str(item.get("body") or item.get("snippet") or "").strip(),
                "url": url,
            }
        )
    if not results:
        raise ToolError(
            f"web_search returned no results for query {query!r}. "
            "Broaden the query: remove site: filters and overly "
            "specific terms, or try different keywords."
        )
    return results


def fetch_page(
    url: str,
    *,
    client: httpx.Client | None = None,
    timeout: float = 30.0,
    max_bytes: int = 1_000_000,
    max_redirects: int = 5,
    ca_bundle: str | None = None,
) -> str:
    """Retrieve a public webpage's text, enforcing SSRF and size limits.

    ``client`` is injectable for tests; when omitted, a client is
    created with the requested timeout, no automatic redirects
    (every hop is validated manually) and a TLS context from
    :func:`build_ssl_context` — certificate verification is always
    on; ``ca_bundle`` selects an explicit CA bundle when given.
    """
    own_client = client is None
    if own_client:
        kwargs: dict[str, Any] = {
            "timeout": httpx.Timeout(timeout),
            "follow_redirects": False,
        }
        ssl_context = build_ssl_context(ca_bundle)
        if ssl_context is not None:
            kwargs["verify"] = ssl_context
        # When no context is built, httpx's default (certifi) applies —
        # verification stays enabled either way; never verify=False.
        client = httpx.Client(**kwargs)
    try:
        try:
            return _fetch_with_client(
                client, url, max_bytes=max_bytes, max_redirects=max_redirects
            )
        except httpx.HTTPError as exc:
            raise ToolError(_describe_http_failure(url, exc)) from exc
    finally:
        if own_client:
            client.close()


def _fetch_with_client(
    client: httpx.Client,
    url: str,
    *,
    max_bytes: int,
    max_redirects: int,
) -> str:
    current_url = url
    for _hop in range(max_redirects + 1):
        _validate_url(current_url)
        request = client.build_request("GET", current_url)
        response = client.send(request, stream=True)
        try:
            if response.is_redirect:
                location = response.headers.get("location")
                if not location:
                    raise ToolError(
                        f"redirect without a Location header from {current_url}"
                    )
                current_url = urljoin(current_url, location)
                continue
            raw, truncated = _read_capped(response, max_bytes)
        finally:
            response.close()
        text = _html_to_text(raw)
        if truncated:
            text = f"{text} [content truncated]"
        return text
    raise ToolError(f"exceeded {max_redirects} redirects starting from {url}")


def _read_capped(response: httpx.Response, max_bytes: int) -> tuple[str, bool]:
    """Read at most ``max_bytes`` of the response body."""
    chunks: list[bytes] = []
    total = 0
    truncated = False
    for chunk in response.iter_bytes():
        total += len(chunk)
        if total > max_bytes:
            truncated = True
            break
        chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", errors="replace"), truncated


def _validate_url(url: str) -> None:
    """Reject unsafe schemes, localhost and non-public destinations."""
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise UnsafeURLError(
            f"blocked scheme {scheme!r} (only http/https allowed): {url}"
        )
    hostname = parsed.hostname
    if not hostname:
        raise UnsafeURLError(f"URL has no hostname: {url}")
    host = hostname.lower().strip("[]")
    if host == "localhost":
        raise UnsafeURLError(f"localhost access is blocked: {url}")

    try:
        literal_ip = ipaddress.ip_address(host)
    except ValueError:
        literal_ip = None

    if literal_ip is not None:
        if not literal_ip.is_global:
            raise UnsafeURLError(f"non-public IP address is blocked: {url}")
        return  # literal public IP — nothing to resolve

    # Hostname: resolve and reject if ANY address is non-public.
    for family, _, _, _, sockaddr in socket.getaddrinfo(host, None):
        address = sockaddr[0]
        if not ipaddress.ip_address(address).is_global:
            raise UnsafeURLError(
                f"hostname {host!r} resolves to non-public address "
                f"{address}: {url}"
            )


_SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_SITE_FILTER_RE = re.compile(r"\bsite:\S+", re.IGNORECASE)


def _without_site_filter(query: str) -> str:
    """Remove ``site:`` operators (empty-result fallback)."""
    return " ".join(_SITE_FILTER_RE.sub("", query).split())


def _html_to_text(markup: str) -> str:
    """Best-effort, dependency-free HTML-to-text extraction."""
    text = _SCRIPT_STYLE_RE.sub(" ", markup)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


class ToolExecutor:
    """Explicit, validated tool execution — the graph's tool layer.

    The LLM proposes tool calls; this class dispatches them safely.
    Unknown tools, non-object arguments and tool failures all become
    controlled error results the LLM can react to — never exceptions
    that crash the graph.
    """

    def __init__(self, functions: dict[str, Callable[..., Any]]):
        self._functions = functions

    def execute(self, tool_call: dict[str, Any]) -> ToolResult:
        name = tool_call.get("name")
        args = tool_call.get("args")
        if not isinstance(name, str) or name not in self._functions:
            return ToolResult(
                tool=str(name), ok=False, data=None,
                error=f"unknown tool: {name!r}", args={},
            )
        if not isinstance(args, dict):
            return ToolResult(
                tool=name, ok=False, data=None,
                error="tool arguments must be an object", args={},
            )
        try:
            data = self._functions[name](**args)
        except TypeError as exc:
            return ToolResult(
                tool=name, ok=False, data=None,
                error=f"invalid arguments for {name}: {exc}", args=dict(args),
            )
        except Exception as exc:
            return ToolResult(
                tool=name, ok=False, data=None,
                error=f"tool {name} failed: {exc}", args=dict(args),
            )
        return ToolResult(tool=name, ok=True, data=data, error=None, args=dict(args))


class ResearchTools:
    """The agent's toolset, bound to application settings."""

    def __init__(self, settings: Settings):
        self._settings = settings

    def web_search(self, query: str) -> list[SearchResult]:
        return web_search(
            query,
            max_results=self._settings.search_results_per_query,
            timeout=self._settings.request_timeout_seconds,
        )

    def fetch_page(self, url: str) -> str:
        return fetch_page(
            url,
            timeout=self._settings.request_timeout_seconds,
            max_bytes=self._settings.max_page_bytes,
            max_redirects=self._settings.max_redirects,
            ca_bundle=self._settings.ssl_ca_bundle,
        )

    def tool_functions(self) -> dict[str, Callable[..., Any]]:
        """Executable tool implementations for the explicit executor."""
        return {"web_search": self.web_search, "fetch_page": self.fetch_page}

    def langchain_tools(self) -> list[BaseTool]:
        """LangChain tool wrappers (JSON schemas) exposed to the LLM."""

        def search(query: str) -> list[dict[str, str]]:
            """Search the web for a query. Returns results with titles, snippets and source URLs."""
            return list(self.web_search(query))

        def fetch(url: str) -> str:
            """Fetch a public HTTP(S) webpage and return its text content."""
            return self.fetch_page(url)

        return [
            StructuredTool.from_function(
                name="web_search", description=search.__doc__, func=search
            ),
            StructuredTool.from_function(
                name="fetch_page", description=fetch.__doc__, func=fetch
            ),
        ]


def to_findings(result: ToolResult) -> list[Finding]:
    """Convert a successful tool result into verifiable findings.

    Findings are built ONLY from actual tool output — the agent can
    never fabricate citations.
    """
    if not result["ok"]:
        return []
    if result["tool"] == "web_search":
        return [
            Finding(
                text=f"{item['title']} — {item['snippet']}".strip(" —"),
                source_url=item["url"],
                evidence="snippet",  # discovered, not inspected
            )
            for item in result["data"]
        ]
    if result["tool"] == "fetch_page":
        url = str(result["args"].get("url", ""))
        excerpt = str(result["data"])[:500]
        return [
            Finding(
                text=excerpt,
                source_url=url,
                evidence="retrieved",  # full page inspected
            )
        ]
    return []
