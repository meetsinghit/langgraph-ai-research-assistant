"""Research tools: web search and webpage retrieval.

Security rules enforced here:

- HTTP/HTTPS schemes only; every other scheme is rejected
- localhost, private, link-local and reserved IP ranges are blocked —
  for literal IPs AND for hostnames after DNS resolution, which
  blocks DNS-based SSRF bypasses
- redirects are followed manually, and EVERY hop is re-validated,
  which blocks unsafe redirects
- URLs with embedded credentials (userinfo) and non-web ports
  (only 80/443) are refused
- non-HTML/text content types are refused (missing header tolerated)
- cleartext-HTTP requests are PINNED to the address that just passed
  validation (Host header preserved), closing the DNS
  time-of-check/time-of-use window for http:// — https cannot be
  pinned without breaking SNI/verification, so its residual window
  is documented instead (docs/SECURITY.md)
- per-request timeout, response-size cap and redirect cap

Note: DNS is resolved for validation; for cleartext HTTP the fetch
then uses exactly that validated address, while HTTPS re-resolves
inside the client under full TLS verification (a small residual
rebind window exists there — never traded for verify=False).

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
import time
from typing import Any, Callable, TypedDict
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from ddgs import DDGS
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.completeness import _requested_terms
from agent.config import Settings
from agent.deadline import RunDeadline
from agent.metrics import categorize_error
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


class PolicyRefusal(ToolError):
    """A deterministic security policy refused the tool call.

    Carries an explicit category (preserved by the executor into
    metrics) so policy refusals are distinguishable from operational
    failures — currently raised when ``fetch_page`` is asked for a
    URL that ``web_search`` did not discover in this run.
    """

    category = "policy"


#: Search-failure taxonomy (fixed vocabulary — never error text).
#: ddgs raises a GENERIC ``DDGSException`` both for "the index had
#: nothing" and for engine-side HTTP failures, so class alone cannot
#: separate them; see :func:`classify_search_failure`.
SEARCH_ERROR_CATEGORIES = frozenset(
    {
        "empty_results",        # valid request, index returned nothing
        "rate_limited",         # 429 / backend rate limit
        "timeout",              # backend or network timeout
        "upstream_unavailable", # 5xx / connection faults (transient)
        "invalid_query",        # backend rejected the query (permanent)
        "unknown",              # anything else — not retried blindly
        "deadline",             # aborted: the run's global deadline hit
    }
)


class SearchError(ToolError):
    """A search failure carrying its taxonomy ``category``.

    The executor records ``category`` (a fixed vocabulary) as the
    metrics error category, so "no results" is never conflated with
    "backend down" in operational reporting.
    """

    def __init__(self, message: str, *, category: str) -> None:
        super().__init__(message)
        if category not in SEARCH_ERROR_CATEGORIES:
            raise ValueError(f"unknown search category: {category!r}")
        self.category = category


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


# -- tool-call argument schemas (WS-A) ---------------------------------
#
# Live bug this fixes: Gemini called ``web_search(queries=...)`` while
# the Python callable expected ``query``. The previous schema was
# INFERRED from a bare closure signature with a vague description, so
# the model had no explicit contract to follow. These Pydantic models
# are advertised verbatim in the JSON schema the provider sees
# (required fields, types, per-field descriptions) AND enforced before
# execution, so a malformed call becomes one structured, actionable
# tool error the model can correct — never a TypeError crash or a
# silent reinterpretation.


class WebSearchArgs(BaseModel):
    """Explicit argument contract for ``web_search``."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=1,
        max_length=400,
        description=(
            "Search terms as ONE string, e.g. "
            "'LangGraph checkpointers'. The argument name must be "
            "'query' (not 'queries')."
        ),
    )


class FetchPageArgs(BaseModel):
    """Explicit argument contract for ``fetch_page``."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(
        min_length=1,
        max_length=2048,
        description=(
            "Full http(s) URL to fetch. The argument name must be "
            "'url'."
        ),
    )


TOOL_ARG_SCHEMAS: dict[str, type[BaseModel]] = {
    "web_search": WebSearchArgs,
    "fetch_page": FetchPageArgs,
}

#: Deterministic argument-NAME aliases: rename-only, allow-listed,
#: applied only when the canonical field is absent. Decision (WS-A):
#: normalizing ``queries`` -> ``query`` is justified because it is
#: unambiguous (exactly one canonical target, one provided key), it
#: reproduces an observed live mis-call, and the VALUE is passed
#: through untouched — no type coercion, no invented arguments, no
#: retry loop. Anything outside this table is rejected as a schema
#: error so genuinely malformed calls stay visible instead of being
#: silently "repaired".
_ARG_ALIASES: dict[str, dict[str, str]] = {
    "web_search": {"queries": "query", "search_query": "query"},
    "fetch_page": {"page_url": "url", "target_url": "url"},
}


def _normalize_arg_names(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """Apply the allow-listed rename-only aliases for ``tool``."""
    aliases = _ARG_ALIASES.get(tool)
    if not aliases:
        return args
    normalized = dict(args)
    for alias, canonical in aliases.items():
        if alias in normalized and canonical not in normalized:
            logger.info(
                "tool %s: normalized argument name %r -> %r "
                "(value passed through unchanged)",
                tool,
                alias,
                canonical,
            )
            normalized[canonical] = normalized.pop(alias)
    return normalized


def _schema_error_text(
    tool: str,
    schema: type[BaseModel],
    exc: ValidationError,
    provided: dict[str, Any],
) -> str:
    """One structured, actionable error for a malformed tool call.

    Names and types only — values are summarized by the schema, never
    echoed back at length.
    """
    details = "; ".join(
        ".".join(str(part) for part in error.get("loc", ()))
        + f": {error.get('msg', 'invalid')}"
        for error in exc.errors()[:3]
    )
    properties = schema.model_json_schema().get("properties", {})
    expected = (
        "{"
        + ", ".join(f"{name}: {spec.get('type', 'any')}" for name, spec in properties.items())
        + "}"
    )
    names = sorted(str(key) for key in provided)[:5]
    return (
        f"invalid arguments for {tool}: {details}. "
        f"Expected arguments {expected}; provided argument names {names}. "
        f"Call {tool} again with the exact argument names."
    )


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


#: Hard cap on backend queries per web_search tool call: the ladder
#: is original -> site-filter removed -> terms simplified (<=3
#: variants) and a transient failure may repeat the current variant
#: exactly once. One tool call therefore costs at most 3 network
#: requests, keeping recovery inside the existing tool-call budget.
_MAX_SEARCH_ATTEMPTS = 3

#: Politeness pause before repeating a rate-limited request. Kept
#: small on purpose; always clamped to the run's remaining global
#: deadline budget (``RunDeadline.pause_budget``).
_RATE_LIMIT_PAUSE_SECONDS = 1.5


def classify_search_failure(exc: BaseException | str) -> str:
    """Map a backend failure to the search-error taxonomy.

    ddgs raises a generic ``DDGSException`` for three very different
    situations — no results (message ``"No results found."``),
    engine-side HTTP errors (message carries the status) and wrapped
    transport faults — so exception class alone is insufficient;
    well-known message markers are inspected (the same way ddgs
    itself special-cases ``TimeoutException``). Returns one of
    :data:`SEARCH_ERROR_CATEGORIES` — never the error text.
    """
    if isinstance(exc, SearchError):
        return exc.category
    name = type(exc).__name__.lower()
    text = f"{type(exc).__name__}: {exc}".lower()
    if "ratelimit" in name or "429" in text or "rate limit" in text or "too many requests" in text:
        return "rate_limited"
    if "timeout" in name or "timed out" in text or "timeout" in text:
        return "timeout"
    if "no results" in text:
        return "empty_results"
    if any(
        marker in text
        for marker in (
            " 500",
            " 502",
            " 503",
            " 504",
            "http 5",
            "bad gateway",
            "service unavailable",
            "unavailable",
            "connection reset",
            "connection aborted",
            "temporarily",
        )
    ):
        return "upstream_unavailable"
    if any(
        marker in text
        for marker in ("400", "bad request", "invalid", "unsupported")
    ):
        return "invalid_query"
    return "unknown"


def _search_error_message(
    category: str, query: str, detail: str
) -> str:
    """Actionable, model-facing error text per category.

    Every message tells the LLM what to do next; none of them is
    shown to metrics (only ``category`` is).
    """
    if category == "empty_results":
        return (
            f"web_search returned no results for query {query!r}. "
            "Broaden the query: remove site: filters and overly "
            "specific terms, or try different keywords."
        )
    if category == "timeout":
        return (
            f"web_search failed for query {query!r}: timed out "
            f"({detail}). The search backend timed out — retry once "
            "or simplify the query."
        )
    if category == "rate_limited":
        return (
            f"web_search failed for query {query!r}: rate limited by "
            f"the search backend ({detail}). Wait briefly and retry "
            "with fewer queries."
        )
    if category == "upstream_unavailable":
        return (
            f"web_search failed for query {query!r}: search backend "
            f"temporarily unavailable ({detail}). Retry once shortly "
            "or rephrase the query."
        )
    if category == "invalid_query":
        return (
            f"web_search failed for query {query!r}: the query was "
            f"rejected as invalid ({detail}). Simplify it — drop "
            "quotes and boolean operators."
        )
    if category == "deadline":
        return (
            "web_search skipped: the run's global execution deadline "
            "has expired."
        )
    return f"web_search failed for query {query!r}: {detail}"


def _search_variants(query: str) -> list[str]:
    """Bounded query-transformation ladder for empty/invalid results.

    1. the original query;
    2. with ``site:`` filters removed (an index lacking that site
       returns nothing — this was the original Phase 1 fallback);
    3. with quoting/boolean operators stripped (free-text indexes
       commonly reject them).

    The tool NEVER invents new terms: alternative queries are the
    planner/model's job, keeping recovery deterministic.
    """
    variants = [query]
    broader = _without_site_filter(query)
    if broader != query:
        variants.append(broader)
    simplified = _simplify_query(variants[-1])
    if simplified and simplified != variants[-1]:
        variants.append(simplified)
    return variants[:_MAX_SEARCH_ATTEMPTS]


def _simplify_query(query: str) -> str:
    """Drop quoting and boolean/paren operators; keep every term."""
    text = query.replace('"', " ").replace("'", " ")
    text = re.sub(r"[(){}\[\]]", " ", text)
    text = re.sub(
        r"\b(AND|OR|NOT|NEAR)\b", " ", text, flags=re.IGNORECASE
    )
    return " ".join(text.split())


def _map_search_results(raw: list[dict[str, Any]] | None) -> list[SearchResult]:
    """Normalize backend items; results without a URL are dropped
    (no verifiable provenance = not usable evidence)."""
    results: list[SearchResult] = []
    for item in raw or []:
        url = str(item.get("href") or item.get("url") or "").strip()
        if not url:
            continue
        results.append(
            {
                "title": str(item.get("title") or "").strip(),
                "snippet": str(item.get("body") or item.get("snippet") or "").strip(),
                "url": url,
            }
        )
    return results


def web_search(
    query: str,
    *,
    max_results: int = 5,
    timeout: float = 30.0,
    ddgs_factory: Callable[..., Any] | None = None,
    deadline: RunDeadline | None = None,
    transient_pause: float = _RATE_LIMIT_PAUSE_SECONDS,
) -> list[SearchResult]:
    """Search the web via DuckDuckGo (free, no API key), with a
    bounded, category-aware recovery ladder.

    An empty result set is a FAILURE to gather evidence, not
    evidence: the call raises :class:`SearchError` with a taxonomy
    category so the caller (LLM) gets an actionable message and the
    metrics collector records WHICH kind of failure occurred.

    Recovery is bounded and category-driven:

    - ``empty_results`` / ``invalid_query``  -> advance the query
      transformation ladder (site-filter removal, operator stripping),
      at most ``_MAX_SEARCH_ATTEMPTS`` backend calls total;
    - ``timeout`` / ``rate_limited`` / ``upstream_unavailable``
      -> repeat the SAME query exactly once (rate limits pause
      ``transient_pause`` seconds first, clamped to the run's global
      deadline budget);
    - ``unknown`` -> no blind retry; surface the failure.

    ``ddgs_factory`` is the backend seam (see
    :class:`SearchBackendSession`): alternative providers plug in
    behind the same factory contract without touching recovery logic.
    """
    if deadline is not None and deadline.expired():
        raise SearchError(
            "web_search skipped: the run's global execution deadline "
            "has expired",
            category="deadline",
        )
    if not query.strip():
        raise SearchError(
            "web_search: query must not be empty", category="invalid_query"
        )
    factory = ddgs_factory or DDGS
    variants = _search_variants(query)

    variant_index = 0
    attempts = 0
    transient_retry_used = False
    last_category = "unknown"
    last_detail = ""

    while (
        attempts < _MAX_SEARCH_ATTEMPTS
        and variant_index < len(variants)
    ):
        if attempts and deadline is not None and deadline.expired():
            break  # no new backend request after the deadline
        attempts += 1
        variant = variants[variant_index]
        try:
            with factory(timeout=timeout) as ddgs:
                raw = ddgs.text(variant, max_results=max_results)
            results = _map_search_results(raw)
        except Exception as exc:  # noqa: BLE001 - classified below
            last_category = classify_search_failure(exc)
            last_detail = f"{type(exc).__name__}: {exc}"
            results = []
        else:
            if results:
                if variant_index:
                    logger.info(
                        "web_search: recovered on attempt %d with a "
                        "query variant (%d results)",
                        attempts,
                        len(results),
                    )
                return results
            last_category = "empty_results"
            last_detail = "the search index returned no results"

        if last_category in ("empty_results", "invalid_query"):
            variant_index += 1  # transform the query, bounded ladder
            continue
        if (
            last_category
            in ("timeout", "rate_limited", "upstream_unavailable")
            and not transient_retry_used
        ):
            transient_retry_used = True
            pause = (
                transient_pause if last_category == "rate_limited" else 0.0
            )
            if deadline is not None:
                pause = deadline.pause_budget(pause)
                if deadline.expired():
                    break
            if pause > 0:
                time.sleep(pause)
            continue  # repeat the current variant exactly once
        break  # unknown category, or the transient retry is spent

    raise SearchError(
        _search_error_message(last_category, query, last_detail),
        category=last_category,
    )


def fetch_page(
    url: str,
    *,
    client: httpx.Client | None = None,
    timeout: float = 30.0,
    max_bytes: int = 1_000_000,
    max_redirects: int = 5,
    ca_bundle: str | None = None,
    deadline: RunDeadline | None = None,
) -> str:
    """Retrieve a public webpage's text, enforcing SSRF and size limits.

    ``client`` is injectable for tests; when omitted, a client is
    created with the requested timeout, no automatic redirects
    (every hop is validated manually) and a TLS context from
    :func:`build_ssl_context` — certificate verification is always
    on; ``ca_bundle`` selects an explicit CA bundle when given.

    ``deadline`` (the run's global execution budget) is checked
    before every redirect hop: no new request starts after the
    deadline expires. The caller is expected to cap ``timeout`` by
    the remaining budget (``ResearchTools.with_budget``), so both
    in-flight time and hop count stay bounded by the global budget.
    """
    if deadline is not None and deadline.expired():
        raise ToolError(
            f"fetch_page skipped for {url}: the run's global "
            "execution deadline has expired"
        )
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
                client,
                url,
                max_bytes=max_bytes,
                max_redirects=max_redirects,
                deadline=deadline,
                # Pinning rewrites the request target for OWN clients
                # only: an injected client is a test transport whose
                # fixtures match original URLs.
                pin_http=own_client,
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
    deadline: RunDeadline | None = None,
    pin_http: bool = False,
) -> str:
    current_url = url
    for _hop in range(max_redirects + 1):
        if deadline is not None and deadline.expired():
            # No new request after expiry (the in-flight hop, if any,
            # was itself bounded by the capped timeout).
            raise ToolError(
                f"fetch_page aborted for {url}: the run's global "
                "execution deadline has expired"
            )
        addresses = _validate_url(current_url)
        request_url, request_headers = _connect_target(
            current_url, addresses, pin=pin_http
        )
        request = client.build_request(
            "GET", request_url, headers=request_headers or None
        )
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
            if not _allowed_content_type(
                response.headers.get("content-type")
            ):
                main_type = (
                    (response.headers.get("content-type") or "")
                    .split(";", 1)[0]
                    .strip()
                    .lower()
                )
                raise ToolError(
                    f"fetch refused for {current_url}: unsupported "
                    f"content type {main_type!r} (only HTML or "
                    "plain-text documents are processed)"
                )
            raw, truncated = _read_capped(response, max_bytes)
        finally:
            response.close()
        text = _html_to_text(raw)
        if truncated:
            text = f"{text} [content truncated]"
        return text
    raise ToolError(f"exceeded {max_redirects} redirects starting from {url}")


#: Content types the fetcher will process. HTML and plain text are
#: research material; anything else (PDF, images, archives, octet
#: streams) is refused with a controlled error instead of being
#: decoded to mojibake. A MISSING header is tolerated (legacy
#: servers and test transports) — the body is then parsed as text.
_ALLOWED_CONTENT_TYPES = frozenset({"application/xhtml+xml"})


def _allowed_content_type(header: str | None) -> bool:
    if not header:
        return True  # absent header: tolerate and parse as text
    main_type = header.split(";", 1)[0].strip().lower()
    return main_type.startswith("text/") or main_type in _ALLOWED_CONTENT_TYPES


def _connect_target(
    url: str, addresses: list[str], *, pin: bool
) -> tuple[str, dict[str, str]]:
    """Cleartext-HTTP destination pinning (DNS time-of-check/time-of-use
    closure).

    For ``http://`` URLs whose host is a NAME (not a literal IP), the
    request is sent to the address that just passed validation, with
    the original host in the ``Host`` header — a DNS answer changing
    between validation and connection can no longer steer the fetch
    to a forbidden internal address.

    ``https://`` cannot be pinned this way: TLS requires SNI and
    certificate hostname verification for the ORIGINAL name, and
    connecting to a raw IP would either break verification or force
    ``verify=False`` (never acceptable). For https the validation
    happens one line above the request, inside the same hop, keeping
    the residual rebind window minimal; it is documented as a
    limitation in docs/SECURITY.md with egress filtering as the
    production-grade control.
    """
    if not pin:
        return url, {}
    parsed = urlparse(url)
    if parsed.scheme.lower() != "http":
        return url, {}
    host = (parsed.hostname or "").lower()
    if not host:
        return url, {}
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return url, {}  # literal IP: validation already proved it public
    target = addresses[0]
    authority = f"[{target}]" if ":" in target else target
    if parsed.port is not None:
        authority = f"{authority}:{parsed.port}"
    pinned = urlunparse(parsed._replace(netloc=authority))
    # ``parsed.netloc`` cannot contain userinfo here — _validate_url
    # refused it earlier in this same hop.
    return pinned, {"Host": parsed.netloc}


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


#: Only standard web ports are fetchable. This is not the primary
#: SSRF control (non-public ADDRESSES are), but it stops a public
#: host from steering the fetcher at unusual services (internal
#: admin panels commonly exposed on 8080/8443, SMTP on 25, etc.).
_ALLOWED_PORTS = frozenset({80, 443})


def _validate_url(url: str) -> list[str]:
    """Reject unsafe schemes, credentials, ports, localhost and
    non-public destinations.

    Returns the VALIDATED global addresses the host resolves to
    (the literal address itself for IP URLs) so callers can pin the
    actual connection target to what was just validated — closing
    the DNS time-of-check/time-of-use window for cleartext HTTP
    (see :func:`_fetch_with_client`).
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise UnsafeURLError(
            f"blocked scheme {scheme!r} (only http/https allowed): {url}"
        )
    if parsed.username is not None or parsed.password is not None:
        # The URL is deliberately NOT echoed back: userinfo can carry
        # attacker- or user-supplied secrets we must not reproduce in
        # tool messages, logs or metrics.
        raise UnsafeURLError(
            "URLs with embedded credentials (userinfo) are blocked"
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise UnsafeURLError(f"malformed port in URL: {url}") from exc
    if port is not None and port not in _ALLOWED_PORTS:
        raise UnsafeURLError(
            f"port {port} is blocked (only 80/443 allowed): {url}"
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
        return [str(literal_ip)]  # literal public IP — nothing to resolve

    # Hostname: resolve and reject if ANY address is non-public.
    addresses: list[str] = []
    for family, _, _, _, sockaddr in socket.getaddrinfo(host, None):
        address = sockaddr[0]
        if not ipaddress.ip_address(address).is_global:
            raise UnsafeURLError(
                f"hostname {host!r} resolves to non-public address "
                f"{address}: {url}"
            )
        if address not in addresses:
            addresses.append(address)
    if not addresses:
        raise UnsafeURLError(f"hostname {host!r} did not resolve: {url}")
    return addresses


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


# -- model-visible evidence shaping ------------------------------------

#: Characters reserved for the truncation marker so
#: ``shaped + marker <= max_chars`` always holds.
_EVIDENCE_MARKER_RESERVE = 240

#: Sentence segmentation for relevance selection. ``_html_to_text``
#: collapses ALL whitespace, so fetched pages are one long line —
#: blank-line "paragraphs" do not exist and sentences are the
#: natural lightweight unit.
_EVIDENCE_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

#: Fixed-size fallback segments for terminator-free content
#: (minified text, base64 blobs, single-word dumps).
_EVIDENCE_CHUNK_CHARS = 400


def evidence_marker(shown: int, total: int) -> str:
    """Explicit, content-free truncation marker."""
    return (
        f"…[evidence bounded: {shown} of {total} characters kept "
        "— query-relevant sections selected; the full page was "
        "not retained]"
    )


def shape_evidence(text: str, query: str, max_chars: int) -> str:
    """Bound webpage text to ``max_chars``, keeping QUERY-RELEVANT
    sections instead of a naive first-N-character cut.

    Lightweight, deterministic and dependency-free:

    - text already within budget passes through unchanged;
    - the text is segmented into sentences (fixed-size chunks
      when a document has no sentence terminators);
    - the first segment is always kept (title/intro context),
      then segments containing any of the question's topic terms
      (``_requested_terms`` — same terms the coverage gate uses)
      are added in original order until the budget is spent;
    - boilerplate with no topic term (navigation, cookie notices)
      is dropped;
    - the result always carries an explicit marker stating the
      original size.

    The returned text is EVIDENCE for the model to read, not a
    correctness claim; source URL and retrieved/snippet provenance
    are attached by callers and never affected here.
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    budget = max(1, max_chars - _EVIDENCE_MARKER_RESERVE)
    segments = [
        segment.strip()
        for segment in _EVIDENCE_SENTENCE_RE.split(text)
        if segment.strip()
    ]
    if not segments:
        segments = [
            text[index : index + _EVIDENCE_CHUNK_CHARS]
            for index in range(0, len(text), _EVIDENCE_CHUNK_CHARS)
        ]

    def _bounded(head: str) -> str:
        if len(head) > budget:
            head = head[:budget]
        return head + "\n\n" + evidence_marker(len(head), len(text))

    lead = segments[0]
    if len(lead) > budget:
        # oversized lead (terminator-free document): bounded fallback
        return _bounded(lead)
    kept: list[str] = [lead]
    used = len(lead)
    terms = _requested_terms(query)
    if terms:
        lowered = [segment.lower() for segment in segments]
        for index in range(1, len(segments)):
            if not any(term in lowered[index] for term in terms):
                continue  # boilerplate: no topic term
            segment = segments[index]
            if used + len(segment) + 1 > budget:
                continue  # too large — a later, smaller match may fit
            kept.append(segment)
            used += len(segment) + 1
    return _bounded(" ".join(kept))


class ToolExecutor:
    """Explicit, validated tool execution — the graph's tool layer.

    The LLM proposes tool calls; this class dispatches them safely.
    Unknown tools, non-object arguments, SCHEMA violations (missing
    required fields, wrong types, unexpected keywords), tool failures
    and search-taxonomy failures all become controlled error results
    the LLM can react to — never exceptions that crash the graph.

    Argument handling (WS-A):

    1. allow-listed rename-only aliases are applied (``queries`` ->
       ``query``) — values are never reinterpreted;
    2. the tool's Pydantic schema is validated BEFORE execution, so
       malformed calls fail fast with an actionable message;
    3. ``SearchError.category`` (search taxonomy) is preserved as the
       metrics error category; other exceptions fall back to
       :func:`categorize_error` — schema errors land in
       ``validation``, keeping them distinct from network/provider/
       search failures.

    An optional :class:`~agent.metrics.MetricsCollector` records
    each execution's name/outcome/latency/result SIZE — never
    arguments or result contents.
    """

    def __init__(
        self,
        functions: dict[str, Callable[..., Any]],
        metrics: Any = None,
        node: str = "research",
        *,
        arg_schemas: dict[str, type[BaseModel]] | None = None,
        sensitive_values: tuple[str, ...] | list[str] = (),
    ):
        self._functions = functions
        self._metrics = metrics
        self._node = node
        # ``None`` -> the built-in schemas; a dict (possibly empty)
        # -> caller's explicit choice (tests inject unschemed tools).
        self._arg_schemas = (
            TOOL_ARG_SCHEMAS if arg_schemas is None else arg_schemas
        )
        # DLP (Phase 4C): known secret VALUES (API keys) that must
        # never appear in tool arguments. Trivially short values are
        # dropped — a 3-character "secret" would refuse every call.
        # The values themselves are never logged, echoed or recorded.
        self._sensitive_values = tuple(
            value
            for value in sensitive_values
            if isinstance(value, str) and len(value) >= 8
        )

    def execute(self, tool_call: dict[str, Any]) -> ToolResult:
        name = tool_call.get("name")
        args = tool_call.get("args")
        if not isinstance(name, str) or name not in self._functions:
            result = ToolResult(
                tool=str(name), ok=False, data=None,
                error=f"unknown tool: {name!r}", args={},
            )
            self._record(result, 0.0, error_category="validation")
            return result
        if not isinstance(args, dict):
            result = ToolResult(
                tool=name, ok=False, data=None,
                error="tool arguments must be an object", args={},
            )
            self._record(result, 0.0, error_category="validation")
            return result
        schema = self._arg_schemas.get(name)
        if schema is not None:
            normalized = _normalize_arg_names(name, args)
            try:
                validated = schema.model_validate(normalized)
            except ValidationError as exc:
                result = ToolResult(
                    tool=name, ok=False, data=None,
                    error=_schema_error_text(name, schema, exc, normalized),
                    args=dict(args),
                )
                self._record(result, 0.0, error_category="validation")
                return result
            args = validated.model_dump()
        if self._sensitive_values and self._args_contain_secret(args):
            # DLP refusal (Phase 4C): a tool call carrying a known
            # secret value (e.g. an API key in a query or URL) is
            # refused BEFORE execution — this is the outbound-channel
            # control that keeps an injected page from turning the
            # fetcher into an exfiltration vector. The refused args
            # are deliberately NOT echoed into the result, never
            # recorded in metrics and never passed to the function.
            result = ToolResult(
                tool=name, ok=False, data=None,
                error=(
                    "tool call refused: the arguments contain a "
                    "configured sensitive value (data-loss "
                    "prevention). Credentials and configuration "
                    "values are never needed in tool arguments — "
                    "drop them and call the tool again without them."
                ),
                args={},
            )
            self._record(result, 0.0, error_category="dlp")
            return result
        started = time.monotonic()
        try:
            data = self._functions[name](**args)
        except TypeError as exc:
            result = ToolResult(
                tool=name, ok=False, data=None,
                error=f"invalid arguments for {name}: {exc}",
                args=dict(args),
            )
            self._record(
                result, time.monotonic() - started,
                error_category="validation",
            )
            return result
        except Exception as exc:
            # ToolErrors already carry their own actionable message
            # (search taxonomy, SSRF, TLS) — pass it through verbatim
            # instead of double-prefixing it. Structured errors keep
            # their category: SearchError contributes its taxonomy
            # value, everything else is categorized coarsely.
            category = getattr(exc, "category", None) or categorize_error(exc)
            error = (
                str(exc)
                if isinstance(exc, ToolError)
                else f"tool {name} failed: {exc}"
            )
            result = ToolResult(
                tool=name, ok=False, data=None, error=error, args=dict(args)
            )
            self._record(
                result, time.monotonic() - started,
                error_category=category,
            )
            return result
        result = ToolResult(
            tool=name, ok=True, data=data, error=None, args=dict(args)
        )
        self._record(result, time.monotonic() - started)
        return result

    def _args_contain_secret(self, args: dict[str, Any]) -> bool:
        """True when the serialized arguments embed a known secret value.

        Exact, case-sensitive substring match against the configured
        secret VALUES (API keys). The serialization is local and
        never stored — only the boolean leaves this method.
        """
        try:
            serialized = json.dumps(args, default=str)
        except (TypeError, ValueError):
            serialized = str(args)
        return any(
            value in serialized for value in self._sensitive_values
        )

    def _record(
        self,
        result: ToolResult,
        latency_seconds: float,
        *,
        error_category: str | None = None,
    ) -> None:
        if self._metrics is None:
            return
        data = result.get("data")
        try:
            result_chars = (
                len(json.dumps(data, default=str))
                if data is not None
                else 0
            )
        except (TypeError, ValueError):
            result_chars = 0
        category = error_category or (
            categorize_error(result.get("error") or "")
            if not result.get("ok")
            else None
        )
        self._metrics.record_tool_call(
            self._node,
            str(result.get("tool")),
            ok=bool(result.get("ok")),
            latency_ms=latency_seconds * 1000.0,
            result_chars=result_chars,
            error_category=category,
        )


class ResearchTools:
    """The agent's toolset, bound to application settings."""

    def __init__(self, settings: Settings):
        self._settings = settings
        #: Run-level execution budget (None outside graph runs):
        #: gates tool-internal retries and redirect hops.
        self._deadline: RunDeadline | None = None

    def with_budget(
        self,
        *,
        timeout_seconds: float | None = None,
        deadline: RunDeadline | None = None,
    ) -> "ResearchTools":
        """Per-run copy whose request timeouts are capped by the
        remaining global budget and whose internal retries stop at
        the run deadline. Returns a NEW instance so a shared,
        graph-level toolset is never mutated across runs.
        """
        settings = self._settings
        if timeout_seconds is not None:
            # model_copy bypasses pydantic validation on purpose:
            # the capped value may be far below the configured
            # default and must stay usable (it is a runtime budget,
            # not user configuration).
            settings = settings.model_copy(
                update={"request_timeout_seconds": timeout_seconds}
            )
        clone = ResearchTools(settings)
        clone._deadline = deadline
        return clone

    def web_search(self, query: str) -> list[SearchResult]:
        return web_search(
            query,
            max_results=self._settings.search_results_per_query,
            timeout=self._settings.request_timeout_seconds,
            deadline=self._deadline,
        )

    def fetch_page(self, url: str) -> str:
        return fetch_page(
            url,
            timeout=self._settings.request_timeout_seconds,
            max_bytes=self._settings.max_page_bytes,
            max_redirects=self._settings.max_redirects,
            ca_bundle=self._settings.ssl_ca_bundle,
            deadline=self._deadline,
        )

    def tool_functions(self) -> dict[str, Callable[..., Any]]:
        """Executable tool implementations for the explicit executor."""
        return {"web_search": self.web_search, "fetch_page": self.fetch_page}

    def langchain_tools(self) -> list[BaseTool]:
        """LangChain tool wrappers exposed to the LLM.

        Each wrapper advertises the EXPLICIT Pydantic argument schema
        (``WebSearchArgs`` / ``FetchPageArgs``) instead of a schema
        inferred from a bare closure signature — the model sees exact
        argument names, required fields, types and descriptions
        (WS-A). The executor enforces the same schemas before the
        function runs, so provider and code can never disagree about
        the contract.
        """

        def search(query: str) -> list[dict[str, str]]:
            """Search the web. REQUIRED ARGUMENTS: {"query": "<search terms as one string>"}. Returns results with titles, snippets and source URLs."""
            return list(self.web_search(query))

        def fetch(url: str) -> str:
            """Fetch a public HTTP(S) webpage and return its text content. REQUIRED ARGUMENTS: {"url": "<full http(s) URL>"}."""
            return self.fetch_page(url)

        return [
            StructuredTool.from_function(
                name="web_search",
                description=search.__doc__,
                func=search,
                args_schema=WebSearchArgs,
            ),
            StructuredTool.from_function(
                name="fetch_page",
                description=fetch.__doc__,
                func=fetch,
                args_schema=FetchPageArgs,
            ),
        ]


def to_findings(
    result: ToolResult,
    *,
    query: str = "",
    max_chars: int = 500,
) -> list[Finding]:
    """Convert a successful tool result into verifiable findings.

    Findings are built ONLY from actual tool output — the agent can
    never fabricate citations.

    ``query`` and ``max_chars`` control how much of a fetched page
    is retained as finding text: an oversized page is reduced to
    query-relevant sections (:func:`shape_evidence`), never a naive
    first-N cut. Only this bounded excerpt is kept in state — the
    full raw document is deliberately not retained.
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
        excerpt = shape_evidence(str(result["data"]), query, max_chars)
        return [
            Finding(
                text=excerpt,
                source_url=url,
                evidence="retrieved",  # full page inspected
            )
        ]
    return []
