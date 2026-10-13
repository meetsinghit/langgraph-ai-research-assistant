"""Adversarial network-layer tests for fetch_page (Phase 4C, WS-D).

All tests run against mocked transports and controlled DNS fixtures —
no live requests, no probing of real or internal infrastructure.

Covers: userinfo URLs, non-web ports, unsupported content types,
DNS time-of-check/time-of-use pinning for cleartext HTTP, HTTPS
SNI/verification preservation, redirects to forbidden addresses,
and the never-verify=False invariant.
"""
from __future__ import annotations

import socket
from typing import Any

import httpx
import pytest

import agent.tools as tools
from agent.tools import UnsafeURLError, ToolError, fetch_page


def _public_dns(
    monkeypatch: pytest.MonkeyPatch, address: str = "93.184.216.34"
) -> None:
    def fake_getaddrinfo(host: str, *args: Any, **kwargs: Any):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))
        ]

    monkeypatch.setattr(tools.socket, "getaddrinfo", fake_getaddrinfo)


def _client(handler: Any) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(handler), follow_redirects=False
    )


# --- userinfo & malformed URLs ------------------------------------------


def test_userinfo_url_blocked():
    with pytest.raises(UnsafeURLError, match="embedded credentials"):
        tools._validate_url("https://user:pass@example.com/page")


def test_userinfo_never_echoed_back():
    """The refusal message must not reproduce the credentials."""
    with pytest.raises(UnsafeURLError) as excinfo:
        tools._validate_url("https://leaked:SECRET123@example.com/")
    assert "SECRET123" not in str(excinfo.value)


def test_userinfo_with_no_password_still_blocked():
    with pytest.raises(UnsafeURLError, match="embedded credentials"):
        tools._validate_url("https://admin@example.com/page")


def test_malformed_port_blocked():
    with pytest.raises(UnsafeURLError, match="malformed port"):
        tools._validate_url("http://example.com:notaport/x")


def test_unexpected_port_blocked(monkeypatch):
    _public_dns(monkeypatch)
    with pytest.raises(UnsafeURLError, match="port 8080 is blocked"):
        tools._validate_url("http://example.com:8080/admin")


@pytest.mark.parametrize("port", [80, 443])
def test_standard_ports_allowed(monkeypatch, port: int):
    _public_dns(monkeypatch)
    tools._validate_url(f"http://example.com:{port}/page")


def test_literal_public_ip_with_standard_port_allowed():
    tools._validate_url("http://93.184.216.34:80/page")


def test_literal_public_ip_with_odd_port_blocked():
    with pytest.raises(UnsafeURLError, match="port 6379 is blocked"):
        tools._validate_url("http://93.184.216.34:6379/")


# --- content types -------------------------------------------------------


def test_pdf_content_type_refused(monkeypatch):
    _public_dns(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"%PDF-1.7 not really text",
            headers={"content-type": "application/pdf"},
        )

    with pytest.raises(ToolError, match="unsupported content type"):
        fetch_page("https://example.com/doc", client=_client(handler))


def test_octet_stream_content_type_refused(monkeypatch):
    _public_dns(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"\x00\x01\x02",
            headers={"content-type": "application/octet-stream"},
        )

    with pytest.raises(ToolError, match="unsupported content type"):
        fetch_page("https://example.com/blob", client=_client(handler))


def test_plain_text_content_type_allowed(monkeypatch):
    _public_dns(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b"plain text body",
            headers={"content-type": "text/plain; charset=utf-8"},
        )

    text = fetch_page("https://example.com/notes", client=_client(handler))
    assert "plain text body" in text


def test_missing_content_type_header_tolerated(monkeypatch):
    _public_dns(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>hello</html>")

    text = fetch_page("https://example.com/page", client=_client(handler))
    assert "hello" in text


# --- redirects -----------------------------------------------------------


def test_redirect_to_private_address_blocked(monkeypatch):
    _public_dns(monkeypatch)
    hops: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hops.append(str(request.url))
        if request.url.host == "example.com":
            return httpx.Response(
                302, headers={"location": "http://10.0.0.1/secret"}
            )
        return httpx.Response(200, content=b"should never render")

    with pytest.raises(UnsafeURLError, match="non-public"):
        fetch_page("https://example.com/start", client=_client(handler))
    # only the first hop ran; the forbidden hop never executed
    assert len(hops) == 1


def test_redirect_to_loopback_blocked(monkeypatch):
    _public_dns(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302, headers={"location": "http://127.0.0.1:8500/admin"}
        )

    with pytest.raises(UnsafeURLError):
        fetch_page("https://example.com/go", client=_client(handler))


def test_redirect_to_file_scheme_blocked(monkeypatch):
    _public_dns(monkeypatch)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302, headers={"location": "file:///etc/passwd"}
        )

    with pytest.raises(UnsafeURLError, match="blocked scheme"):
        fetch_page("https://example.com/go", client=_client(handler))


# --- DNS time-of-check/time-of-use pinning -------------------------------


def test_http_requests_pinned_to_validated_address(monkeypatch):
    """Cleartext HTTP connects to the address that just passed
    validation (Host header preserved) — a DNS answer flipping
    between check and connect can no longer redirect the fetch."""
    _public_dns(monkeypatch, address="93.184.216.34")
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["host"] = request.url.host
        seen["host_header"] = request.headers.get("host", "")
        return httpx.Response(200, content=b"pinned ok")

    text = tools._fetch_with_client(
        _client(handler),
        "http://example.com/page",
        max_bytes=1000,
        max_redirects=5,
        pin_http=True,
    )
    assert "pinned ok" in text
    assert seen["host"] == "93.184.216.34"
    assert seen["host_header"] == "example.com"


def test_http_pinning_preserves_explicit_port_in_host_header(
    monkeypatch,
):
    _public_dns(monkeypatch, address="93.184.216.34")
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["host"] = request.url.host
        seen["host_header"] = request.headers.get("host", "")
        return httpx.Response(200, content=b"ok")

    tools._fetch_with_client(
        _client(handler),
        "http://example.com:80/page",
        max_bytes=1000,
        max_redirects=5,
        pin_http=True,
    )
    assert seen["host"] == "93.184.216.34"
    assert seen["host_header"] == "example.com:80"


def test_https_never_pinned_sni_and_verification_preserved(monkeypatch):
    """HTTPS must keep the ORIGINAL hostname: pinning would break
    SNI and certificate hostname verification (which we never trade
    for verify=False)."""
    _public_dns(monkeypatch, address="93.184.216.34")
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["host"] = request.url.host
        return httpx.Response(200, content=b"tls ok")

    tools._fetch_with_client(
        _client(handler),
        "https://example.com/page",
        max_bytes=1000,
        max_redirects=5,
        pin_http=True,
    )
    assert seen["host"] == "example.com"


def test_pinning_disabled_for_injected_clients_by_fetch_page(monkeypatch):
    """fetch_page only pins when IT owns the transport — injected
    (test) clients keep original URLs."""
    _public_dns(monkeypatch, address="93.184.216.34")
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["host"] = request.url.host
        return httpx.Response(200, content=b"ok")

    fetch_page("http://example.com/page", client=_client(handler))
    assert seen["host"] == "example.com"


def test_literal_ip_url_not_rewritten():
    request_url, headers = tools._connect_target(
        "http://93.184.216.34/x", ["93.184.216.34"], pin=True
    )
    assert request_url == "http://93.184.216.34/x"
    assert headers == {}


def test_pinning_noop_when_disabled():
    request_url, headers = tools._connect_target(
        "http://example.com/x", ["93.184.216.34"], pin=False
    )
    assert request_url == "http://example.com/x"
    assert headers == {}


# --- slow responses / timeouts -------------------------------------------


def test_own_client_uses_bounded_timeout_and_manual_redirects(
    monkeypatch,
):
    captured: dict[str, Any] = {}

    def fake_client(**kwargs: Any):
        captured.update(kwargs)
        raise RuntimeError("captured; not connecting")

    monkeypatch.setattr(tools.httpx, "Client", fake_client)
    with pytest.raises(RuntimeError, match="captured"):
        fetch_page("https://example.com/page", timeout=7.5)
    assert isinstance(captured["timeout"], httpx.Timeout)
    assert captured["timeout"].connect == 7.5
    # redirects are followed manually, hop by hop, each re-validated
    assert captured["follow_redirects"] is False


# --- TLS invariants -------------------------------------------------------


def test_verification_never_disabled(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_client(**kwargs: Any):
        captured.update(kwargs)
        raise RuntimeError("captured")

    monkeypatch.setattr(tools.httpx, "Client", fake_client)
    with pytest.raises(RuntimeError):
        fetch_page("https://example.com/page")
    verify = captured.get("verify")
    # either our explicit SSL context or httpx's default — never False
    assert verify is not False


def test_ssl_context_verifies_certificates():
    import ssl

    context = tools.build_ssl_context()
    if context is not None:
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True


def test_no_verify_false_in_fetch_source():
    """Static guard: fetch code never passes verify=False."""
    import inspect

    source = inspect.getsource(tools.fetch_page)
    code_lines = [
        line
        for line in source.splitlines()
        if "verify=False" in line.replace(" ", "")
        and not line.strip().startswith("#")
        # prose in comments/docstrings may mention the invariant
        and "never" not in line
        and "not" not in line
    ]
    assert code_lines == []
