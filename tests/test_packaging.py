"""Phase 4D.3 packaging & integration tests (offline, mocked).

Covers the new local-packaging surface without touching a network or a
real provider:

- built-UI serving from FastAPI (same-origin production mode):
  unauthenticated static files, cache headers, SPA fallback, path
  traversal containment, JSON 404 contract for /api and for
  not-built installs;
- auth preservation: every /api route still requires the Bearer
  token even with the UI mounted; the token never appears in any UI
  response;
- session/history persistence and token stability across application
  restarts (two separate app instances on the same data files);
- Windows launcher helper functions (skipped on other platforms).
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent.config import Settings
from app.main import create_app, resolve_token
from tests.app_fakes import TEST_TOKEN, make_settings, model_factory, tools_factory

AUTH = {"Authorization": f"Bearer {TEST_TOKEN}"}

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_POWERSHELL = REPO_ROOT / "scripts" / "launcher.ps1"


def _dist_with_ui(tmp_path: Path) -> Path:
    """A minimal but realistic built bundle (hashed asset + shell)."""
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text(
        "<!doctype html><html><head><title>AI Research Assistant</title>"
        '<link rel="stylesheet" href="/assets/index-abc123.css">'
        "</head><body><div id='root'></div>"
        "<script type='module' src='/assets/index-abc123.js'></script>"
        "</body></html>",
        encoding="utf-8",
    )
    (dist / "assets" / "index-abc123.js").write_text(
        "console.log('ui bundle');", encoding="utf-8"
    )
    (dist / "assets" / "index-abc123.css").write_text(
        ":root{--x:1}", encoding="utf-8"
    )
    return dist


def _packaged_app(tmp_path: Path, **overrides: object):
    settings = make_settings(tmp_path, **overrides)
    app = create_app(
        settings,
        db_path=tmp_path / "data" / "app.db",
        checkpointer=None,
        model_factory=model_factory([]),
        tools_factory=tools_factory([]),
    )
    return app


# --- built-UI serving --------------------------------------------------------


class TestBuiltUiServing:
    def test_index_and_assets_served_without_auth(self, tmp_path: Path):
        dist = _dist_with_ui(tmp_path)
        app = _packaged_app(tmp_path, frontend_dist=str(dist))
        with TestClient(app) as client:
            index = client.get("/")
            assert index.status_code == 200
            assert "text/html" in index.headers["content-type"]
            asset = client.get("/assets/index-abc123.js")
            assert asset.status_code == 200
        app.state.manager.shutdown()

    def test_cache_headers_match_content(self, tmp_path: Path):
        dist = _dist_with_ui(tmp_path)
        app = _packaged_app(tmp_path, frontend_dist=str(dist))
        with TestClient(app) as client:
            hashed = client.get("/assets/index-abc123.js")
            assert hashed.headers["cache-control"] == (
                "public, max-age=31536000, immutable"
            )
            shell = client.get("/")
            # The shell must always be revalidated so new builds load.
            assert shell.headers["cache-control"] == "no-cache"
        app.state.manager.shutdown()

    def test_ui_responses_are_hardened(self, tmp_path: Path):
        dist = _dist_with_ui(tmp_path)
        app = _packaged_app(tmp_path, frontend_dist=str(dist))
        with TestClient(app) as client:
            shell = client.get("/")
            assert shell.headers["x-content-type-options"] == "nosniff"
            assert "default-src 'self'" in shell.headers["content-security-policy"]
            assert shell.headers["referrer-policy"] == "no-referrer"
        app.state.manager.shutdown()

    def test_deep_links_fall_back_to_the_app_shell(self, tmp_path: Path):
        dist = _dist_with_ui(tmp_path)
        app = _packaged_app(tmp_path, frontend_dist=str(dist))
        with TestClient(app) as client:
            deep = client.get("/some/bookmarked/route")
            assert deep.status_code == 200
            assert "<div id='root'>" in deep.text
        app.state.manager.shutdown()

    def test_unknown_api_paths_keep_the_json_contract(self, tmp_path: Path):
        dist = _dist_with_ui(tmp_path)
        app = _packaged_app(tmp_path, frontend_dist=str(dist))
        with TestClient(app) as client:
            unknown = client.get("/api/does-not-exist", headers=AUTH)
            assert unknown.status_code == 404
            assert unknown.json()["error"]["category"] == "not_found"
            # never the HTML shell
            assert "text/html" not in unknown.headers["content-type"]
        app.state.manager.shutdown()

    def test_path_traversal_is_contained(self, tmp_path: Path):
        dist = _dist_with_ui(tmp_path)
        secret = tmp_path / "secret.txt"
        secret.write_text("sensitive", encoding="utf-8")
        app = _packaged_app(tmp_path, frontend_dist=str(dist))
        with TestClient(app) as client:
            for probe in (
                "/..%2F..%2Fsecret.txt",
                "/assets/..%2F..%2F..%2Fsecret.txt",
                "/%2e%2e/%2e%2e/secret.txt",
            ):
                response = client.get(probe)
                assert response.status_code in (404, 400), probe
                assert "sensitive" not in response.text
        app.state.manager.shutdown()

    def test_missing_dist_returns_actionable_hint_and_api_still_works(
        self, tmp_path: Path
    ):
        app = _packaged_app(
            tmp_path, frontend_dist=str(tmp_path / "never_built")
        )
        with TestClient(app) as client:
            root = client.get("/")
            assert root.status_code == 404
            assert "npm run build" in root.json()["error"]["message"]
            assert client.get("/api/health", headers=AUTH).status_code == 200
        app.state.manager.shutdown()

    def test_token_never_appears_in_ui_responses(self, tmp_path: Path):
        dist = _dist_with_ui(tmp_path)
        app = _packaged_app(tmp_path, frontend_dist=str(dist))
        with TestClient(app) as client:
            for path in ("/", "/assets/index-abc123.js", "/assets/index-abc123.css"):
                response = client.get(path)
                assert TEST_TOKEN not in response.text, path
                joined = " ".join(f"{k}:{v}" for k, v in response.headers.items())
                assert TEST_TOKEN not in joined, path
        app.state.manager.shutdown()


# --- auth preservation -------------------------------------------------------


class TestAuthPreservedWithUiMounted:
    def test_every_api_route_still_requires_the_token(self, tmp_path: Path):
        dist = _dist_with_ui(tmp_path)
        app = _packaged_app(tmp_path, frontend_dist=str(dist))
        with TestClient(app) as client:
            assert client.get("/api/sessions").status_code == 401
            assert client.get("/api/config").status_code == 401
            assert client.get("/api/runs/x/events").status_code == 401
            assert (
                client.get("/api/sessions", headers={"Authorization": "Bearer wrong"})
                .status_code
                == 401
            )
            # and the real token still works
            assert client.get("/api/sessions", headers=AUTH).status_code == 200
        app.state.manager.shutdown()

    def test_health_stays_unauthenticated(self, tmp_path: Path):
        dist = _dist_with_ui(tmp_path)
        app = _packaged_app(tmp_path, frontend_dist=str(dist))
        with TestClient(app) as client:
            assert client.get("/api/health").status_code == 200
        app.state.manager.shutdown()


# --- persistence across restarts --------------------------------------------


class TestRestartPersistence:
    def test_sessions_and_history_survive_a_backend_restart(
        self, tmp_path: Path
    ):
        db = tmp_path / "data" / "app.db"

        def instance() -> object:
            return create_app(
                make_settings(tmp_path),
                db_path=db,
                checkpointer=None,
                model_factory=model_factory([]),
                tools_factory=tools_factory([]),
            )

        first = instance()
        with TestClient(first) as client:
            created = client.post(
                "/api/sessions", json={"title": "Durable session"}, headers=AUTH
            )
            assert created.status_code == 201
            session_id = created.json()["id"]
        first.state.manager.shutdown()

        # A completely separate application instance (fresh process
        # boundary in production) on the same data files.
        second = instance()
        with TestClient(second) as client:
            sessions = client.get("/api/sessions", headers=AUTH).json()
            assert [s["title"] for s in sessions] == ["Durable session"]
            detail = client.get(f"/api/sessions/{session_id}", headers=AUTH)
            assert detail.status_code == 200
        second.state.manager.shutdown()

    def test_token_bootstrap_is_stable_and_never_overwritten(
        self, tmp_path: Path
    ):
        settings = Settings(
            _env_file=None,
            data_dir=str(tmp_path / "data"),
            frontend_dist=str(tmp_path / "no_ui"),
        )
        data_dir = tmp_path / "data"
        first = resolve_token(settings, data_dir)
        assert first  # bootstrapped
        token_file = data_dir / "app_token.txt"
        assert token_file.read_text(encoding="utf-8").strip() == first
        # Second start (restart) must reuse the SAME token.
        assert resolve_token(settings, data_dir) == first
        # And an explicitly configured token always wins untouched.
        configured = Settings(
            _env_file=None, app_token="explicit-token", data_dir=str(data_dir)
        )
        assert resolve_token(configured, data_dir) == "explicit-token"
        assert token_file.read_text(encoding="utf-8").strip() == first


# --- Windows launcher helpers -----------------------------------------------


@pytest.mark.skipif(os.name != "nt", reason="launcher helpers are Windows-only")
class TestLauncherHelpers:
    """Pure-function coverage of scripts/launcher.ps1."""

    def _ps(self, command: str) -> str:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(LAUNCHER_POWERSHELL),
                "-Action",
                "selftest",
                "-Command",
                command,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    def test_env_file_value_parsing(self):
        assert self._ps("Get-EnvValueTest") == "ok"

    def test_port_probe_and_health_wait(self):
        assert self._ps("Test-PortProbeTest") == "ok"

    def test_safe_stop_refuses_foreign_processes(self):
        assert self._ps("Test-SafeStopTest") == "ok"
