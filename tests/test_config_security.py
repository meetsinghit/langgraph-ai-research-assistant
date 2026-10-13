"""Supply-chain and configuration regression guards (Phase 4C, WS-F).

These are cheap, deterministic checks that keep the security-relevant
configuration properties from silently regressing:

- every dependency in requirements.txt is EXACTLY pinned (==);
- secrets files stay git-ignored;
- Settings ignore unknown keys (no config-key smuggling) and keep
  their secret-field registry aligned with real fields;
- no agent source line disables TLS verification.
"""
from __future__ import annotations

import pathlib
import re

import agent.config as config

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _requirements_lines() -> list[str]:
    # requirements.txt is UTF-16 LE with BOM (see repo notes)
    text = (_ROOT / "requirements.txt").read_text(encoding="utf-16")
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def test_every_dependency_is_exactly_pinned():
    pinned = re.compile(r"^[A-Za-z0-9._\-]+\s*==\s*[A-Za-z0-9._\-]+$")
    for line in _requirements_lines():
        assert pinned.match(line), f"unpinned dependency: {line!r}"


def test_secrets_and_reports_are_git_ignored():
    ignored = (_ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in (
        ".env",
        "evals/report.json",
        "evals/security_report.json",
        "run_metrics*.json",
    ):
        assert pattern in ignored, f"{pattern!r} missing from .gitignore"


def test_settings_ignore_unknown_keys():
    """extra='ignore' keeps unknown/renamed keys from erroring or —
    worse — from being silently treated as settings."""
    assert config.Settings.model_config.get("extra") == "ignore"


def test_secret_field_registry_covers_every_api_key_field():
    """_SECRET_FIELDS drives redaction; a newly added key field that
    is not registered here would leak into logs."""
    secret_fields = set(config._SECRET_FIELDS)
    key_fields = {
        name
        for name in config.Settings.model_fields
        if name.endswith("_api_key")
    }
    assert key_fields <= secret_fields
    # and every registered field actually exists
    assert secret_fields <= set(config.Settings.model_fields)


def test_no_agent_source_disables_tls_verification():
    """Static guard across agent/ — no verify=False anywhere in code.
    Prose in comments/docstrings may mention the invariant by saying
    'never verify=False'; a bare `verify=False` code usage must not
    exist."""
    offenders: list[str] = []
    for path in (_ROOT / "agent").rglob("*.py"):
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if re.search(r"verify\s*=\s*False", line):
                # allow prose that negates it (never/not/insecure)
                if any(
                    word in line
                    for word in ("never", "not", "No ", "no ", "insecure")
                ):
                    continue
                offenders.append(f"{path.name}:{lineno}")
    assert offenders == [], f"verify=False found in {offenders}"
