"""Tests for structured logging and secret redaction."""
import json

from agent.config import Settings
from agent.logging_config import redact, setup_logging


def test_redact_masks_credential_pairs():
    message = "calling LLM with api_key=supersecret123 and token=abc.def"
    redacted = redact(message)
    assert "supersecret123" not in redacted
    assert "abc.def" not in redacted
    assert "***redacted***" in redacted


def test_redact_leaves_normal_messages_untouched():
    message = "research iteration 2 completed with 4 findings"
    assert redact(message) == message


def test_setup_logging_writes_json_lines_and_redacts_secrets(tmp_path):
    log_file = tmp_path / "agent.log"
    settings = Settings(_env_file=None, log_level="INFO", log_file=str(log_file))

    logger = setup_logging(settings)
    logger.info("run started with GEMINI_API_KEY=leaked-secret-value")
    for handler in logger.handlers:
        handler.flush()

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["level"] == "INFO"
    assert entry["logger"] == "agent"
    assert "leaked-secret-value" not in lines[0]
    assert "***redacted***" in lines[0]


def test_redact_masks_json_style_credentials():
    message = 'connecting with "api_key": "super-secret-json" ok'
    redacted = redact(message)
    assert "super-secret-json" not in redacted
    assert "***redacted***" in redacted


def test_redact_masks_bearer_authorization_header():
    message = "request sent with Authorization: Bearer eyJhbGciOiJIUzI1NiJ9"
    redacted = redact(message)
    assert "eyJhbGciOiJIUzI1NiJ9" not in redacted
    assert "***redacted***" in redacted
