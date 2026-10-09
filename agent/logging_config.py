"""Structured (JSON-lines) logging with secret redaction.

Security guarantee: API keys and other credentials are never written to
logs. ``RedactingJsonFormatter`` scrubs the fully formatted message, so
secrets are caught even if they arrive via ``%s`` arguments or an
exception traceback. This is a defensive layer on top of the rule that
secrets are only ever read through ``Settings``.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent.config import Settings

# Matches patterns like  api_key=xyz  or  "token": "abc"  (case-insensitive).
_SECRET_PATTERN = re.compile(
    r"(?i)((?:api[_-]?key|apikey|token|secret|password|passwd|credential)s?)"
    r"\s*[:=]\s*[^\s,\"'}]+"
)


def redact(message: str) -> str:
    """Mask credential-looking values inside an arbitrary string."""
    return _SECRET_PATTERN.sub(r"\1=***redacted***", message)


class RedactingJsonFormatter(logging.Formatter):
    """JSON-lines formatter that redacts secrets from the final output."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": redact(record.getMessage()),
        }
        if record.exc_info:
            payload["exception"] = redact(self.formatException(record.exc_info))
        return json.dumps(payload)


def setup_logging(settings: Settings) -> logging.Logger:
    """Configure and return the ``agent`` logger (console + file, JSON lines).

    Idempotent: calling it twice does not duplicate handlers.
    """
    logger = logging.getLogger("agent")
    logger.setLevel(settings.log_level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = RedactingJsonFormatter()

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(settings.log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
