"""Structured (JSON-lines) logging with best-effort secret redaction.

Redaction is a defense-in-depth layer, NOT a guarantee. The primary
control is simpler and stronger: credential-bearing payloads are never
logged in the first place — secrets are read only through ``Settings``
and must never be passed to a log call. As a secondary measure, the
formatter below scrubs common ``key=value`` / ``"key": "value"`` /
``Authorization: Bearer ...`` patterns from the fully formatted message
(so secrets arriving via ``%s`` arguments or tracebacks are usually
masked too). It cannot catch every possible encoding — e.g. a secret
split across words — so treat all log output as potentially sensitive.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from agent.config import Settings

# Matches patterns like  api_key=xyz , "token": "abc" or
# Authorization: Bearer xyz  (case-insensitive). Best-effort: values
# containing spaces may only be partially masked.
_SECRET_PATTERN = re.compile(
    r"(?i)((?:api[_-]?key|apikey|token|secret|password|passwd|credential"
    r"|authorization)s?)"
    r"[\"']?\s*[:=]\s*[\"']?\s*(?:[^\s,\"'}]+(?:\s+[^\s,\"'}]+)?)"
)


def redact(message: str) -> str:
    """Mask credential-looking values inside an arbitrary string."""
    return _SECRET_PATTERN.sub(r"\1=***redacted***", message)


class RedactingJsonFormatter(logging.Formatter):
    """JSON-lines formatter with best-effort secret redaction."""

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
