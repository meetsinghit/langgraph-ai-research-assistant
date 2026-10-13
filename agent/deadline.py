"""End-to-end run deadline — the global execution budget.

A :class:`RunDeadline` is created once per graph invocation by the
graph's ``start`` node and stored in state, so every node, tool call,
research retry and provider backoff consults the SAME monotonic
expiry instead of each operation getting its own budget.

Design rules:

- **Cooperative, checked at operation boundaries.** A synchronous,
  blocking call already in flight cannot be cancelled by this object;
  the wall-clock overrun after expiry is therefore bounded by the
  duration of the single in-flight operation. Request timeouts that
  we own (HTTPX fetch/search) are capped by the remaining budget via
  :meth:`timeout_budget`; provider SDK request timeouts are set at
  client construction and cannot be narrowed per call — the README
  states this limitation explicitly. The deadline is never described
  as a strict kill switch.
- **Monotonic clock, injectable.** ``time.monotonic`` by default;
  tests pass a fake clock so timeout behaviour is deterministic and
  never sleeps.
- **Never negative budgets.** ``pause_budget``/``timeout_budget``
  clamp to the remaining time so a retry can never overshoot the
  budget deliberately.
"""
from __future__ import annotations

import time
from typing import Callable

#: Floor for a capped request timeout: handing httpx a ~0s timeout
#: would fail instantly; a 0.1s overrun is the accepted worst case.
_MIN_TIMEOUT_FLOOR = 0.1


class RunDeadline:
    """One run's shared, monotonic end-to-end deadline."""

    __slots__ = ("total_seconds", "_clock", "_expires_at")

    def __init__(
        self,
        total_seconds: float,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if total_seconds <= 0:
            raise ValueError("deadline must be a positive duration")
        self.total_seconds = float(total_seconds)
        self._clock: Callable[[], float] = clock or time.monotonic
        self._expires_at = self._clock() + self.total_seconds

    @property
    def expires_at(self) -> float:
        return self._expires_at

    def remaining(self) -> float:
        """Seconds left (may be negative once expired)."""
        return self._expires_at - self._clock()

    def expired(self) -> bool:
        return self.remaining() <= 0

    def pause_budget(self, desired: float) -> float:
        """Sleep time actually allowed: ``desired``, clamped to the
        remaining budget and to zero when already expired."""
        if desired <= 0:
            return 0.0
        remaining = self.remaining()
        if remaining <= 0:
            return 0.0
        return min(desired, remaining)

    def timeout_budget(self, requested: float) -> float:
        """Request timeout actually allowed: ``requested`` clamped to
        the remaining budget (floor of 0.1s — see module docstring)."""
        if requested <= 0:
            return requested
        remaining = self.remaining()
        if remaining <= 0:
            return 0.0
        return max(_MIN_TIMEOUT_FLOOR, min(requested, remaining))
