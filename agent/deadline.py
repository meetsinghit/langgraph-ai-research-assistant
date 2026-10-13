"""End-to-end run deadline — the global execution budget.

A :class:`RunDeadline` is created once per graph invocation (by the
caller, carried in the runtime-only ``RunContext`` — see
``agent.runtime``) so every node, tool call, research retry and
provider backoff consults the SAME monotonic expiry instead of each
operation getting its own budget.

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
- **Also the run's cooperative CANCEL surface (Phase 4D).**
  :meth:`cancel` flags the run; from then on :meth:`expired` reports
  True exactly as if time had run out, so every existing guard,
  retry pause and request budget stops new work with the same
  proven bounds. A cancelled run is reported with the distinct
  ``CANCELLED`` outcome (never ``TIMED_OUT``). Cancellation is a
  single-attribute flag write — safe to call from another thread
  (the API server's request handler) without a lock, the same
  single-write signal pattern LangGraph's ``RunControl`` uses.

Runtime-only object (Phase 4D): a ``RunDeadline`` carries a live
clock and a live flag, so it is NEVER placed in graph state — state
holds only the serializable budget primitive
(``deadline_total_seconds``). The object is passed to a graph
invocation via the runtime-only ``RunContext`` (`agent.runtime`),
which the checkpointer never serializes.
"""
from __future__ import annotations

import time
from typing import Callable

#: Floor for a capped request timeout: handing httpx a ~0s timeout
#: would fail instantly; a 0.1s overrun is the accepted worst case.
_MIN_TIMEOUT_FLOOR = 0.1


class RunDeadline:
    """One run's shared, monotonic end-to-end deadline.

    Doubles as the cooperative cancel flag (see module docstring):
    ``cancel()`` from any thread makes ``expired()`` report True for
    every guard, while ``cancelled`` lets the terminal nodes report
    the distinct CANCELLED outcome.
    """

    __slots__ = ("total_seconds", "_clock", "_expires_at", "_cancelled")

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
        self._cancelled = False

    @property
    def expires_at(self) -> float:
        return self._expires_at

    @property
    def cancelled(self) -> bool:
        """True once :meth:`cancel` was called (cooperative flag)."""
        return self._cancelled

    def cancel(self) -> None:
        """Request cooperative cancellation of the run.

        Single-attribute write: callable from another thread without
        a lock. After this, :meth:`expired` reports True, so no new
        operation starts; an operation already in flight finishes
        (same bound as expiry — the in-flight request's own timeout).
        Idempotent.
        """
        self._cancelled = True

    def remaining(self) -> float:
        """Seconds left (may be negative once expired)."""
        return self._expires_at - self._clock()

    def expired(self) -> bool:
        return self._cancelled or self.remaining() <= 0

    def pause_budget(self, desired: float) -> float:
        """Sleep time actually allowed: ``desired``, clamped to the
        remaining budget and to zero when expired or cancelled."""
        if desired <= 0 or self.expired():
            return 0.0
        remaining = self.remaining()
        if remaining <= 0:
            return 0.0
        return min(desired, remaining)

    def timeout_budget(self, requested: float) -> float:
        """Request timeout actually allowed: ``requested`` clamped to
        the remaining budget (floor of 0.1s — see module docstring).
        Zero once expired or cancelled: no new request may start."""
        if requested <= 0:
            return requested
        if self.expired():
            return 0.0
        remaining = self.remaining()
        if remaining <= 0:
            return 0.0
        return max(_MIN_TIMEOUT_FLOOR, min(requested, remaining))
