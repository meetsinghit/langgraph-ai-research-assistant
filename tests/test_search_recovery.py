"""WS-B: search-error taxonomy and bounded recovery.

The live bug: ddgs raises a GENERIC ``DDGSException`` for very
different situations (index empty vs engine-side HTTP failure), so
exception class alone cannot drive recovery — message markers are
classified into a fixed taxonomy and recovery is bounded:

- ``empty_results``/``invalid_query`` advance the query ladder
  (site-filter removal, operator stripping) — at most 3 backend
  calls, never inventing terms;
- ``timeout``/``rate_limited``/``upstream_unavailable`` repeat the
  SAME query exactly once (rate limits pause first, clamped by the
  run's global deadline);
- ``unknown`` is never retried blindly;
- the executor records the taxonomy category — never error text.
"""
from __future__ import annotations

from typing import Any

import pytest

import agent.tools as tools
from agent.deadline import RunDeadline
from agent.metrics import MetricsCollector
from agent.tools import (
    SEARCH_ERROR_CATEGORIES,
    SearchError,
    ToolExecutor,
    classify_search_failure,
    web_search,
)
from tests.fakes import FakeClock

_OK_RESULT = [{"title": "T", "body": "B", "href": "https://example.com/1"}]
_OK_MAPPED = [{"title": "T", "snippet": "B", "url": "https://example.com/1"}]


class _Session:
    def __init__(self, recorder: "_Recorder") -> None:
        self._recorder = recorder

    def __enter__(self) -> "_Session":
        return self

    def __exit__(self, *args: Any) -> bool:
        return False

    def text(self, query: str, max_results: int = 5) -> list[dict]:
        self._recorder.queries.append(query)
        outcomes = self._recorder.outcomes
        outcome = outcomes[min(self._recorder.calls, len(outcomes) - 1)]
        self._recorder.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome  # type: ignore[return-value]


class _Recorder:
    """Scripted backend: one outcome per backend call (last repeats).

    Each outcome is an Exception (raise), ``None`` (empty index) or a
    raw result list.
    """

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.queries: list[str] = []
        self.calls = 0

    def factory(self, *, timeout: float = 0.0) -> _Session:
        return _Session(self)


# -- taxonomy classification -------------------------------------------


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (RuntimeError("No results found."), "empty_results"),
        (RuntimeError("429 Too Many Requests"), "rate_limited"),
        (RuntimeError("rate limit exceeded"), "rate_limited"),
        (TimeoutError("backend timeout"), "timeout"),
        (RuntimeError("503 Service Unavailable"), "upstream_unavailable"),
        (RuntimeError("connection reset by peer"), "upstream_unavailable"),
        (RuntimeError("400 Bad Request"), "invalid_query"),
        (RuntimeError("something else entirely"), "unknown"),
        (
            SearchError("already classified", category="rate_limited"),
            "rate_limited",
        ),
    ],
)
def test_classify_search_failure_taxonomy(exc: BaseException, expected: str):
    category = classify_search_failure(exc)
    assert category == expected
    assert category in SEARCH_ERROR_CATEGORIES


def test_classify_returns_category_never_error_text():
    category = classify_search_failure(
        RuntimeError("failed at https://secret.internal/page?q=term")
    )
    assert category == "unknown"
    assert "secret.internal" not in category


# -- bounded recovery ---------------------------------------------------


def test_rate_limited_query_repeats_once_with_politeness_pause(
    monkeypatch,
):
    sleeps: list[float] = []
    monkeypatch.setattr(tools.time, "sleep", sleeps.append)
    recorder = _Recorder([RuntimeError("429 Too Many Requests"), _OK_RESULT])

    results = web_search("langgraph", timeout=1.0, ddgs_factory=recorder.factory)

    assert results == _OK_MAPPED
    # same query, exactly twice — one transient retry
    assert recorder.queries == ["langgraph", "langgraph"]
    assert sleeps == [pytest.approx(1.5)]


def test_timeout_repeats_once_without_pause(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(tools.time, "sleep", sleeps.append)
    recorder = _Recorder([TimeoutError("backend timeout"), _OK_RESULT])

    results = web_search("langgraph", timeout=1.0, ddgs_factory=recorder.factory)

    assert results == _OK_MAPPED
    assert recorder.queries == ["langgraph", "langgraph"]
    assert sleeps == []  # only rate limits pause


def test_rate_limit_pause_clamped_to_remaining_deadline_budget(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(tools.time, "sleep", sleeps.append)
    clock = FakeClock()
    deadline = RunDeadline(1.0, clock=clock)
    clock.advance(0.8)  # 0.2s of budget left
    recorder = _Recorder([RuntimeError("429 rate limit"), _OK_RESULT])

    web_search(
        "q",
        timeout=5.0,
        ddgs_factory=recorder.factory,
        deadline=deadline,
    )

    # the 1.5s politeness pause must never exceed the run's budget
    assert sleeps == [pytest.approx(0.2)]


def test_unknown_failure_is_not_retried():
    recorder = _Recorder([RuntimeError("something else entirely")])

    with pytest.raises(SearchError) as excinfo:
        web_search("q", timeout=1.0, ddgs_factory=recorder.factory)

    assert excinfo.value.category == "unknown"
    assert recorder.calls == 1  # no blind retry
    assert "web_search failed" in str(excinfo.value)


def test_empty_ladder_is_bounded_to_three_backend_calls():
    recorder = _Recorder([None])  # always-empty index
    query = 'site:example.com "exact phrase"'

    with pytest.raises(SearchError) as excinfo:
        web_search(query, timeout=1.0, ddgs_factory=recorder.factory)

    assert excinfo.value.category == "empty_results"
    # original -> site filter removed -> operators stripped
    assert recorder.calls == 3
    assert recorder.queries[1] == '"exact phrase"'
    assert recorder.queries[2] == "exact phrase"
    assert "no results" in str(excinfo.value)
    assert "Broaden the query" in str(excinfo.value)


def test_operator_stripping_recovers_without_inventing_terms():
    recorder = _Recorder([None, _OK_RESULT])

    results = web_search(
        '"langgraph" AND checkpointers',
        timeout=1.0,
        ddgs_factory=recorder.factory,
    )

    assert results == _OK_MAPPED
    assert recorder.queries == [
        '"langgraph" AND checkpointers',
        "langgraph checkpointers",
    ]


def test_empty_query_is_invalid_query():
    with pytest.raises(SearchError) as excinfo:
        web_search("   ", timeout=1.0)

    assert excinfo.value.category == "invalid_query"


def test_expired_deadline_refuses_before_any_backend_call():
    clock = FakeClock()
    deadline = RunDeadline(1.0, clock=clock)
    clock.advance(2.0)
    recorder = _Recorder([_OK_RESULT])

    with pytest.raises(SearchError) as excinfo:
        web_search(
            "q",
            timeout=5.0,
            ddgs_factory=recorder.factory,
            deadline=deadline,
        )

    assert excinfo.value.category == "deadline"
    assert recorder.calls == 0
    assert "deadline" in str(excinfo.value).lower()


# -- executor integration ----------------------------------------------


def test_executor_records_search_category_not_error_text():
    def failing_search(query: str) -> list[Any]:
        raise SearchError(
            "web_search failed: backend down at https://example.org/x",
            category="upstream_unavailable",
        )

    collector = MetricsCollector()
    executor = ToolExecutor(
        {"web_search": failing_search}, metrics=collector
    )

    result = executor.execute(
        {"name": "web_search", "args": {"query": "q"}}
    )

    assert result["ok"] is False
    counts = collector.snapshot()["counts"]["tool_error_categories"]
    assert counts == {"upstream_unavailable": 1}
    # the message (with its URL) goes to the MODEL, never to metrics
    assert "example.org" not in str(counts)


def test_executor_empty_results_category_for_no_results():
    def empty_search(query: str) -> list[Any]:
        raise SearchError(
            "web_search returned no results for query 'q'. "
            "Broaden the query.",
            category="empty_results",
        )

    collector = MetricsCollector()
    executor = ToolExecutor(
        {"web_search": empty_search}, metrics=collector
    )

    result = executor.execute(
        {"name": "web_search", "args": {"query": "q"}}
    )

    assert result["ok"] is False
    assert "Broaden the query" in (result["error"] or "")
    counts = collector.snapshot()["counts"]["tool_error_categories"]
    assert counts == {"empty_results": 1}
