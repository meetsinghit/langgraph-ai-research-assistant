"""WS-D: provider-failure classification and bounded retries.

One retry layer only: SDK retries are disabled at provider
construction (``max_retries=1``), and this layer retries strictly
what ``classify_provider_error`` marks retryable —

- retryable: 429/quota, 5xx, timeout, transient network faults;
- never retried: auth (bad key), invalid request, UNKNOWN errors;
- bounded exponential backoff + jitter, Retry-After honoured but
  capped, every sleep clamped to the remaining global deadline;
- every attempt (failed or not) is recorded with its category and
  honest usage (``None`` = unmeasured, never a fake zero).

No live calls: everything runs on the scripted fake model.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent.deadline import RunDeadline
from agent.graph import _invoke_with_retry, _response_usage
from agent.llm import PROVIDER_FAILURE_CATEGORIES, classify_provider_error
from agent.metrics import MetricsCollector
from tests.fakes import FakeClock, ScriptedChatModel

_MESSAGES = [HumanMessage(content="hello")]


def _model(**kwargs) -> ScriptedChatModel:
    return ScriptedChatModel(**kwargs)  # type: ignore[arg-type]


# -- classification ------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "category", "retryable"),
    [
        (RuntimeError("429 Too Many Requests"), "rate_limit", True),
        (RuntimeError("quota exceeded for model"), "rate_limit", True),
        (RuntimeError("503 Service Unavailable"), "unavailable", True),
        (RuntimeError("model overloaded"), "unavailable", True),
        (RuntimeError("500 internal server error"), "internal", True),
        (TimeoutError("request deadline exceeded"), "timeout", True),
        (ConnectionError("connection reset by peer"), "network", True),
        (RuntimeError("401 UNAUTHENTICATED"), "auth", False),
        (RuntimeError("API key not valid"), "auth", False),
        (RuntimeError("400 invalid argument"), "invalid_request", False),
        (RuntimeError("provider exploded"), "unknown", False),
    ],
)
def test_classify_provider_error(exc, category, retryable):
    failure = classify_provider_error(exc)
    assert failure.category == category
    assert failure.retryable is retryable
    assert failure.category in PROVIDER_FAILURE_CATEGORIES


def test_retry_after_parsed_from_message():
    failure = classify_provider_error(
        RuntimeError("429 rate limit; retry-after: 45")
    )
    assert failure.category == "rate_limit"
    assert failure.retry_after == pytest.approx(45.0)


def test_retry_after_from_exception_attribute():
    exc = RuntimeError("429 slow down")
    exc.retry_after = 7.0  # type: ignore[attr-defined]
    failure = classify_provider_error(exc)
    assert failure.retry_after == pytest.approx(7.0)


# -- retry behaviour ------------------------------------------------------


def test_retryable_error_retried_with_deterministic_backoff():
    sleeps: list[float] = []
    model = _model(
        responses=[AIMessage(content="ok")],
        raise_on_call={0: RuntimeError("503 Service Unavailable")},
    )

    response = _invoke_with_retry(
        model,
        _MESSAGES,
        2.0,
        sleep=sleeps.append,
        rng=lambda: 0.5,  # jitter factor 1.0
        max_retries=1,
        backoff_max=30.0,
    )

    assert response.content == "ok"
    # one retry: 2.0s base * 2^0 * 1.0 jitter
    assert sleeps == [pytest.approx(2.0)]


def test_backoff_doubles_per_retry_and_is_capped():
    sleeps: list[float] = []
    model = _model(
        responses=[AIMessage(content="ok")],
        raise_on_call={
            0: RuntimeError("500 internal error"),
            1: RuntimeError("500 internal error"),
            2: RuntimeError("500 internal error"),
        },
    )

    _invoke_with_retry(
        model,
        _MESSAGES,
        10.0,
        sleep=sleeps.append,
        rng=lambda: 0.5,
        max_retries=3,
        backoff_max=12.0,
    )

    # 10 -> 20 (capped 12) -> 40 (capped 12)
    assert sleeps == [pytest.approx(10.0), pytest.approx(12.0), pytest.approx(12.0)]


def test_jitter_decorrelates_the_computed_backoff():
    sleeps: list[float] = []
    model = _model(
        responses=[AIMessage(content="ok")],
        raise_on_call={0: RuntimeError("429 rate limit")},
    )

    _invoke_with_retry(
        model,
        _MESSAGES,
        4.0,
        sleep=sleeps.append,
        rng=lambda: 0.0,  # lower jitter bound: 0.5x
        max_retries=1,
        backoff_max=30.0,
    )

    assert sleeps == [pytest.approx(2.0)]


def test_retry_after_honoured_but_capped_by_backoff_max():
    sleeps: list[float] = []
    model = _model(
        responses=[AIMessage(content="ok")],
        raise_on_call={
            0: RuntimeError("429 rate limit; retry-after: 45")
        },
    )

    _invoke_with_retry(
        model,
        _MESSAGES,
        2.0,
        sleep=sleeps.append,
        rng=lambda: 0.5,
        max_retries=1,
        backoff_max=30.0,
    )

    # provider guidance (45s) wins over the 2s backoff but the
    # configured cap still bounds the sleep
    assert sleeps == [pytest.approx(30.0)]


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("401 UNAUTHENTICATED invalid key"),
        RuntimeError("API key not valid"),
        RuntimeError("400 invalid argument: field 'x'"),
        RuntimeError("provider exploded"),
    ],
)
def test_permanent_and_unknown_failures_never_retry(exc):
    sleeps: list[float] = []
    model = _model(
        responses=[AIMessage(content="never reached")],
        raise_on_call={0: exc},
    )

    with pytest.raises(RuntimeError):
        _invoke_with_retry(
            model,
            _MESSAGES,
            1.0,
            sleep=sleeps.append,
            rng=lambda: 0.5,
            max_retries=3,
        )

    assert sleeps == []
    assert model.call_index == 1  # exactly one attempt


def test_retry_stops_when_the_global_deadline_has_expired():
    sleeps: list[float] = []
    clock = FakeClock()
    deadline = RunDeadline(1.0, clock=clock)
    clock.advance(2.0)  # expired
    model = _model(
        responses=[AIMessage(content="never reached")],
        raise_on_call={0: RuntimeError("503 Service Unavailable")},
    )

    with pytest.raises(RuntimeError):
        _invoke_with_retry(
            model,
            _MESSAGES,
            1.0,
            sleep=sleeps.append,
            rng=lambda: 0.5,
            max_retries=3,
            deadline=deadline,
        )

    assert sleeps == []  # no backoff sleep beyond an expired budget


def test_backoff_sleep_is_clamped_to_remaining_deadline_budget():
    sleeps: list[float] = []
    clock = FakeClock()
    deadline = RunDeadline(10.0, clock=clock)
    clock.advance(9.6)  # 0.4s left
    model = _model(
        responses=[AIMessage(content="ok")],
        raise_on_call={0: RuntimeError("503 Service Unavailable")},
    )

    _invoke_with_retry(
        model,
        _MESSAGES,
        2.0,
        sleep=sleeps.append,
        rng=lambda: 0.5,
        max_retries=1,
        backoff_max=30.0,
        deadline=deadline,
    )

    assert sleeps == [pytest.approx(0.4)]


# -- accounting ------------------------------------------------------------


def test_every_attempt_is_recorded_with_category_and_honest_usage():
    metrics = MetricsCollector()
    model = _model(
        responses=[
            AIMessage(
                content="ok",
                usage_metadata={
                    "input_tokens": 7,
                    "output_tokens": 3,
                    "total_tokens": 10,
                },
            )
        ],
        raise_on_call={0: RuntimeError("503 Service Unavailable")},
    )

    _invoke_with_retry(
        model,
        _MESSAGES,
        0.0,
        metrics=metrics,
        node="research",
        sleep=lambda _s: None,
        rng=lambda: 0.5,
        max_retries=1,
    )

    snapshot = metrics.snapshot()
    calls = snapshot["model_calls"]
    assert len(calls) == 2
    # failed attempt: category recorded, usage honestly unavailable
    assert calls[0]["error_category"] == "unavailable"
    assert calls[0]["input_tokens"] is None
    assert calls[0]["usage_source"] == "unavailable"
    # successful retry: provider-reported usage captured
    assert calls[1]["error_category"] is None
    assert calls[1]["input_tokens"] == 7
    assert calls[1]["usage_source"] == "provider"
    assert snapshot["totals"]["retries"] == 1
    retry = snapshot["retries"][0]
    assert retry["reason_category"] == "unavailable"


def test_structured_include_raw_usage_is_captured():
    """WS-E: with_structured_output(include_raw=True) keeps the raw
    AIMessage reachable, so provider usage is recorded instead of
    being lost with the parsed object."""
    metrics = MetricsCollector()
    model = _model(
        structured_outputs=["parsed"],
        structured_usage=True,
    )
    runnable = model.with_structured_output(dict, include_raw=True)

    result = _invoke_with_retry(
        runnable,
        _MESSAGES,
        0.0,
        metrics=metrics,
        node="plan",
    )

    assert isinstance(result, dict)
    assert result["parsed"] == "parsed"
    call = metrics.snapshot()["model_calls"][0]
    assert call["input_tokens"] and call["input_tokens"] > 0
    assert call["usage_source"] == "provider"


def test_response_usage_absent_is_none_not_zero():
    # plain message without usage: (None, None, None), never (0, 0, 0)
    assert _response_usage(AIMessage(content="hi")) == (None, None, None)
    # usage block present: measured values (zero is a real value);
    # no thinking breakdown reported -> thinking stays None (4E.3)
    message = AIMessage(
        content="hi",
        usage_metadata={
            "input_tokens": 0,
            "output_tokens": 5,
            "total_tokens": 5,
        },
    )
    assert _response_usage(message) == (0, 5, None)
    # provider-reported thinking breakdown (Gemini thoughts_token_count
    # via output_token_details.reasoning) is read honestly
    thinking_message = AIMessage(
        content="hi",
        usage_metadata={
            "input_tokens": 10,
            "output_tokens": 25,
            "total_tokens": 35,
            "output_token_details": {"reasoning": 15},
        },
    )
    assert _response_usage(thinking_message) == (10, 25, 15)
