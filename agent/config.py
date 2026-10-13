"""Typed application configuration.

Configuration is loaded from environment variables (and an optional ``.env``
file in the project root) and validated by pydantic-settings. Secrets are
held in memory only and are masked by :meth:`Settings.redacted` before they
are ever logged or printed.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["gemini", "openrouter", "openai"]
ThinkingLevel = Literal["low", "medium", "high"]

_SECRET_FIELDS = (
    "gemini_api_key",
    "openrouter_api_key",
    "openai_api_key",
    "app_token",
)


class Settings(BaseSettings):
    """Central, typed configuration for the AI Research Assistant."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # GEMINI_API_KEY maps to gemini_api_key
        extra="ignore",  # unrelated env vars are ignored, not fatal
    )

    # --- LLM backend (wired up in Phase 3) ---
    llm_provider: ProviderName = "gemini"
    llm_model: str = "gemini-3.8-flash"
    llm_base_url: str | None = None

    # Optional reasoning-depth override for normal research runs
    # (Gemini 3 models; unset = "medium"). Verification always
    # uses "low".
    llm_thinking_level: ThinkingLevel | None = None

    # API keys: read from env/.env only, never logged (see redacted()).
    gemini_api_key: str | None = None
    openrouter_api_key: str | None = None
    openai_api_key: str | None = None

    # --- Agent behaviour (validated: a zero/negative loop budget or
    # timeout would either never run or spin without progress) ---
    request_timeout_seconds: float = Field(default=30.0, gt=0)
    connection_verify_seconds: float = Field(default=30.0, gt=0)
    connection_verify_max_output_tokens: int = Field(default=256, ge=1)
    max_research_iterations: int = Field(default=3, ge=1)
    max_tool_calls_per_research: int = Field(default=4, ge=1)
    # Wall-clock budget PER RESEARCH PASS: every pass gets its own
    # budget. The full run is bounded by max_research_iterations x
    # this value, plus plan/synthesize/review LLM calls, which this
    # limit does NOT cover — see max_total_execution_seconds for the
    # end-to-end budget that DOES cover those.
    max_execution_seconds: float = Field(default=120.0, gt=0)
    # END-TO-END execution deadline for ONE graph run (WS-C),
    # covering plan + research + tools + synthesize + review +
    # retries + finalization. Separate from (and stricter than)
    # max_execution_seconds: one monotonic deadline is stamped at
    # run start; no new operation starts after it expires. A call
    # already in flight cannot be cancelled (sync architecture), so
    # actual overrun is bounded by the in-flight operation — tool
    # request timeouts are capped by the remaining budget; provider
    # request timeouts are fixed at client construction (documented
    # limitation). Keep it >= max_execution_seconds so a single
    # research pass can finish within the global budget.
    max_total_execution_seconds: float = Field(default=180.0, gt=0)

    # -- provider retry policy (WS-D) ----------------------------------
    # Retries AFTER the first attempt, applied only to failures
    # classified retryable (429/quota, 5xx, timeout, transient
    # network). Auth and invalid-request errors are never retried.
    # Default 1 preserves the historical rate-limit behaviour.
    # SDK-level retries stay disabled (max_retries=1 at provider
    # construction, see agent/llm.py) so the layers never multiply.
    llm_max_retries: int = Field(default=1, ge=0, le=6)
    # Upper bound for ONE backoff sleep (exponential base doubles per
    # retry; provider Retry-After wins over the computed backoff but
    # is still clamped by this bound). Every sleep is additionally
    # clamped to the remaining global deadline budget.
    retry_backoff_max_seconds: float = Field(default=30.0, gt=0)
    max_redirects: int = Field(default=5, ge=1)
    max_page_bytes: int = Field(default=1_000_000, ge=1)
    # Per-SOURCE character cap for evidence embedded in model
    # prompts. Fetched pages can be up to max_page_bytes, but
    # every research round re-sends all findings to the model
    # (then synthesize and review send them again) — without
    # this cap a few large pages amplify into tens of thousands
    # of cumulative input tokens. Truncation keeps the head of
    # each source (where documentation intros live) and marks
    # the cut explicitly; the full text stays in state.
    max_evidence_chars: int = Field(default=6000, ge=500)
    # Per-RESULT cap on MODEL-VISIBLE tool output: a tool result
    # is serialized into a ToolMessage in the research transcript,
    # so an oversized fetched page (up to max_page_bytes) would
    # otherwise be re-sent to the provider on the next invocation
    # (observed live: one 183K-character page = ~90% of a run's
    # input tokens). Oversized payloads are reduced to
    # query-relevant sections with an explicit marker; the metrics
    # collector still records the REAL result size. Separate from
    # max_evidence_chars (which budgets how findings are rendered
    # into research/synthesis prompts).
    max_tool_result_chars: int = Field(default=8000, ge=500)
    max_plan_queries: int = Field(default=4, ge=1)
    search_results_per_query: int = Field(default=5, ge=1)
    min_findings: int = Field(default=3, ge=1)
    # Minimum number of source PAGES that must be fetched and
    # inspected (Finding.evidence == "retrieved") before an answer
    # counts as verified. Search snippets never satisfy this
    # threshold. 0 disables the requirement: no search/fetch budget
    # partitioning and no unverified-answer label.
    min_retrieved_sources: int = Field(default=1, ge=0)

    # Optional path to a CA bundle (PEM) used to VERIFY server
    # certificates when fetching pages — e.g. an enterprise
    # TLS-inspection root that is not in certifi's store. When set,
    # it is the verification store; when unset, the OS trust store
    # (truststore) or certifi applies. Certificate verification is
    # never disabled either way; verify=False does not exist here.
    ssl_ca_bundle: str | None = None

    # Paid providers (OpenAI) are disabled unless explicitly enabled.
    allow_paid_providers: bool = False

    # --- Observability ---
    log_level: str = "INFO"
    log_file: str = "research_agent.log"

    # --- Application server (Phase 4D) ---
    # Bearer token for the local API. Unset = the server bootstraps
    # one into ``<data_dir>/app_token.txt`` on first start (git-
    # ignored; treat the file like a credential).
    app_token: str | None = None
    # Bind address — 127.0.0.1 by default. This is a SINGLE-USER,
    # LOCAL-ONLY service: do not bind or publish it publicly without
    # proper access controls (see docs/SECURITY.md).
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)
    # Comma-separated browser origins allowed to call the API (the
    # Vite dev server during development).
    cors_origins: str = "http://127.0.0.1:5173,http://localhost:5173"
    # Concurrent research runs; each run owns its model/tools/metrics.
    max_concurrent_runs: int = Field(default=2, ge=1, le=8)
    # Directory for the SQLite databases and the bootstrap token
    # file. The database files contain research content — treat the
    # directory as sensitive (see docs/SECURITY.md).
    data_dir: str = "data"
    # SSE keepalive interval for idle streams.
    sse_heartbeat_seconds: float = Field(default=15.0, gt=0)
    # Follow-up context (server-derived from prior runs of the same
    # session; never client-supplied) bounds.
    followup_context_max_turns: int = Field(default=3, ge=0, le=10)
    followup_context_max_chars: int = Field(default=4000, ge=0, le=20000)

    @property
    def cors_origin_list(self) -> list[str]:
        """Parsed :attr:`cors_origins` (trimmed, blanks dropped)."""
        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        level = value.upper()
        if not hasattr(logging, level):
            raise ValueError(f"invalid log level: {value!r}")
        return level

    def redacted(self) -> dict[str, Any]:
        """Configuration with every secret masked — the only safe way to log it."""
        data: dict[str, Any] = self.model_dump()
        for field in _SECRET_FIELDS:
            if data.get(field):
                data[field] = "***redacted***"
        return data

    def __repr__(self) -> str:
        """Redacted representation — safe for logs, tracebacks and REPLs.

        ``model_dump()`` still returns real secrets for internal use by
        the LLM factory; never pass its output to a log call or print.
        """
        return f"Settings({self.redacted()!r})"

    def __str__(self) -> str:
        return self.__repr__()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings (env vars are read once per process)."""
    return Settings()
