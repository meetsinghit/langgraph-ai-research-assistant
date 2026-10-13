"""Pydantic request/response contracts for the local API (4D.1).

Deliberately narrow: responses expose research content (question,
answer, sources, content-free diagnostics) and NEVER configuration
secrets, system prompts, raw exception text or internal state.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

QuestionMax = 2000


def _clean_question(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("question must not be empty")
    # Control characters have no meaning in a research question and
    # only ever help terminal/UI smuggling.
    return "".join(char for char in value if char.isprintable() or char in " \t")


class SessionCreate(BaseModel):
    title: str = Field(default="New research", min_length=1, max_length=120)

    @field_validator("title")
    @classmethod
    def _strip_title(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("title must not be empty")
        return "".join(char for char in value if char.isprintable())


class RunCreate(BaseModel):
    question: str = Field(min_length=1, max_length=QuestionMax)

    @field_validator("question")
    @classmethod
    def _clean(cls, value: str) -> str:
        return _clean_question(value)


class SessionOut(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    run_count: int | None = None


class RunSummaryOut(BaseModel):
    id: str
    session_id: str
    question: str
    status: str
    has_answer: bool
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None


class SessionDetailOut(BaseModel):
    session: SessionOut
    runs: list[RunSummaryOut]


class SourceOut(BaseModel):
    sid: str
    url: str
    host: str = ""
    evidence: Literal["retrieved", "snippet"] = "snippet"
    position: int = 0


class RunDetailOut(BaseModel):
    id: str
    session_id: str
    question: str
    status: str
    answer: str | None = None
    review_status: str | None = None
    outcome_status: str | None = None
    outcome_warnings: list[str] = Field(default_factory=list)
    error_category: str | None = None
    iterations: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    estimated_cost_usd: float = 0.0
    total_execution_seconds: float = 0.0
    sources: list[SourceOut] = Field(default_factory=list)
    diagnostics: dict[str, Any] | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    events_url: str | None = None


class RunAcceptedOut(BaseModel):
    run_id: str
    status: str
    events_url: str


class CancelOut(BaseModel):
    run_id: str
    status: str


class ResumeOut(BaseModel):
    run_id: str
    resumed: bool = True


class ErrorDetail(BaseModel):
    category: str
    message: str


class ErrorOut(BaseModel):
    error: ErrorDetail


class HealthOut(BaseModel):
    ok: bool = True


class ConfigOut(BaseModel):
    """Public runtime configuration for the UI — never secrets."""

    provider: str
    model: str
    thinking_level: str | None = None
    max_total_execution_seconds: float
    max_research_iterations: int
    max_tool_calls_per_research: int
    min_findings: int
    min_retrieved_sources: int
    max_concurrent_runs: int
    resume_enabled: bool
