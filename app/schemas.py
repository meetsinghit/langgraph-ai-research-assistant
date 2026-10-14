"""Pydantic request/response contracts for the local API (4D.1).

Deliberately narrow: responses expose research content (question,
answer, sources, content-free diagnostics) and NEVER configuration
secrets, system prompts, raw exception text or internal state.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

QuestionMax = 2000

#: Per-run research depth and evidence policy (Phase 4E.1). The
#: Literal types are the API-level validation — an unknown value is
#: a 422, never a silent downgrade.
ResearchDepthLiteral = Literal["quick", "standard", "deep"]
EvidencePolicyLiteral = Literal["balanced", "verified"]
#: Thinking selector vocabulary (Phase 4E.3). Which values a given
#: model ACCEPTS is registry-validated server-side (400 with a fixed
#: category), never inferred by the client.
ThinkingSelectLiteral = Literal[
    "auto", "minimal", "low", "medium", "high"
]


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


#: Rename payload (4E.3 WS-F) — identical validation to creation, so
#: a title can never be made un-displayable after the fact. The
#: default is deliberately dropped: an explicit rename without a
#: title is a 422, never a silent rename to some generated name.
class SessionRename(SessionCreate):
    title: str = Field(min_length=1, max_length=120)
    """PATCH body: {"title": "..."} (1-120 chars, control chars stripped)."""


class RunCreate(BaseModel):
    question: str = Field(min_length=1, max_length=QuestionMax)
    #: research depth for this run (persisted as run metadata so
    #: historical answers retain their original settings)
    research_depth: ResearchDepthLiteral = "standard"
    #: evidence policy for this run (backend-enforced; see
    #: agent.citations.validate_citations)
    evidence_policy: EvidencePolicyLiteral = "balanced"
    #: Gemini model for this run (4E.3). Must name a selectable entry
    #: of the backend registry — arbitrary model IDs from the client
    #: are rejected; omitted = the configured default.
    model_id: str | None = Field(default=None, max_length=120)
    #: thinking effort for this run; omitted = configured default
    #: ("medium" when unset). Per-model support is validated
    #: server-side before the run is created.
    thinking_level: ThinkingSelectLiteral | None = None

    @field_validator("question")
    @classmethod
    def _clean(cls, value: str) -> str:
        return _clean_question(value)

    @field_validator("model_id")
    @classmethod
    def _clean_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        if not all(char.isprintable() for char in value):
            raise ValueError("model_id must not contain control characters")
        return value


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
    #: run metadata (Phase 4E.1): the settings this run USED
    research_depth: str = "standard"
    evidence_policy: str = "balanced"
    #: run metadata (Phase 4E.3): model + thinking configuration
    #: frozen at creation. NULL for pre-4E.3 runs — displayed as
    #: "not recorded", never back-filled with a guess.
    model_id: str | None = None
    thinking_level: str | None = None


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
    #: run metadata (Phase 4E.1): depth and evidence policy this
    #: run used — persisted so history keeps its original settings
    research_depth: str = "standard"
    evidence_policy: str = "balanced"
    #: run metadata (Phase 4E.3): model + thinking this run used
    #: (None = pre-4E.3 run, never back-filled)
    model_id: str | None = None
    thinking_level: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    events_url: str | None = None


class RunAcceptedOut(BaseModel):
    run_id: str
    status: str
    events_url: str


# -- model catalogue (4E.3) ----------------------------------------------


class ModelThinkingOut(BaseModel):
    """Per-model thinking capabilities (registry-backed)."""

    control: Literal["level", "budget", "none"]
    levels: list[str]
    default: str


class ModelOut(BaseModel):
    """One entry of GET /api/models.

    ``selectable`` is the ONLY flag the UI may enable; it is true
    exclusively for registry entries with confirmed free-tier
    eligibility that are not retired. Discovery can annotate
    ``availability`` but can never make a model selectable.
    """

    model_id: str
    display_name: str
    family: str
    status: str  # stable | preview | deprecated | shutdown
    free_tier: str  # eligible | paid_only | unknown
    selectable: bool
    exclusion_reason: str | None = None
    thinking: ModelThinkingOut | None = None  # None = unknown
    availability: str  # not_checked | confirmed | not_listed
    verified_on: str = ""
    docs_url: str = ""
    notes: str = ""
    source: str = "registry"  # registry | configured_default | discovery


class ModelDiscoveryOut(BaseModel):
    """Bounded live-discovery status — never raw error text."""

    status: str  # disabled | available | unavailable
    checked_at: str | None = None
    error_category: str | None = None  # timeout | provider | auth | ...
    model_count: int | None = None


class ModelsOut(BaseModel):
    default_model_id: str
    default_thinking_level: str
    discovery: ModelDiscoveryOut
    models: list[ModelOut]
    #: Observed via live discovery but NOT in the backend registry:
    #: reported for transparency, never selectable (eligibility and
    #: capabilities unknown).
    unregistered: list[ModelOut] = Field(default_factory=list)


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
