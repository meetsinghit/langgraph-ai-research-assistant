"""Graph state schema for the AI Research Assistant.

GraphState is the single shared data structure every node reads from and
writes to. LangGraph backs each field with a channel: plain fields behave
like "last write wins", while fields annotated with a reducer merge updates
(e.g. append to a list) so several nodes can contribute without clobbering
each other.
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, TypedDict


class ReviewStatus(str, Enum):
    """Outcome of the quality-review gate."""

    PENDING = "pending"
    APPROVED = "approved"
    NEEDS_MORE_RESEARCH = "needs_more_research"
    FAILED = "failed"


class Finding(TypedDict):
    """A single research finding with its provenance.

    ``source_url`` must come from a real tool result — the agent never
    invents citations. ``evidence`` distinguishes the strength of the
    provenance: ``retrieved`` (a full page was fetched and inspected)
    vs ``snippet`` (a search-result excerpt — a discovered URL whose
    content was not verified).
    """

    text: str
    source_url: str | None
    evidence: str  # "retrieved" | "snippet"


def append_items(current: list[Any] | None, new: list[Any]) -> list[Any]:
    """Reducer: merge list updates by concatenation.

    LangGraph calls this as (existing_value, update). Handling ``None``
    keeps the first append working when the channel starts empty.
    """
    if not current:
        return list(new)
    return current + list(new)


class GraphState(TypedDict, total=False):
    """Shared state flowing through the research graph.

    ``total=False`` makes every key optional: nodes return only the fields
    they change, which is what makes partial updates possible.
    """

    question: str
    research_plan: Annotated[list[str], append_items]
    findings: Annotated[list[Finding], append_items]
    sources: Annotated[list[str], append_items]
    iteration: int
    errors: Annotated[list[str], append_items]
    review_status: ReviewStatus
    final_answer: str
    tokens_in: int
    tokens_out: int
    estimated_cost_usd: float
