"""Deterministic completeness check for explicitly requested topics.

Keyword-coverage only: it verifies that the distinctive terms of the
question appear in the drafted answer, so an answer that silently
ignores an explicitly requested topic (the live-run defect: the
question asked for conditional routing, the draft never mentioned
it) is flagged and routed back to targeted follow-up research.

It deliberately does NOT verify correctness, support or provenance —
a covered topic can still be explained wrongly or with bad
citations. URL/citation provenance lives in ``agent.citations``;
semantic judgement stays with the LLM reviewer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "CompletenessReport",
    "check_completeness",
    "format_completeness_report",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")

#: Question-framing and generic words that are not topics. The list
#: is intentionally conservative: a missed stopword can only cause a
#: false "missing topic" that routes the draft back to research,
#: which the iteration budget bounds.
_QUESTION_STOPWORDS = frozenset(
    {
        "about", "after", "again", "also", "and", "any", "are", "based",
        "could", "describe", "describes", "did", "does", "docs",
        "documentation", "during", "explain", "explains", "explicitly",
        "first", "from", "generate", "give", "happen", "happens",
        "happening", "have", "help", "into", "its", "latest", "list",
        "means", "meaning", "mention", "must", "need", "official",
        "party", "please", "question", "reference", "references",
        "request", "requested", "should", "site", "specifically",
        "source", "sources", "tell", "that", "the", "their", "there",
        "these", "they", "this", "those", "through", "topic",
        "understand", "using", "want", "was", "were", "what", "when",
        "where", "which", "who", "why", "with", "work", "works",
        "working", "would", "your", "website",
    }
)


@dataclass(frozen=True)
class CompletenessReport:
    """Result of the deterministic topic-coverage check."""

    requested_terms: tuple[str, ...]
    missing: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing


def _requested_terms(question: str) -> tuple[str, ...]:
    """Distinctive topic terms of the question, first-appearance order.

    Tokens shorter than 4 characters, stopwords and digit-only tokens
    are dropped; a trailing plural ``s`` is stripped so "routes"
    covers "route" (word-prefix matching does the rest).
    """
    terms: list[str] = []
    for raw in _TOKEN_RE.findall((question or "").lower()):
        term = raw[:-1] if len(raw) >= 6 and raw.endswith("s") else raw
        if len(term) < 4 or term in _QUESTION_STOPWORDS:
            continue
        if not any(ch.isalpha() for ch in term):
            continue
        if term not in terms:
            terms.append(term)
    return tuple(terms)


def check_completeness(question: str, answer: str) -> CompletenessReport:
    """Flag requested terms the answer never mentions (word-prefix,
    case-insensitive match)."""
    text = answer or ""
    terms = _requested_terms(question)
    missing = tuple(
        term
        for term in terms
        if not re.search(rf"\b{re.escape(term)}", text, re.IGNORECASE)
    )
    return CompletenessReport(requested_terms=terms, missing=missing)


def format_completeness_report(report: CompletenessReport) -> str:
    """LLM-readable report for the review prompt — with the
    coverage-is-not-correctness disclaimer."""
    requested = ", ".join(report.requested_terms) or "(none)"
    missing = ", ".join(report.missing) or "(none)"
    return "\n".join(
        [
            "Topic-coverage check (deterministic keyword matching only — "
            "it does NOT verify correctness, factual support or "
            "citations):",
            f"- explicitly requested topic term(s): {requested}",
            f"- missing from the draft: {missing}",
            "result: "
            + (
                "PASS"
                if report.ok
                else "FAIL — the draft does not address: " + missing
            ),
        ]
    )
