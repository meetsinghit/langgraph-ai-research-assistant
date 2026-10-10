"""Deterministic completeness check for explicitly requested topics.

Lexical coverage only: it verifies that the distinctive terms of
the question appear in the drafted answer OUTSIDE known
disclaimer/unavailability phrasings, so an answer that silently
ignores an explicitly requested topic — or only mentions it to say
the details "were not available" — is flagged and routed back to
targeted follow-up research.

It deliberately does NOT verify that a covered topic is explained
correctly, substantively or adequately: lexical coverage is not
completeness. Citation provenance lives in ``agent.citations``;
semantic adequacy stays with the LLM reviewer.
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

#: Sentence boundary (period/question/exclamation followed by
#: whitespace). Good enough for drafted answers; conservative on
#: abbreviations.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

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

#: Phrasings that disclaim rather than explain. A requested term
#: that appears only in sentences containing one of these does not
#: count as covered — "conditional routing details were not
#: available" is not an explanation of conditional routing. The
#: list is deterministic and conservative: a sentence that both
#: explains and disclaims is treated as a disclaimer (a false
#: "missing" only costs one more bounded research pass).
_DISCLAIMER_RE = re.compile(
    r"(?:"
    # negated availability/provision: "was not available",
    # "details were not included", "not covered", "n't provided"
    r"\b(?:not|n't|never)\s+(?:be\s+)?(?:"
    r"available|provide[ds]?|cover(?:ed|s)?|include[ds]?|found|shown"
    r"|verified|documented|mentioned|discussed|explained?"
    r")\b"
    # explicit unavailability adjectives
    r"|\bunavailable\b|\bunverified\b"
    # "no details / no information ..."
    r"|\bno\s+(?:details|information|further|such)\b"
    # retrieval/knowledge failures
    r"|\bcould\s+not\s+be\s+(?:found|determined|verified|retrieved|located|established)\b"
    r"|\bcouldn't\s+(?:find|determine|verify|retrieve|locate|establish)\b"
    r"|\bunable\s+to\s+(?:find|determine|verify|retrieve|locate|provide|explain|cover|establish)\b"
    r"|\binsufficient\s+(?:information|details?|evidence|findings)\b"
    r")",
    re.IGNORECASE,
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


def _substantive_sentences(answer: str) -> list[str]:
    """Sentences that do not contain an unavailability disclaimer."""
    return [
        sentence
        for sentence in _SENTENCE_SPLIT_RE.split(answer or "")
        if sentence.strip() and not _DISCLAIMER_RE.search(sentence)
    ]


def check_completeness(question: str, answer: str) -> CompletenessReport:
    """Flag requested terms the answer never mentions substantively.

    A term counts as covered only when it appears (word-prefix,
    case-insensitive) in at least one sentence WITHOUT a known
    disclaimer phrasing. Mentions that merely disclaim availability
    do not count. This is still lexical — it makes no claim that
    the explanation is correct or adequate.
    """
    terms = _requested_terms(question)
    substantive = _substantive_sentences(answer or "")
    missing = tuple(
        term
        for term in terms
        if not any(
            re.search(rf"\b{re.escape(term)}", sentence, re.IGNORECASE)
            for sentence in substantive
        )
    )
    return CompletenessReport(requested_terms=terms, missing=missing)


def format_completeness_report(report: CompletenessReport) -> str:
    """LLM-readable report for the review prompt — with the
    coverage-is-not-adequacy disclaimer."""
    requested = ", ".join(report.requested_terms) or "(none)"
    missing = ", ".join(report.missing) or "(none)"
    return "\n".join(
        [
            "Topic-coverage check (deterministic LEXICAL matching only — a "
            "term counts as covered only when it appears outside known "
            "unavailability disclaimers; this does NOT verify that any "
            "explanation is correct, substantive or adequate — semantic "
            "adequacy is the reviewer's judgement):",
            f"- explicitly requested topic term(s): {requested}",
            f"- missing from the draft: {missing}",
            "result: "
            + (
                "PASS"
                if report.ok
                else "FAIL — the draft does not substantively address: "
                + missing
            ),
        ]
    )
