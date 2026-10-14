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
from typing import Any

__all__ = [
    "CompletenessReport",
    "ExplanationReport",
    "check_completeness",
    "check_explanation_depth",
    "format_completeness_report",
    "format_explanation_report",
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
#: available" or "… were not present in the retrieved text
#: excerpts" are not explanations of conditional routing. The
#: list is deterministic and conservative: a sentence that both
#: explains and disclaims is treated as a disclaimer (a false
#: "missing" only costs one more bounded research pass).
_DISCLAIMER_RE = re.compile(
    r"(?:"
    # negated availability/provision/presence: "was not available",
    # "were not present", "details were not included", "not covered",
    # "not contained", "not listed", "n't provided"
    r"\b(?:not|n't|never)\s+(?:be\s+)?(?:"
    r"available|provide[ds]?|present|contain(?:ed|s)?|cover(?:ed|s)?"
    r"|include[ds]?|listed|found|shown"
    r"|verified|documented|mentioned|discussed|explained?"
    r")\b"
    # explicit unavailability adjectives
    r"|\bunavailable\b|\bunverified\b|\babsent\b"
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


# --- explanation depth (Phase 4E.1, WS-B) -------------------------------
#
# Deterministic approximation of "the topic was NAMED but never
# EXPLAINED": a covering sentence that merely places the term in a
# generic-category definition ("Conditional routing is a core
# capability for directing flow") and contains no explanatory marker
# (subordinate clause, causal/conditional/structural word, or process
# verb) counts as naming-only. A term whose ONLY covering sentences
# are naming-only is SHALLOW — the reviewer is told, the draft gets a
# deterministic label, and follow-up research targets it.
#
# Boundaries (deliberate, documented):
# - it is LEXICAL, not semantic: it cannot judge whether an
#   explanation is correct, only whether one was attempted
# - it runs ONLY when at least one source page was actually
#   RETRIEVED — answers honestly labeled UNVERIFIED (snippets only)
#   keep the existing verification governance instead of also being
#   failed for depth, and simple questions with thin evidence are
#   never forced into extra research by this check
# - it runs ONLY for explanation-intent questions ("Explain…",
#   "How…", "What…"): a resource request such as "LangGraph
#   official documentation." is complete when it points at the
#   right source, not when it "explains" documentation
# - terms missing entirely stay in :func:`check_completeness`;
#   this check only catches the "mentioned but not explained" gap
# - one bounded follow-up pass is the worst case (the iteration
#   budget already bounds repeats)

#: Generic category nouns that classify a subject without explaining
#: it. "X is a capability/approach/..." states membership of a
#: category, not a mechanism.
_GENERIC_CATEGORY_NOUNS = (
    "capability", "feature", "ability", "concept", "approach",
    "type", "kind", "form", "way", "method", "framework",
    "mechanism", "technique", "option", "mode", "tool",
)

#: Copula + (article) + <=2 filler words + generic category noun.
#: Bounded window: only short definitional predicates count.
_NAMING_ONLY_RE = re.compile(
    r"\b(?:is|are|was|were)\s+(?:a|an|the)\b"
    r"[^.?!]{0,48}?\b(?:"
    + "|".join(_GENERIC_CATEGORY_NOUNS)
    + r")\b",
    re.IGNORECASE,
)

#: Explanatory markers: subordinate/relative clauses, causal and
#: conditional structure, and process verbs (stems). A sentence
#: containing any of these does something beyond naming the subject.
#: Deliberately does NOT include vague action paraphrases of the
#: topic itself (e.g. "directing flow" for routing) — those are the
#: superficial pattern this check exists to catch.
_EXPLANATION_MARKER_RE = re.compile(
    r"\b(?:that|which|when|where|whenever|if|then|because|so|via|"
    r"through|by|using|how)\b"
    r"|\b(?:call|invok|read|writ|persist|guard|rout|branch|return|"
    r"execut|run|emit|stor|valid|requir|enabl|prevent|check|select|"
    r"compar|generat|pars|fetch|send|keep|appl|redirect|wrap|add|"
    r"work|mean|allow|ensur|combin|integrat|transform|convert|"
    r"filter|merg|split|delegat|pass|loop|iterat|decid|evalu|"
    r"verif|handl|process|communicat)\w*\b",
    re.IGNORECASE,
)


def _is_naming_only(sentence: str, terms: tuple[str, ...] = ()) -> bool:
    """True when the sentence only classifies, never explains.

    ``terms`` (the question's requested topic vocabulary) are removed
    first: the marker list is there to find EXPLANATION, not more
    question vocabulary — a topic word that merely looks like a
    process verb ("routing" for ``rout*``, "checkpointing" for
    ``check*``) can never exempt its own superficial definition, and
    an answer that merely ECHOES the question never counts as
    substance. A sentence that reduces to almost nothing once the
    question's vocabulary is removed is naming-only by definition.
    """
    scope = sentence
    for term in terms:
        scope = re.sub(
            rf"\b{re.escape(term)}\w*", " ", scope, flags=re.IGNORECASE
        )
    if terms and len(scope.split()) < 4:
        return True
    if not _NAMING_ONLY_RE.search(scope):
        return False
    return _EXPLANATION_MARKER_RE.search(scope) is None


#: Questions this check applies to: explanation/intent wording.
#: Resource requests ("LangGraph official documentation.") are NOT
#: explanation demands — answering them with a pointer to the right
#: source is complete, and forcing an "explanation" would be wrong.
_EXPLANATION_INTENT_RE = re.compile(
    r"\b(?:explain|describes?|how|why|what|where|when|who|define|"
    r"works?|working|mechanism|difference|compar\w*|implications?|"
    r"impacts?|benefits?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExplanationReport:
    """Requested terms the draft only names or defines (WS-B)."""

    shallow: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.shallow


def check_explanation_depth(
    question: str, answer: str, findings: list[Any] | None = None
) -> ExplanationReport:
    """Flag explicitly requested terms that are named but not explained.

    Runs the lexical machinery of :func:`check_completeness` one level
    deeper: every requested term that IS covered (outside disclaimers)
    is checked for at least one covering sentence that does more than
    place the term in a generic-category definition. See the module
    section above for the exact boundaries — including why the check
    is skipped when no page was ever retrieved or the question does
    not ask for an explanation.
    """
    has_retrieved = any(
        str(finding.get("evidence")) == "retrieved"
        for finding in (findings or [])
        if isinstance(finding, dict)
    )
    if not has_retrieved:
        return ExplanationReport(shallow=())
    if not _EXPLANATION_INTENT_RE.search(question or ""):
        return ExplanationReport(shallow=())
    terms = _requested_terms(question)
    substantive = _substantive_sentences(answer or "")
    shallow: list[str] = []
    for term in terms:
        covering = [
            sentence
            for sentence in substantive
            if re.search(rf"\b{re.escape(term)}", sentence, re.IGNORECASE)
        ]
        if covering and all(
            _is_naming_only(sentence, terms) for sentence in covering
        ):
            shallow.append(term)
    return ExplanationReport(shallow=tuple(shallow))


def format_explanation_report(report: ExplanationReport) -> str:
    """LLM-readable report for the review prompt."""
    shallow = ", ".join(report.shallow) or "(none)"
    return "\n".join(
        [
            "Explanation-depth check (deterministic LEXICAL heuristic — it "
            "detects requested terms whose covering sentences only place "
            "them in a generic definition, with no explanatory marker; it "
            "does NOT judge whether any explanation is correct):",
            f"- named-but-not-explained requested term(s): {shallow}",
            "result: "
            + (
                "PASS"
                if report.ok
                else "FAIL — the draft names these topics without "
                "explaining how they work: " + shallow
            ),
        ]
    )
