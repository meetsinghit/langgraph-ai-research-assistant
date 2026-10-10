"""Deterministic citation validation: URL provenance, not semantics.

This module answers ONE narrow question about a final answer: do the
URLs it cites correspond to sources the research actually gathered,
and how strong is each cited source's provenance (``retrieved`` page
fetched/inspected vs ``snippet`` search excerpt)?

It deliberately does NOT verify that the cited sources factually
support the claims made about them — that is a semantic judgement the
review node's LLM makes, guided by this module's report. URL matching
proves provenance, never correctness.

It also hosts the first-party/official-source heuristics, which are
shared by the graph's verification ordering and by citation checking.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

from agent.state import Finding

__all__ = [
    "Citation",
    "CitationReport",
    "extract_cited_urls",
    "citation_note",
    "format_citation_report",
    "is_first_party",
    "requests_official_sources",
    "validate_citations",
]

# --- official / first-party heuristics ----------------------------

_OFFICIAL_REQUEST_RE = re.compile(r"\b(official|first[- ]party)\b", re.IGNORECASE)

#: Curated list of first-party documentation domains (heuristic —
#: a domain on this list, or any of its subdomains, is treated as
#: an official source when the question asks for official docs).
_FIRST_PARTY_DOC_DOMAINS = frozenset(
    {
        "langchain-ai.github.io",
        "ai.google.dev",
        "developers.google.com",
        "cloud.google.com",
        "openai.com",
        "platform.openai.com",
        "python.org",
        "docs.python.org",
        "learn.microsoft.com",
        "developer.mozilla.org",
        "docs.aws.amazon.com",
        "huggingface.co",
        "pytorch.org",
        "react.dev",
        "kubernetes.io",
        "docs.github.com",
        "fastapi.tiangolo.com",
        "pandas.pydata.org",
        "numpy.org",
    }
)

_GENERIC_TOKENS = frozenset(
    {
        "about", "after", "again", "also", "and", "any", "api", "are",
        "based", "best", "between", "does", "doc", "docs", "documentation",
        "for", "from", "guide", "guides", "have", "help", "how", "into",
        "its", "latest", "official", "party", "reference", "references",
        "should", "site", "some", "source", "sources", "than", "that",
        "their", "there", "these", "they", "this", "those", "using",
        "want", "was", "were", "what", "when", "where", "which", "with",
        "would", "your", "website",
    }
)


def requests_official_sources(question: str) -> bool:
    """True when the user explicitly asks for official/first-party sources."""
    return bool(_OFFICIAL_REQUEST_RE.search(question))


def _product_tokens(question: str) -> list[str]:
    """Likely product/vendor names in the question (heuristic)."""
    return [
        token
        for token in re.findall(r"[a-z0-9]+", question.lower())
        if len(token) >= 4 and token not in _GENERIC_TOKENS
    ]


def is_first_party(url: str, question: str) -> bool:
    """Heuristic first-party check for a source URL.

    A URL is first-party when its hostname is a curated official
    documentation domain, or when a product token from the question
    appears in the hostname (e.g. ``fastapi`` in
    ``fastapi.tiangolo.com``).
    """
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False
    if not host:
        return False
    if host.startswith("www."):
        host = host[4:]
    for domain in _FIRST_PARTY_DOC_DOMAINS:
        if host == domain or host.endswith("." + domain):
            return True
    return any(token in host for token in _product_tokens(question))


# --- URL extraction & normalization -------------------------------

#: URLs in answer text: bare, Markdown links ``[text](url)`` and
#: ``<url>``. ``)`` ``]`` ``}`` terminate a match so Markdown link
#: syntax is not swallowed; URLs containing those characters are
#: truncated (documented heuristic — rare for sourced pages).
_URL_RE = re.compile(r"https?://[^\s<>\"'\)\]\}]+", re.IGNORECASE)
_TRAILING_PUNCT_RE = re.compile(r"[.,;:!?]+$")


def extract_cited_urls(text: str) -> list[str]:
    """URLs cited in ``text``, deduplicated, order-preserving."""
    seen: set[str] = set()
    urls: list[str] = []
    for match in _URL_RE.findall(text or ""):
        url = _TRAILING_PUNCT_RE.sub("", match)
        if not url or url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def _normalize_url(url: str) -> str:
    """Citation-identity key for a URL.

    Scheme (http vs https), ``www.`` prefixes, fragments and query
    strings are ignored; host (lowercased, non-default port kept)
    and trailing-slash-stripped path must match. This is a forgiving
    heuristic: it reduces false "unknown citation" flags for trivial
    variants of a real source URL at the cost of collapsing pages
    distinguished only by a query string.
    """
    try:
        parsed = urlparse(url.strip())
    except ValueError:
        return url.strip().lower()
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    port = f":{parsed.port}" if parsed.port and parsed.port not in (80, 443) else ""
    path = parsed.path.rstrip("/")
    return f"{host}{port}{path}"


# --- validation ---------------------------------------------------

_EVIDENCE_RANK = {"snippet": 1, "retrieved": 2}
EvidenceStatus = Literal["retrieved", "snippet", "unknown"]


@dataclass(frozen=True)
class Citation:
    """One URL cited in the answer and its provenance status."""

    url: str
    status: EvidenceStatus


@dataclass(frozen=True)
class CitationReport:
    """Result of deterministic citation-provenance validation."""

    citations: tuple[Citation, ...]
    official_required: bool
    official_retrieved_available: tuple[str, ...]
    official_cited: tuple[str, ...]
    reasons: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.reasons

    @property
    def has_citations(self) -> bool:
        return bool(self.citations)

    @property
    def retrieved(self) -> list[str]:
        return [c.url for c in self.citations if c.status == "retrieved"]

    @property
    def snippet_only(self) -> list[str]:
        return [c.url for c in self.citations if c.status == "snippet"]

    @property
    def unknown(self) -> list[str]:
        return [c.url for c in self.citations if c.status == "unknown"]


def _source_index(findings: list[Finding]) -> dict[str, str]:
    """Normalized source URL -> strongest recorded evidence class."""
    index: dict[str, str] = {}
    for finding in findings:
        url = finding.get("source_url")
        if not url:
            continue
        key = _normalize_url(url)
        evidence = str(finding.get("evidence", "snippet"))
        if (
            key not in index
            or _EVIDENCE_RANK.get(evidence, 0) > _EVIDENCE_RANK.get(index[key], 0)
        ):
            index[key] = evidence
    return index


def validate_citations(
    answer: str, findings: list[Finding], question: str
) -> CitationReport:
    """Compare the URLs cited in ``answer`` with recorded sources.

    Deterministic rules (see module docstring: this is provenance
    matching, NOT factual verification):

    - a cited URL not matching any recorded source is ``unknown``
      and fails the check (the agent must never cite sources it did
      not gather)
    - a cited URL matching only a ``snippet`` finding is flagged
      ``snippet`` — unverified, not automatically verified
    - when the question asks for official documentation and a
      retrieved first-party source was gathered, the check fails
      unless the answer cites at least one of those retrieved
      official sources
    """
    index = _source_index(findings)
    citations_list: list[Citation] = []
    for url in extract_cited_urls(answer):
        evidence = index.get(_normalize_url(url))
        status: EvidenceStatus = "unknown" if evidence is None else _as_status(evidence)
        citations_list.append(Citation(url=url, status=status))
    citations = tuple(citations_list)

    official_required = requests_official_sources(question)
    official_retrieved_available = tuple(
        sorted(
            {
                finding["source_url"]
                for finding in findings
                if finding.get("source_url")
                and finding.get("evidence") == "retrieved"
                and is_first_party(finding["source_url"], question)
            }
        )
    ) if official_required else ()
    official_cited = tuple(
        citation.url
        for citation in citations
        if citation.status == "retrieved"
        and is_first_party(citation.url, question)
    ) if official_required else ()

    reasons: list[str] = []
    unknown_count = sum(1 for c in citations if c.status == "unknown")
    if unknown_count:
        reasons.append(
            f"{unknown_count} cited URL(s) are not among the recorded sources"
        )
    if official_required and official_retrieved_available and not official_cited:
        reasons.append(
            "the question asks for official documentation and retrieved "
            "official source(s) exist, but the answer cites none of them"
        )
    return CitationReport(
        citations=citations,
        official_required=official_required,
        official_retrieved_available=official_retrieved_available,
        official_cited=official_cited,
        reasons=tuple(reasons),
    )


def _as_status(evidence: str) -> EvidenceStatus:
    """Map a recorded evidence class to a citation status.

    Anything that is not a recorded ``retrieved`` page counts as
    ``snippet`` (unverified) — conservative by design.
    """
    return "retrieved" if evidence == "retrieved" else "snippet"


def _listing(urls: tuple[str, ...] | list[str]) -> str:
    return ", ".join(urls) if urls else "(none)"


def format_citation_report(report: CitationReport) -> str:
    """LLM-readable report for the review prompt (with the
    provenance-vs-semantics disclaimer required for honesty)."""
    return "\n".join(
        [
            "Citation validation (deterministic URL/provenance matching "
            "only — it does NOT verify that the cited sources factually "
            "support the claims):",
            (
                f"- cited URL(s): {len(report.citations)} "
                f"({len(report.retrieved)} retrieved, "
                f"{len(report.snippet_only)} snippet-only/unverified, "
                f"{len(report.unknown)} unknown)"
            ),
            f"- retrieved citations: {_listing(report.retrieved)}",
            f"- snippet-only (unverified) citations: {_listing(report.snippet_only)}",
            f"- unknown citations (not among recorded sources): {_listing(report.unknown)}",
            (
                f"- official documentation asked for: "
                f"{'yes' if report.official_required else 'no'}; retrieved "
                f"official source(s) available: "
                f"{_listing(report.official_retrieved_available)}; cited "
                f"official retrieved source(s): "
                f"{_listing(report.official_cited)}"
            ),
            "result: "
            + ("PASS" if report.ok else "FAIL — " + "; ".join(report.reasons)),
        ]
    )


def citation_note(report: CitationReport) -> str:
    """Deterministic marking for answers that failed validation.

    Counts and reasons only — never the offending URLs, so the note
    cannot feed URLs back into a later validation pass.
    """
    return (
        "> ⚠️ **CITATION QUALITY: INSUFFICIENTLY VERIFIED** — "
        + "; ".join(report.reasons)
        + ". This deterministic check compares cited URLs against "
        "recorded source provenance only; it does NOT verify that the "
        "sources factually support the claims."
    )
