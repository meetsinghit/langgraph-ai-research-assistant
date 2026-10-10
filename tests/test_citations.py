"""Unit tests for deterministic citation validation (agent.citations).

These tests cover URL/provenance matching ONLY: whether URLs cited in
an answer correspond to recorded sources and how strong each cited
source's provenance is. They deliberately assert the module's own
disclaimer too — a URL match never proves a claim factually supported.
"""
from agent.citations import (
    assign_source_ids,
    citation_note,
    extract_cited_urls,
    format_citation_report,
    format_source_refs,
    is_first_party,
    requests_official_sources,
    validate_citations,
)

OFFICIAL = "https://langchain-ai.github.io/langgraph/docs/"
THIRD_PARTY = "https://blog.example.com/langgraph-tutorial"


def _finding(url: str, evidence: str, text: str = "finding") -> dict:
    return {"text": text, "source_url": url, "evidence": evidence}


# --- URL extraction & normalization -------------------------------


def test_extract_cited_urls_handles_bare_markdown_and_trailing_punct():
    text = (
        "See https://example.com/a, then [the guide](https://example.com/b.) "
        "and <https://example.com/a> — also https://example.com/c."
    )
    assert extract_cited_urls(text) == [
        "https://example.com/a",
        "https://example.com/b",
        "https://example.com/c",
    ]


def test_extract_cited_urls_from_empty_text():
    assert extract_cited_urls("") == []
    assert extract_cited_urls("no links here at all.") == []


def test_trivial_url_variants_match_the_recorded_source():
    """Scheme, www., fragment and trailing slash are forgiven when
    matching a cited URL against a recorded source URL."""
    findings = [_finding("https://www.example.com/docs/", "retrieved")]
    answer = "See http://example.com/docs#intro for the manual."
    report = validate_citations(answer, findings, question="question")
    assert report.ok
    assert report.retrieved == ["http://example.com/docs#intro"]
    assert report.unknown == []


# --- valid / unknown / snippet / mixed ----------------------------


def test_valid_retrieved_citation_passes():
    findings = [_finding("https://example.com/a", "retrieved")]
    report = validate_citations(
        "Details at https://example.com/a.", findings, question="question"
    )
    assert report.ok
    assert report.has_citations
    assert report.citations[0].status == "retrieved"
    assert report.unknown == []


def test_unknown_citation_fails_the_check():
    findings = [_finding("https://example.com/a", "retrieved")]
    report = validate_citations(
        "See https://other.example/page.", findings, question="question"
    )
    assert not report.ok
    assert report.unknown == ["https://other.example/page"]
    assert any("not among the recorded sources" in r for r in report.reasons)


def test_snippet_only_citation_is_flagged_unverified_not_verified():
    """A citation backed only by a search snippet is classified
    'snippet' (unverified) — it is never auto-promoted to verified."""
    findings = [_finding("https://example.com/a", "snippet")]
    report = validate_citations(
        "See https://example.com/a.", findings, question="question"
    )
    assert report.citations[0].status == "snippet"
    assert report.snippet_only == ["https://example.com/a"]
    assert report.retrieved == []
    # flagged, but for a non-official question snippet backing alone
    # is not a hard failure (the retrieved threshold handles that)
    assert report.ok


def test_mixed_evidence_reports_each_class():
    findings = [
        _finding("https://example.com/a", "retrieved"),
        _finding("https://example.com/b", "snippet"),
    ]
    answer = (
        "Compare https://example.com/a, https://example.com/b and "
        "https://other.example/page."
    )
    report = validate_citations(answer, findings, question="question")
    statuses = {c.url: c.status for c in report.citations}
    assert statuses == {
        "https://example.com/a": "retrieved",
        "https://example.com/b": "snippet",
        "https://other.example/page": "unknown",
    }
    # the unknown citation fails the check; the snippet citation is
    # flagged unverified without failing on its own
    assert not report.ok
    assert report.snippet_only == ["https://example.com/b"]
    assert report.retrieved == ["https://example.com/a"]


def test_same_source_with_mixed_evidence_counts_as_retrieved():
    """A source both surfaced as a snippet and fetched later is
    indexed at its strongest evidence: retrieved."""
    findings = [
        _finding("https://example.com/a", "snippet"),
        _finding("https://example.com/a", "retrieved"),
    ]
    report = validate_citations(
        "See https://example.com/a.", findings, question="question"
    )
    assert report.citations[0].status == "retrieved"
    assert report.ok


def test_answer_without_citations_passes_for_non_official_question():
    report = validate_citations(
        "A plain answer with no URLs.",
        [_finding("https://example.com/a", "retrieved")],
        question="question",
    )
    assert not report.has_citations
    assert report.ok


# --- official-documentation requirement ---------------------------


def _official_findings() -> list[dict]:
    return [
        _finding(OFFICIAL, "retrieved", "official docs page"),
        _finding(THIRD_PARTY, "snippet", "third-party tutorial"),
    ]


def test_official_question_citing_retrieved_official_source_passes():
    report = validate_citations(
        f"Docs at {OFFICIAL}", _official_findings(), question="LangGraph official documentation"
    )
    assert report.official_required
    assert report.official_retrieved_available == (OFFICIAL,)
    assert report.official_cited == (OFFICIAL,)
    assert report.ok


def test_official_question_answering_from_third_party_snippets_only_fails():
    """Retrieved official source exists, but the answer cites only a
    snippet-based third-party URL: the citation-quality check fails."""
    report = validate_citations(
        f"Try this tutorial: {THIRD_PARTY}",
        _official_findings(),
        question="LangGraph official documentation",
    )
    assert report.official_required
    assert report.official_retrieved_available == (OFFICIAL,)
    assert report.official_cited == ()
    assert not report.ok
    assert any("official" in r for r in report.reasons)


def test_official_question_citing_nothing_fails_when_official_available():
    report = validate_citations(
        "An answer that cites nothing.",
        _official_findings(),
        question="LangGraph official documentation",
    )
    assert not report.ok


def test_official_reason_absent_when_no_official_source_was_retrieved():
    """If official docs were never retrieved (only snippets), the
    official-citation rule cannot fire — the UNVERIFIED/verification
    mechanisms cover that case instead."""
    findings = [_finding(THIRD_PARTY, "snippet")]
    report = validate_citations(
        f"Try {THIRD_PARTY}",
        findings,
        question="LangGraph official documentation",
    )
    assert report.official_required
    assert report.official_retrieved_available == ()
    assert report.ok


def test_official_reason_not_required_for_plain_questions():
    report = validate_citations(
        "See https://example.com/a",
        [_finding("https://example.com/a", "retrieved")],
        question="how do retries work?",
    )
    assert not report.official_required
    assert report.official_retrieved_available == ()
    assert report.official_cited == ()
    assert report.ok


# --- report / note honesty ----------------------------------------


def test_report_disclaims_semantic_verification():
    """Requirement: URL matching must never be presented as proof of
    factual correctness."""
    report = validate_citations(
        "https://other.example/page",
        [_finding("https://example.com/a", "retrieved")],
        question="question",
    )
    formatted = format_citation_report(report)
    assert "does NOT verify" in formatted
    assert "FAIL" in formatted
    assert "PASS" in format_citation_report(
        validate_citations(
            "https://example.com/a",
            [_finding("https://example.com/a", "retrieved")],
            question="question",
        )
    )


def test_citation_note_is_counts_only_and_carries_the_disclaimer():
    """The deterministic marking lists reasons/counts, never the
    offending URLs, so it cannot feed URLs back into a later
    validation pass."""
    findings = [_finding("https://example.com/a", "retrieved")]
    report = validate_citations(
        "See https://other.example/page", findings, question="question"
    )
    note = citation_note(report)
    assert "CITATION QUALITY" in note
    assert "https://other.example/page" not in note
    assert "1 cited URL(s)" in note
    assert "does NOT verify" in note


def test_citation_note_not_generated_for_passing_reports():
    report = validate_citations(
        "See https://example.com/a",
        [_finding("https://example.com/a", "retrieved")],
        question="question",
    )
    assert report.ok
    assert report.reasons == ()


# --- moved official-source heuristics -----------------------------


def test_requests_official_sources_heuristic():
    assert requests_official_sources("LangGraph official documentation")
    assert requests_official_sources("the first-party API reference")
    assert not requests_official_sources("how do retries work?")


def test_is_first_party_heuristic():
    question = "LangGraph official documentation"
    assert is_first_party("https://langchain-ai.github.io/langgraph/", question)
    assert is_first_party("https://ai.google.dev/gemini-api/docs", question)
    assert not is_first_party(THIRD_PARTY, question)
    assert not is_first_party("", question)


# --- canonical source identifiers & malformed citations -----------


def test_source_ids_are_deterministic_and_use_strongest_evidence():
    """Conflicting snippet/retrieved labels for the same URL resolve
    to ONE identifier at its strongest recorded evidence — the
    attribution inconsistency the live run showed (answer said
    [snippet], source list said [retrieved]) cannot happen."""
    findings = [
        _finding("https://example.com/a", "snippet"),
        _finding("https://example.com/b", "snippet"),
        _finding("https://example.com/a", "retrieved"),  # conflict
    ]
    refs = assign_source_ids(findings)
    assert [(r.sid, r.url, r.evidence) for r in refs] == [
        ("S1", "https://example.com/a", "retrieved"),
        ("S2", "https://example.com/b", "snippet"),
    ]


def test_source_refs_render_labels_separately_from_citations():
    """Provenance labels are application-rendered — identifier first,
    URL, then the label in its own bracket."""
    findings = [
        _finding("https://example.com/a", "retrieved"),
        _finding("https://example.com/b", "snippet"),
    ]
    rendered = format_source_refs(assign_source_ids(findings))
    assert rendered == (
        "- [S1] https://example.com/a [retrieved]\n"
        "- [S2] https://example.com/b [snippet]"
    )
    assert format_source_refs(()) == "(no sources)"


def test_sid_citations_resolve_to_recorded_sources():
    findings = [_finding("https://example.com/a", "retrieved")]
    report = validate_citations("See [S1] for details.", findings, "question")
    assert report.ok
    assert report.retrieved == ["[S1]"]
    assert report.citations[0].url == "https://example.com/a"


def test_unknown_sid_is_not_silently_accepted():
    findings = [_finding("https://example.com/a", "retrieved")]
    report = validate_citations("See [S9].", findings, "question")
    assert not report.ok
    assert report.unknown == ["[S9]"]


def test_malformed_citation_flagged_and_uses_recorded_evidence():
    """Live regression: the model pasted '[snippet]' inside the URL
    even though the source was recorded [retrieved]. The recorded
    evidence wins; the malformed rendering fails the check."""
    findings = [_finding("https://docs.example.com/overview", "retrieved")]
    answer = "See [https://docs.example.com/overview [snippet]]."
    report = validate_citations(answer, findings, "question")
    assert report.malformed == ("https://docs.example.com/overview",)
    assert report.retrieved == ["https://docs.example.com/overview"]
    assert not report.ok
    assert any("malformed" in r for r in report.reasons)
    assert "malformed" in format_citation_report(report)
    assert "malformed" in citation_note(report)


def test_live_wrapped_malformed_citation_repairs_and_maps():
    official = "https://docs.langchain.com/oss/python/langgraph/overview"
    findings = [_finding(official, "retrieved")]
    answer = f"See [{official} [snippet]] for details."
    report = validate_citations(answer, findings, "question")
    assert report.malformed == (official,)
    assert report.retrieved == [official]  # recorded evidence, not the label
    assert not report.ok


def test_clean_citations_are_never_flagged_malformed():
    findings = [
        _finding("https://example.com/a", "retrieved"),
        _finding("https://example.com/b", "snippet"),
    ]
    answer = (
        "As [S1] shows, and per https://example.com/b "
        "(unverified), see also [1] for context."
    )
    report = validate_citations(answer, findings, "question")
    assert report.malformed == ()
    assert report.ok


def test_official_source_cited_by_identifier_satisfies_the_gate():
    findings = [
        _finding(THIRD_PARTY, "snippet"),
        _finding(OFFICIAL, "retrieved"),
    ]
    report = validate_citations(
        "See [S2] for LangGraph details.",
        findings,
        question="LangGraph official documentation",
    )
    assert report.official_cited == (OFFICIAL,)
    assert report.ok
