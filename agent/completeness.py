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
    "CodeExampleReport",
    "CompletenessReport",
    "ExplanationReport",
    "check_code_example",
    "check_completeness",
    "check_explanation_depth",
    "format_code_example_report",
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

#: Instruction / framing verbs: they say what the assistant must DO,
#: never what the answer must COVER. "Include a practical example"
#: asks for a deliverable — "include" is not a topic, and treating it
#: as one produced the observed false "missing topic: include". This
#: is a generalizable CLASS (imperative request verbs), deliberately
#: not a single patched word, and it is distinct from the
#: question-framing stopwords above.
_INSTRUCTION_VERBS = frozenset(
    {
        "include", "includes", "including", "included",
        "provide", "provides", "providing", "provided",
        "add", "adds", "adding", "added",
        "show", "shows", "showing", "shown",
        "cover", "covers", "covering", "covered",
        "discuss", "discusses", "discussing", "discussed",
        "write", "writes", "writing", "written",
        "create", "creates", "creating", "created",
        "demonstrate", "demonstrates", "demonstrating", "demonstrated",
        "illustrate", "illustrates", "illustrating", "illustrated",
        "ensure", "ensures", "ensuring",
        "outline", "outlines", "outlining",
        "summarize", "summarizes", "summarizing", "summarized",
        "specify", "specifies", "specifying",
        "detail", "details", "detailed",
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


# --- requested deliverables (live-acceptance Problem B) -------------------
#
# The observed question asked for "a practical example, implementation
# considerations, and citations to official documentation" — three
# DELIVERABLES, not three ordinary topics. Matching them as isolated
# keywords ("practical", "example", "citation") both flagged answers
# that DID deliver them and said nothing about whether the deliverable
# was substantively present. A deliverable is therefore recognised in
# the question and verified structurally in the answer (a code fence
# for an example, citation markers for references, consideration
# vocabulary for trade-offs) — never by the bare presence of the word.


@dataclass(frozen=True)
class _Deliverable:
    """A requested deliverable: detected in the question, verified in
    the answer, and absorbed out of the ordinary topic terms."""

    key: str
    label: str
    #: matches the REQUEST in the question (the deliverable was asked
    #: for)
    request: re.Pattern[str]
    #: normalised question words that belong to the request and must
    #: NOT be treated as required topics (they are checked structurally
    #: instead)
    absorb: frozenset[str]
    #: matches SUBSTANTIVE presence in the answer
    present: re.Pattern[str]


_DELIVERABLES: tuple[_Deliverable, ...] = (
    _Deliverable(
        key="example",
        label="a requested example",
        request=re.compile(
            r"\b(?:examples?|samples?|snippets?|demos?|for\s+example|e\.g\.?)\b",
            re.IGNORECASE,
        ),
        # "practical"/"working" describe the example, not a topic
        absorb=frozenset({"example", "sample", "snippet", "demo", "practical"}),
        present=re.compile(
            # a real code block (runnable example), or inline code, or
            # an explicit example/instance/scenario construction
            r"```|`[^`\n]+`"
            r"|\bfor\s+example\b|\be\.g\.?|\bfor\s+instance\b"
            r"|\bsuch\s+as\b|\bsuch\s+an?\s+example\b|\binstance\b"
            r"|\bimagine\b|\bsuppose\b|\bscenario\b|\bwalk[- ]through\b"
            r"|\bstep\s+\d",
            re.IGNORECASE,
        ),
    ),
    _Deliverable(
        key="considerations",
        label="requested implementation considerations",
        request=re.compile(r"\b(?:considerations?|caveats?)\b", re.IGNORECASE),
        # "implementation" scopes WHICH considerations are requested —
        # it belongs to the deliverable, not to the topic terms (its
        # substance is verified by ``present`` below instead).
        absorb=frozenset({"consideration", "caveat", "implementation"}),
        present=re.compile(
            r"\bconsider\w*|\bcaveat\w*|\btrade-?off\w*|\blimitation\w*"
            r"|\bpitfall\w*|\bshould\b|\bmust\b|\bimportant\b|\bnote\s+that\b"
            r"|\bbe\s+careful\b|\bprefer\b|\bavoid\b|\bkeep\s+in\s+mind\b"
            r"|\bensure\b|\bin\s+practice\b|\bwatch\s+out\b|\bworth\s+(?:noting|keeping)\b",
            re.IGNORECASE,
        ),
    ),
    _Deliverable(
        key="citations",
        label="requested citations to sources",
        request=re.compile(
            r"\b(?:citations?|cited?|references?|sources?|links?)\b",
            re.IGNORECASE,
        ),
        absorb=frozenset({"citation", "cite", "cited"}),
        present=re.compile(
            # [S1]/[1] markers, a URL, or an attributive phrase
            r"\[\s*S?\d+\s*\]|\[S\d+\]|https?://|www\."
            r"|\(\d{4}\)|according\s+to|\bas\s+cited\b|\bcited\s+in\b",
            re.IGNORECASE,
        ),
    ),
)


def _detect_deliverables(question: str) -> list[_Deliverable]:
    """Deliverables this question explicitly asks for (order-stable)."""
    text = question or ""
    return [d for d in _DELIVERABLES if d.request.search(text)]


def _absorb_words(deliverables: list[_Deliverable]) -> frozenset[str]:
    words: set[str] = set()
    for deliverable in deliverables:
        words |= deliverable.absorb
    return frozenset(words)


def _content_term(raw: str, absorb: frozenset[str]) -> str | None:
    """Normalised topic term for a question token, or ``None``.

    Applies the same rules as :func:`_requested_terms` (length,
    stopwords, plural stripping, must contain a letter) plus the two
    Problem-B exclusions: instruction verbs and deliverable-absorbed
    words are never ordinary topics.
    """
    term = raw[:-1] if len(raw) >= 6 and raw.endswith("s") else raw
    if len(term) < 4:
        return None
    if term in _QUESTION_STOPWORDS or term in _INSTRUCTION_VERBS:
        return None
    if term in absorb:
        return None
    if not any(ch.isalpha() for ch in term):
        return None
    return term


def _topic_runs(question: str, absorb: frozenset[str]) -> list[tuple[str, ...]]:
    """Maximal runs of ADJACENT content words in the question.

    A run is a compound technical phrase as written — "state
    management", "conditional routing" — recognised as ONE unit rather
    than isolated tokens (the observed false "missing: management").
    Stopwords, short tokens, instruction verbs and deliverable words
    all break a run, AND so do punctuation boundaries (commas,
    semicolons, sentence terminators): "streaming, checkpointing and
    memory sharing" lists DISTINCT topics, so a comma separates them —
    merging list items into one phrase would let a genuinely absent
    topic hide behind a sibling that is present.
    """
    runs: list[tuple[str, ...]] = []
    current: list[str] = []
    for clause in re.split(r"[.?!;,]", (question or "").lower()):
        for raw in _TOKEN_RE.findall(clause):
            term = _content_term(raw, absorb)
            if term is None:
                if current:
                    runs.append(tuple(current))
                    current = []
                continue
            if term not in current:
                current.append(term)
        # a clause/punctuation boundary ends the current phrase run
        if current:
            runs.append(tuple(current))
            current = []
    if current:
        runs.append(tuple(current))
    return runs


def _topic_terms(question: str) -> tuple[str, ...]:
    """Distinct topic terms (first-appearance order) — the phrase runs
    flattened. This is what completeness reports as requested."""
    deliverables = _detect_deliverables(question)
    absorb = _absorb_words(deliverables)
    terms: list[str] = []
    for run in _topic_runs(question, absorb):
        for term in run:
            if term not in terms:
                terms.append(term)
    return tuple(terms)


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
    """Flag requested topics the answer never mentions substantively.

    Problem-B correction (generalizable, not word-specific):

    - a compound technical PHRASE ("state management", "conditional
      routing") is a single unit — it counts as covered when all but
      one of its adjacent content words appear, so an answer that
      explains the concept without echoing every isolated token is not
      falsely flagged, while a run whose words are almost entirely
      absent (the existing "conditional routing" regressions) still is;
    - instruction verbs ("include", "provide", …) are never topics;
    - requested DELIVERABLES (example, considerations, citations) are
      verified structurally — a code fence, citation markers or
      consideration vocabulary — not by the bare word.

    A single content word with no neighbour stays strictly required
    (unchanged behaviour), and every mention must be outside a known
    disclaimer. This is still lexical: it makes no claim that a covered
    topic is correct or adequate.
    """
    deliverables = _detect_deliverables(question)
    absorb = _absorb_words(deliverables)
    runs = _topic_runs(question, absorb)
    requested: list[str] = []
    for run in runs:
        for term in run:
            if term not in requested:
                requested.append(term)

    substantive = _substantive_sentences(answer or "")
    missing: list[str] = []
    for run in runs:
        # A phrase is covered when all but at most ONE of its content
        # words is present: "state management" is addressed if the
        # answer substantively discusses "state" even without the
        # literal token "management". A lone word needs itself.
        need = max(1, len(run) - 1)
        present = [
            term
            for term in run
            if any(
                re.search(rf"\b{re.escape(term)}", sentence, re.IGNORECASE)
                for sentence in substantive
            )
        ]
        if len(present) < need:
            for term in run:
                if term not in present and term not in missing:
                    missing.append(term)

    # Requested deliverables: verified structurally in the whole answer
    # (a code fence is not a "sentence", and citation markers live
    # wherever the answer put them).
    answer_text = answer or ""
    for deliverable in deliverables:
        if not deliverable.present.search(answer_text) and (
            deliverable.label not in missing
        ):
            missing.append(deliverable.label)

    return CompletenessReport(
        requested_terms=tuple(requested), missing=tuple(missing)
    )


def format_completeness_report(report: CompletenessReport) -> str:
    """LLM-readable report for the review prompt — with the
    coverage-is-not-adequacy disclaimer."""
    requested = ", ".join(report.requested_terms) or "(none)"
    missing = ", ".join(report.missing) or "(none)"
    lines = [
        "Topic-coverage check (deterministic LEXICAL matching only — a "
        "term counts as covered only when it appears outside known "
        "unavailability disclaimers; compound phrases are matched as "
        "units and requested deliverables structurally; this does NOT "
        "verify that any explanation is correct, substantive or "
        "adequate — semantic adequacy is the reviewer's judgement):",
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
    return "\n".join(lines)


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


# --- code-example completeness (live-acceptance Problem C) ---------------
#
# The observed defect: a question asked for a practical working example
# and the drafted LangGraph code commented out the required node
# registrations and never established a complete graph entry path — it
# READ like an example but could not run. This check reports what the
# code text structurally contains (definitions, commented-out
# statements, invocation) so a misleadingly incomplete example is
# flagged. It is a STRUCTURAL heuristic, deliberately: it never claims
# the code was executed or is API-correct — semantic adequacy and API
# correctness stay with the reviewer and the cited documentation.

_EXEC_METHOD_RE = re.compile(
    r"\.(?:invoke|stream|run|execute|predict|process|apply|start)\s*\("
)
_MAIN_GUARD_RE = re.compile(r"if\s+__name__")
_BARE_CALL_RE = re.compile(r"^\s*[A-Za-z_]\w*\([^=]*\)\s*$")
_DEFINED_NAME_RE = re.compile(r"^\s*(?:def|class)\s+([A-Za-z_]\w*)")
_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+\w")


def _code_blocks(answer: str) -> list[str]:
    """Contents of every fenced code block (language tag stripped)."""
    return re.findall(r"```[^\n`]*\n(.*?)```", answer or "", re.DOTALL)


def _is_stmt_comment(line: str) -> bool:
    """True when a comment line looks like commented-out CODE.

    A descriptive comment ("# build the graph") is prose; a commented-out
    statement ("# graph.add_node('a', a)") is suppressed code. We
    distinguish them by whether the comment body STARTS like a statement.
    """
    body = line.strip().lstrip("#").strip()
    if not body:
        return False
    return (
        re.match(
            r"(?:def|class|import|from|return|if|elif|else|for|while|with|"
            r"try|except|raise|yield|pass)\b"
            r"|[\w.]+\s*=|[\w.]+\s*\(|\.[\w]+\s*\(",
            body,
        )
        is not None
    )


@dataclass(frozen=True)
class CodeExampleReport:
    """Structural verdict on a requested, code-bearing example."""

    requested: bool
    blocks: int
    has_import: bool
    has_definition: bool
    has_invocation: bool
    commented_out: bool
    reasons: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.reasons


def check_code_example(question: str, answer: str) -> CodeExampleReport:
    """Flag a requested example that is illustrative rather than runnable.

    Runs only when the question explicitly requested an example AND the
    answer actually contains a code block (an absent example is the
    completeness deliverable's job, not this one). Signals, aggregated
    across every block so usage in a later block counts:

    - definitions (``def`` / ``class``) present;
    - an INVOCATION — a known run method (``.invoke``/``.stream``/…), a
      ``main()``/``__main__`` guard, a bare call statement, or a call to a
      name defined in the block (``total = add(1, 2)`` after ``def add``);
      construction of an EXTERNAL class (``g = StateGraph(S)``) is
      deliberately NOT an invocation;
    - commented-out statements (suppressed definitions / wiring).

    A requested example is flagged when it has NO invocation yet either
    defines code (dead definitions) or has commented-out structure — the
    exact shape of the observed defect.
    """
    example_requested = any(d.key == "example" for d in _detect_deliverables(question))
    blocks = _code_blocks(answer)
    if not example_requested or not blocks:
        return CodeExampleReport(
            requested=example_requested,
            blocks=len(blocks),
            has_import=False,
            has_definition=False,
            has_invocation=False,
            commented_out=False,
            reasons=(),
        )

    active: list[str] = []
    comments: list[str] = []
    for block in blocks:
        for line in block.splitlines():
            if not line.strip():
                continue
            if line.strip().startswith("#"):
                comments.append(line)
            else:
                active.append(line)

    defined = {
        match.group(1)
        for block in blocks
        for match in (
            _DEFINED_NAME_RE.match(line) for line in block.splitlines()
        )
        if match
    }

    has_import = any(_IMPORT_RE.match(line) for line in active)
    has_definition = bool(defined)
    commented_out = any(_is_stmt_comment(line) for line in comments)

    def _is_invocation(line: str) -> bool:
        if line.lstrip().startswith(("def ", "class ")):
            return False
        if _EXEC_METHOD_RE.search(line) or _MAIN_GUARD_RE.search(line):
            return True
        if _BARE_CALL_RE.match(line):
            return True
        return any(
            re.search(rf"\b{re.escape(name)}\s*\(", line) for name in defined
        )

    has_invocation = any(_is_invocation(line) for line in active)

    reasons: list[str] = []
    if not has_invocation and commented_out:
        reasons.append(
            "definitions or wiring are commented out and there is no "
            "invocation, so the example is illustrative rather than runnable"
        )
    elif not has_invocation and has_definition:
        reasons.append(
            "the example defines functions/classes but never invokes them, "
            "so there is no entry point that actually runs"
        )

    return CodeExampleReport(
        requested=example_requested,
        blocks=len(blocks),
        has_import=has_import,
        has_definition=has_definition,
        has_invocation=has_invocation,
        commented_out=commented_out,
        reasons=tuple(reasons),
    )


def format_code_example_report(report: CodeExampleReport) -> str:
    """LLM-readable report for the review prompt."""
    if not report.requested:
        return "Code-example check: (no example requested)"
    if not report.blocks:
        return (
            "Code-example check: an example was requested but the draft "
            "contains no code block (see the topic-coverage check)."
        )
    flags = ", ".join(
        sorted(
            {
                "imports" if report.has_import else None,
                "definitions" if report.has_definition else None,
                "invocation" if report.has_invocation else None,
                "commented-out code" if report.commented_out else None,
            }
            - {None}
        )
    ) or "nothing substantive"
    return "\n".join(
        [
            "Code-example check (deterministic STRUCTURAL heuristic — it "
            "reports what the code text contains; it does NOT prove the "
            "code executes or is API-correct):",
            f"- code block(s): {report.blocks}; present: {flags}",
            "result: "
            + (
                "PASS"
                if report.ok
                else "FAIL — requested example looks incomplete: "
                + "; ".join(report.reasons)
            ),
        ]
    )
