# AI Research Assistant (POC)

A modular LangGraph agent that answers questions with verifiable,
web-sourced findings. Built as a phased learning project in
practical Agentic AI engineering.

## Quick start

```powershell
# from the project root (AI_POC)
.\ai_poc\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # then edit .env
python main.py "What is LangGraph?"
```

## Configuration (.env)

| Variable | Purpose | Default |
| --- | --- | --- |
| `LLM_PROVIDER` | `gemini` \| `openrouter` \| `openai` | `gemini` |
| `LLM_MODEL` | model name | `gemini-3.8-flash` |
| `GEMINI_API_KEY` | free-tier key from aistudio.google.com/apikey | – |
| `OPENROUTER_API_KEY` | key from openrouter.ai/keys (free models exist) | – |
| `OPENAI_API_KEY` | paid key — only used with `ALLOW_PAID_PROVIDERS=true` | – |
| `ALLOW_PAID_PROVIDERS` | explicit opt-in for paid providers | `false` |
| `REQUEST_TIMEOUT_SECONDS` | per-request timeout | `30` |
| `SSL_CA_BUNDLE` | optional PEM CA bundle used to **verify** page-fetch certificates (e.g. enterprise TLS-inspection root); unset = OS trust store via `truststore`, then certifi — verification is never disabled | – |
| `MAX_RESEARCH_ITERATIONS` | research passes (graph-level budget) | `3` |
| `MIN_FINDINGS` | findings required to synthesize | `3` |
| `MIN_RETRIEVED_SOURCES` | source **pages** that must be fetched/inspected before an answer counts as verified (snippets never satisfy it; `0` disables verification) | `1` |
| `MAX_TOOL_CALLS_PER_RESEARCH` | executed tool invocations per research pass (counts batches in one LLM response); split ~half search / ~half fetch | `4` |
| `MAX_EXECUTION_SECONDS` | wall-clock budget **per research pass** (each pass gets its own budget; the full run is bounded by `MAX_RESEARCH_ITERATIONS` × this, plus plan/synthesize/review LLM calls, which it does not cover) | `120` |
| `MAX_REDIRECTS` / `MAX_PAGE_BYTES` | fetch safety limits | `5` / `1000000` |
| `MAX_EVIDENCE_CHARS` | per-source text budget injected into model prompts (snippet + page, truncated with an explicit marker beyond it) — keeps prompt size and input tokens bounded regardless of page size; full text stays in state for citations | `6000` |

## Running

- `python main.py "question"` — research, then print the answer,
  sources, iteration count, token usage and estimated cost
- `python main.py "question" --metrics run.json` — additionally
  write the machine-readable metrics snapshot (per-node/per-call
  latency, prompt sizes, token counts, tool outcomes; sizes and
  counts only — never content)
- `python main.py --verify-connection` — opt-in live smoke test
  (exactly one minimal request to the configured provider)
- `python -m pytest` — full mocked suite (no network, no API keys)
- `python -m evals` — offline evaluation benchmark (21 scripted
  cases, no network, no API keys; see below)
- `RUN_LIVE_TESTS=1 python -m pytest -m live` — opt-in live
  integration test (requires a configured provider key)

## Architecture

```
START → plan → research ⇢(conditional)⇢ research | synthesize | handle_error
                synthesize → review ⇢(conditional)⇢ research | finalize
                finalize → END,  handle_error → END
```

- **State** (`agent/state.py`): `GraphState` TypedDict. List fields
  use reducers (append semantics); scalar fields are last-write-wins.
- **Routing** (`agent/routing.py`): pure, deterministic functions —
  budgets, error handling and termination policy live here and never
  call the LLM.
- **LLM layer** (`agent/llm.py`): provider factory — Gemini (native
  `langchain-google-genai` integration, free tier), OpenRouter
  (OpenAI-compatible client), OpenAI (paid, disabled by default).
- **Tools** (`agent/tools.py`): `web_search` (DuckDuckGo via
  the maintained `ddgs` package, free; an empty result set is
  a meaningful error, not silent success) and
  `fetch_page` (SSRF-hardened: http/https only, localhost/private-IP
  blocking including DNS-resolution checks, per-hop redirect
  validation, timeouts, size caps).
- **Graph** (`agent/graph.py`): `plan` (LLM structured output),
  `research` (bounded, phase-partitioned ReAct loop — the tool
  budget is deterministically split between discovery and
  verification: the model is bound to only the currently-allowed
  tools, out-of-phase calls are refused, and a pass that would
  end on snippets alone force-fetches the best candidates;
  first-party domains are verified first when the question asks
  for official documentation), `synthesize` (LLM answer from
  findings only; answers below the `MIN_RETRIEVED_SOURCES`
  threshold are labeled UNVERIFIED/INCOMPLETE), `review` (LLM
  quality gate plus three deterministic gates — citation
  provenance, topic coverage and official-source retrieval; a
  failed gate downgrades an `approved` verdict to
  `needs_more_research`), `finalize`, `handle_error`.
- **Evidence provenance**: findings are tagged `retrieved` (page
  fetched and inspected) or `snippet` (discovered URL, content
  unverified); `sources` in state is deduplicated, and the CLI
  prints each source as `[S1] … [retrieved]` / `[discovered]`.
- **Canonical citations**: sources get deterministic `[S#]`
  identifiers (first-appearance order, strongest recorded
  evidence wins); the model cites `[S1]` or a plain URL, and
  provenance labels are rendered only by application code —
  citations with a label glued into the URL
  (`https://… [snippet]`), unknown URLs and fabricated `[S#]`
  identifiers fail the deterministic citation gate and mark the
  answer as insufficiently verified.
- **Official-source quality**: for questions that explicitly ask
  for official documentation, a deterministic status is computed —
  `VERIFIED_OFFICIAL` (a first-party page was fetched),
  `OFFICIAL_DISCOVERED_ONLY` (first-party URL found but every
  fetch failed) or `NO_OFFICIAL_SOURCE` (nothing first-party
  found). Anything but `VERIFIED_OFFICIAL` marks the answer
  (`OFFICIAL SOURCES: NOT RETRIEVED`) and blocks an unqualified
  `APPROVED` verdict regardless of the reviewer; if retrieval stays
  impossible, the bounded retries end with the honest labeled
  result. Research prompts steer queries to the current domains
  (`docs.langchain.com`, `reference.langchain.com`) instead of
  legacy ones. This is provenance only — citation validation stays
  separate and neither check proves semantic accuracy.
- **Topic completeness**: a deterministic lexical check verifies
  that explicitly requested question terms appear in the draft
  outside known unavailability disclaimers (a mention that details
  "were not available" is not coverage); missing topics are marked
  (`TOPIC COVERAGE: INCOMPLETE`), sent back through the review
  gate, and the next research pass is given a targeted follow-up
  focus. Coverage is not adequacy — the LLM reviewer still judges
  whether explanations are substantive and sources support claims.
- **Limits are separate**: tool calls per pass (partitioned
  search/fetch), research passes, retrieved-source threshold, and
  wall-clock time — plus one retry for rate limits and controlled
  errors for provider failures.

## Offline evaluation (`python -m evals`)

A deterministic benchmark in `evals/` drives the **real graph**
with scripted model responses and canned search/page fixtures —
no network, no API keys, no live provider (the test suite
enforces this by making any httpx client construction fail the
run).

- `python -m evals` — human-readable summary
  (`evals/report.json` is written by default;
  `--out PATH` / `--no-out` to change)
- `python -m evals --no-out` — summary only

The report keeps **two independent verdicts per case**:

| Verdict | What it measures |
| --- | --- |
| `deterministic_pass` | The run met its fixture expectations: workflow completed, tool budgets respected, deterministic gates (citation provenance, topic coverage, official-source retrieval) produced the expected decisions, expected markers present, expected `review_status`. |
| `semantic` | Whether the draft actually *explains* each requested topic, scored against human-authored substantive markers per topic (disclaimer-aware: naming a topic inside "were not available" does not count). Also records `reviewer_false_approval`. |

**How to interpret the numbers**

- A case `PASS` means *expectations were met* — not that the
  answer is objectively correct. Fixture markers are eval
  design, not ground truth.
- `summary.reviewer_false_approvals` lists cases the reviewer
  approved that the semantic rubric scores inadequate — the
  false-approval failure mode observed live in Phase 3
  (Workstream 4). The `ws4-routing-superficial` case reproduces
  it: every deterministic gate passes, the reviewer approves,
  and only the rubric flags that conditional routing is merely
  *named*, not explained.
- Semantic scores are **diagnostics, not proofs**: keyword/URL
  matching never establishes factual correctness, and citation
  checks only establish provenance. An optional LLM-as-judge
  backend (`EVAL_LLM_JUDGE=1`) exists for rubric scoring but is
  disabled by default and never runs in tests.
- The topic-coverage checker is deliberately lexical: it only
  verifies requested terms appear outside known unavailability
  disclaimers. It cannot distinguish an explanation from a
  mention, and **no escalating regex rules were added** to try —
  regex "superficiality" arms-races into overfitting and would
  blur routing enforcement with quality scoring. Instead, the
  review prompt now demands mechanism-level answers, the
  semantic rubric measures them, and the deterministic gates
  keep enforcing only what they can verify honestly.

## Observability & token accounting

Every run collects per-node, per-model-call and per-tool-call
metrics (`agent/metrics.py`): latency, prompt size in characters,
input/output tokens when the provider reports them, tool result
sizes, retry counts and coarse error categories. The collector is
content-free **by construction** — sizes, counts, categories and
timings only; never message content, tool results, URLs or raw
error text.

- `python main.py "question" --metrics run.json` writes the full
  JSON snapshot (also embedded per case in `evals/report.json`
  under `run_metrics`).
- **Where the 61K tokens went (Workstream 2)**: the live Phase 3
  run jumped 18,923 → 61,565 input tokens for one extra fetch.
  Cause: research prompts embedded full fetched-page text
  (up to `MAX_PAGE_BYTES` = 1 MB per page), which was then
  re-sent on every later model call — each fetched page
  multiplied across remaining research rounds, synthesis and
  review. Mitigation: `MAX_EVIDENCE_CHARS` (default 6000) caps
  each source's text in prompts with an explicit truncation
  marker, and duplicate findings render once. The offline
  benchmark proves the cap holds (a 100 KB page keeps every
  prompt under 20K chars — `tests/test_evals.py`).
- **Known limitations**: token counts come from provider
  `usage_metadata`; plan and review use `with_structured_output`,
  which discards the `AIMessage`, so those calls record
  `0` tokens (prompt size and latency are still captured).
  Eval-report token totals are synthetic estimates from the fake
  model (labeled in report metadata) — an exact per-node
  breakdown of the live 61K run requires one live run with
  `--metrics`, which was not performed in Phase 4A.

## Security notes

- Retrieved web content is treated as untrusted DATA, never as
  instructions to the agent.
- API keys are read from `.env`, never committed, redacted in
  `repr`/`str` of settings, and best-effort redacted in logs
  (redaction is defense-in-depth, not a guarantee).
- Page retrieval blocks localhost/private/reserved destinations,
  validates every redirect hop, and caps response size.
- Page-fetch TLS verification is always on: the OS trust store
  (`truststore`) or an explicit `SSL_CA_BUNDLE` CA bundle is used
  when the default certifi store cannot verify a chain (e.g.
  enterprise TLS inspection). There is no `verify=False` path —
  failed verifications refuse the fetch and are reported.

## Project layout

```
AI_POC/
├── main.py                  CLI entry point (+ --metrics JSON report)
├── requirements.txt
├── .env.example             config template (no secrets)
├── agent/
│   ├── config.py            typed settings (pydantic-settings)
│   ├── state.py             GraphState + reducers
│   ├── routing.py           deterministic conditional routing
│   ├── logging_config.py    JSON-lines logging + redaction
│   ├── llm.py               provider-specific LLM factory
│   ├── tools.py             web_search / fetch_page + SSRF protection
│   ├── metrics.py           per-run metrics (content-free by design)
│   └── graph.py             node implementations + graph wiring
├── evals/
│   ├── cases.py             ~21-case benchmark (fixtures + expectations + rubrics)
│   ├── semantic.py          fixture rubric scorer + opt-in LLM judge
│   ├── harness.py           offline runner + JSON report
│   └── __main__.py          `python -m evals` CLI
└── tests/                   mocked unit tests + opt-in live test
```
