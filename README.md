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
| `MAX_EXECUTION_SECONDS` | wall-clock budget **per research pass** (each pass gets its own budget; the full run is bounded by `MAX_RESEARCH_ITERATIONS` × this, plus plan/synthesize/review LLM calls, which it does not cover — see `MAX_TOTAL_EXECUTION_SECONDS` for the run-wide bound) | `120` |
| `MAX_TOTAL_EXECUTION_SECONDS` | **global** end-to-end deadline for one run (Phase 4B/WS-C) — stamped when the graph starts; every node, retry, backoff sleep and tool call consults this same budget, and owned request timeouts are capped by the time remaining. Cooperative: checked at operation boundaries, so an in-flight blocking call may over-run it by at most its own duration — not a pre-emptive kill switch. On expiry the run returns a controlled `TIMED_OUT` result (sources preserved, no post-deadline synthesis) | `180` |
| `LLM_MAX_RETRIES` | provider retries **after** the first attempt (WS-D), applied only to failures classified retryable (429/quota, 5xx, timeout, transient network). Auth, invalid-request and unknown errors are never retried; SDK-level retries stay disabled so the layers never multiply | `1` |
| `RETRY_BACKOFF_MAX_SECONDS` | cap on a single backoff sleep (WS-D): exponential base doubling per retry with jitter; a provider `Retry-After` hint is honoured but still capped; every sleep is additionally clamped to the remaining global deadline | `30` |
| `MAX_REDIRECTS` / `MAX_PAGE_BYTES` | fetch safety limits | `5` / `1000000` |
| `MAX_EVIDENCE_CHARS` | per-source text budget injected into model prompts (snippet + page, truncated with an explicit marker beyond it) — keeps prompt size and input tokens bounded regardless of page size; findings retain the bounded text | `6000` |
| `MAX_TOOL_RESULT_CHARS` | per-result cap on **model-visible tool output**: oversized fetched pages are reduced to query-relevant sections (explicit marker) before entering the transcript ToolMessage — a raw page would otherwise be re-sent to the provider on the next research invocation | `8000` |

## Running

- `python main.py "question"` — research, then print the answer,
  sources, iteration count, token usage, estimated cost and the
  run outcome (`COMPLETED` / `COMPLETED_WITH_WARNINGS` / `INCOMPLETE`
  / `TIMED_OUT` / `FAILED`) with warnings
- `python main.py "question" --metrics run.json` — additionally
  write the machine-readable metrics snapshot (per-node/per-call
  latency, prompt sizes, token counts, tool outcomes, outcome and
  deadline status; sizes and counts only — never content). Written
  even when an unhandled exception aborts the run (recorded as
  `FAILED` with a category, never the error text)
- `python main.py --verify-connection` — opt-in live smoke test
  (exactly one minimal request to the configured provider)
- `python -m pytest` — full mocked suite (no network, no API keys)
- `python -m evals` — offline evaluation benchmark (23 scripted
  cases, no network, no API keys; see below)
- `python -m evals.compare baseline.json candidate.json` — diff two
  metrics/eval reports (token, latency, retrieval and quality
  deltas; `--json` for machine output, `--fail-on-regression` for CI
  gates)
- `RUN_LIVE_TESTS=1 python -m pytest -m live` — opt-in live
  integration test (requires a configured provider key)

## Architecture

```
START → start → plan → research ⇢(conditional)⇢ research | synthesize | handle_error
                       synthesize → review ⇢(conditional)⇢ research | finalize
                       finalize → END,  handle_error → END
```

The `start` node stamps the global run deadline once per invocation
(every later node/step consults that same monotonic expiry); all
other nodes are deadline-guarded and degrade to a controlled
`TIMED_OUT`/error result rather than raising (routing stays pure).

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
  per-pass wall-clock time — **plus one global deadline for the
  whole run** (`MAX_TOTAL_EXECUTION_SECONDS`), category-aware
  bounded provider retries, and controlled errors for provider
  failures.

## Reliability (Phase 4B)

Five bounded-recovery mechanisms added after the Phase 3 live run
(`5be3dfb`) exposed tool-arg mis-calls, opaque search failures and
provider flakiness. All are offline-tested; none made a live call.

- **Tool contracts (WS-A)**: every tool call is validated against
  the tool's Pydantic schema *before* execution, so a malformed
  call fails fast with one actionable, structured error naming the
  expected argument contract (category `validation` in metrics —
  distinct from network/provider/search failures). A narrow,
  allow-listed alias table normalizes *rename-only* legacy names
  (`web_search: queries|search_query → query`,
  `fetch_page: page_url|target_url → url`) with values untouched —
  this fixes the exact live mis-call (`queries=`) without silently
  reinterpreting anything else; with both alias and canonical name
  present the call stays a visible error. Tool-call ids and
  ordering are preserved: a refused call is still answered with a
  `ToolMessage` carrying the provider's original id.
- **Search-failure taxonomy (WS-B)**: ddgs raises a generic
  `DDGSException` for very different situations (empty index vs
  engine-side HTTP failure vs wrapped transport faults), so
  exception class alone cannot drive recovery — well-known message
  markers are classified into a fixed taxonomy
  (`empty_results`, `rate_limited`, `timeout`,
  `upstream_unavailable`, `invalid_query`, `unknown`, `deadline`)
  that is recorded in metrics instead of error text. Recovery is
  bounded and never invents terms: `empty_results`/`invalid_query`
  advance a query ladder (site-filter removal, then operator
  stripping — at most 3 backend calls per tool call);
  `timeout`/`rate_limited`/`upstream_unavailable` repeat the
  *same* query exactly once (rate limits pause briefly first,
  clamped to the remaining deadline budget); `unknown` is never
  retried blindly.
- **Global deadline (WS-C)**: `MAX_TOTAL_EXECUTION_SECONDS=180`
  is stamped once per run on a monotonic clock (injectable —
  tests drive it with fake clocks) and shared by every node, tool
  call, research retry and backoff sleep, in addition to the
  existing per-pass budget. Owned request timeouts (HTTPX
  fetch/search) are capped by the time remaining; after expiry no
  new work starts and the run ends with a controlled `TIMED_OUT`
  result — sources gathered before the deadline are preserved and
  rendered as provenance-only (no post-deadline synthesis or
  review). **Caveat**: the check is cooperative, at operation
  boundaries — an in-flight blocking call cannot be cancelled, so
  actual wall-clock overrun is bounded by the single longest
  in-flight operation (provider SDK timeouts are set at client
  construction and cannot be narrowed per call).
- **Provider retries (WS-D)**: `classify_provider_error`
  (`agent/llm.py`) maps SDK failures to a fixed taxonomy and a
  retryable flag. Only transient categories are retried (429/quota,
  5xx/unavailable/internal, timeout, transient network faults) with
  bounded exponential backoff (`retry_delay × 2^(n−1)`), jitter and
  a capped, clamped-to-deadline sleep; a provider `Retry-After`
  hint is honoured instead of the computed backoff (still capped).
  Auth (bad key), invalid-request and *unknown* errors are never
  retried. SDK retries stay disabled at provider construction so
  the SDK and application layers never multiply, and every failed
  attempt is recorded with its category.
- **Run outcomes & comparison (WS-F)**: every run terminates in
  exactly one of `COMPLETED`, `COMPLETED_WITH_WARNINGS`,
  `INCOMPLETE`, `TIMED_OUT`, `FAILED`, with deterministic warnings
  merged in (tool failures, provider retries, partial token
  coverage, deadline expiry) — a clean `COMPLETED` is only
  reported when nothing degraded. `python -m evals.compare
  baseline.json candidate.json` diffs two metrics/eval reports
  (token, latency, call-count, retrieval, outcome-severity, quality
  and per-category tool-failure deltas; case pass→fail flips and
  false approvals for benchmark reports) and prints a console
  table, with `--json` and `--fail-on-regression` (exit 1) for CI.
  Stdlib only — no databases or dashboards.

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
- `python -m evals --security` — the Phase 4C **adversarial security
  suite**: 25 offline cases (injection, tool spoofing, SSRF-adjacent
  fetches, secret-DLP, citation manipulation, blocked sources), each
  tagged with attacker objective, entry point, defense layer
  (deterministic vs model-behavior), severity and residual limits.
  Writes `evals/security_report.json` including a machine-readable
  `security_summary` (severity/layer/entry-point counts, refusal
  categories, signal and redaction totals). Honesty rule: a missing
  injection signal is never treated as proof of safety, and the
  printed note reminds that mocked fixtures cannot prove real-model
  prompt-injection resistance.
- `python -m evals.compare baseline.json candidate.json` — diff two
  reports produced by `--out` (or any `--metrics` snapshot):
  token/latency/call/retrieval/quality/outcome deltas, per-category
  tool failures, benchmark case pass→fail flips and false-approval
  additions — regressions and improvements listed separately.
  `--json` for machine output; `--fail-on-regression` exits 1 on
  any regression (CI gate). The catalog includes two recovery cases
  (`invalid-tool-args-recovery` for the WS-A alias path,
  `search-failure-recovery` for the WS-B taxonomy path) whose
  failure would signal a broken recovery, not just a broken fixture.

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
- **Where the tokens went (Workstream 2, two amplification
  paths found and fixed)**:
  1. *Prompt rendering* (Phase 4A): research prompts embedded
     full fetched-page text (up to `MAX_PAGE_BYTES` = 1 MB per
     page) in the findings block, re-sent on every later model
     call. Fixed by `MAX_EVIDENCE_CHARS` (default 6000) — a
     per-source prompt budget with an explicit truncation marker.
  2. *Tool transcript* (live-run follow-up): the research loop
     answers every tool call with a ToolMessage built from the
     full tool result, so a large fetched page re-entered the
     transcript on the next invocation. The instrumented live run
     measured a single 183,164-character page producing a
     189,546-character research prompt — ~90% of that run's
     43,033 input tokens. Fixed by `MAX_TOOL_RESULT_CHARS`
     (default 8000): oversized payloads are reduced to
     query-relevant sections (sentence-level selection by the
     question's topic terms, lead segment always kept, explicit
     marker) before entering the transcript; findings retain the
     same bounded, relevant excerpt — the raw document is not
     retained. Offline reproduction with the same ~183K fixture:
     research prompts drop from >150,000 to <20,000 characters
     (`tests/test_evidence_bounds.py`).
- **Tradeoff**: evidence is selected by lexical relevance to the
  question, so a page whose key passage shares no topic terms
  with the question may be under-represented; the truncation
  marker makes the reduction explicit, and re-fetching with a
  better query remains possible within budget.
- Metrics reports are finalized per run (`finished_at` and
  `total_execution_seconds` are stamped before the JSON snapshot
  is written), including on an unhandled exception, which is
  recorded as outcome `FAILED` with a category (never the error
  text). Each snapshot carries the terminal `outcome`
  (status + warnings), `deadline` (configured seconds + whether it
  was exceeded) and a content-free `quality` summary.
- **Token honesty (WS-E)**: usage comes from provider
  `usage_metadata` where available. Plan and review call
  `with_structured_output(..., include_raw=True)` so the raw
  `AIMessage` stays reachable and those calls record real
  provider-reported usage. Every model call carries an explicit
  `usage_source` — `provider` (reported), `estimated` (only when
  an estimator is explicitly applied) or `unavailable` — and
  unmeasured calls are recorded as JSON `null`, never a fake zero.
  The `token_usage` block reports the coverage triad
  (`provider_reported_calls` / `estimated_calls` /
  `unavailable_calls` + `coverage_ratio`), so a run with mixed
  measurement is visible as a `partial_token_coverage` warning
  instead of silently under-counting. Eval-report token totals are
  synthetic estimates from the fake model (labeled in report
  metadata); the instrumented live run (`run_metrics.json`,
  gitignored) provides real per-call attribution.

## Security (Phase 4C)

Full threat model (9 trust boundaries, control matrix, residual
risks): **[docs/SECURITY.md](docs/SECURITY.md)**. The design rule:
**retrieved web content is evidence, never authority.** Layers:

1. **Prompt framing** — all system prompts end with a fixed trust
   boundary clause; evidence enters model context wrapped in
   `UNTRUSTED QUOTED EVIDENCE` framing.
2. **Signal detection (`agent/injection.py`)** — a 7-category
   pattern detector scans snippets, pages and tool errors and
   records *content-free* signals (category + source counts) plus a
   `security_note` inside the ToolMessage JSON. It is explicitly
   **signal, not proof**: it never blocks or rewrites evidence, so
   legitimate security documentation keeps working.
3. **Deterministic policy layer** — where the guarantees live:
   - tool allow-list + strict schemas in `ToolExecutor` (unknown
     tools / oversized or malformed args are refused, never run);
   - **fetch-target allow-list**: `fetch_page` only accepts URLs
     `web_search` returned this run, so a fully attacker-compliant
     model still cannot reach an attacker URL;
   - **secret-DLP**: tool arguments are scanned for configured
     secret values and refused before execution, without echoing.
4. **Network hardening (`fetch_page`)** — scheme/port/userinfo
   checks, every address of every hop validated against
   non-public ranges, manual redirect validation, cleartext-HTTP
   connection pinning to the validated address (DNS-TOCTOU), size/
   timeout/content-type caps, TLS verification always on (no
   `verify=False` path — static guard test).
5. **Output backstop** — credential-shaped values are redacted at
   synthesis and again at `finalize` (`[REDACTED-CREDENTIAL]`,
   counted in metrics, value never logged).

Observability: metrics gained a content-free `security` block
(injection signals by category/source, refusals by category,
answer-redaction count) — never payload text, keys, or full URLs.

Verification is **offline**: 453 passing tests (1 skipped opt-in
live test), benchmark 23/23, security suite 25/25 — mocked
fixtures cannot prove real-model prompt-injection resistance or
replace a live penetration test (see the doc's residual risks).

## Project layout

```
AI_POC/
├── main.py                  CLI entry point (+ --metrics JSON report)
├── requirements.txt
├── .env.example             config template (no secrets)
├── docs/
│   └── SECURITY.md          Phase 4C threat model (9 trust boundaries,
│                            controls, residual risks)
├── agent/
│   ├── config.py            typed settings (pydantic-settings)
│   ├── state.py             GraphState + reducers
│   ├── routing.py           deterministic conditional routing
│   ├── logging_config.py    JSON-lines logging + redaction
│   ├── llm.py               provider factory + error taxonomy/retries
│   ├── tools.py             web_search / fetch_page + SSRF protection,
│   │                        fetch-target policy, secret-DLP executor
│   ├── injection.py         injection-signal detector + credential
│   │                        redaction (content-free by design)
│   ├── deadline.py          global run deadline (monotonic, injectable clock)
│   ├── metrics.py           per-run metrics (content-free by design)
│   └── graph.py             node implementations + graph wiring
├── evals/
│   ├── cases.py             ~23-case benchmark (fixtures + expectations + rubrics)
│   ├── security_cases.py    25-case adversarial suite (+ security_summary)
│   ├── semantic.py          fixture rubric scorer + opt-in LLM judge
│   ├── harness.py           offline runner + JSON report
│   ├── compare.py           `python -m evals.compare` report diffing
│   └── __main__.py          `python -m evals` CLI (--security flag)
└── tests/                   mocked unit tests + opt-in live test
                             (tool schema, search recovery, deadline,
                             provider retries, compare — all offline;
                             security: injection, tool policy, network,
                             secrets, config, security evals)
```
