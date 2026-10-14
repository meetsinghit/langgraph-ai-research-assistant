# AI Research Assistant (POC)

A modular LangGraph agent that answers questions with verifiable,
web-sourced findings. Built as a phased learning project in
practical Agentic AI engineering.

## Quick start

**Browser app (recommended for daily research):**

```powershell
# from the project root (AI_POC) — no terminal needed afterwards
.\start-research-assistant.bat
```

One click builds/serves everything locally and opens
`http://127.0.0.1:8000/`. On first run the launcher offers to copy
the access token to your clipboard for the connection dialog.
Operations guide (stop/restart, backups, troubleshooting, Docker):
`docs/OPERATIONS.md`.

**CLI (unchanged):**

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
| `LLM_THINKING_LEVEL` | default thinking effort for runs that do not pick one: `auto` (provider decides) \| `minimal` \| `low` \| `medium` \| `high`; per-run overrides are capability-validated against the selected model's registry entry | unset (= `medium`) |
| `MODEL_DISCOVERY_ENABLED` · `MODEL_DISCOVERY_TTL_SECONDS` · `MODEL_DISCOVERY_BACKOFF_SECONDS` · `MODEL_DISCOVERY_TIMEOUT_SECONDS` | opt-in live availability annotation for `GET /api/models` (Phase 4E.3): TTL-cached, single-flight, failure-backed-off, capped — **off by default** so nothing touches the network during tests/launches; discovery never changes what is selectable | `false` / `900` / `300` / `10` |
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
| `QUICK_*` / `DEEP_*` | per-depth budget overrides (Phase 4E.1, selected per run via `--depth` / `research_depth`): plan queries, research passes, tool calls, findings, retrieved pages (`QUICK_MAX_TOOL_CALLS_PER_RESEARCH=3`, `DEEP_MAX_RESEARCH_ITERATIONS=5`, `DEEP_MIN_RETRIEVED_SOURCES=3`, …). `standard` uses the historical rows above unchanged; all of it stays under `MAX_TOTAL_EXECUTION_SECONDS` | see `.env.example` |

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
- `python main.py "question" --depth quick` — research depth for
  this run (4E.1; default `standard`): `quick` = minimal sufficient
  retrieval and a short direct answer, `standard` = the default
  developed answer, `deep` = broader explicit budgets (more passes,
  more independently retrieved sources) and a longer structured
  answer. Depth changes budgets and structure — never the quality
  gates, and never the global execution deadline
- `python main.py "question" --evidence-policy verified` —
  citation evidence policy for this run (4E.1; default `balanced`):
  `verified` requires every factual claim to rest on a retrieved
  (fetched-and-inspected) page and deterministically fails
  citation validation on snippet-only or absent citations, so a
  thin-evidence run ends clearly labeled rather than overstated;
  `balanced` permits clearly-labeled snippet-only support
- `python main.py --verify-connection` — opt-in live smoke test
  (exactly one minimal request to the configured provider)
- `python -m pytest` — full mocked suite (no network, no API keys)
- `python -m evals` — offline evaluation benchmark (30 scripted
  cases, no network, no API keys; see below)
- `python -m evals.compare baseline.json candidate.json` — diff two
  metrics/eval reports (token, latency, retrieval and quality
  deltas; `--json` for machine output, `--fail-on-regression` for CI
  gates)
- `RUN_LIVE_TESTS=1 python -m pytest -m live` — opt-in live
  integration test (requires a configured provider key)
- `cd frontend && npm run dev` — browser UI for the local server
  (see **Web UI (Phase 4D.2)** below)

## Server API (Phase 4D.1)

The same graph, behind a local FastAPI service — sessions, runs,
resumable checkpoints and a server-sent-event progress stream.
The CLI above is unchanged; the engine is shared.

**Start the server** (loopback-only by default):

```bash
python -m app.main                 # 127.0.0.1:8000 (api_host/api_port in .env)
# then, in another shell:
TOKEN=$(cat data/app_token.txt)    # bootstrapped on first start (or set APP_TOKEN)
curl -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8000/api/health
```

Every endpoint except `GET /api/health` requires the Bearer token —
**including SSE streams**. `data/` (SQLite DBs, WAL, token file) is
git-ignored and sensitive at rest; deleting it is *not* secure
erasure (see [docs/SECURITY.md §10](docs/SECURITY.md)).

| Method & path | Purpose |
|---|---|
| `GET /api/health` | liveness (unauthenticated) |
| `GET /api/config` | content-free capabilities (e.g. `resume_enabled`), budgets — never secrets |
| `GET /api/models` | backend-managed model catalogue (4E.3): free-tier registry with canonical ids, friendly names, thinking capabilities and per-model `selectable` flag, plus discovery status; discovery is opt-in and off by default |
| `POST /api/sessions` · `GET /api/sessions` | create / list sessions |
| `GET /api/sessions/{id}` | session + its runs (summary rows) |
| `PATCH /api/sessions/{id}` | rename a session (required `title`, 1–120 chars, trimmed; metadata only — run history and `updated_at` untouched) |
| `DELETE /api/sessions/{id}` | delete a session, its runs, sources, events and checkpoints (cascade) |
| `POST /api/sessions/{id}/runs` | start a run; optional `research_depth` (`quick` / `standard` / `deep`, default `standard`), `evidence_policy` (`balanced` / `verified`, default `balanced`), `model_id` (registry-selectable; `null`/omitted = the configured default) and `thinking_level` (`auto` / `minimal` / `low` / `medium` / `high`, capability-validated per model) — 4E.1/4E.3 per-run settings, **frozen at creation**, persisted and visible on `GET /api/runs/{id}`; optional `Idempotency-Key` header replays the same run instead of starting a duplicate (409 `idempotency_conflict` on key reuse with a different question **or different settings**); invalid selections → 400 `unknown_model` / `model_not_selectable` / `unsupported_thinking_level` |
| `GET /api/runs/{id}` | run detail: status, answer, review, sources, outcome, content-free diagnostics |
| `POST /api/runs/{id}/cancel` | cooperative cancel (202); terminal runs → 409 |
| `POST /api/runs/{id}/resume` | continue an interrupted run from its checkpoint (202), see below |
| `GET /api/runs/{id}/events` | SSE stream of sequenced, content-free progress events |

**Run lifecycle:** `queued → running → (cancelling →) completed |
failed | cancelled`, plus `interrupted` — a row left `running` by a
crash/restart is marked `interrupted` at startup and becomes
resumable.

**SSE integrity:** every event is persisted first with a monotonically
increasing `seq` before it is delivered. Reconnect with the standard
`Last-Event-ID` header (or `?after=<seq>`) to replay exactly the
events after that id — no gaps, no duplicates, race-free against
live publishing. Streams send `: heartbeat` comments while idle and
end with the terminal event; disconnecting detaches the subscriber.

**Resume semantics (capability-gated):** the server advertises
`resume_enabled` only when a LangGraph checkpointer is configured;
otherwise resume returns **409 `resume_unverified`** (an
unverifiable resume is disabled, not guessed at). Other refusals:
**409 `not_interrupted`** (only interrupted runs resume) and
**409 `no_checkpoint`** (nothing to continue from). Guarantees,
all covered by cross-instance crash tests (a second app instance on
the same DB files recovers the run):

- completed nodes and their tool actions are **not replayed** — the
  resumed execution continues at the interrupted node;
- the *interrupted node itself* re-executes (LangGraph at-least-once
  semantics — plan for idempotency before adding side effects);
- the resumed execution gets a **fresh deadline and budget**; the
  persisted `deadline_total_seconds` records the run's original
  configured budget, it is not an expiry timestamp;
- provenance (sources/findings), review state and security
  diagnostics survive the restart;
- concurrent resume attempts elect **exactly one** worker (winner
  202, loser 409).

**Not exposed (deliberately):** no prompt/argument/page-text fields
in any response or event, no raw exception text (type + coarse
category only), no public bind, no TLS/multi-user auth (loopback
POC scope, residual risk R8).

## Web UI (Phase 4D.2 + 4D.3 packaging)

A browser front-end for the same API — three-region responsive
layout (session history · conversation · sources/diagnostics),
live SSE progress with resume-after-drop, safe markdown rendering,
light/dark themes. React + TypeScript + Vite + Tailwind v4 in
`frontend/`; it talks to the unmodified Phase 4D.1 backend.

**Production mode (Phase 4D.3): one process, one origin.**
`start-research-assistant.bat` runs the FastAPI server, which also
serves the built bundle from `frontend/dist` on the same origin —
no dev server, no proxy, no second port. Static files are public
(they contain no secrets; the token is never in the bundle) while
every `/api` route stays Bearer-protected. Hashed `/assets/*` are
cached immutably for a year; `index.html` is always revalidated so
new builds load on the next refresh; UI responses carry a
restrictive CSP, `nosniff` and `Referrer-Policy: no-referrer`.

**Local development:**

```bash
# shell 1 — the backend (unchanged from 4D.1)
python -m app.main                       # 127.0.0.1:8000

# shell 2 — the UI
cd frontend
npm install                              # first time only
npm run dev                              # http://127.0.0.1:5173
```

Open http://127.0.0.1:5173 and paste the access token from
`data/app_token.txt` (bootstrapped on first backend start) or your
`APP_TOKEN` value into the connection dialog.

- **Same-origin proxy:** the Vite dev server proxies `/api` to
  `http://127.0.0.1:8000`, so the browser only ever talks to one
  origin — no CORS changes, no service worker, and the Bearer
  header works on the SSE stream (plain `EventSource` cannot send
  auth headers; the UI uses `fetch` streaming).
- **Token handling:** the token lives **in memory only** (never
  `localStorage`, URLs, or analytics) and is sent as
  `Authorization: Bearer …`. Because it is memory-only, a page
  refresh requires reconnecting — the UI says so explicitly. A 401
  or unreachable backend surfaces a friendly reconnect prompt; the
  backend's auth is never weakened or bypassed.
- **Sessions:** created lazily on your first question; the title is
  derived from the question (≤120 chars) because the API has no
  title-update endpoint (documented contract workaround — no
  backend change). Reopening a session is read-only (GETs); a
  follow-up question starts a new run in the same session, and
  citations are always scoped to the run that produced them.
- **Interrupted runs:** a "Resume research" card appears only when
  the backend reports the run as `interrupted` *and*
  `resume_enabled: true` — with the caveat that resumed execution
  may repeat the interrupted step.
- **Safe rendering:** answers render as markdown with raw HTML
  inert (no `rehype-raw`), `javascript:`/`data:` links stripped,
  external links `noopener noreferrer`, and unknown `[S#]`
  citations degrading to plain text.

**Commands** (from `frontend/`; no network, no API keys):

- `npm run typecheck` — `tsc --noEmit`
- `npm test` — Vitest + React Testing Library + an in-memory MSW
  backend (82 tests: auth/connection, conversation lifecycle, SSE
  dedup/reconnect/backoff, progress/cancel/outcome states, resume
  gating, panels/citations, safe markdown, a11y + drawers,
  theme persistence)
- `npm run build` — production bundle into `frontend/dist`
  (serving it with the API is packaging scope = Phase 4D.3)
- `python ..\frontend\scripts\dump_api_fixtures.py` — regenerate
  `frontend/src/test/fixtures/backend/*.json` from the real FastAPI
  app after a backend contract change (`schemaCompat.test.ts` then
  verifies the UI types still match reality)

**Real-browser E2E** (Phase 4D.3; Windows, Microsoft Edge, no API
keys, no network — the backend runs scripted fakes through the real
auth/SSE/SQLite stack while the browser drives the production bundle):

```powershell
cd frontend          # npm run build   (once: creates dist/)
cd e2e               # npm install     (once: installs Playwright)
npx playwright test
```

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
  Model and thinking selection is driven by the maintained free-tier
  registry (`agent/gemini_models.py`, Phase 4E.3): canonical ids,
  capability-validated thinking levels, tombstoned retired models —
  never a silent model substitution.
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

## Research depth & evidence quality (Phase 4E.1)

Two independent per-run axes, both plain strings in graph state
(checkpointed, so a resume keeps its original settings), both
enforced in backend code — not by prompting — and both persisted
as run metadata on the runs table so history keeps the settings
each run actually used.

- **Research depth** (`--depth` / `research_depth`), WS-G:
  `quick` (minimal sufficient retrieval, short direct answer,
  ~150-300-word target), `standard` (the default, ~500-900), and
  `deep` (broader explicit budgets — more plan queries, passes,
  tool calls and independently retrieved pages — with a longer
  structured answer, ~1,200-2,500). Each depth resolves to an
  explicit `DepthPolicy` from *configurable* settings
  (`QUICK_*` / `DEEP_*` in `.env`); `standard` uses the historical
  settings unchanged. Word targets are synthesis/reviewer
  **guidance, never gates**: no length-based pass/fail in either
  direction, padding is explicitly forbidden, and thin evidence
  makes the answer shorter and more qualified, not longer. The
  global execution deadline is the unchanged hard ceiling — depth
  never extends it.
- **Evidence policy** (`--evidence-policy` / `evidence_policy`),
  WS-A: `balanced` (default) permits clearly-labeled snippet-only
  citations (discovered sources stay usable for discovery, with
  the honest UNVERIFIED labeling carrying their weakness);
  `verified` deterministically fails citation validation when a
  claim rests only on a search snippet — or on nothing at all —
  so insufficient retrieval ends as a clearly-labeled incomplete
  result instead of an overstated one. Snippets are never silently
  promoted to verified, and no fabricated `[S#]`/URL is accepted
  in either policy.
- **Claim-to-source grounding** (advisory, WS-A): a lexical
  overlap check reports citations whose citing sentence shares no
  vocabulary with the cited page (paraphrase can cause it, so it
  never auto-fails — semantic judgement stays with the reviewer),
  and retrieved-but-uncited sources are reported so the answer
  prefers them over snippet-only ones.
- **Explanation depth** (WS-B): a deterministic lexical check
  catches the Workstream-4 failure mode — a requested topic that
  is merely *named* ("X is a core capability for…") while
  retrieved evidence existed to explain it. Such drafts are
  marked (`EXPLANATION DEPTH: SUPERFICIAL`), fail the review
  gate, and feed follow-up research. The check runs only for
  explanation-intent questions with at least one retrieved page,
  so honest UNVERIFIED answers and legitimate short definitions
  ("What is SecOps?") are never forced into extra research.

## Search efficiency & honest diagnostics (Phase 4E.2)

Two fixes for what live runs exposed: follow-up passes wasting
time re-searching identical queries, and a diagnostics panel that
could contradict the progress view after a crash+resume.

- **Per-run search dedup** (WS-C): a repeat of an already
  *successfully* searched query — the classic follow-up-pass
  pattern — is served from a per-run query cache instead of
  re-hitting the search backend (casefold + whitespace
  normalization only; `site:`/operator differences are preserved
  because they change what the backend returns). The cache is
  live-only (a `RunContext` field, like the deadline): fresh per
  invocation, never checkpointed, so one run can never serve
  another's queries and a resumed run may legitimately search
  again. Only successful, non-empty results are cached — failures
  stay retryable (the query-recovery ladder is unaffected) —
  cached results are copied on store and serve, and the executed
  call still counts against every quota (bounded), still emits
  its progress events, and still counts in `counts.searches`;
  the saved backend work is surfaced separately as
  `counts.search_cache_hits` (rendered as "N searches (K
  cached)" in the diagnostics panel). Fetches are never cached
  (force-verify must be able to retry a failed fetch).
- **Diagnostics/progress reconciliation** (WS-D): after a
  crash+resume the metrics collector only covers the RESUMED
  segment while the durable event channel (and therefore the
  progress timeline) spans the whole run — the two views could
  legitimately disagree ("2 fetch failed" in progress vs "0
  failed tool calls" in diagnostics). At finish, the run manager
  re-attaches the earlier segments' tool calls from their durable
  `tool.finished` events — the only record that survived the
  crash — so the persisted counters match what the progress view
  showed. Reconstructed calls carry `result_chars: null` (their
  size is unknown, never a fake zero). The snapshot states the
  scope explicitly: `segments` (1 = uninterrupted) and
  `token_usage.scope` — `whole_run`, or `since_resume` because a
  crashed segment's token usage died with its in-process
  collector and is honestly labeled, never fabricated. The
  diagnostics panel annotates both: retrieval counters "cover the
  whole run (reconciled from the durable event log)", token usage
  "covers the resumed segment only".
- **Definition alignment**: "Web searches"/"Pages fetched" are
  ATTEMPTS (failures included, now broken out inline as "N (M
  failed · K cached)"), progress "pages verified" = successes,
  and the timeline counts only exact tool names (`fetch_page` is
  never inferred from "anything else"). `totals.tool_latency_ms`
  adds combined search+fetch wall time ("Tool time") so latency
  can be read separately from model time.

## Model & thinking selection (Phase 4E.3)

Runs are configured from the UI (or API) with four settings —
**model, thinking effort, research depth, evidence policy** — all
frozen into the run row at creation, so history, resume and
diagnostics keep exactly what the run used. The composer inherits
a session's most recent run settings when you open it.

- **Backend-managed catalogue** (WS-A), `GET /api/models`: a
  separately *maintained* registry (`agent/gemini_models.py`) maps
  canonical ids to friendly names, families, thinking capabilities
  (`level` / `budget` / `none` + supported levels) and **free-tier
  eligibility** — the registry is the only authority for what is
  selectable; a live listing (opt-in discovery, off by default)
  can annotate `availability` but never selectability. Entries are
  verified against official docs (`verified_on`), retired models
  remain as tombstones so a historical selection fails fast with
  `model_unavailable` — **no automatic substitution**, ever. Only
  free-tier-eligible models are offered; `unregistered` findings
  are listing-only.
- **Thinking effort** (WS-B): `Auto` (parameter omitted — the
  provider decides), `Minimal`, `Low`, `Medium`, `High`,
  capability-validated per model (3.x level models omit
  `minimal`; 2.5-family models map to `thinking_budget` with
  documented token values). The selector only offers levels the
  selected model supports; unsupported combinations are rejected
  (422/400), never coerced. Reasoning stays private — tokens are
  reported as counts (`thinking_tokens`), never content.
- **Per-run isolation** (WS-C): each run gets its own settings
  object (`model_copy`) so concurrent runs with different models
  or levels can never race; the configured default is submitted
  as `null` (legacy rows created before 4E.3 show *Not recorded*,
  never a back-filled guess). Resuming a run continues with its
  **original** frozen configuration, or fails with a clear
  configuration error if that configuration is no longer
  supported. `Idempotency-Key` reuse is now checked against the
  full setting set: identical → replay, different → 409
  `idempotency_conflict`.
- **Session rename** (WS-F): `PATCH /api/sessions/{id}` with a
  required 1–120-char title (control characters stripped), via an
  inline edit in the sidebar — metadata only, run history and
  `updated_at` untouched.
- **Presentation** (WS-G/H): the composer exposes the four
  selectors (capability-aware, disabled while submitting); the
  diagnostics panel gains a **Run configuration** section
  (model / thinking / depth / evidence, *Not recorded* for legacy
  rows) plus provider-reported **thinking tokens** with an honest
  "Not reported" when the provider gave no breakdown; selection
  and availability errors surface as friendly category copy, never
  raw exceptions.

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
| `deterministic_pass` | The run met its fixture expectations: workflow completed, tool budgets respected, deterministic gates (citation provenance + evidence policy, topic coverage, explanation depth, official-source retrieval) produced the expected decisions, expected markers present, expected `review_status`. |
| `semantic` | Whether the draft actually *explains* each requested topic, scored against human-authored substantive markers per topic (disclaimer-aware: naming a topic inside "were not available" does not count). Also records `reviewer_false_approval`. |

**How to interpret the numbers**

- A case `PASS` means *expectations were met* — not that the
  answer is objectively correct. Fixture markers are eval
  design, not ground truth.
- `summary.reviewer_false_approvals` lists cases the reviewer
  approved that the semantic rubric scores inadequate — the
  false-approval failure mode observed live in Phase 3
  (Workstream 4). The `ws4-routing-superficial` case reproduced
  it through Phase 4D: every deterministic gate passed, the
  reviewer approved, and only the rubric flagged that conditional
  routing was merely *named*, not explained. Since Phase 4E.1 the
  deterministic explanation-depth gate catches it (see below),
  the approval is downgraded, and this list is empty — the
  benchmark asserts that, so a new false approval is a regression.
- Semantic scores are **diagnostics, not proofs**: keyword/URL
  matching never establishes factual correctness, and citation
  checks only establish provenance. An optional LLM-as-judge
  backend (`EVAL_LLM_JUDGE=1`) exists for rubric scoring but is
  disabled by default and never runs in tests.
- The topic-coverage checker is deliberately lexical: it only
  verifies requested terms appear outside known unavailability
  disclaimers. It cannot distinguish an explanation from a
  mention. Phase 4E.1 added a second, deliberately narrow lexical
  check for the one superficiality pattern that is honestly
  detectable without semantics — a requested term covered ONLY by
  generic-category definitions ("X is a core capability for…",
  with the question's own vocabulary stripped and no explanatory
  marker present), and only for explanation-intent questions when
  at least one page was actually retrieved. It never judges
  correctness, never fires on honest UNVERIFIED answers, and its
  result is a labeled note plus a review-gate failure — not a
  claim that the explanation is wrong. The review prompt still
  demands mechanism-level answers and the semantic rubric still
  measures them.

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

Verification is **offline**: 506 passing tests (1 skipped opt-in
live test), benchmark 23/23, security suite 25/25 — mocked
fixtures cannot prove real-model prompt-injection resistance or
replace a live penetration test (see the doc's residual risks).
Phase 4D extended the model with the API surface (TB10) and the
persistence/erasure policy (§10).

## Project layout

```
AI_POC/
├── main.py                  CLI entry point (+ --metrics JSON report)
├── requirements.txt
├── .env.example             config template (no secrets)
├── start-research-assistant.bat      one-click production launch (4D.3)
├── start-research-assistant-dev.bat  dev launch (Vite hot reload)
├── stop-research-assistant.bat       stops only launcher-started processes
├── scripts/
│   └── launcher.ps1         launcher core logic + self-tests
├── docker/                  4D.3 container packaging (STATIC REVIEW ONLY,
│   │                        not runtime-verified — see docs/OPERATIONS.md)
│   ├── Dockerfile           multi-stage: Node UI build → non-root Python
│   └── docker-compose.yml   loopback-only publish, data volume, healthcheck
├── e2e/
│   └── server.py            real-server E2E harness (scripted fakes,
│                            test-only shutdown endpoint; no network)
├── docs/
│   ├── OPERATIONS.md        run/stop/backup/troubleshoot guide (4D.3)
│   └── SECURITY.md          Phase 4C threat model (10 trust boundaries,
│                            controls, residual risks) + Phase 4D
│                            API surface (TB10) & erasure policy (§10)
├── app/                     Phase 4D FastAPI service
│   ├── main.py              app factory, auth middleware (/api only; static
│   │                        UI is public), token bootstrap, checkpointer
│   │                        wiring, `python -m app.main`
│   ├── ui.py                serves the built frontend/dist same-origin
│   │                        (cache headers, CSP, SPA fallback, traversal-safe)
│   ├── api.py               routers + SSE stream (Last-Event-ID replay)
│   ├── runs.py              run lifecycle manager (threads, CAS state
│   │                        transitions, cancel/resume, event sanitizing)
│   ├── events.py            durable sequenced event channel
│   ├── repository.py        SQLite CRUD + status vocabulary
│   ├── db.py                schema (sessions/runs/sources/events), WAL
│   ├── model_discovery.py   opt-in model availability annotation (4E.3;
│   │                        TTL/backoff/single-flight, off by default)
│   ├── schemas.py           request/response contracts
│   └── __init__.py
├── agent/
│   ├── config.py            typed settings (pydantic-settings)
│   ├── state.py             GraphState + reducers
│   ├── routing.py           deterministic conditional routing
│   ├── depth.py             research depth + evidence policy (4E.1)
│   ├── gemini_models.py     free-tier model registry + thinking mapping
│   │                        (4E.3: selectability, validation, tombstones)
│   ├── logging_config.py    JSON-lines logging + redaction
│   ├── llm.py               provider factory + error taxonomy/retries
│   ├── tools.py             web_search / fetch_page + SSRF protection,
│   │                        fetch-target policy, secret-DLP executor
│   ├── injection.py         injection-signal detector + credential
│   │                        redaction (content-free by design)
│   ├── deadline.py          global run deadline (monotonic, injectable clock)
│   ├── runtime.py           RunContext: LIVE deadline/cancel/sink passed
│   │                        via LangGraph context (never persisted)
│   ├── metrics.py           per-run metrics (content-free by design)
│   └── graph.py             node implementations + graph wiring
├── data/                    runtime state (git-ignored): app.db,
│                            checkpoints.db, app_token.txt — sensitive,
│                            deletion ≠ secure erasure
├── evals/
│   ├── cases.py             30-case benchmark (fixtures + expectations + rubrics)
│   ├── security_cases.py    25-case adversarial suite (+ security_summary)
│   ├── semantic.py          fixture rubric scorer + opt-in LLM judge
│   ├── harness.py           offline runner + JSON report
│   ├── compare.py           `python -m evals.compare` report diffing
│   └── __main__.py          `python -m evals` CLI (--security flag)
├── frontend/                Phase 4D.2 web UI (React + TS + Vite + Tailwind)
│   ├── src/api/             typed API client + fetch-based SSE stream
│   │                        (Bearer auth, Last-Event-ID reconnect, seq dedup)
│   ├── src/lib/             timeline/citations/format/messages (pure fns)
│   ├── src/state/           in-memory auth context + app store
│   ├── src/components/      layout, conversation, panels, safe markdown
│   ├── src/test/            Vitest + RTL + MSW suite + real-API fixtures
│   ├── e2e/                 Playwright real-browser suite (system Edge;
│   │                        drives e2e/server.py through the production path)
│   └── scripts/             dump_api_fixtures.py (fixtures from real FastAPI)
└── tests/                   mocked unit tests + opt-in live test
                             (tool schema, search recovery, deadline,
                             provider retries, compare — all offline;
                             security: injection, tool policy, network,
                             secrets, config, security evals;
                             Phase 4D app: auth, lifecycle, idempotency,
                             SSE replay/races, crash recovery/resume,
                             cancellation, API redaction;
                             Phase 4E.3: model registry, discovery,
                             per-run model/thinking config, rename)
```
