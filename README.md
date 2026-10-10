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
| `MAX_RESEARCH_ITERATIONS` | research passes (graph-level budget) | `3` |
| `MIN_FINDINGS` | findings required to synthesize | `3` |
| `MIN_RETRIEVED_SOURCES` | source **pages** that must be fetched/inspected before an answer counts as verified (snippets never satisfy it; `0` disables verification) | `1` |
| `MAX_TOOL_CALLS_PER_RESEARCH` | executed tool invocations per research pass (counts batches in one LLM response); split ~half search / ~half fetch | `4` |
| `MAX_EXECUTION_SECONDS` | wall-clock budget **per research pass** (each pass gets its own budget; the full run is bounded by `MAX_RESEARCH_ITERATIONS` × this, plus plan/synthesize/review LLM calls, which it does not cover) | `120` |
| `MAX_REDIRECTS` / `MAX_PAGE_BYTES` | fetch safety limits | `5` / `1000000` |

## Running

- `python main.py "question"` — research, then print the answer,
  sources, iteration count, token usage and estimated cost
- `python main.py --verify-connection` — opt-in live smoke test
  (exactly one minimal request to the configured provider)
- `python -m pytest` — full mocked suite (no network, no API keys)
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
  quality gate plus two deterministic gates — citation provenance
  and topic coverage; a failed gate downgrades an `approved`
  verdict to `needs_more_research`), `finalize`, `handle_error`.
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
- **Topic completeness**: a deterministic keyword check verifies
  that explicitly requested question terms appear in the draft;
  missing topics are marked (`TOPIC COVERAGE: INCOMPLETE`), sent
  back through the review gate, and the next research pass is
  given a targeted follow-up focus. Coverage is not correctness —
  the LLM reviewer still judges whether sources support claims.
- **Limits are separate**: tool calls per pass (partitioned
  search/fetch), research passes, retrieved-source threshold, and
  wall-clock time — plus one retry for rate limits and controlled
  errors for provider failures.

## Security notes

- Retrieved web content is treated as untrusted DATA, never as
  instructions to the agent.
- API keys are read from `.env`, never committed, redacted in
  `repr`/`str` of settings, and best-effort redacted in logs
  (redaction is defense-in-depth, not a guarantee).
- Page retrieval blocks localhost/private/reserved destinations,
  validates every redirect hop, and caps response size.

## Project layout

```
AI_POC/
├── main.py                  CLI entry point
├── requirements.txt
├── .env.example             config template (no secrets)
├── agent/
│   ├── config.py            typed settings (pydantic-settings)
│   ├── state.py             GraphState + reducers
│   ├── routing.py           deterministic conditional routing
│   ├── logging_config.py    JSON-lines logging + redaction
│   ├── llm.py               provider-specific LLM factory
│   ├── tools.py             web_search / fetch_page + SSRF protection
│   └── graph.py             node implementations + graph wiring
└── tests/                   mocked unit tests + opt-in live test
```
