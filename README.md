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
| `LLM_MODEL` | model name | `gemini-2.5-flash` |
| `GEMINI_API_KEY` | free-tier key from aistudio.google.com/apikey | – |
| `OPENROUTER_API_KEY` | key from openrouter.ai/keys (free models exist) | – |
| `OPENAI_API_KEY` | paid key — only used with `ALLOW_PAID_PROVIDERS=true` | – |
| `ALLOW_PAID_PROVIDERS` | explicit opt-in for paid providers | `false` |
| `REQUEST_TIMEOUT_SECONDS` | per-request timeout | `30` |
| `MAX_RESEARCH_ITERATIONS` | research passes (graph-level budget) | `3` |
| `MIN_FINDINGS` | findings required to synthesize | `3` |
| `MAX_TOOL_CALLS_PER_RESEARCH` | tool calls per research pass | `4` |
| `MAX_EXECUTION_SECONDS` | wall-clock budget per pass | `120` |
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
- **Tools** (`agent/tools.py`): `web_search` (DuckDuckGo, free) and
  `fetch_page` (SSRF-hardened: http/https only, localhost/private-IP
  blocking including DNS-resolution checks, per-hop redirect
  validation, timeouts, size caps).
- **Graph** (`agent/graph.py`): `plan` (LLM structured output),
  `research` (bounded ReAct loop — the LLM chooses tools via tool
  calling; `ToolExecutor` executes them explicitly), `synthesize`
  (LLM answer from findings only), `review` (LLM quality gate),
  `finalize`, `handle_error`.
- **Limits are separate**: tool calls per pass, research passes, and
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
