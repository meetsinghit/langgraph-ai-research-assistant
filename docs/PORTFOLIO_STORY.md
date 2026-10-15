# AI Research Assistant — project story and interview walkthrough

**One-line description:** A stateful, evidence-aware, browser-operated agentic research system that transforms a question into a reviewed, cited report—with measurable execution and explicit provenance.

This guide explains the *engineering problem, implementation choices, trade-offs, results and limitations* in a portfolio- or interview-friendly way. It reflects the source snapshot and user-reported test results at `aec1648` (October 2026); it is **not** proof that the project is production-hardened.

---

## 1. Problem and product value

**Problem:** LLMs can answer fluently while omitting requested topics, attaching weak or fabricated citations, wasting tokens on entire webpages, or offering little visibility into what the agent actually did. A command-line script also creates friction for repeated use.

**Product response:** Build a structured workflow that separates discovery, retrieval, synthesis and review; differentiates retrieved-page evidence from discovered snippets; and exposes state, citations, warnings and run diagnostics through a persistent browser app.

| Pain point | Engineering intervention | Intended benefit |
|---|---|---|
| Plausible but unsupported claims | Canonical citations, provenance labels, bounded source extraction, review gates | Better traceability; transparent weak evidence |
| Repeated/manual research workflow | LangGraph state and conditional routing | Recoverable and inspectable multi-stage process |
| Expensive/slow prompts | Capped tool messages/evidence, bounded passes, within-run search dedup | Controlled latency and token growth |
| Crashes or disconnects | SQLite history/checkpointing, SSE replay, idempotent run creation | Retain completed work; safe reconnect without duplicate research |
| Untrusted web instructions | Deterministic tool authorization, SSRF/DLP controls, prompt source boundaries | Reduced unauthorized-operation paths |
| CLI-only experience | React UI served by FastAPI, local launchers | Research from the browser with history and diagnostics |

These are **engineering goals and implemented mechanisms**, not claims of independently measured business ROI.

## 2. Architecture in 60 seconds

> “I built an AI Research Assistant with React and FastAPI on top of a LangGraph state machine. The graph plans research, searches the web, fetches allow-listed pages, synthesizes evidence-grounded answers and reviews them against deterministic citation and completeness gates. Each run freezes its Gemini model, thinking level, research depth and evidence policy. SQLite stores sessions, outcomes and checkpoints, while authenticated SSE streams progress into the UI. I emphasized operational safeguards—timeouts, retries, SSRF prevention, prompt-injection signals, DLP, and content-minimized metrics—and evaluated the system with offline quality/security fixtures and real browser tests.”

See [the architecture illustration](assets/architecture.svg) and [the complete handbook](technical-handbook.html).

## 3. Demo scenario (5–7 minutes)

**Before demo:** Ensure the browser app is running, provider access is configured, and all questions use public non-sensitive topics. Avoid disclosing application tokens or `.env`.

1. **Show research settings (30s).** Explain model selection, supported thinking effort, depth and evidence policy. Run configuration is frozen at creation and persisted.
2. **Ask a technical question (60s).** Example: “Explain LangGraph state management and conditional routing, with an executable example and official sources.”
3. **Watch the timeline (30s).** Highlight planned queries, search and fetch events, synthesis, review and the final result—actual lifecycle events, not fabricated progress percentages.
4. **Inspect citations (60s).** Show retrieved/verified versus discovered/snippet-only source badges. Explain that “verified” is *retrieval provenance*, not proof of factual truth.
5. **Open diagnostics (60s).** Discuss tool time vs total wall-clock, provider retries, token reporting coverage, thinking tokens when available, review outcome and security signals.
6. **Reopen saved history (30s).** Close and reopen a session without re-running Gemini; explain SQLite persistence and at-least-once interrupted-node recovery.
7. **Explain trade-offs (60–120s).** Why separate planning/research/review? Why deterministic gates? Why local single-user auth? Why doesn't the system automatically retry with a paid model?

If the live provider fails, demonstrate graceful classified failure behavior and use **already-saved research** rather than consuming API quota repeatedly.

## 4. Real-world performance — what can and cannot be claimed

The user observed two successful runs of the same LangGraph technical question:

| Metric | Gemini 3.5 Flash-Lite (Low) | Gemini 3.8 Flash (Medium) |
|---|---:|---:|
| End-to-end runtime | 32.1 s | 77 s |
| Model calls | 7 | 7 |
| Retrieved pages | 1 | 2 |
| Input tokens | 14,392 | 17,877 |
| Output tokens | 2,148 | 3,411 |
| Thinking tokens | Not reported | 1,541 |
| Outcome | Completed / approved | Completed / approved |

**Valid claim:** Both selected configurations executed the same end-to-end workflow and reported complete token metadata coverage for their tracked model calls.

**Invalid claim:** “Gemini 3.5 is always faster/better/cheaper.” The runs were not controlled experiments; thinking settings, provider latency and retrieval paths differed.

## 5. Architecture decisions and rejected alternatives

| Decision | Why | Trade-off |
|---|---|---|
| LangGraph over a simple sequential script | State, conditional feedback loop, explicit retry/review transitions and checkpoints | More orchestration overhead |
| Strict **Verified** mode alongside **Balanced** mode | Trust policy should be explicit and enforceable in code | Stricter runs can finish incomplete when retrieval fails |
| `RunContext` for live deadline/cancellation | Clock objects and callbacks are not appropriate serialized checkpoint values | Resume gets a documented fresh execution budget |
| SQLite + local single-process manager | Minimal infrastructure for a personal application | Not designed for horizontally scaled/multi-user production |
| Authenticated SSE | Progressive UI without WebSockets and without token-in-URL | Client handles reconnect and replay sequencing |
| Registered fetch targets only | Prevent arbitrary model-directed external fetches | A discovered URL is still untrusted; SSRF/network checks remain necessary |
| No raw prompt/page content in metrics | Better privacy, lower leak risk | Investigations sometimes have less context; structured categories matter |
| Model registry instead of arbitrary IDs | Validate capabilities and documented free-tier eligibility | Registry needs maintenance; eligible ≠ accessible or free for all projects |

## 6. Technical questions to prepare for

<details><summary><strong>How does the research agent avoid hallucinated citations?</strong></summary>

It assigns canonical `[S#]` IDs from recorded sources, validates references against discovery/retrieval records, and applies policy-sensitive quality checks. This reduces fabricated IDs and unattributed statements; it cannot establish that every cited sentence is semantically correct.
</details>

<details><summary><strong>Why not accept web page instructions as agent commands?</strong></summary>

Retrieved pages are untrusted evidence. Even if a model becomes persuaded by malicious content, the execution boundary still validates the tool name, arguments, URL destination and fetch authorization. Detection of suspicious text is supplementary—not the enforcement mechanism.
</details>

<details><summary><strong>Does checkpointing guarantee exactly-once execution?</strong></summary>

No. Completed checkpointed steps are preserved, but if an interruption happens inside a node, that node may re-execute on resume. The application documents *at-least-once interrupted-node semantics*. Side-effecting workflows would need additional idempotency controls.
</details>

<details><summary><strong>What if Google Gemini is unavailable or quota-limited?</strong></summary>

The app categorizes failures and applies bounded, deadline-aware retries for transient categories. Permanent model-access failures do not silently switch models, especially not to paid alternatives. A new run may use a different *user-selected* eligible configuration.
</details>

<details><summary><strong>How do you make Deep Research actually deeper?</strong></summary>

Depth affects planning quotas, research iterations, tool/fetch budgets, source minimums and synthesis/review guidance; it isn't just “write a longer answer.” All modes remain bounded by the run-wide deadline, and inadequate evidence should produce a qualified rather than padded answer.
</details>

<details><summary><strong>What would be required for an enterprise production deployment?</strong></summary>

Identity and authorization per tenant, rate limits, secure network egress, hardened secret management, database access policy and encryption, background job coordination, measured live-model adversarial resistance, deployed observability, container integration and reproducible CI. The current implementation is a local-first, single-user system.
</details>

## 7. Verifiable artifacts to reference

- [`agent/graph.py`](../agent/graph.py) — node implementations and guarded orchestration.
- [`agent/citations.py`](../agent/citations.py) — evidence identifiers and citation validation.
- [`agent/tools.py`](../agent/tools.py) — search and hardened page retrieval.
- [`app/runs.py`](../app/runs.py) — lifecycle, checkpoint/resume and metrics snapshot.
- [`frontend/src/`](../frontend/src/) — actual browser frontend and research settings.
- [`evals/`](../evals/) — offline research-quality and adversarial fixtures.
- [`docs/SECURITY.md`](SECURITY.md) — threat boundaries and documented residual risk.
- [`docs/technical-handbook.html`](technical-handbook.html) — interactive, long-form study guide.

**Reference posture:** Code and documentation were inspected in an uploaded tracked-source archive. Full runtime assertions and historical Git scans were not rerun independently during creation of this portfolio pack.
