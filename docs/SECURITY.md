# Threat Model & Security Design — AI Research Assistant (Phase 4C, API surface added in Phase 4D)

**Status:** POC, verified *offline* against mocked transports,
scripted models and fixture sources. Passing tests and eval cases
demonstrate that the implemented controls behave as specified — they
do **not** prove the application is production-secure, and they say
nothing about how a live model responds to real adversarial content.
See [Verification status](#7-verification-status) and
[Residual risks](#8-residual-risks) before drawing conclusions.

---

## 1. Scope and assets

The agent researches a user question with two tools (`web_search`,
`fetch_page`), synthesizes an answer with citations, and reviews it
before returning it. **Assets in scope:**

| Asset | Why it matters |
|---|---|
| Provider API keys (Gemini/OpenRouter/OpenAI) | Cost, quota, account compromise |
| The run's final answer | Decisions made by its readers; credibility of citations |
| The host machine / local files | Any path from web content to local reads or execution |
| Outbound network position | SSRF into loopback, private ranges, cloud metadata |
| Logs, metrics, reports | Must stay content-free (no payloads, keys, or full URLs) |
| Prompt integrity | The system prompts are the control plane for every gate |
| **Runtime data (Phase 4D)** — `data/app.db`, `data/checkpoints.db` (+ WAL), `data/app_token.txt` | Conversation content, answers/sources, LangGraph checkpoints and the local API token: sensitive at rest; checkpoint files additionally hold prompt-derived state |

**Out of scope (explicitly):** deployment infrastructure, provider
side, the operator's own machine hygiene, and *semantic truth* of
sources — citation validation proves references exist and are
provenanced, not that a page's claims are true.

---

## 2. Actors

| Actor | Capability | Trust |
|---|---|---|
| Operator (POC user) | Chooses questions, config, keys | Trusted (but subject to honest failure) |
| Model (Gemini) | Produces plans, tool calls, drafts | **Semi-trusted**: follows schemas, but can be influenced by content it reads (indirect injection) |
| Search provider | Returns titles/snippets/URLs | **Untrusted** (index can host attacker pages) |
| Fetched web page | Arbitrary attacker-controlled text | **Untrusted** (primary adversary) |
| Tool/error messages | Echo model- or URL-influenced text | **Untrusted** (relay channel) |
| Dependencies | Code that runs in-process | Trusted only via pinning/review |

The design follows one rule: **retrieved content is evidence, never
authority.** Everything untrusted may be *quoted* to the model; it may
never *authorize* an action.

---

## 3. Trust boundaries

Nine boundaries, each with its failure mode, controls, and residual
risk. "Deterministic" = enforced in code, testable offline; "signal"
= advisory detection; "model" = depends on model behavior.

### TB1 — Operator question → agent

*Adversary:* a malicious *prompter* (in a shared/multi-tenant future)
or a question that merely *contains* hostile instructions.

- **Failure:** direct prompt injection through the question itself.
- **Controls:** system prompts are fixed in code and end with a trust
  boundary clause (`_TRUST_BOUNDARY`); the question is placed as user
  input, not instruction; schema validation bounds argument sizes
  (`query ≤ 400`, `url ≤ 2048`); tool surface is an allow-list of two.
- **Residual:** the operator is trusted here. Multi-tenancy would
  require per-tenant isolation not present in the POC.

### TB2 — Model output → tool invocation *(tool-authorization gate)*

*Adversary:* a model coerced by injected content (or a buggy model).

- **Failure:** calls to tools that don't exist (`read_file`,
  `run_shell`), oversized/smuggled arguments, or tools authorized by
  *producing* their names.
- **Controls (deterministic):**
  - `ToolExecutor`'s function table **is** the allow-list — exactly
    `web_search` and `fetch_page` exist; unknown names and non-string
    names are refused with `error_category="validation"` and never
    execute;
  - strict pydantic schema per tool (types, required keys, extra keys
    rejected, length limits) — failures are `validation`, not
    partial-execution;
  - `bind_tools` only ever advertises the two tools (asserted by test);
  - per-pass call budgets and a global deadline bound volume/time.
- **Evidence:** `tests/test_tool_policy.py`,
  `sec-tool-spoof-read-file`, `sec-tool-spoof-shell`,
  `sec-text-form-tool-call-inert`, `sec-oversized-tool-args`.
- **Residual:** none known for this control — refusal is structural,
  not model-mediated.

### TB3 — Search results (titles/snippets/URLs) → agent

*Adversary:* an attacker who can rank a page or poison a snippet.

- **Failure:** instruction override, fake roles, credential requests,
  attacker URLs, suppressed citations delivered as *data*.
- **Controls:**
  - snippets enter state only as bounded, provenance-labeled
    `"snippet"` findings — never as `retrieved` evidence;
  - the signal detector scans titles+snippets
    (`scan_search_results`) and attaches a **content-free**
    `security_note` to the ToolMessage (category names only);
  - the fetch-target allow-list registers only structured
    `result["url"]` values — an attacker URL *mentioned inside* a
    snippet is **not** fetchable;
  - system prompts carry the `UNTRUSTED QUOTED EVIDENCE` framing.
- **Evidence:** `sec-snippet-ignore-instructions`,
  `sec-snippet-exfil-url`, `sec-snippet-citation-suppression`,
  `sec-snippet-role-spoof-authority`.
- **Residual:** snippet text still reaches the model verbatim
  (trimmed) — see [Residual risks](#8-residual-risks) on why the
  detector is signal-only.

### TB4 — Fetched page content → agent *(the primary attack surface)*

*Adversary:* any page the search led to (indirect prompt injection).

- **Failure:** fake `[SYSTEM]` directives, conclusion dictates,
  credential/env exfiltration requests, encoded payloads, instructions
  to disable TLS, citation hijacking.
- **Controls:**
  - **prompt layer** — all four system prompts end with the trust
    boundary clause; evidence blocks are wrapped in
    `UNTRUSTED QUOTED EVIDENCE` framing (`_evidence_blocks`);
  - **detector layer** — `agent/injection.py` scans page text and
    errors with 7 fixed categories
    (`instruction_override`, `role_spoof`, `credential_request`,
    `policy_tamper`, `exfil_url_request`, `conclusion_directive`,
    `encoded_payload`), capped per category and in bytes; results are
    *signals* recorded in metrics + a content-free ToolMessage note —
    **the evidence is never blocked, stripped, or rewritten**;
  - **deterministic gates downstream** — even a fully compliant model
    cannot exfiltrate (TB3/TB5/TB6 controls) or cite nonexistent
    sources (citation validator);
  - **HTML handling** — comments/scripts are stripped before text
    enters state, and only bounded relevant excerpts are retained.
- **Evidence:** `sec-page-*` cases, `tests/test_injection_signals.py`
  (including `sec-legit-security-doc-preserved`: legitimate security
  documentation that *quotes* attack phrases keeps working).
- **Residual:** what a real model *does* with quoted hostile text is
  not established offline (see §7).

### TB5 — Tool calls/errors → outbound network *(SSRF & egress)*

*Adversary:* a URL chosen by a coerced model, redirects, DNS answers.

- **Failure:** loopback/private/link-local/metadata targets, DNS
  rebinding (TOCTOU), redirect chains to internal hosts, userinfo or
  odd ports hiding targets, oversized/slow responses, wrong content
  types, leaked credentials in requests.
- **Controls (deterministic):**
  - `_validate_url` on **every** URL and **every redirect hop**:
    http/https only; no userinfo; ports restricted to 80/443;
    hostname resolved and **every** address checked against
    loopback/private/link-local/reserved sets (IPv4 and IPv6);
  - redirects are never followed automatically
    (`follow_redirects=False`); each hop is re-validated, deadline
    checked, hop count capped;
  - **fetch-target allow-list** (`_DiscoveredSources`): `fetch_page`
    refuses any URL `web_search` did not return *this run*
    (`PolicyRefusal`, category `policy`) — attacker URLs embedded in
    content can't be fetched even by a compliant model;
  - **DNS TOCTOU pinning** for cleartext HTTP: `_connect_target`
    connects to the *validated* address with a `Host` header
    (own-client only), so a re-bound DNS answer between check and
    connect cannot redirect the request;
  - response caps: size (`MAX_PAGE_BYTES`), redirects, timeouts; only
    `text/*` and `application/xhtml+xml` accepted;
  - TLS: `build_ssl_context` — OS trust store / `SSL_CA_BUNDLE` /
    certifi, always verifying; **no `verify=False` path** (static
    guard test).
- **Evidence:** `tests/test_network_security.py` (mocked transports +
  controlled DNS only), `sec-fetch-unlisted-attacker-url`,
  `sec-fetch-metadata-service`, `sec-fetch-file-scheme`.
- **Residual:** the HTTPS pinning window (§8).

### TB6 — Model output → final answer *(exit point)*

*Adversary:* credential-shaped values reaching the answer (hallucinated
or echoed from a page), fabricated citations, suppressed provenance.

- **Controls (deterministic):**
  - `redact_credentials()` at synthesis **and** again at `finalize`
    (defense in depth for whichever path produced the answer); counts
    recorded in metrics, values never logged;
  - citation validator resolves every `[Sn]` against actual findings —
    unknown references are counted (`unknown_citations`), and the
    deterministic quality gate downgrades a scripted approval to
    `needs_more_research` rather than letting it ride;
  - verification gate: snippets alone never pass silently — failed
    verification appends the `UNVERIFIED / INCOMPLETE` marker.
- **Evidence:** `sec-answer-credential-redaction`,
  `sec-fabricated-citation`, `sec-citation-manipulation`,
  `sec-blocked-source-unverified`, `tests/test_secret_protection.py`.
- **Residual:** shape-based detection misses exotic credential formats
  (§8).

### TB7 — Secrets → prompts, logs, metrics, errors

*Adversary:* any content that asks the model to reveal or transmit a
key; any code path that formats secrets into visible strings.

- **Controls (deterministic):**
  - **the model never needs a secret** — prompts are built from the
    question, evidence, and plan; API keys are not part of any prompt
    (verified end-to-end with a synthetic canary in
    `test_canary_secret_never_reaches_model_prompts`);
  - **secret-DLP in `ToolExecutor`** — tool arguments are scanned for
    *configured secret values* (≥ 8 chars) and refused *before*
    execution (`error_category="dlp"`); refused arguments are **not**
    echoed (`args={}`) and the error names no values;
  - metrics record tool name/outcome/size/category — **never
    arguments or results** (`ToolCallMetrics`);
  - `Settings` secret fields are redacted in `repr`/`str`; logging
    redacts known key shapes; the security metrics block is
    counts/categories only;
  - errors never interpolate the offending value (userinfo URLs are
    refused without echoing credentials).
- **Evidence:** `sec-dlp-key-in-tool-args`, `sec-page-env-dump-request`
  and the whole `test_secret_protection.py` (canary assertions over
  prompts, logs, metrics, report JSON).
- **Residual:** DLP only covers values the app *loaded* — a secret it
  never held can't leak from it either (that's the actual guarantee:
  **secrets stay out of model context entirely**).

### TB8 — Configuration & supply chain → runtime

*Adversary:* dependency confusion, unpinned updates, committed
secrets, unsafe defaults.

- **Controls:** every dependency in `requirements.txt` is exactly
  pinned (`==`); `.env`/`.env.*` are git-ignored (guard test); eval
  reports are ignored artifacts; `Settings` uses `extra="ignore"`
  (no config-key smuggling) with a `_SECRET_FIELDS` registry kept
  aligned by test; paid calls are gated (live tests opt-in, LLM judge
  off by default); CA bundle comes from config/env, not content.
- **Evidence:** `tests/test_config_security.py`.
- **Residual:** no automated vulnerability scanner was run (approval
  required for installs/scanners); pinning ≠ audit — review of each
  pinned package's release was manual and shallow (§8).

### TB9 — Application → logs, metrics, reports *(observability channel)*

*Adversary:* anyone reading logs/reports who shouldn't see payloads,
keys, or full URLs.

- **Controls:** metrics and logging are content-free by design —
  category names, counts, sizes, durations only; injection signals are
  recorded as category+source counts; security refusal counts by
  category; report writer asserts (by test) that the synthetic
  credential never appears in report JSON or console summaries.
- **Residual:** tool result *sizes* and *latencies* are still visible
  (accepted: they're operational metadata, not content).

---

### TB10 — API client → server *(Phase 4D HTTP surface)*

*Adversary:* anything that can reach the API port with requests it
shouldn't make (other local processes, or a remote client if the
operator exposes the port).

- **Controls:**
  - **Bearer auth on everything except `GET /api/health`** — the
    token comes from `APP_TOKEN` or is bootstrapped once into
    `data/app_token.txt` (owner-only file on POSIX; git-ignored
    everywhere). SSE streams require the same header (a browser
    `EventSource` alone is therefore not enough — no silent
    unauthenticated event leak).
  - **`127.0.0.1` default bind** — the server is loopback-only
    unless the operator explicitly changes `api_host`; it is never
    intended for public exposure (no TLS, single-user token).
  - Empty CORS allow-list by default; docs/OpenAPI/schema routes
    disabled; request bodies capped at 64 KiB.
  - **Responses and SSE events are contract-typed**: progress
    events are content-free (node/tool/category/latency only —
    never arguments, page text or prompts), error responses carry a
    fixed message plus a coarse `error_category`, and provider
    exception text is type-only in state (raw text goes to logs).
  - Session/run IDs are unguessable UUIDs and always scoped — a
    404/403 is returned for cross-session access attempts.
- **Residual:** the token is stored in plaintext next to the data
  it protects (acceptable while loopback-only); no per-user authn,
  no rate limiting beyond concurrency limits (see R8).

---

## 4. Threat catalog → controls (OWASP LLM mapping)

| Threat (style) | Boundary | Primary control(s) | Layer | Test evidence |
|---|---|---|---|---|
| LLM01 Direct prompt injection | TB1 | Fixed system prompts + trust-boundary clause, tool allow-list, schema limits | deterministic | schema/policy tests |
| LLM01 **Indirect** prompt injection | TB3/TB4 | Untrusted-evidence framing, signal detector, content-free security notes, downstream gates | signal + model | `sec-page-*`, `sec-snippet-*`, `test_injection_signals` |
| LLM02 Sensitive information disclosure | TB7/TB9 | No secrets in prompts, DLP refusal, content-free logs/metrics, redaction | deterministic | `test_secret_protection`, `sec-dlp-*` |
| LLM05 Improper output handling | TB6 | Credential-shape redaction at synthesis + finalize, citation validation, verification gate | deterministic | `sec-answer-credential-redaction`, `sec-fabricated-citation` |
| LLM06 Excessive agency | TB2 | Two-tool allow-list, strict schemas, refusal categories, budgets, no shell/fs/code tools | deterministic | `test_tool_policy`, spoof cases |
| LLM07 System prompt leakage | TB4 | Prompt extraction attempts are signaled; **design**: prompts contain no secrets to steal | signal + design | `sec-page-prompt-extraction` |
| LLM09 Misinformation / citation abuse | TB3/TB6 | Provenance-labeled evidence, citation validator, `UNVERIFIED` marker | deterministic | citation cases, `sec-blocked-source-*` |
| LLM10 Unbounded consumption | TB2/TB5 | Per-pass call budgets, global deadline, response/redirect/timeout caps | deterministic | existing WS-C tests, `test_network_security` |
| SSRF / internal network access | TB5 | URL validation per hop, address-set checks, HTTP pinning, fetch-target allow-list | deterministic | `test_network_security`, `sec-fetch-*` |
| Secrets in transit | TB5 | No credential-bearing requests (headers tested), userinfo blocked | deterministic | `test_fetch_sends_no_credential_headers` |
| Supply chain | TB8 | Full pinning, git-ignored secrets, guard tests | deterministic | `test_config_security` |

---

## 5. Layered defense design (and why)

1. **Prompt layer** — cheap, improves *real-model* behavior, but not
   a security boundary on its own: it is exactly what prompt injection
   attacks. It exists to *reinforce*, never to *carry* a guarantee.
2. **Signal layer (detector)** — deliberately **signal, not proof**:
   - it never blocks or rewrites evidence (legitimate security docs
     that quote attack phrases must remain researchable —
     `sec-legit-security-doc-preserved`);
   - a *clean* result means "no known pattern matched", never "safe";
   - categories are fixed vocabulary; notes/metrics carry names and
     counts only (content-free), so the detector itself can't become
     a relay channel.
3. **Deterministic policy layer** — where the actual guarantees live:
   fetch-target allow-list (breaks the *outbound* leg of the
   exfiltration "lethal trifecta"), executor allow-list + schema
   validation, secret-DLP. These hold **even if the model fully
   cooperates with the attacker** — which is why the security eval
   cases are scripted to be maximally obedient to the attack: the
   deterministic layer must make the objective unattainable anyway.
4. **Output layer** — credential-shape redaction and citation
   validation at the exit point, as a backstop for anything that got
   through earlier layers.

**Rejected alternatives:** stripping/filtering untrusted text (breaks
legitimate research, creates false confidence); blocking fetches on
*detector* verdicts (pattern matching would then gate evidence —
denial-of-research by obfuscation); putting secrets behind a
"don't print them" prompt (prompts are not a boundary).

---

## 6. Security evaluation suite (WS-G)

`python -m evals --security` runs **25 offline adversarial cases**
through the real graph (scripted model, fixture sources, no network,
no API keys). Each case records attacker objective, entry point,
expected behavior, defense layer, severity, and residual limits;
the JSON report carries a machine-readable `security_summary`
(by severity, defense layer, entry point, refusal categories).

Last run: **25/25 passed** — severity high=13, medium=9, low=3;
defense layer deterministic=20, model_behavior=5; refusals fired
policy=3, validation=3, dlp=1; 19 injection signals; 2 answer
redactions.

Honesty rules baked into the suite:

- cases where the model *cooperates with the attack* pass only via
  deterministic controls;
- `model_behavior` cases exist to **document** the untestable
  surface, not to claim it's defended;
- no case ever asserts "zero signals = safe".

---

## 7. Verification status

**Verified offline (deterministic):** 506 passing tests (1 skipped —
the opt-in live test), including the five security suites and the
Phase 4D API suites (auth, lifecycle, idempotency, SSE replay/race,
crash-recovery/resume, cancellation, canary exfiltration over every
response surface); benchmark 23/23 with no unexplained regressions;
security suite 25/25; mocked-transport SSRF/redirect/pinning/TLS
tests; canary-based exfiltration tests over prompts, logs, metrics,
report JSON, API responses, SSE streams and the SQLite files.

**NOT verified:**

- **live model behavior** against real adversarial content (no live
  Gemini calls were made in this phase by instruction);
- **real network adversarial conditions** (DNS rebinding against a
  live resolver, real redirect chains) — only simulated with
  controlled fixtures;
- **a real penetration test** (no probing of any system, internal or
  external, was performed or is implied);
- **dependency vulnerabilities** (no scanner ran; pinning ≠ audited).

---

## 8. Residual risks

| # | Risk | Why it remains | Mitigation if it fires |
|---|---|---|---|
| R1 | **HTTPS DNS-rebinding window** — HTTP fetches are address-pinned, HTTPS are not: pinning would break SNI and certificate hostname verification, and we never trade verification for pinning (`verify=False` is worse). | Cryptographic constraint (small real-world window: attacker must control DNS between two adjacent lookups *and* serve a valid cert for the hostname — which requires CA compromise or a misissued cert). | HSTS-preloaded targets, OS resolver caching, and the fetch-target allow-list (attacker can't choose arbitrary hosts) shrink exposure; document any future change. |
| R2 | **Detector is pattern-based** — paraphrased, translated, split or steganographic instructions may yield zero signals. | Signal layer is explicitly advisory; relying on it as a gate would make evasion = censorship of evidence. | The guarantees sit in the deterministic layers (allow-list, DLP, schemas); detector signals are triage metadata. |
| R3 | **Real-model compliance unknown** — a live model may still *repeat* hostile text or *attempt* refused actions more often than fixtures suggest. | Offline scripting cannot model persuasion. | Refusals are deterministic regardless of attempts; live-run validation (pending approval) should watch `security.*` metrics. |
| R4 | **Shape-based redaction gaps** — exotic credential formats may pass the answer filter. | Format space is open-ended. | Secrets never enter model context (the actual control); redaction is the backstop, not the primary. |
| R5 | **No outbound rate limiting / destination reputation** — budgets bound *this run*, not a compromised long-lived deployment. | POC scope. | Deploy behind egress controls with allow-listing. |
| R6 | **Supply-chain depth** — pins prevent drift but don't attest provenance (no hashes/SBOM/signatures). | Approval-gated tooling constraint. | Add hash-pinning + SBOM scan before any production pilot. |
| R7 | **Single-tenant trust** — the operator's question is trusted input. | POC usage model. | Multi-tenant would need per-tenant isolation and TB1 hardening. |
| R8 | **Loopback HTTP + single bearer token** (Phase 4D API) — no TLS, no per-user authn, no request rate limiting; the token sits in `data/app_token.txt`. | Single-user, localhost-by-design POC scope (TB10). | Keep `api_host=127.0.0.1`; never port-forward without a TLS-terminating reverse proxy that adds real authn; rotate `APP_TOKEN` by deleting the token file and restarting. |

---

## 9. Operational guidance (pre-pilot checklist)

- Keep API keys in `.env` (git-ignored) or a secret manager; rotate on
  suspicion of exposure.
- Watch the metrics `security` block: non-zero `tool_refusals`
  (esp. `policy`/`dlp`) and rising `injection_signals` are triage
  signals to inspect *content-free* — request the source manually,
  don't log payloads.
- Re-run `pytest` + `python -m evals` + `python -m evals --security`
  before any change to prompts, tools, or network code.
- Treat R1–R8 as open items for a production readiness review.

---

## 10. Persistence, deletion and secure erasure (Phase 4D)

**What is persisted where:**

| File | Contents | Lifetime |
|---|---|---|
| `data/app.db` (+ `-wal`/`-shm`) | Sessions, questions, answers, review status, sources, sequenced progress events, run diagnostics | Until deleted; rows cascade with their session |
| `data/checkpoints.db` (+ `-wal`/`-shm`) | LangGraph checkpoints (full node state, including prompt-derived content) for resumable runs | Checkpoints are **pruned when a run reaches a terminal state**; only interrupted runs keep one (and it is deleted when the resumed run finishes) |
| `data/app_token.txt` | The local API bearer token | Until deleted (then a new one is bootstrapped) |

**Deletion is not secure erasure.** Deleting these files (or a
session row) removes them from the filesystem/logic path but does
**not** guarantee the bytes are gone: SQLite pages survive in free
pages and WAL journals until overwritten, SSD wear-levelling and
journaling file systems keep stale copies, and page-level
backups/snapshots may hold any of it. The data is treated as
**sensitive at rest by policy**: `data/` is git-ignored, files are
created owner-readable where the OS supports it (POSIX), and no
plaintext secrets are ever written to them (credentials are redacted
before state is stored; the API token is the one deliberate secret,
kept in its own file rather than in the database). Where erasure
must be cryptographically guaranteed, use an encrypted volume and
destroy the key — file deletion alone is documented here as
insufficient.

**API-side invariants** (enforced by tests): configuration secrets
and the bearer token never appear in any response, event stream or
SQLite byte range; provider exceptions surface as type + category
only; answers are credential-redacted before persistence; progress
events are whitelisted and content-free.
