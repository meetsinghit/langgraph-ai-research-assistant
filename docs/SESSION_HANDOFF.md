# Session Handoff — Phase 4E.3 Complete + Final Review Corrections

> Updated after the Phase 4E.3 **final review corrections**
> (default model restored, model-access error clarity, eligibility
> vs availability wording, 3.1-flash-lite verification, wire-level
> thinking tests). Phase 4E.1 was committed as `9bf5a4b`, 4E.2 as
> `1342259`, 4E.3 as `f404c49`, the 4E.3 handoff checkpoint as
> `0b689f8`; the 4E.3 review corrections are the commit that
> contains this file's update.

## Git state

- Branch: `main`
- Chain: `93bbabc (4D.1) → e6b7ec3 (4D.2) → 70c97ca (4D.3) → 9bf5a4b (4E.1)
  → 60407e3 (handoff) → 1342259 (4E.2) → f404c49 (4E.3) → 0b689f8 (handoff)
  → review-correction commit (4E.3 final review corrections)`
- Baseline for the corrections: `0b689f8` — verified clean before starting.

## Current phase

Phase 4E.3 (model/thinking registry + per-run frozen configuration +
frontend settings UI + session rename + diagnostics/errors UX),
including the **final review corrections**, is **implemented,
gated, documented, and committed** — awaiting user review. **No
further phase has been started** (explicitly out of scope until
approved).

## Final review corrections (focused configuration review)

1. **Default model investigated (items 1–2).** `f404c49` changed
   nothing about the default: `agent/config.py`'s fallback
   (`gemini-3.8-flash`, from the user's own commit `36d26d6`,
   Oct 9) and `.env.example` were untouched, and the live server
   honoured the `.env` `LLM_MODEL=gemini-3.5-flash-lite` all along
   (launcher smoke printed `default=gemini-3.5-flash-lite`). The
   apparent "change to 3.8-flash" was a **reporting error** in the
   4E.3 summary. Correction: the built-in fallback and
   `.env.example` now say `gemini-3.5-flash-lite` (explicit user
   configuration still overrides it), registry notes mark 3.5-lite
   as default, `gemini-3.8-flash` **stays selectable**. Google's
   docs independently recommend "3.5 Flash-Lite or 3.8 Flash" for
   new projects.
2. **Eligibility ≠ access in the frontend (item 3).** The UI never
   renders `free_tier`/`availability` — a new test asserts model
   options carry no availability/eligibility wording; `types.ts`
   documents `free_tier` as documented pricing status and
   `availability` as `not_checked` until opt-in discovery runs.
3. **Google's 2.5 restriction (item 4).** Verified from official
   docs: 2.5 models are limited to projects that actively used
   them; new projects get 403/404. Correction: `is_model_access_failure`
   (auth + 404-model-gone) raises the clear, type-only
   `"Model access failed: <type>"` state error at every model
   invocation site (research/synthesize/review);
   `categorize_error` → new `model_access` category; frontend copy
   states "No other model is used automatically". Retries were
   already permanent for these classes and there is no
   substitution path anywhere (registry `model_unavailable`,
   frozen rows, single-model graph). Registry notes document the
   2.5 restriction; `docs/OPERATIONS.md` §7 documents the
   operator-facing remedy.
4. **`gemini-3.1-flash-lite` (item 5).** It was already in the
   backend registry AND the real `/api/models` fixture (selectable,
   one of the 8) — it was absent only from the frontend **MSW mock
   catalogue**. Its compatibility is now documented in the registry
   notes from the official model card (function calling
   **Supported**, structured outputs **Supported**, thinking
   **Supported**, free tier documented): mock now includes it, and
   tests assert it is offered/selectable.
5. **Thinking → exact provider config (item 6).** Verified offline:
   `thinking_level` binds to `reasoning_effort` (pydantic alias) /
   `thinking_budget` to its field, and langchain-google-genai maps
   them to `GenerateContentConfig.thinking_config`
   (`thinkingLevel`/`thinkingBudget`); `auto` → no `thinking_config`
   on the wire. Existing tests already assert wire-level
   `ThinkingLevel.LOW/MEDIUM`; new tests assert wire-level budget
   (8192), `auto` omission and `minimal` for a model that supports
   it.
6. **Preserved (item 7):** SQLite migration/columns, frozen per-run
   settings, security controls and legacy-row handling are untouched
   — full gates below.

## Completed in Phase 4E.3

### WS-A — backend-managed model catalogue
- **Registry** `agent/gemini_models.py`: 11 active + 2 tombstone
  ModelSpec entries (verified against official docs 2026-10-14).
  Only `free_tier == "eligible"` models are selectable:
  **3.5-flash-lite (default, restored in the review)**,
  3.8-flash (still selectable), 3.6-flash,
  3.1-flash-lite, 3-flash-preview, 2.5-pro/flash/flash-lite.
  3.7-flash & 3.5-flash
  excluded (eligibility unknown), 4-argon & 3.1-pro excluded (paid),
  2.0-flash/-lite kept as tombstones so historical selections fail
  fast. The registry — never a live listing — decides selectability.
- **`GET /api/models`** (`app/model_discovery.py` ModelCatalog):
  default model id, default thinking level, registry entries with
  capabilities + `selectable` + `exclusion_reason`, `unregistered`
  listing-only bucket, discovery status. Discovery is **off by
  default** (`MODEL_DISCOVERY_ENABLED=false`, TTL 900s, backoff
  300s, timeout 10s, single-flight, caps 200 models / 50
  unregistered); it only annotates `availability`, never
  selectability, and never stores raw error text (fixed categories).

### WS-B — thinking effort
- Vocabulary `auto | minimal | low | medium | high` (config
  `ThinkingLevel`). `auto` = parameter omitted (provider decides).
  `agent/llm.py` maps per family from the registry: 3.x level models
  → `thinking_level` (`minimal` rejected for 3.7/3.8 which don't
  offer it), 2.5-family → `thinking_budget` (minimal/low 1024,
  medium 8192, high 24576, auto = 0/omit), verification stays
  `low`. Unsupported levels are rejected at validation (422 literal
  / 400 registry), never coerced. Reasoning content stays private —
  only token counts (`thinking_tokens` from `thoughts_token_count`,
  `None` when unreported) are surfaced.

### WS-C — per-run frozen configuration
- `runs.model_id` / `runs.thinking_level` columns (migration;
  legacy rows stay NULL = "not recorded", never back-filled).
- Validation order at creation: literal well-formedness (422) →
  registry resolution (`unknown_model` / `model_not_selectable` /
  `unsupported_thinking_level` → 400) → freeze. Explicit
  `model_id == LLM_MODEL` is allowed even if not registry-listed
  (you can always keep your configured default); `null`/omitted =
  configured default.
- Execution: `RunManager._run_settings` re-resolves the row —
  tombstone/unavailable → run fails `model_unavailable` (no
  substitution, no auto-fallback); unsupported frozen thinking →
  `configuration`. Legacy NULL rows execute with base settings.
- Each run gets `settings.model_copy(update=…)` — concurrency-safe
  isolation through the untouched `model_factory(settings)` /
  `graph_factory(settings, …)` contracts.
- **Idempotency**: same key + identical question/depth/evidence/
  model/thinking → replay; any difference → 409
  `idempotency_conflict`. Legacy NULL rows compare via effective
  defaults (old clients keep replaying cleanly).

### WS-D/E/F/G/H — frontend
- **`ResearchControls`** in the composer: Model (registry
  selectable only, "Default (name)" = configured default →
  submits `null`), Thinking (capability-filtered per model;
  budget-family exposes `minimal`), Depth (Quick/Standard/Deep,
  standard default), Evidence (Balanced/Verified — independent
  axis). Disabled while submitting. Inheritance: selecting a
  session adopts its latest run's settings; a stale model no
  longer selectable is sanitized to default, unsupported thinking
  falls back to the model's default (UI never offers a combo the
  backend must reject). Catalogue failure degrades to "Default"
  with a visible note.
- **Session rename**: pencil button per sidebar row → inline edit
  (prefilled input, Save/Cancel/Escape, empty blocked,
  1–120 chars, `aria-label`s, Enter submits) → `PATCH
  /api/sessions/{id}` (auth; required title; control chars
  stripped; does not touch `updated_at` or run history).
- **Diagnostics**: new **Run configuration** section (model /
  thinking / depth / evidence, "Not recorded" for pre-4E.3 rows,
  "never back-filled" hint) + **Thinking tokens** row (provider-
  reported value or honest "Not reported"; never fabricated) +
  "N with thinking" coverage part.
- **Errors/a11y**: `ApiErrorCategory` gains `invalid_selection` /
  `model_unavailable`; `submissionMessage()` gives actionable
  copy for selection/idempotency/validation submit errors;
  `ERROR_CATEGORY_LABELS` + `categoryMessage` cover run-level
  `model_unavailable` / `configuration`. All new controls are
  labelled; selects wrap on mobile.

## Gate results at commit time (all green)

- Python: **656 passed, 1 skipped** (4E.3 baseline 640 → +16 review tests)
- Offline quality benchmark: **30/30**
- Offline security benchmark: **25/25**
- Frontend: typecheck ✓, **106/106** (baseline 104 → +2 messages
  tests), build ✓
- Real-browser E2E: **20/20**
- Windows launcher smoke ✓ — unattended start → health 200 → UI
  200 → `/api/models` 401-without/200-with token
  (`default=gemini-3.5-flash-lite`, `first=gemini-3.5-flash-lite`,
  `thinking=low` per local `.env`, **8 selectable**
  = 3.5-flash-lite, 2.5-flash, 2.5-flash-lite, 2.5-pro,
  3-flash-preview, 3.1-flash-lite, 3.6-flash, 3.8-flash; discovery
  `disabled`) → clean stop
- No live Gemini calls; no dependency changes; fixtures regenerated
  from the real FastAPI app (`dump_api_fixtures.py`).

## Key decisions (for the reviewer)

1. **Registry over listing**: free-tier eligibility/selectability is
   a separately maintained, docs-verified registry; API discovery
   (opt-in, off by default) annotates availability only. A listing
   cannot prove eligibility and is never the source of truth.
2. **Configured default is always allowed**: an explicit
   `model_id == LLM_MODEL` bypasses the selectability check (only
   for naming the configured default itself) so the deployment's own
   default can never be locked out by a registry edit.
3. **No substitution ever**: unavailable/retired models fail the run
   with `model_unavailable` and clear copy ("pick another model for
   a NEW run") rather than silently running a different model.
4. **Per-run isolation via `model_copy`**, not global mutation —
   concurrent runs with different models/thinking can't race; the
   `model_factory(settings)` single-argument contract is unchanged.
5. **Legacy rows are honest**: NULL model/thinking display as "Not
   recorded" and execute with effective defaults; they are never
   back-filled, and idempotency treats them as their effective
   defaults so old clients keep replaying.
6. **`auto` = omit**: expressed by sending no thinking parameter
   (not a sentinel value), matching provider semantics; the stored
   level is the literal `auto` for display/idempotency.
7. **Justified test edits (Stage 1)**:
   `test_gemini_2_5_models_use_thinking_budget` (medium → 8192, the
   documented budget) and `test_response_usage_absent_is_none_not_zero`
   (3-tuple + reasoning breakdown); `_crash_first_app` gained an
   optional `payload` param (defaults to the original body).
8. **Rename doesn't bump `updated_at`**: it's metadata editing, not
   research activity — sorting/history semantics unchanged.

## Errors / flakes encountered

- Known spurious first-run exits from `python -m evals` (quality and
  security) — clean consecutive reruns: 30/30 and 25/25, exit 0.
- One frontend-suite run failed 13 tests while pytest + 2 eval
  suites ran concurrently (environment setup 1051s under load);
  solo rerun: 104/104 in 50s. Not product failures — load flakes.
- Mid-development: 5 new component tests failed on my own test bugs
  (raw vs title-cased thinking label in the panel; prefilled rename
  input assumed empty; multi-match text queries; inherited-thinking
  expectation). Fixed the product casing inconsistency
  (`settingLabel` for thinking) and corrected the tests.

## Pending / next steps

1. **User reviews the 4E.3 correction commit** (restored default,
   `model_access` error path, eligibility-vs-availability wording,
   3.1-flash-lite verification, wire-level thinking tests).
2. Still permanently pending: Docker runtime validation (not
   installed), live-Gemini-dependent verification (needs explicit
   user approval).

## Resume command sequence (after restart)

```powershell
cd <project-root>
git log --oneline -3        # expect HEAD = 4E.3 correction commit
.\ai_poc\Scripts\python.exe -m pytest -q   # optional sanity: 656 passed, 1 skipped
```

Do not start another phase until the user confirms review of the
4E.3 corrections.
