# Session Handoff — Phase 4E.3 Complete, Awaiting Review

> Updated after Phase 4E.3 (absorbed the old 4E.4 scope: browser
> control of model + thinking, depth/evidence, session rename,
> diagnostics/progress/errors UX, tests, docs). Phase 4E.1 was
> committed as `9bf5a4b`, 4E.2 as `1342259`, 4E.3 as `f404c49`;
> the handoff checkpoint for 4E.1 is `60407e3`.

## Git state

- Branch: `main`
- Chain: `93bbabc (4D.1) → e6b7ec3 (4D.2) → 70c97ca (4D.3) → 9bf5a4b (4E.1)
  → 60407e3 (handoff) → 1342259 (4E.2) → f404c49 (4E.3) → <this handoff>`
- Baseline for 4E.3: `1342259` — verified clean before starting.

## Current phase

Phase 4E.3 (model/thinking registry + per-run frozen configuration +
frontend settings UI + session rename + diagnostics/errors UX) is
**implemented, gated, documented, and committed** — awaiting user
review. **No further phase has been started** (explicitly out of
scope until approved).

## Completed in Phase 4E.3

### WS-A — backend-managed model catalogue
- **Registry** `agent/gemini_models.py`: 11 active + 2 tombstone
  ModelSpec entries (verified against official docs 2026-10-14).
  Only `free_tier == "eligible"` models are selectable: 3.8-flash
  (default), 3.6-flash, 3.5-flash-lite, 3.1-flash-lite,
  3-flash-preview, 2.5-pro/flash/flash-lite. 3.7-flash & 3.5-flash
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

- Python: **640 passed, 1 skipped** (4E.2 baseline 585 → +55)
- Offline quality benchmark: **30/30**
- Offline security benchmark: **25/25**
- Frontend: typecheck ✓, **104/104** (baseline 84 → +20), build ✓
- Real-browser E2E: **20/20** (baseline 18 → +2: frozen run
  settings, inline rename persistence; ran because the UI changed)
- Windows launcher smoke ✓ — unattended start → health 200 → UI
  200 → `/api/models` 401-without/200-with token (default
  `gemini-3.5-flash-lite`, 8 selectable, discovery `disabled`) →
  clean stop
- No live Gemini calls; no dependency changes; fixtures regenerated
  from the real FastAPI app (`dump_api_fixtures.py`, +`models.json`
  +`session_renamed.json`).

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

1. **User reviews the 4E.3 commit** (registry boundaries, frozen-
   config semantics, rename contract, composer UX).
2. Still permanently pending: Docker runtime validation (not
   installed), live-Gemini-dependent verification (needs explicit
   user approval).

## Resume command sequence (after restart)

```powershell
cd C:\Users\meet.a.singh\Downloads\OpenCode\AI_POC
git log --oneline -3        # expect HEAD = 4E.3 handoff on f404c49
.\ai_poc\Scripts\python.exe -m pytest -q   # optional sanity: 640 passed, 1 skipped
```

Do not start another phase until the user confirms review of 4E.3.
