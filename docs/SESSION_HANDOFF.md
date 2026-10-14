# Session Handoff — Phase 4E.1 Complete, Awaiting Review

> Written at session end before an imminent laptop restart (Oct 14,
> 2026). No work was in flight: the working tree was clean and every
> planned 4E.1 change is committed. This file is itself the only
> post-4E.1 artifact (checkpoint commit — see Git state below).

## Git state

- Branch: `main`
- HEAD: `9bf5a4b` — "Phase 4E.1: research trustworthiness - evidence
  policy, grounding checks, explanation-depth gate, adaptive research
  depth" (23 files, +1981/−79; new: `agent/depth.py`,
  `tests/test_depth.py`)
- Baseline for 4E.1: `70c97ca` (Phase 4D.3) — verified clean before
  starting.
- Chain: `93bbabc (4D.1) → e6b7ec3 (4D.2) → 70c97ca (4D.3) → 9bf5a4b (4E.1)`
- Working tree: clean at checkpoint time except this handoff file
  (committed as a WIP checkpoint).

## Current phase

Phase 4E.1 (Research Trustworthiness + Adaptive Research Depth: WS-A,
WS-B, WS-G, and the related WS-F fixtures) is **implemented, gated,
documented, and committed**. It is **awaiting user review**. Phase
4E.2 (search/latency + diagnostics: WS-C/WS-D) and 4E.3 (frontend UX:
WS-E) are **explicitly NOT started** — out of scope until the user
approves.

## Completed in Phase 4E.1

- **WS-A evidence policy**: per-run `evidence_policy`
  (`balanced` default / `verified`) enforced in
  `agent/citations.validate_citations()` — verified deterministically
  fails snippet-only and citation-less answers (with findings present)
  → review downgrade → labeled incomplete result; balanced keeps
  discovered sources usable with honest UNVERIFIED labeling. Advisory
  lexical claim-to-source grounding (`GroundingCheck`) reported to the
  reviewer, never auto-fails. Retrieved-but-uncited sources reported.
- **WS-B explanation depth**: `agent/completeness.check_explanation_depth()`
  flags requested topics covered only by generic-category definitions
  ("X is a core capability for…") when a page was retrieved and the
  question is explanation-intent; produces the `EXPLANATION DEPTH:
  SUPERFICIAL` note, fails the review gate, and feeds follow-up
  research. Fixes the ws4 conditional-routing false approval.
- **WS-G adaptive depth**: new `agent/depth.py` — `quick` / `standard` /
  `deep` resolve to explicit `DepthPolicy` budgets from configurable
  `QUICK_*` / `DEEP_*` settings (`agent/config.py`); standard uses the
  historical settings unchanged; the global deadline
  (`MAX_TOTAL_EXECUTION_SECONDS`) remains the hard ceiling. Depth
  drives plan-query cap, passes, tool/fetch quotas, retrieved
  threshold, evidence chars, and synthesis/review structure guidance
  (word targets are ranges, never gates). Depth and evidence policy are
  independent axes, plain state strings (checkpointed, resume-safe).
- **Plumbing**: `RunCreate`/run-detail/summary schemas (Literal-validated,
  422 on unknown values), SQLite additive migration
  (`Database._MIGRATIONS`: `runs.research_depth`, `runs.evidence_policy`),
  repository/runs wiring, CLI `--depth` / `--evidence-policy`
  (backward compatible), persisted as run metadata.
- **Evals/tests**: harness carries per-case depth/policy and
  depth-aware budget checks; 7 new fixtures (superficial-keyword-answer,
  verified-snippet-citation, verified-retrieved-citation,
  balanced-snippet-allowed, quick-mode-secops, deep-mode-budget,
  citation-mismatch-grounding); benchmark 23 → 30; new unit tests
  (`tests/test_depth.py`, extended citations/completeness/app-api tests).
- **Docs**: README (new "Research depth & evidence quality (Phase 4E.1)"
  section, config table, CLI list, eval interpretation), `.env.example`
  (QUICK_*/DEEP_* knobs), `docs/OPERATIONS.md` limitation note.

## Gate results at commit time (all green)

- Python: **568 passed, 1 skipped** (baseline 521 → +47)
- Offline quality benchmark: **30/30** (was 23)
- Offline security benchmark: **25/25** (unchanged)
- Frontend: typecheck pass, **82/82**, build pass
- Real-browser E2E: **18/18** (run because the API schema changed)
- No live Gemini calls; no dependency changes.

## Key decisions (for the reviewer)

1. **Evidence policy is backend-enforced** in `validate_citations`,
   not prompt-only; balanced ≠ strict — snippets stay usable when
   labeled.
2. **Grounding is advisory only** (lexical overlap can false-flag
   honest paraphrase); semantic judgement stays with the reviewer.
3. **Explanation-depth check is deliberately narrow**: question
   vocabulary stripped, generic-category copula pattern, requires
   retrieved evidence + explanation-intent question — so honest
   UNVERIFIED answers and legitimate short definitions ("What is
   SecOps?") are never forced into extra research.
4. **Benchmark expectation change (needs sign-off)**:
   `ws4-routing-superficial` flipped `approved → needs_more_research`.
   This is the requested WS-B fix (a strengthening — the gate now
   catches what only the semantic rubric caught); the 3 tests
   documenting the old false approval were rewritten to assert the
   gate fires and that `reviewer_false_approvals` is empty (a new one
   is now a regression). Rubric assertions kept.
5. **Depth budgets are explicit config** (`QUICK_*`/`DEEP_*`), never
   exceeding the global deadline; no length-based gates in either
   direction (no verbosity reward, no short-answer penalty).

## Errors / flakes encountered (not code regressions)

- A DNS outage mid-session (`getaddrinfo failed`) failed two
  network-dependent `tests/test_tools.py` cases once; a PowerShell
  launcher self-test flaked once under full-suite load. All passed on
  retry and in the final clean full run (568 passed). If a post-restart
  run shows these failing, suspect environment, not the commit.

## Pending / next steps

1. **User reviews commit `9bf5a4b`** (especially the ws4 expectation
   change and the new fixtures).
2. On approval, begin **Phase 4E.2** (WS-C search/latency work, WS-D
   diagnostics reconciliation — e.g. the "1 page verified · 2 fetch
   failed" vs "1 page fetched · 0 failed tool calls" discrepancy noted
   in the SecOps session; investigate definitions before changing
   metrics).
3. Phase 4E.3 (WS-E) afterwards: frontend depth/policy selector, title
   rename, error-message UX.
4. Still permanently pending: Docker runtime validation (not installed),
   live-Gemini-dependent verification (needs explicit user approval).

## Resume command sequence (after restart)

```powershell
cd C:\Users\meet.a.singh\Downloads\OpenCode\AI_POC
git log --oneline -2        # expect HEAD = handoff commit on top of 9bf5a4b
.\ai_poc\Scripts\python.exe -m pytest -q   # optional sanity: 568 passed, 1 skipped
```

Do not start Phase 4E.2 until the user confirms review of 4E.1.
