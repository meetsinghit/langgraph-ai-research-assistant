# GitHub publication readiness — source snapshot audit

**Snapshot:** Local Git HEAD reported as `aec1648` (Phase 4E.4), uploaded as a `git archive` ZIP on 15 October 2026. 177 entries. The ZIP includes the tracked file tree only; **it does not include the `.git` directory or historical commits**.

**Purpose:** Inform a decision about publication of a personal engineering portfolio project. This is **not** a professional security assessment, a dependency vulnerability scan or legal/IP clearance.

## Executive decision

| Destination | Assessment | Rationale |
|---|---|---|
| **Private GitHub repository** | Suitable for continued development, subject to account access controls | The preliminary tracked-snapshot scan found no common secret-pattern matches; source files can remain private |
| **Public GitHub repository** | **Hold pending actions below** | Historic commits were not available for full re-scan; ownership, personal identifiers and licensing still require review |
| **Live hosted deployment** | **Not recommended without separate engineering review** | Application is designed and tested for single-user localhost; Docker runtime and multi-user/public exposure are not verified |

## Review evidence and limits

| Inspection | Result | Limitation |
|---|---|---|
| Archive traversal/name safety | 177 entries inspected and extracted with traversal checks | Current snapshot only |
| Common embedded secret signatures | **0 matches** for Google `AIza…`, GitHub token prefixes, AWS access keys, PEM private key headers, long literal Bearer tokens and high-risk key assignments in readable snapshot | Regex misses unusual formats, obfuscation and unrecognized credentials; false negatives possible |
| `.env` | **Not included in archive** | The owner previously also confirmed `.env` is ignored; historical blob content is not in this archive |
| `data/app_token.txt` / SQLite | **Not included in archive** | Local runtime secrets/database content were not audited (correctly excluded) |
| `.env.example` | Included as expected | Must remain placeholder-only; never replace placeholders with real credentials |
| Company/project name patterns | No obvious hits for the checked client/company names in readable source | Does **not** prove IP ownership, public-release rights or absence of company-confidential concepts |
| Personal OS path | **1 instance found** in `docs/SESSION_HANDOFF.md` | The proposed improvement patch replaces that *current* path with a generic path; old committed history still exists |
| Test email-like strings | Examples in `tests/test_network_security.py` use `example.com` in SSRF payload tests | Not real personal email addresses |
| Dependency vulnerability scan | **Not performed** | Pins are present but do not guarantee secure dependencies |
| Automated tests / live model behavior | **Not independently executed** against this archive | Existing README reports earlier OpenCode test results; user provided live browser screenshots/metrics |
| Docker deployment | **Not run** | No Docker on original test machine |

### Important historical-commit limitation

The owner previously ran filename and common-Google-key pattern checks across local Git history and reported no concerning matches. That is reassuring but **not equivalent to an exhaustive scan of every historical file and credential format**. A Git archive cannot independently verify the historic commit tree. Publishing the repository makes historic commits and deleted files inspectable; removing a secret from today's HEAD would not remove it from earlier commits.

## Required before changing visibility to Public

- [ ] **Run a full-history secret scan** with a trusted scanner such as Gitleaks or an equivalent (approved for the environment); review results locally without pasting secret values into chat, tickets or CI logs. If any real secret appears, rotate it and purge affected history *before* public release.
- [ ] **Inspect `docs/SESSION_HANDOFF.md` and older commits** for personal machine/account paths. The current-file patch anonymizes one path, but Git history still contains the earlier copy. Decide whether to leave harmless history, keep the repo private, or intentionally rewrite history with proper backup and coordination.
- [ ] **Manually inspect `.env.example`**, the entire committed README/docs and images for actual tokens or internal configuration values. Automated patterns are not comprehensive.
- [ ] **Confirm ownership/permission to release** source code, screenshots, diagrams and any work produced on employer-managed equipment or accounts, under applicable agreements/policies. Nothing in a content scan grants publication rights.
- [ ] **Choose a license deliberately** (or explicitly keep the repository all-rights-reserved with no license). No LICENSE file existed in the uploaded snapshot.
- [ ] **Review screenshots and HTML handbook** for sensitive research prompts, browser notifications, client information, hidden screenshots, or user-specific file paths. The proposed screenshots were selected from user-provided app images and the handbook path was sanitized.
- [ ] **Run dependency/security scanning and current tests** in the owner's environment, then record dates/results clearly. Pay attention to Python/Node versions and dependency updates.
- [ ] **Review public risk boundaries:** local-only app; auth token cannot replace proper Internet-exposed identity and authorization; Gemini quotas/billing vary; downloaded source does not include actual keys.
- [ ] **Double-check GitHub settings** (visibility, collaborator access, branch protection if desired, secret scanning availability, GitHub Actions permissions) before release.

## Content improvements delivered separately

| Deliverable | Path |
|---|---|
| High-impact GitHub landing page | `README.md` |
| Interactive engineering reference | `docs/technical-handbook.html` |
| New architecture diagrams and real screenshots | `docs/assets/` |
| GitHub description, tags and social image instructions | `docs/GITHUB_METADATA.md` |
| Interview/demo narrative | `docs/PORTFOLIO_STORY.md` |
| Current-head personal path redaction | `docs/SESSION_HANDOFF.md` |

## What was not changed

- No LangGraph/FastAPI/React/Python application source or tests were modified.
- No network calls, API keys, GitHub push, repository visibility changes or license changes were performed.
- No Git history rewrite was performed. The owner controls any later commit and push.

## Safe-first publication process

1. Review the proposed file changes and images locally.
2. Apply the documentation patch; verify the app and CI/tests remain intact.
3. Run a **history-aware** credential scan before any visibility change.
4. Confirm employer/IP rights and decide on licensing.
5. Commit and push to the **private** GitHub repository.
6. Re-check the public-facing README and GitHub About fields.
7. Only then consider making the repository public.

**Bottom line:** The tracked source snapshot is a credible basis for a public-facing portfolio presentation, but current-snapshot cleanliness is **not** enough by itself to certify public-release readiness.
