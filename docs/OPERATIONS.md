# Operations guide (Phase 4D.3)

How to run, troubleshoot and back up the AI Research Assistant on
Windows. Everyday research needs **no terminal**: the launchers start
everything and open your browser.

---

## 1. One-click startup (Windows)

| Want | Double-click |
| --- | --- |
| Normal daily use | `start-research-assistant.bat` |
| UI development (hot reload) | `start-research-assistant-dev.bat` |
| Stop what the launcher started | `stop-research-assistant.bat` |

What the production launcher does, in order:

1. Verifies the `ai_poc` virtual environment and its dependencies
   (actionable error with the exact fix command if missing).
2. Builds the browser UI on first run if needed (requires Node.js
   once; afterwards the built files are reused).
3. Warns (without failing) when no `GEMINI_API_KEY` is configured —
   the app starts, but runs report a provider error until one exists.
4. Refuses to start a duplicate: if the app already answers on the
   configured port, it just opens the browser.
5. Starts the server in the background, waits for its health check,
   then opens `http://127.0.0.1:8000/` (or your `API_PORT`).
6. Offers — never prints — the access token: press `C` + Enter at the
   prompt to copy it to the clipboard for the browser dialog.

The server keeps running if you close the launcher window. Press
Enter in the launcher, or run `stop-research-assistant.bat`, to stop
it. The stop script kills **only** processes it started (PID file +
command-line verification); unrelated programs are never touched.
No administrator privileges are required.

Automation knobs (also useful for scripted smoke tests):
`RESEARCH_ASSISTANT_NO_BROWSER=1` skips opening the browser;
`RESEARCH_ASSISTANT_UNATTENDED=1` skips the interactive prompts.

## 2. First-run authentication

- The browser asks for the **access token** once per page load
  (the UI keeps it in memory only — never in the page URL, storage
  or logs; a refresh simply asks again).
- The token is bootstrapped automatically on first server start into
  `data/app_token.txt` (a cryptographically secure random value,
  git-ignored, never overwritten). The launcher can copy it to your
  clipboard; otherwise open the file and paste its contents.
- Advanced: set `APP_TOKEN` in `.env` to choose your own value. It is
  never embedded in the browser bundle.
- Behind the token: every `/api` route requires it (including the SSE
  stream); only the health probe and the static UI files are public,
  and they reveal nothing.

## 3. Everyday research in the browser

- **Ask**: type a question in the composer; `Enter` submits,
  `Shift+Enter` adds a line. Progress streams live into the timeline;
  `Stop research` cancels cooperatively.
- **Research settings**: above the question box, four selectors are
  frozen into each run when it starts — **Model** (free-tier
  catalogue from the backend registry; *Default* = the configured
  `LLM_MODEL`), **Thinking** (Auto/Minimal/Low/Medium/High — only
  levels the selected model supports are offered), **Depth**
  (Quick/Standard/Deep) and **Evidence** (Balanced/Verified, see
  §3 of the README for what each does). Opening a session inherits
  its most recent run's settings. If the model list cannot load,
  the composer keeps working with the configured default.
- **Citations**: `[S#]` chips in answers are buttons — clicking one
  highlights that source in the right panel. Sources are badged
  **Verified** (page fetched, evidence inspected) or **Discovered**
  (search snippet only).
- **Diagnostics**: the right panel's Diagnostics tab shows duration,
  iterations, model/tool calls, token usage (including
  provider-reported **thinking tokens**, or an honest *Not
  reported*), coverage, retrieval counts, provider retries, the
  security summary and **Run configuration** — the model, thinking
  level, depth and evidence policy the run actually used (older
  runs created before model selection show *Not recorded*).
- **History**: sessions are stored in SQLite (`data/app.db`). The
  left sidebar lists them; clicking one reopens the stored answer —
  never a re-run. Follow-up questions run in the same session;
  `Start a new research session` begins a new one. Each session
  has a pencil button for inline **renaming** (1–120 characters;
  renaming never touches run history). Deleting asks for
  confirmation.
- **Interrupted runs**: if the server restarted mid-run, the run card
  offers `Resume research` with its caveat: the interrupted step is
  repeated from the last checkpoint.

## 4. Stop, restart, and data persistence

- Stop: `stop-research-assistant.bat` (or Enter in the launcher).
- Restart: run `start-research-assistant.bat` again — sessions,
  answers, runs and the access token all survive; nothing is re-run.
- SQLite lives in `data/` (`app.db`, `checkpoints.db`,
  `app_token.txt`). Start/stop never touches it.

## 5. Backing up SQLite safely

With the app **stopped** (avoids copying a mid-write WAL):

```powershell
.\stop-research-assistant.bat
Copy-Item data\app.db      "$env:USERPROFILE\Desktop\backup-app.db"
Copy-Item data\checkpoints.db "$env:USERPROFILE\Desktop\backup-checkpoints.db"
# optional (treat like a credential; do not share):
Copy-Item data\app_token.txt "$env:USERPROFILE\Desktop\backup-app-token.txt"
```

Restore by copying the files back into `data/` while stopped. The
backup contains your conversation content — store it accordingly.

## 6. Port conflicts

- Symptom: *"Port 8000 is used by another program"*.
- Fix: close the conflicting program, or set a free port in `.env`
  (`API_PORT=8123`) and start again. The browser URL updates
  automatically (the launcher opens the configured port).
- If the app is already running on that port, the launcher detects it
  (health check) and only opens the browser — no duplicate instance.

## 7. Troubleshooting

**"Python virtual environment not found"**
The `ai_poc` folder is missing or incomplete. From the project root:

```powershell
python -m venv ai_poc
ai_poc\Scripts\python.exe -m pip install -r requirements.txt
```

**"Backend dependencies are missing or broken"**
Re-run the install command above; then start again.

**The browser says "The research service is not reachable"**
The server is not running (or is on another port). Check the launcher
window for startup errors, look at `logs\server-error.log`, then start
again. If you changed `API_PORT`, use the new port in the browser.

**"The access token was rejected"**
Use the current token: press `C` + Enter in the launcher (or open
`data\app_token.txt`) and paste it. If you set `APP_TOKEN` in `.env`,
use that value and restart the server after changing it.

**The UI says the browser UI is not built**
Run `cd frontend && npm install && npm run build` once (or use the
production launcher, which does it for you).

**Gemini/provider failures ("provider", "auth", "rate limit" errors)**
- Missing key: add `GEMINI_API_KEY=...` to `.env` (get a free-tier
  key at aistudio.google.com/apikey) and restart. The key is never
  printed or sent anywhere but the configured provider.
- Invalid key: verify it, then restart.
- Rate limits: the app retries transient 429/5xx automatically
  (`LLM_MAX_RETRIES`); persistent limits need a wait or another
  provider (`LLM_PROVIDER=openrouter` with `OPENROUTER_API_KEY`).
- Model access refused (`model_access` category): Google limits the
  2.5 models to projects that have actively used them, so a
  project without access gets a 403/404 on those models. The run
  fails with a fixed, non-sensitive message and its frozen model
  stays on the record — fix the key/permissions or start a new run
  with a different model. The app never switches models by itself.
- Runs always end with a categorized error in the run card and
  Diagnostics — the raw provider error text is never shown or stored.

**Logs**
`logs\server.log` / `logs\server-error.log` (also `vite.log` in dev
mode). They are redacted at the source: tokens and provider keys are
masked.

## 8. Docker (UNVERIFIED — static review only)

> Docker was not installed on the machine where this phase was
> reviewed. The files in `docker/` are **statically validated only**
> and have not been built or run. Treat a first launch as a test.

Files: `docker/Dockerfile` (multi-stage: Node builds the UI, then a
non-root Python 3.12 slim runtime), `docker/docker-compose.yml`,
`.dockerignore` (keeps `.env`, `data/` and build junk out of images).

```powershell
# from the repository root (Docker Desktop must be running)
docker compose -f docker/docker-compose.yml up -d --build
# UI + API: http://127.0.0.1:8000/   (loopback-only publish)
docker compose -f docker/docker-compose.yml logs -f
docker compose -f docker/docker-compose.yml down        # keeps data
docker compose -f docker/docker-compose.yml down -v     # DELETES data
```

- Configuration comes from `../.env` at runtime (Compose v2.24+;
  the file is optional). No credentials are baked into images.
- SQLite persists in the `assistant-data` volume; the access token is
  bootstrapped there on first boot. Read it with:
  `docker compose -f docker/docker-compose.yml exec research-assistant cat /app/data/app_token.txt`
  (or set `APP_TOKEN` in `.env` before first start).
- `restart: unless-stopped` brings it back after a reboot/Docker
  restart; `down` stops it. Health checks use `/api/health`.
- The container binds `0.0.0.0` internally so the published port
  works, but compose publishes on `127.0.0.1` only — the app stays
  machine-local, same as the native install.

## 9. Security posture (local-only)

- Single-user local tool: binds `127.0.0.1`; never publish it.
- Bearer token on every `/api` route; constant-time comparison; the
  static UI contains no secrets.
- Provider keys and the app token are held in memory, masked in logs
  and errors, and never embedded in the browser bundle, URLs or
  analytics (there is no analytics).
- Retrieved web content is treated as untrusted (prompt-injection
  signals, tool policy, answer redaction — Phase 4C; benchmark:
  25/25 offline security cases).
- SQLite files and backups contain your conversations — they are the
  sensitive data boundary.
- The token in `data/app_token.txt` grants full local access: don't
  commit, share or screenshot it.

## 10. Known limitations

- Session titles are derived from the first question (no rename API).
- The access token is memory-only in the browser: a page refresh asks
  for it again (use the launcher's clipboard copy).
- Resuming an interrupted run repeats the interrupted step (backend
  checkpoint semantics); the UI says so before you resume.
- Cancel is cooperative: it takes effect at the next graph boundary
  (an in-flight provider call finishes first).
- Research depth and evidence policy (Phase 4E.1) are selectable
  per run via the CLI (`--depth`, `--evidence-policy`) and the API
  (`research_depth`, `evidence_policy`), but the browser UI has no
  selector yet — browser runs use the defaults (`standard`,
  `balanced`).
- After a crash + resume, retrieval counters in the diagnostics
  panel are reconciled from the durable event log and cover the
  whole run; token usage covers only the resumed segment (the
  panel says so explicitly — pre-interruption usage died with the
  interrupted process and is never fabricated).
- Docker packaging is statically reviewed but not runtime-tested.
- Windows launchers use Windows PowerShell 5.1+; Linux/macOS users
  run `python -m app.main` and `cd frontend && npm run dev` directly.
