# GitHub portfolio metadata — proposed settings

> **For GitHub's repository settings / About panel.** No remote settings were changed by this documentation pack.

## Repository name

`langgraph-ai-research-assistant`

## Short description (copy into GitHub → About)

> Evidence-grounded Agentic AI research assistant with LangGraph, Gemini, FastAPI, React, SQLite, live progress and security-aware tool execution.

## Suggested GitHub topics

`agentic-ai` · `langgraph` · `gemini-api` · `fastapi` · `react` · `typescript` · `sqlite` · `ai-agents` · `llm` · `research-assistant` · `prompt-injection` · `python`

GitHub generally supports up to **20 repository topics**. Keep tags truthful: this agent retrieves live webpages for evidence, but it is not a vector-database-centric RAG service, so `rag` is optional rather than automatic.

## Social preview image

Upload [`docs/assets/social-preview.png`](assets/social-preview.png) in **Repository → Settings → General → Social preview**. This was designed at 1280 × 640 px specifically for GitHub sharing cards.

## Suggested repository homepage

While private or without a public deployment, **leave the Website field blank**. Do not point to `http://127.0.0.1:8000`: that address only works on the owner's computer.

If a public demo/website is authorized later, link its public, security-reviewed URL here.

## Recommended GitHub repository structure

```text
langgraph-ai-research-assistant/
├── README.md                     ← GitHub portfolio landing page
├── agent/                        ← LangGraph research engine
├── app/                          ← FastAPI + SQLite + SSE
├── frontend/                     ← React/TypeScript app
├── evals/                        ← Offline quality and adversarial tests
├── tests/                        ← Python regression suite
├── docs/
│   ├── technical-handbook.html   ← Full interactive engineering field guide
│   ├── PORTFOLIO_STORY.md        ← Interview / project demonstration guide
│   ├── GITHUB_PUBLICATION_REVIEW.md
│   ├── GITHUB_METADATA.md
│   ├── OPERATIONS.md
│   ├── SECURITY.md
│   └── assets/                   ← Original diagrams + screenshots
├── docker/
├── scripts/
└── start-research-assistant.bat
```

## Visibility and license

1. **Keep Private** until the secret/history review and ownership/rights review are completed.
2. Decide whether to add a license. **No license** keeps rights reserved; it doesn't signal that visitors can fork/reuse code as an open-source project. If authorized for open source, choose a license deliberately (MIT or Apache-2.0 are common options, with different patent/notice considerations).
3. Ensure all screenshots and handbook material are appropriate to show publicly. Avoid client data, employer-confidential code or personal research histories.
4. GitHub Pages is **not necessary** to share the repository. The self-contained HTML handbook can be downloaded from `docs/` and opened locally. If enabling Pages later, review static-site security and repository visibility rules.

## Profile presentation ideas

- Pin the repository to your GitHub profile after making it public.
- In the GitHub About panel, emphasize **citation grounding, trustworthy research, per-run model configuration and resumable execution** over generic “AI chatbot” language.
- Link the README, the technical handbook, and one polished screenshot in a relevant portfolio or resume entry.
- For demos, use synthetic/public research questions; never upload credentials or private research records.
