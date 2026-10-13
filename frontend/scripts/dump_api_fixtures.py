"""Dump real API responses to JSON fixtures for the frontend tests.

Run from the repository root with the project venv:

    ai_poc\\Scripts\\python.exe frontend\\scripts\\dump_api_fixtures.py

The fixtures are produced by the REAL FastAPI app (same code path as
production) with the scripted test fakes — no network, no live model
calls, synthetic token only. `frontend/src/test/schemaCompat.test.ts`
validates the TypeScript contract shapes against these files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from agent.graph import ResearchPlan, ReviewVerdict  # noqa: E402
from app.main import create_app  # noqa: E402
from tests.app_fakes import (  # noqa: E402
    TEST_TOKEN,
    make_settings,
    model_factory,
    tools_factory,
)
from tests.test_app_api import AUTH, PAGES, SEARCH_RESULTS  # noqa: E402
from tests.test_tool_schema import _tool_call_response  # noqa: E402

from langchain_core.messages import AIMessage  # noqa: E402

OUT_DIR = ROOT / "frontend" / "src" / "test" / "fixtures" / "backend"


def main() -> None:
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="api_fixtures_"))
    models: list = []
    tools: list = []
    app = create_app(
        make_settings(tmp),
        db_path=tmp / "app.db",
        token=TEST_TOKEN,
        checkpointer=None,
        model_factory=model_factory(
            models,
            responses=[
                _tool_call_response("call_1", "web_search", {"query": "q"}),
                AIMessage(content="A cited answer [S1] with an unknown ref [S9]."),
            ],
            structured_outputs=[
                ResearchPlan(queries=["q"]),
                ReviewVerdict(status="approved", reason="ok"),
            ],
        ),
        tools_factory=tools_factory(tools, search_results=SEARCH_RESULTS, pages=PAGES),
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with TestClient(app) as client:
        (OUT_DIR / "config.json").write_text(
            json.dumps(client.get("/api/config", headers=AUTH).json(), indent=2),
            encoding="utf-8",
        )
        (OUT_DIR / "health.json").write_text(
            json.dumps(client.get("/api/health").json(), indent=2),
            encoding="utf-8",
        )
        session = client.post("/api/sessions", json={"title": "Fixture session"}, headers=AUTH)
        (OUT_DIR / "session_created.json").write_text(
            json.dumps(session.json(), indent=2), encoding="utf-8"
        )
        session_id = session.json()["id"]
        run = client.post(
            f"/api/sessions/{session_id}/runs",
            json={"question": "question"},
            headers=AUTH,
        )
        (OUT_DIR / "run_accepted.json").write_text(
            json.dumps(run.json(), indent=2), encoding="utf-8"
        )
        run_id = run.json()["run_id"]
        app.state.manager.wait_idle()
        detail = client.get(f"/api/runs/{run_id}", headers=AUTH)
        (OUT_DIR / "run_detail_completed.json").write_text(
            json.dumps(detail.json(), indent=2), encoding="utf-8"
        )
        (OUT_DIR / "session_detail.json").write_text(
            json.dumps(
                client.get(f"/api/sessions/{session_id}", headers=AUTH).json(), indent=2
            ),
            encoding="utf-8",
        )
        (OUT_DIR / "sessions_list.json").write_text(
            json.dumps(client.get("/api/sessions", headers=AUTH).json(), indent=2),
            encoding="utf-8",
        )
        (OUT_DIR / "error_404.json").write_text(
            json.dumps(client.get("/api/runs/missing", headers=AUTH).json(), indent=2),
            encoding="utf-8",
        )
        resume = client.post("/api/runs/missing/resume", headers=AUTH)
        (OUT_DIR / "error_404_resume.json").write_text(
            json.dumps(resume.json(), indent=2), encoding="utf-8"
        )
    app.state.manager.shutdown()
    for path in sorted(OUT_DIR.glob("*.json")):
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
