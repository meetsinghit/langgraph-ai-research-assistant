"""Real-server E2E harness (Phase 4D.3) -- test-only, never shipped.

Runs the REAL FastAPI application under uvicorn with scripted fakes:

- real Bearer auth, real SSE streaming, real SQLite persistence;
- the real production serving path (the server also serves the built
  UI from frontend/dist, exactly like a normal launch);
- no network access, no live model calls, synthetic token only.

Question keywords steer the scripted model (checked against the
prompt text, which contains the question):

- ``SLOWCANCEL``  the first model call sleeps ~6s -> a long-running
  run the UI test can cancel (honored at the next node boundary);
- ``CRASHDEATH``  the FIRST model instance of the server lifetime
  raises a BaseException that nothing catches, leaving the run
  non-terminal -- restarting this server on the same data dir marks
  it ``interrupted`` (crash recovery), and a resume then completes
  with a fresh model instance (the crash is one-shot per data dir,
  guarded by a marker file, so it cannot re-kill the resumed run);
- ``UNSAFELINK``  the answer is hostile markdown (javascript: link,
  raw HTML) to exercise safe rendering end-to-end.

Endpoints added here for tests only (never part of app/):

- ``POST /__e2e__/shutdown`` (Bearer-protected) stops the process so
  a test can simulate "backend went away" and restart it.

Run (from the repository root, after ``cd frontend && npm run build``):

    ai_poc\\Scripts\\python.exe e2e\\server.py --port 8123 \\
        --data-dir %TEMP%\\aipoc-e2e --token e2e-token
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langchain_core.messages import AIMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from agent.config import Settings  # noqa: E402
from agent.graph import ResearchPlan, ReviewVerdict  # noqa: E402
from app.main import create_app  # noqa: E402
from tests.fakes import FakeTools, ScriptedChatModel  # noqa: E402
from tests.test_tool_schema import _tool_call_response  # noqa: E402

URL_FETCHED = "https://example.com/paris"
URL_SNIPPET = "https://example.org/insee"

SEARCH_RESULTS = [
    {
        "title": "Paris - City Data",
        "snippet": "Official statistics about the city of Paris.",
        "url": URL_FETCHED,
    },
    {
        "title": "INSEE overview",
        "snippet": "Demographic overview mentioning Paris.",
        "url": URL_SNIPPET,
    },
]
PAGES = {
    URL_FETCHED: "Paris is the capital and most populous city of France.",
}

#: Rich answer: markdown heading, citation chips, a code block and a
#: normal link -- enough for rendering/badge/citation assertions.
ANSWER = (
    "Paris is the capital of France [S1], with demographics in the "
    "national overview [S2].\n\n"
    "Example:\n\n```python\nprint('hello research')\n```\n\n"
    "More detail is on the [official page](https://example.com/paris)."
)

#: Hostile answer: a javascript: link, a data: link and raw HTML.
UNSAFE_ANSWER = (
    "Click [here](javascript:alert('xss')) and [raw](data:text/html,<script>1</script>) "
    "or visit <img src=x onerror=alert(1)> <b>bold?</b>."
)


class SimulatedProcessDeath(BaseException):
    """Escapes everything (like a killed process); leaves the run
    non-terminal so startup recovery decides its fate."""


class E2EModel(ScriptedChatModel):
    """Scripted model with question-keyword behavior (see module doc).

    Subclassing means every invoke path (invoke/batch/async) funnels
    through ``_generate``; structured calls (plan/review) go through
    ``with_structured_output`` and stay fast.
    """

    crash: bool = False
    #: one-shot guard: once a crash has fired for this data dir, a
    #: restarted server lets the resumed run proceed normally
    crash_marker: Any = None

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        """Schema-driven replay, robust to resume.

        The base class replays by call INDEX; resuming from the plan
        node's checkpoint skips the plan call, which would hand the
        review node a ResearchPlan instead of a ReviewVerdict. A real
        provider returns whatever the requested schema is, so match on
        the schema here instead of on position.
        """
        from langchain_core.runnables import RunnableLambda

        include_raw = bool(kwargs.get("include_raw"))
        if isinstance(schema, type) and issubclass(schema, ResearchPlan):
            parsed: Any = ResearchPlan(queries=["paris capital city"])
        else:
            parsed = ReviewVerdict(status="approved", reason="ok")

        def _invoke(messages: Any) -> Any:
            if not include_raw:
                return parsed
            raw = AIMessage(content=f"[structured output] {parsed!r}")
            return {"raw": raw, "parsed": parsed, "parsing_error": None}

        return RunnableLambda(_invoke)

    def _generate(
        self,
        messages: list[Any],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = str(messages)
        if (
            "CRASHDEATH" in text
            and self.crash
            and self.crash_marker is not None
            and not Path(self.crash_marker).exists()
        ):
            Path(self.crash_marker).touch()
            time.sleep(0.2)
            raise SimulatedProcessDeath("simulated process death")
        if "UNSAFELINK" in text and self.call_index >= 2:
            # first two invokes play the scripted search+fetch (so
            # evidence exists), then the hostile answer arrives
            time.sleep(0.2)
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=UNSAFE_ANSWER))]
            )
        if "SLOWCANCEL" in text and self.call_index == 0:
            # one long first call; cancel lands during it and is
            # honored at the next node boundary
            time.sleep(6.0)
        else:
            time.sleep(0.4)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def build_app(data_dir: Path, dist: Path, token: str):
    settings = Settings(
        _env_file=None,
        data_dir=str(data_dir),
        app_token=token,
        frontend_dist=str(dist),
        min_findings=1,
        max_research_iterations=2,
        max_tool_calls_per_research=3,
        sse_heartbeat_seconds=1.0,
    )
    instances = {"models": 0}

    def model_factory(_: Settings) -> E2EModel:
        # Instance 0 is the one allowed to "die" (CRASHDEATH); a
        # resumed run gets a later instance and completes.
        model = E2EModel(
            responses=[
                _tool_call_response(
                    "call_search", "web_search", {"query": "paris"}
                ),
                _tool_call_response(
                    "call_fetch", "fetch_page", {"url": URL_FETCHED}
                ),
                AIMessage(content=ANSWER),
            ],
            structured_outputs=[
                ResearchPlan(queries=["paris capital city"]),
                ReviewVerdict(status="approved", reason="ok"),
            ],
            synthesize_usage=True,
            crash=instances["models"] == 0,
            crash_marker=data_dir / ".e2e-crash-done",
        )
        instances["models"] += 1
        return model

    def tools_factory(_: Settings) -> FakeTools:
        return FakeTools(search_results=SEARCH_RESULTS, pages=PAGES)

    return create_app(
        settings,
        model_factory=model_factory,
        tools_factory=tools_factory,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="E2E harness server")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--token", default="e2e-token-not-a-real-credential")
    parser.add_argument("--dist", default=str(ROOT / "frontend" / "dist"))
    args = parser.parse_args()

    dist = Path(args.dist)
    if not (dist / "index.html").is_file():
        print(
            f"E2E ERROR: built UI not found at {dist} -- "
            "run `npm run build` in frontend/ first.",
            flush=True,
        )
        raise SystemExit(2)

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    app = build_app(data_dir, dist, args.token)

    expected = f"Bearer {args.token}"

    @app.post("/__e2e__/shutdown")
    def _shutdown() -> dict[str, bool]:
        threading.Timer(0.3, os._exit, args=(0,)).start()
        return {"ok": True}

    # Guard the test-only endpoint with the same Bearer token.
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    @app.middleware("http")
    async def _e2e_guard(request: Request, call_next: Any) -> Any:
        if request.url.path == "/__e2e__/shutdown" and (
            request.headers.get("authorization", "") != expected
        ):
            return JSONResponse(status_code=401, content={"error": "unauthorized"})
        return await call_next(request)

    import uvicorn

    print(f"E2E SERVER READY on 127.0.0.1:{args.port}", flush=True)
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=args.port,
        log_config=None,
        access_log=False,
    )


if __name__ == "__main__":
    main()
