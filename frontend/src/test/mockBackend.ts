/**
 * In-memory mock of the Phase 4D.1 backend for component/integration
 * tests, served through MSW. Mirrors the REAL contracts:
 *  - Bearer auth on everything except /api/health;
 *  - `{error:{category,message}}` bodies;
 *  - SSE frames `id:/event:/data:` with Last-Event-ID replay and
 *    stream close on terminal events.
 *
 * Tests script runs via `seedRun(...)` and can drive live behavior
 * with `backend.pushEvent(runId, event)`.
 */
import { http, HttpResponse } from "msw";
import { setupServer } from "msw/node";
import type {
  ConfigOut,
  RunDetailOut,
  RunSummaryOut,
  SessionOut,
} from "../api/types";

export const TOKEN = "test-token-123";

export interface MockRun {
  summary: RunSummaryOut;
  detail: RunDetailOut;
  events: string[]; // serialized SSE data payloads (JSON strings)
  /** Drop the stream after N backlog frames without a terminal event. */
  dropAfter?: number;
  /** Refuse new connections once (for reconnect scenarios). */
  failNextConnections?: number;
  /** Live subscribers of the open SSE connections. */
  listeners: Set<(frame: string, terminal: boolean) => void>;
}

export interface MockBackend {
  config: ConfigOut;
  token: string;
  sessions: SessionOut[];
  runs: Record<string, MockRun>;
  /** Requests observed (path + method), for assertions. */
  requests: Array<{ method: string; path: string; headers: Headers }>;
}

export const defaultConfig: ConfigOut = {
  provider: "gemini",
  model: "gemini-3.5-flash-lite",
  thinking_level: "low",
  max_total_execution_seconds: 180,
  max_research_iterations: 3,
  max_tool_calls_per_research: 4,
  min_findings: 3,
  min_retrieved_sources: 2,
  max_concurrent_runs: 2,
  resume_enabled: true,
};

export const backend: MockBackend = {
  config: { ...defaultConfig },
  token: TOKEN,
  sessions: [],
  runs: {},
  requests: [],
};

export function resetBackend(): void {
  backend.config = { ...defaultConfig };
  backend.token = TOKEN;
  backend.sessions = [];
  backend.runs = {};
  backend.requests = [];
}

export function seedSession(
  session: Partial<SessionOut> & { id: string },
): SessionOut {
  const full: SessionOut = {
    id: session.id,
    title: session.title ?? "Session",
    created_at: session.created_at ?? "2026-10-14T10:00:00Z",
    updated_at: session.updated_at ?? "2026-10-14T10:00:00Z",
    run_count: session.run_count ?? 0,
  };
  backend.sessions.unshift(full);
  return full;
}

export function makeSummary(
  overrides: Partial<RunSummaryOut> & { id: string; session_id: string },
): RunSummaryOut {
  return {
    question: "question",
    status: "completed",
    has_answer: true,
    created_at: "2026-10-14T10:00:05Z",
    started_at: "2026-10-14T10:00:06Z",
    finished_at: "2026-10-14T10:00:40Z",
    ...overrides,
  };
}

export function makeDetail(
  overrides: Partial<RunDetailOut> & { id: string; session_id: string },
): RunDetailOut {
  return {
    question: "question",
    status: "completed",
    answer: "An answer [S1].",
    review_status: "approved",
    outcome_status: "COMPLETED",
    outcome_warnings: [],
    error_category: null,
    iterations: 1,
    tokens_in: 1200,
    tokens_out: 80,
    estimated_cost_usd: 0.001,
    total_execution_seconds: 12.5,
    sources: [
      {
        sid: "S1",
        url: "https://example.com/doc",
        host: "example.com",
        evidence: "retrieved",
        position: 1,
      },
    ],
    diagnostics: {
      outcome: { status: "COMPLETED", warnings: [] },
      deadline: { configured_seconds: 180, exceeded: false },
      totals: {
        input_tokens: 1200,
        output_tokens: 80,
        model_calls: 3,
        tool_calls: 2,
        retries: 1,
      },
      token_usage: {
        input_tokens: 1200,
        output_tokens: 80,
        provider_reported_calls: 3,
        estimated_calls: 0,
        unavailable_calls: 0,
        coverage_ratio: 1,
      },
      counts: {
        searches: 1,
        fetches: 1,
        failed_searches: 0,
        failed_fetches: 0,
        search_success_rate: 1,
        fetch_success_rate: 1,
        failed_tool_calls: 0,
        tool_error_categories: {},
      },
      security: {
        injection_signals: { total: 0, by_category: {}, by_source: {} },
        tool_refusals: {},
        answer_redactions: 0,
      },
      retries: [],
    },
    created_at: "2026-10-14T10:00:05Z",
    events_url: `/api/runs/${overrides.id}/events`,
    ...overrides,
  };
}

export function seedRun(
  summary: Partial<RunSummaryOut> & { id: string; session_id: string },
  detail?: Partial<RunDetailOut>,
  events: object[] = [],
): MockRun {
  const run: MockRun = {
    summary: makeSummary(summary),
    detail: makeDetail({ id: summary.id, session_id: summary.session_id, ...detail }),
    events: events.map((event) => JSON.stringify(event)),
    listeners: new Set(),
  };
  backend.runs[summary.id] = run;
  const session = backend.sessions.find((s) => s.id === summary.session_id);
  if (session) {
    session.run_count = (session.run_count ?? 0) + 1;
  }
  return run;
}

/** Deliver a live event to any open SSE connection (and the backlog). */
export function pushEvent(
  runId: string,
  event: { seq: number; type: string; [key: string]: unknown },
): void {
  const run = backend.runs[runId];
  run.events.push(JSON.stringify(event));
  const terminal =
    event.type === "run.finished" || event.type === "run.failed";
  if (terminal) applyTerminal(run, event);
  const frame = sseFrame(event);
  for (const listener of [...run.listeners]) {
    listener(frame, terminal);
  }
}

export function sseFrame(event: object, id?: number): string {
  const seq = id ?? (event as { seq: number }).seq;
  const type = (event as { type: string }).type;
  return `id: ${seq}\nevent: ${type}\ndata: ${JSON.stringify(event)}\n\n`;
}

function authed(request: Request): boolean {
  const header = request.headers.get("Authorization");
  return header === `Bearer ${backend.token}`;
}

/** Scripted happy-path event stream for runs created via POST. */
export function autoRunEvents(): object[] {
  return [
    { seq: 1, type: "status.changed", status: "running" },
    { seq: 2, type: "run.started" },
    { seq: 3, type: "stage.started", node: "plan" },
    { seq: 4, type: "stage.finished", node: "plan" },
    { seq: 5, type: "stage.started", node: "research" },
    { seq: 6, type: "tool.started", tool: "web_search" },
    { seq: 7, type: "tool.finished", tool: "web_search", ok: true, latency_ms: 90 },
    { seq: 8, type: "source.discovered", url: "https://example.com/doc", title: "Doc" },
    { seq: 9, type: "tool.started", tool: "fetch_page" },
    { seq: 10, type: "tool.finished", tool: "fetch_page", ok: true, latency_ms: 250 },
    { seq: 11, type: "source.retrieved", url: "https://example.com/doc" },
    { seq: 12, type: "stage.finished", node: "research" },
    { seq: 13, type: "stage.started", node: "synthesize" },
    { seq: 14, type: "stage.finished", node: "synthesize" },
    { seq: 15, type: "stage.started", node: "review" },
    { seq: 16, type: "stage.finished", node: "review" },
    { seq: 17, type: "run.finished", outcome: "COMPLETED", review_status: "approved" },
    { seq: 18, type: "status.changed", status: "completed" },
  ];
}

/**
 * The persisted run "completes" at the moment the terminal frame is
 * delivered — mirroring the real backend, where the answer only
 * exists in the API after run.finished.
 */
function applyTerminal(
  run: MockRun,
  event: { type: string; outcome?: string; error_category?: string },
): void {
  const current = run.summary.status;
  if (current === "completed" || current === "failed" || current === "cancelled") {
    return;
  }
  const now = new Date().toISOString();
  const failed = event.type === "run.failed";
  const cancelled = event.outcome === "CANCELLED";
  const status = failed ? "failed" : cancelled ? "cancelled" : "completed";
  run.summary.status = status;
  run.summary.has_answer = status === "completed";
  run.summary.finished_at = now;
  run.detail = {
    ...run.detail,
    status,
    answer:
      status === "completed"
        ? (run.detail.answer ?? `Mock answer for: ${run.summary.question}`)
        : (run.detail.answer ?? null),
    error_category: failed
      ? (event.error_category ?? "unknown")
      : run.detail.error_category,
    outcome_status: failed
      ? "FAILED"
      : cancelled
        ? "CANCELLED"
        : (run.detail.outcome_status ?? "COMPLETED"),
    review_status:
      status === "completed"
        ? (run.detail.review_status ?? "approved")
        : run.detail.review_status,
    finished_at: now,
  };
}

function errorBody(category: string, message: string, status: number) {
  return HttpResponse.json({ error: { category, message } }, { status });
}

function runEventsResponse(run: MockRun, request: Request): Response {
  const lastIdHeader = request.headers.get("Last-Event-ID");
  const after = lastIdHeader ? Number(lastIdHeader) : -1;

  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const encoder = new TextEncoder();
      let closed = false;
      let backlogDelivered = 0;
      const finish = () => {
        if (closed) return;
        closed = true;
        try {
          controller.close();
        } catch {
          /* already closed/errored */
        }
      };
      const emit = (frame: string) => {
        if (closed) return;
        try {
          controller.enqueue(encoder.encode(frame));
        } catch {
          closed = true;
        }
      };

      // 0) subscribe FIRST so a live push can never fall between the
      //    replay and the tail (sequence dedup absorbs any overlap)
      const listener = (frame: string, terminal: boolean) => {
        emit(frame);
        if (terminal) {
          run.listeners.delete(listener);
          finish();
        }
      };
      run.listeners.add(listener);
      request.signal?.addEventListener("abort", () => {
        run.listeners.delete(listener);
        closed = true;
      });

      // 1) backlog replay (respecting Last-Event-ID)
      for (const raw of [...run.events]) {
        const event = JSON.parse(raw) as {
          seq: number;
          type: string;
          outcome?: string;
          error_category?: string;
        };
        if (event.seq <= after) continue;
        const terminal =
          event.type === "run.finished" || event.type === "run.failed";
        if (terminal) applyTerminal(run, event);
        emit(sseFrame(event));
        backlogDelivered += 1;
        if (terminal) {
          run.listeners.delete(listener);
          finish();
          return;
        }
      }

      // 2) scripted mid-stream drop (reconnect scenarios)
      if (run.dropAfter != null && backlogDelivered >= run.dropAfter) {
        run.dropAfter = undefined;
        run.listeners.delete(listener);
        finish();
        return;
      }
    },
  });

  return new Response(stream, {
    status: 200,
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
    },
  });
}

export const handlers = [
  http.get("/api/health", () => HttpResponse.json({ ok: true })),

  http.get("/api/config", ({ request }) => {
    backend.requests.push({ method: "GET", path: "/api/config", headers: request.headers });
    if (!authed(request)) {
      return errorBody("unauthorized", "missing or invalid bearer token", 401);
    }
    return HttpResponse.json(backend.config);
  }),

  http.get("/api/sessions", ({ request }) => {
    backend.requests.push({ method: "GET", path: "/api/sessions", headers: request.headers });
    if (!authed(request)) {
      return errorBody("unauthorized", "missing or invalid bearer token", 401);
    }
    return HttpResponse.json(backend.sessions);
  }),

  http.post("/api/sessions", async ({ request }) => {
    backend.requests.push({ method: "POST", path: "/api/sessions", headers: request.headers });
    if (!authed(request)) {
      return errorBody("unauthorized", "missing or invalid bearer token", 401);
    }
    const body = (await request.json()) as { title?: string };
    const session: SessionOut = {
      id: `s-${backend.sessions.length + 1}`,
      title: body.title?.trim() || "New research",
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      run_count: 0,
    };
    backend.sessions.unshift(session);
    return HttpResponse.json(session, { status: 201 });
  }),

  http.get("/api/sessions/:id", ({ params, request }) => {
    backend.requests.push({
      method: "GET",
      path: `/api/sessions/${params.id}`,
      headers: request.headers,
    });
    if (!authed(request)) {
      return errorBody("unauthorized", "missing or invalid bearer token", 401);
    }
    const session = backend.sessions.find((s) => s.id === params.id);
    if (!session) return errorBody("not_found", "session not found", 404);
    const runs = Object.values(backend.runs)
      .filter((run) => run.summary.session_id === session.id)
      .map((run) => run.summary);
    return HttpResponse.json({ session, runs });
  }),

  http.delete("/api/sessions/:id", ({ params, request }) => {
    backend.requests.push({
      method: "DELETE",
      path: `/api/sessions/${params.id}`,
      headers: request.headers,
    });
    if (!authed(request)) {
      return errorBody("unauthorized", "missing or invalid bearer token", 401);
    }
    const index = backend.sessions.findIndex((s) => s.id === params.id);
    if (index === -1) return errorBody("not_found", "session not found", 404);
    backend.sessions.splice(index, 1);
    for (const [runId, run] of Object.entries(backend.runs)) {
      if (run.summary.session_id === params.id) delete backend.runs[runId];
    }
    return new Response(null, { status: 204 });
  }),

  http.post("/api/sessions/:id/runs", async ({ params, request }) => {
    backend.requests.push({
      method: "POST",
      path: `/api/sessions/${params.id}/runs`,
      headers: request.headers,
    });
    if (!authed(request)) {
      return errorBody("unauthorized", "missing or invalid bearer token", 401);
    }
    const session = backend.sessions.find((s) => s.id === params.id);
    if (!session) return errorBody("not_found", "session not found", 404);
    const body = (await request.json()) as { question?: string };
    const question = (body.question ?? "").trim();
    if (!question) {
      return errorBody("validation", "question must not be empty", 422);
    }
    const id = `run-${Object.keys(backend.runs).length + 1}`;
    seedRun(
      {
        id,
        session_id: session.id,
        question,
        status: "queued",
        has_answer: false,
      },
      {
        question,
        answer: null,
        status: "queued",
        review_status: null,
        outcome_status: null,
      },
      autoRunEvents(),
    );
    return HttpResponse.json(
      { run_id: id, status: "queued", events_url: `/api/runs/${id}/events` },
      { status: 202 },
    );
  }),

  http.get("/api/runs/:id", ({ params, request }) => {
    backend.requests.push({
      method: "GET",
      path: `/api/runs/${params.id}`,
      headers: request.headers,
    });
    if (!authed(request)) {
      return errorBody("unauthorized", "missing or invalid bearer token", 401);
    }
    const run = backend.runs[String(params.id)];
    if (!run) return errorBody("not_found", "run not found", 404);
    return HttpResponse.json(run.detail);
  }),

  http.post("/api/runs/:id/cancel", ({ params, request }) => {
    backend.requests.push({
      method: "POST",
      path: `/api/runs/${params.id}/cancel`,
      headers: request.headers,
    });
    if (!authed(request)) {
      return errorBody("unauthorized", "missing or invalid bearer token", 401);
    }
    const run = backend.runs[String(params.id)];
    if (!run) return errorBody("not_found", "run not found", 404);
    const cancellable =
      run.summary.status === "queued" || run.summary.status === "running";
    if (!cancellable) {
      return errorBody(
        "conflict",
        `run is already ${run.summary.status}; only queued or running runs can be cancelled`,
        409,
      );
    }
    run.summary.status = "cancelling";
    return HttpResponse.json({ run_id: run.summary.id, status: "cancelling" });
  }),

  http.post("/api/runs/:id/resume", ({ params, request }) => {
    backend.requests.push({
      method: "POST",
      path: `/api/runs/${params.id}/resume`,
      headers: request.headers,
    });
    if (!authed(request)) {
      return errorBody("unauthorized", "missing or invalid bearer token", 401);
    }
    const run = backend.runs[String(params.id)];
    if (!run) return errorBody("not_found", "run not found", 404);
    if (run.summary.status !== "interrupted") {
      return errorBody(
        "conflict",
        "resume unavailable: not_interrupted",
        409,
      );
    }
    run.summary.status = "queued";
    // A resumed run has a fresh scripted event stream (the backend
    // re-attaches the thread and continues from the checkpoint).
    if (run.events.length === 0) {
      run.events = autoRunEvents().map((event) => JSON.stringify(event));
    }
    return HttpResponse.json({ run_id: run.summary.id, resumed: true }, { status: 202 });
  }),

  http.get("/api/runs/:id/events", ({ params, request }) => {
    backend.requests.push({
      method: "GET",
      path: `/api/runs/${params.id}/events`,
      headers: request.headers,
    });
    if (!authed(request)) {
      return errorBody("unauthorized", "missing or invalid bearer token", 401);
    }
    const run = backend.runs[String(params.id)];
    if (!run) return errorBody("not_found", "run not found", 404);
    if (run.failNextConnections && run.failNextConnections > 0) {
      run.failNextConnections -= 1;
      return new Response(null, { status: 502 });
    }
    return runEventsResponse(run, request);
  }),
];

export const server = setupServer(...handlers);
