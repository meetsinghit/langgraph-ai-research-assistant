/**
 * TypeScript mirrors of the Phase 4D.1 backend contracts.
 *
 * Sources of truth (all verified against the Python code):
 *  - request/response models: `app/schemas.py`
 *  - SSE event whitelist: `RunManager._EVENT_FIELDS` in `app/runs.py`
 *  - diagnostics payload: `MetricsCollector.snapshot()` in `agent/metrics.py`
 *
 * `schemaCompat.test.ts` asserts these shapes against fixtures
 * captured from the real FastAPI app.
 */

// ---- responses (app/schemas.py) -------------------------------------------

export interface SessionOut {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  run_count?: number | null;
}

export type RunStatus =
  | "queued"
  | "running"
  | "cancelling"
  | "completed"
  | "failed"
  | "cancelled"
  | "interrupted";

export const TERMINAL_STATUSES: ReadonlySet<string> = new Set([
  "completed",
  "failed",
  "cancelled",
]);

export interface RunSummaryOut {
  id: string;
  session_id: string;
  question: string;
  status: RunStatus;
  has_answer: boolean;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
}

export interface SessionDetailOut {
  session: SessionOut;
  runs: RunSummaryOut[];
}

export interface SourceOut {
  sid: string;
  url: string;
  host: string;
  evidence: "retrieved" | "snippet";
  position: number;
}

export interface RunDetailOut {
  id: string;
  session_id: string;
  question: string;
  status: RunStatus;
  answer?: string | null;
  review_status?: string | null;
  outcome_status?: string | null;
  outcome_warnings: string[];
  error_category?: string | null;
  iterations: number;
  tokens_in: number;
  tokens_out: number;
  estimated_cost_usd: number;
  total_execution_seconds: number;
  sources: SourceOut[];
  diagnostics?: DiagnosticsSnapshot | null;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  events_url?: string | null;
}

export interface RunAcceptedOut {
  run_id: string;
  status: RunStatus;
  events_url: string;
}

export interface CancelOut {
  run_id: string;
  status: string;
}

export interface ResumeOut {
  run_id: string;
  resumed: boolean;
}

export interface ErrorOut {
  error: { category: string; message: string };
}

export interface HealthOut {
  ok: boolean;
}

export interface ConfigOut {
  provider: string;
  model: string;
  thinking_level?: string | null;
  max_total_execution_seconds: number;
  max_research_iterations: number;
  max_tool_calls_per_research: number;
  min_findings: number;
  min_retrieved_sources: number;
  max_concurrent_runs: number;
  resume_enabled: boolean;
}

// ---- SSE events (RunManager._EVENT_FIELDS whitelist) ----------------------

export type StageNode =
  | "start"
  | "plan"
  | "research"
  | "synthesize"
  | "review"
  | "finalize"
  | "handle_error";

export type ToolName = "web_search" | "fetch_page";

export type RunEvent =
  | { seq: number; type: "stage.started"; node: StageNode }
  | { seq: number; type: "stage.finished"; node: StageNode }
  | { seq: number; type: "tool.started"; tool: ToolName }
  | {
      seq: number;
      type: "tool.finished";
      tool: ToolName;
      ok: boolean;
      error_category?: string;
      latency_ms?: number;
    }
  | { seq: number; type: "source.discovered"; url: string; title?: string }
  | { seq: number; type: "source.retrieved"; url: string }
  | { seq: number; type: "run.started" }
  | {
      seq: number;
      type: "run.finished";
      outcome: string;
      review_status?: string;
    }
  | { seq: number; type: "status.changed"; status: RunStatus }
  | { seq: number; type: "run.failed"; error_category: string };

export type RunEventType = RunEvent["type"];

// ---- diagnostics (MetricsCollector.snapshot) -------------------------------

export interface ModelCallInfo {
  node: string;
  iteration: number | null;
  invocation: number;
  latency_ms: number;
  input_tokens?: number | null;
  output_tokens?: number | null;
  prompt_chars: number;
  usage_source?: string | null;
  error_category?: string | null;
}

export interface ToolCallInfo {
  node: string;
  iteration: number | null;
  tool: string;
  ok: boolean;
  latency_ms: number;
  result_chars: number;
  error_category?: string | null;
}

export interface DiagnosticsSnapshot {
  run_id?: string;
  started_at?: number | null;
  finished_at?: number | null;
  total_execution_seconds?: number | null;
  outcome: { status: string; warnings: string[] };
  deadline?: { configured_seconds?: number | null; exceeded?: boolean };
  totals?: {
    input_tokens: number;
    output_tokens: number;
    model_calls: number;
    tool_calls: number;
    retries: number;
    prompt_chars?: number;
  };
  token_usage?: {
    input_tokens: number;
    output_tokens: number;
    provider_reported_calls: number;
    estimated_calls: number;
    unavailable_calls: number;
    coverage_ratio?: number | null;
    sources?: Record<string, number>;
  };
  tokens_by_node?: Record<
    string,
    {
      calls: number;
      input_tokens: number;
      output_tokens: number;
      usage_unavailable_calls: number;
      prompt_chars: number;
      latency_ms: number;
    }
  >;
  model_calls?: ModelCallInfo[];
  tool_calls?: ToolCallInfo[];
  nodes?: { node: string; iteration: number | null; latency_ms: number }[];
  counts?: {
    searches: number;
    fetches: number;
    failed_searches: number;
    failed_fetches: number;
    search_success_rate?: number | null;
    fetch_success_rate?: number | null;
    failed_tool_calls: number;
    tool_error_categories: Record<string, number>;
  };
  security?: {
    injection_signals?: {
      total: number;
      by_category: Record<string, number>;
      by_source: Record<string, number>;
    };
    tool_refusals?: Record<string, number>;
    answer_redactions?: number;
  };
  quality?: {
    citations_ok?: boolean;
    coverage_ok?: boolean;
    official_status?: string;
    review_status?: string;
    missing_topics?: number;
    retrieved_sources?: number;
    snippet_sources?: number;
  } | null;
  retries?: { node: string; iteration: number | null; reason_category: string }[];
}
