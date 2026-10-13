/**
 * Pure derivation of the progress timeline from backend SSE events.
 *
 * Inputs are ONLY the real event contracts (`stage.*`, `tool.*`,
 * `source.*`, `status.changed`, `run.finished`, `run.failed`) —
 * there are no fabricated percentages anywhere; a step is pending,
 * active (with live activity counters) or done.
 */
import type { RunEvent, RunStatus } from "../api/types";

export type TimelineStepId =
  | "plan"
  | "search"
  | "fetch"
  | "synthesize"
  | "review";

export interface TimelineStep {
  id: TimelineStepId;
  label: string;
  state: "pending" | "active" | "done";
  /** Short live activity line, e.g. "2 searches · 3 pages verified". */
  detail?: string;
}

export interface TimelineResult {
  steps: TimelineStep[];
  terminal:
    | { kind: "completed"; label: string; outcome?: string }
    | { kind: "incomplete"; label: string; outcome?: string }
    | { kind: "failed"; label: string; category?: string }
    | { kind: "cancelled"; label: string }
    | { kind: "interrupted"; label: string }
    | null;
  /** True while any step is active and no terminal state arrived. */
  running: boolean;
}

const STEP_LABELS: Record<TimelineStepId, string> = {
  plan: "Planning the research",
  search: "Searching the web",
  fetch: "Fetching & verifying sources",
  synthesize: "Synthesizing the answer",
  review: "Reviewing citations & coverage",
};

const ORDER: TimelineStepId[] = [
  "plan",
  "search",
  "fetch",
  "synthesize",
  "review",
];

export function deriveTimeline(
  events: RunEvent[],
  runStatus?: RunStatus | null,
): TimelineResult {
  const started = new Set<string>();
  const finished = new Set<string>();
  const nodeFinished = new Set<string>();
  const activeTools = new Set<string>();
  let searches = 0;
  let searchFails = 0;
  let fetches = 0;
  let fetchFails = 0;
  let retrieved = 0;
  let discovered = 0;
  let outcome: string | undefined;
  let reviewStatus: string | undefined;
  let failedCategory: string | undefined;
  let sawRunStarted = false;
  let sawStatus: RunStatus | undefined;

  for (const event of events) {
    switch (event.type) {
      case "run.started":
        sawRunStarted = true;
        break;
      case "stage.started":
        started.add(event.node);
        break;
      case "stage.finished":
        finished.add(event.node);
        nodeFinished.add(event.node);
        break;
      case "tool.started":
        activeTools.add(event.tool);
        break;
      case "tool.finished": {
        activeTools.delete(event.tool);
        if (event.tool === "web_search") {
          searches += 1;
          if (!event.ok) searchFails += 1;
        } else {
          fetches += 1;
          if (!event.ok) fetchFails += 1;
        }
        break;
      }
      case "source.discovered":
        discovered += 1;
        break;
      case "source.retrieved":
        retrieved += 1;
        break;
      case "status.changed":
        sawStatus = event.status;
        break;
      case "run.finished":
        outcome = event.outcome;
        reviewStatus = event.review_status;
        break;
      case "run.failed":
        failedCategory = event.error_category;
        break;
    }
  }

  // The persisted run status is authoritative once terminal: the
  // event stream deliberately stops at the terminal event, so its
  // last status.changed can be stale (e.g. still "running" after
  // run.finished). Events only drive the in-progress display.
  const apiStatus = runStatus ?? null;
  const status =
    apiStatus === "completed" ||
    apiStatus === "failed" ||
    apiStatus === "cancelled" ||
    apiStatus === "interrupted"
      ? apiStatus
      : (sawStatus ?? apiStatus);
  const terminalStatus =
    status === "completed" ||
    status === "failed" ||
    status === "cancelled" ||
    status === "interrupted" ||
    sawRunStarted === false
      ? status
      : null;

  const stepState = (id: TimelineStepId): TimelineStep["state"] => {
    // Both retrieval steps (search + fetch) are driven by the
    // `research` stage node plus their own tool activity.
    const node =
      id === "fetch" || id === "search" ? "research" : id;
    if (nodeFinished.has(node)) return "done";
    if (started.has(node)) return "active";
    // tool activity implies the research stage is live even before
    // its stage event lands in rare ordering cases
    if (id === "search" && activeTools.has("web_search")) return "active";
    if (id === "fetch" && activeTools.has("fetch_page")) return "active";
    return "pending";
  };

  const steps: TimelineStep[] = ORDER.map((id) => {
    const state = stepState(id);
    const detail = activityDetail(id, {
      searches,
      searchFails,
      fetches,
      fetchFails,
      retrieved,
      discovered,
      active: state === "active",
      activeTools,
    });
    return { id, label: STEP_LABELS[id], state, detail };
  });

  let terminal: TimelineResult["terminal"] = null;
  if (status === "completed") {
    terminal = {
      kind: outcome === "INCOMPLETE" ? "incomplete" : "completed",
      label:
        outcome === "INCOMPLETE"
          ? "Completed with incomplete research"
          : "Research completed",
      outcome: reviewStatus ? `${outcome ?? "COMPLETED"} · ${reviewStatus}` : outcome,
    };
  } else if (status === "failed") {
    terminal = {
      kind: "failed",
      label: "Research failed",
      category: failedCategory,
    };
  } else if (status === "cancelled") {
    terminal = { kind: "cancelled", label: "Research cancelled" };
  } else if (status === "interrupted") {
    terminal = {
      kind: "interrupted",
      label: "Run interrupted — can be resumed",
    };
  } else if (outcome === "INCOMPLETE" && terminalStatus === null) {
    terminal = {
      kind: "incomplete",
      label: "Completed with incomplete research",
      outcome,
    };
  }

  const running =
    terminal === null &&
    (sawRunStarted ||
      steps.some((step) => step.state === "active") ||
      status === "running" ||
      status === "queued" ||
      status === "cancelling");

  return { steps, terminal, running: terminal === null && running };
}

interface ActivityInput {
  searches: number;
  searchFails: number;
  fetches: number;
  fetchFails: number;
  retrieved: number;
  discovered: number;
  active: boolean;
  activeTools: Set<string>;
}

function activityDetail(id: TimelineStepId, input: ActivityInput): string | undefined {
  const parts: string[] = [];
  if (id === "search") {
    if (input.searches > 0) {
      parts.push(
        `${input.searches} search${input.searches === 1 ? "" : "es"}` +
          (input.searchFails > 0 ? ` · ${input.searchFails} failed` : ""),
      );
    }
    if (input.active && input.activeTools.has("web_search")) {
      parts.push("searching…");
    }
  } else if (id === "fetch") {
    if (input.retrieved > 0) {
      parts.push(
        `${input.retrieved} page${input.retrieved === 1 ? "" : "s"} verified`,
      );
    } else if (input.discovered > 0) {
      parts.push(
        `${input.discovered} candidate${input.discovered === 1 ? "" : "s"} found`,
      );
    }
    if (input.fetchFails > 0) {
      parts.push(`${input.fetchFails} fetch failed`);
    }
    if (input.active && input.activeTools.has("fetch_page")) {
      parts.push("fetching…");
    }
  } else if (id === "plan" && input.active) {
    parts.push("decomposing the question…");
  } else if (id === "synthesize" && input.active) {
    parts.push("drafting the answer…");
  } else if (id === "review" && input.active) {
    parts.push("checking citations…");
  }
  return parts.length > 0 ? parts.join(" · ") : undefined;
}
