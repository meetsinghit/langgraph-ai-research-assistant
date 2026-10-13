/**
 * Friendly, non-technical copy for run states and error categories.
 *
 * The backend guarantees errors are category + fixed text only —
 * these mappings never render raw exceptions (the UI also has no
 * access to any).
 */

const CATEGORY_COPY: Record<string, string> = {
  rate_limit:
    "The research stopped because of provider rate limiting. Wait a moment and try again.",
  tls: "A secure connection to a source could not be established.",
  ssrf: "A source was blocked by the network safety policy.",
  deadline: "The run exceeded its time budget before finishing.",
  validation: "The model produced an action that was refused by validation.",
  provider: "The AI provider had an error before the answer was finished.",
  tool: "A research tool failed while gathering sources.",
  unknown: "The research could not be completed due to an unexpected error.",
};

export function categoryMessage(category?: string | null): string {
  if (!category) {
    return CATEGORY_COPY.unknown;
  }
  return CATEGORY_COPY[category] ?? CATEGORY_COPY.unknown;
}

export function outcomeMessage(outcome?: string | null): string | null {
  switch (outcome) {
    case "INCOMPLETE":
      return (
        "The answer was produced, but the quality review found it incomplete — " +
        "some topics lacked verified sources. Treat unverified claims with care."
      );
    case "TIMED_OUT":
      return "The run stopped when its time budget ran out. The partial answer, if any, is not shown as final.";
    case "CANCELLED":
      return "This run was cancelled before finishing.";
    case "FAILED":
      return "The research failed before an answer could be produced.";
    case "COMPLETED_WITH_WARNINGS":
      return "Research completed, but with quality warnings — check the review status and sources.";
    default:
      return null;
  }
}

export function reviewLabel(status?: string | null): string | null {
  switch (status) {
    case "approved":
      return "Approved by review";
    case "needs_more_research":
      return "Needs more research";
    case "failed":
      return "Review failed";
    default:
      return null;
  }
}

export function statusLabel(status: string): string {
  switch (status) {
    case "queued":
      return "Queued";
    case "running":
      return "Researching";
    case "cancelling":
      return "Cancelling";
    case "completed":
      return "Completed";
    case "failed":
      return "Failed";
    case "cancelled":
      return "Cancelled";
    case "interrupted":
      return "Interrupted";
    default:
      return status;
  }
}

export const ERROR_CATEGORY_LABELS: Record<string, string> = {
  rate_limit: "Rate limited",
  tls: "TLS error",
  ssrf: "Blocked by network policy",
  deadline: "Time budget exhausted",
  validation: "Validation refusal",
  provider: "Provider error",
  tool: "Tool failure",
  unknown: "Error",
};
