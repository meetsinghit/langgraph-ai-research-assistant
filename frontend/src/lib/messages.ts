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
  model_unavailable:
    "The model this run was started with is no longer available. " +
    "Pick a different model in the composer and start a new run — " +
    "this run is not retried on another model automatically.",
  model_access:
    "The selected model could not be accessed with this project's " +
    "configuration — the model may not be available to this project, " +
    "or the API key may be invalid. No other model is used " +
    "automatically: fix the configuration or pick a different model, " +
    "then start a new run.",
  configuration:
    "This run's saved model settings are no longer supported. " +
    "Pick a valid model and thinking level for a new run.",
  unknown: "The research could not be completed due to an unexpected error.",
};

/**
 * Friendly copy for run-submission (HTTP) errors about the user's
 * model/thinking selection — shown in the composer notice, never a
 * raw exception (the backend contract is category + fixed text).
 */
export function submissionMessage(category?: string | null): string | null {
  switch (category) {
    case "invalid_selection":
      return (
        "That model or thinking selection is no longer valid. " +
        "Choose another option and start the run again."
      );
    case "model_unavailable":
      return (
        "The selected model is no longer available. " +
        "Choose a different model and start the run again."
      );
    case "conflict":
      return (
        "This request was already submitted with different settings. " +
        "Change the question or the settings to start a new run."
      );
    case "validation":
      return "The run request was not valid — check the question and settings.";
    default:
      return null;
  }
}

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
  model_unavailable: "Model no longer available",
  model_access: "Model access failed",
  configuration: "Configuration error",
  unknown: "Error",
};
