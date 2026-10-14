/**
 * One conversation turn: the user's question and the assistant's
 * research result (answer, progress while live, banners for
 * incomplete/failed/cancelled outcomes, resume card for interrupted
 * runs).
 */
import { useState } from "react";
import type {
  RunDetailOut,
  RunEvent,
  RunSummaryOut,
  SourceOut,
} from "../api/types";
import { deriveTimeline } from "../lib/timeline";
import {
  categoryMessage,
  ERROR_CATEGORY_LABELS,
  outcomeMessage,
  reviewLabel,
} from "../lib/messages";
import { formatTimestamp } from "../lib/format";
import { SafeMarkdown } from "./SafeMarkdown";
import { ProgressTimeline } from "./ProgressTimeline";
import { Badge, Button, Icons } from "./ui";

export function RunTurn({
  run,
  detail,
  events,
  isActive,
  isSelected,
  resumeEnabled,
  streamStatus,
  onCancel,
  onResume,
  onReconnect,
  onCite,
  onFocus,
}: {
  run: RunSummaryOut;
  detail?: RunDetailOut;
  events: RunEvent[];
  isActive: boolean;
  isSelected: boolean;
  resumeEnabled: boolean;
  streamStatus?: string;
  onCancel: () => void;
  onResume: () => void;
  onReconnect: () => void;
  onCite: (sid: string) => void;
  onFocus: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const sources: SourceOut[] = detail?.sources ?? [];
  const timeline = deriveTimeline(events, run.status);
  const answer = detail?.answer ?? null;
  const canCancel =
    run.status === "queued" || run.status === "running";
  const showResume =
    run.status === "interrupted" && resumeEnabled;
  const resumeBlocked =
    run.status === "interrupted" && !resumeEnabled;

  const copyAnswer = async () => {
    if (!answer) return;
    try {
      await navigator.clipboard.writeText(answer);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable — silently ignore (nothing to show) */
    }
  };

  return (
    <article
      className={`rounded-xl border bg-surface transition-colors ${
        isSelected ? "border-accent/60" : "border-border"
      }`}
      onFocus={onFocus}
      aria-label={`Research turn: ${run.question.slice(0, 60)}`}
    >
      {/* user question */}
      <div className="flex justify-end border-b border-border px-4 py-3">
        <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-accent-soft px-3.5 py-2">
          <p className="whitespace-pre-wrap text-sm text-ink">{run.question}</p>
          <p className="mt-1 text-[10px] text-ink-muted">
            {formatTimestamp(run.created_at)}
          </p>
        </div>
      </div>

      {/* assistant side */}
      <div className="space-y-3 px-4 py-3">
        <div className="flex items-center gap-2 text-xs font-medium text-ink-muted">
          <span
            className="flex h-5 w-5 items-center justify-center rounded-full bg-accent text-[10px] font-bold text-white"
            aria-hidden="true"
          >
            AI
          </span>
          Research Assistant
          {run.status !== "completed" && (
            <Badge tone="neutral">{run.status}</Badge>
          )}
        </div>

        {isActive || events.length > 0 ? (
          <ProgressTimeline
            timeline={timeline}
            canCancel={canCancel}
            onCancel={onCancel}
            streamStatus={streamStatus}
            onReconnect={onReconnect}
          />
        ) : run.status === "completed" ? (
          detail?.outcome_status === "INCOMPLETE" ||
          detail?.review_status === "needs_more_research" ? (
            // Live-acceptance Problem D: the graph FINISHED but quality
            // validation did not pass. "Research completed" (green)
            // next to an Incomplete/Needs-more-research badge read as a
            // contradiction — so the header says what actually happened.
            // Backend outcome semantics are unchanged; the answer's own
            // warning badges and outcome message still show below.
            <p className="rounded-lg border border-warning/40 bg-warning/10 px-3 py-2 text-sm text-warning">
              Research finished — answer incomplete.
            </p>
          ) : (
            <p className="rounded-lg border border-success/40 bg-success/10 px-3 py-2 text-sm text-success">
              Research completed
            </p>
          )
        ) : run.status === "failed" ? (
          <p className="rounded-lg border border-danger/40 bg-danger/10 px-3 py-2 text-sm text-danger">
            Research failed
            {detail?.error_category
              ? ` — ${ERROR_CATEGORY_LABELS[detail.error_category] ?? detail.error_category}`
              : ""}
          </p>
        ) : run.status === "cancelled" ? (
          <p className="rounded-lg border border-border bg-surface-2 px-3 py-2 text-sm text-ink-muted">
            Research cancelled
          </p>
        ) : null}

        {streamStatus === "error" && (
          <div
            role="alert"
            className="rounded-lg border border-danger/40 bg-danger/10 px-3 py-2 text-sm text-danger"
          >
            Live progress was lost, but the research is still running on the
            service.{" "}
            <button
              type="button"
              onClick={onReconnect}
              className="font-semibold underline"
            >
              Reconnect
            </button>{" "}
            — nothing will be resubmitted.
          </div>
        )}

        {run.status === "failed" && !answer && (
          <div
            role="alert"
            className="rounded-lg border border-danger/40 bg-danger/10 px-3 py-2 text-sm text-danger"
          >
            {categoryMessage(detail?.error_category)}
          </div>
        )}

        {run.status === "cancelled" && !answer && (
          <div className="rounded-lg border border-border bg-surface-2 px-3 py-2 text-sm text-ink-muted">
            This run was cancelled before an answer was produced.
          </div>
        )}

        {showResume && (
          <div className="rounded-lg border border-warning/40 bg-warning/10 px-3 py-2.5 text-sm">
            <p className="font-medium text-warning">
              Research interrupted — resumable
            </p>
            <p className="mt-1 text-xs text-ink-muted">
              The run was interrupted (for example by a restart). Completed
              steps are kept, but the step that was in progress will run
              again — a resumed action may repeat.
            </p>
            <Button variant="primary" className="mt-2" onClick={onResume}>
              Resume research
            </Button>
          </div>
        )}

        {resumeBlocked && (
          <div className="rounded-lg border border-border bg-surface-2 px-3 py-2 text-xs text-ink-muted">
            This run was interrupted and cannot be resumed (resume is disabled
            on this service). Ask the question again to redo the research.
          </div>
        )}

        {answer && (
          <div className="space-y-2">
            <div className="flex items-center justify-between gap-2">
              <div className="flex flex-wrap items-center gap-2">
                {detail?.review_status && reviewLabel(detail.review_status) && (
                  <Badge
                    tone={
                      detail.review_status === "approved"
                        ? "success"
                        : detail.review_status === "needs_more_research"
                          ? "warning"
                          : "danger"
                    }
                  >
                    {reviewLabel(detail.review_status)}
                  </Badge>
                )}
                {detail?.outcome_status === "INCOMPLETE" && (
                  <Badge tone="warning">Incomplete</Badge>
                )}
              </div>
              <Button
                variant="ghost"
                onClick={copyAnswer}
                aria-label="Copy answer to clipboard"
              >
                {copied ? Icons.check : Icons.copy}
                {copied ? "Copied" : "Copy"}
              </Button>
            </div>

            {detail?.outcome_status &&
              detail.outcome_status !== "COMPLETED" &&
              outcomeMessage(detail.outcome_status) && (
                <p className="rounded-lg border border-warning/40 bg-warning/10 px-3 py-2 text-xs text-warning">
                  {outcomeMessage(detail.outcome_status)}
                </p>
              )}

            <SafeMarkdown
              markdown={answer}
              sources={sources}
              onCiteSelect={onCite}
            />

            {sources.length > 0 && (
              <p className="text-[11px] text-ink-muted">
                {sources.length} source
                {sources.length === 1 ? "" : "s"} cited — see the sources panel
                for details.
              </p>
            )}
          </div>
        )}

        {!answer && !isActive && run.status === "completed" && (
          <p className="text-sm text-ink-muted">No answer was stored.</p>
        )}
      </div>
    </article>
  );
}
