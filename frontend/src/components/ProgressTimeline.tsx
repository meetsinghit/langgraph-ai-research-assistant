/**
 * Progress timeline rendered from REAL backend events — each step is
 * pending/active/done with live activity counters; there are no
 * fabricated percentages anywhere.
 */
import type { TimelineResult } from "../lib/timeline";
import { ERROR_CATEGORY_LABELS } from "../lib/messages";
import { Icons, Spinner } from "./ui";

export function ProgressTimeline({
  timeline,
  onCancel,
  canCancel,
  streamStatus,
  onReconnect,
}: {
  timeline: TimelineResult;
  canCancel: boolean;
  onCancel: () => void;
  streamStatus?: string;
  onReconnect: () => void;
}) {
  const { steps, terminal } = timeline;
  return (
    <section
      aria-label="Research progress"
      className="rounded-xl border border-border bg-surface-2/60 p-3"
    >
      <ol className="space-y-1.5">
        {steps.map((step) => (
          <li key={step.id} className="flex items-start gap-2.5 text-sm">
            <span
              className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border text-[10px] ${
                step.state === "done"
                  ? "border-success/50 bg-success/15 text-success"
                  : step.state === "active"
                    ? "border-accent/60 bg-accent-soft text-accent"
                    : "border-border bg-surface text-ink-muted"
              }`}
              aria-hidden="true"
            >
              {step.state === "done"
                ? Icons.check
                : step.state === "active" ? (
                    <Spinner />
                  ) : (
                    <span className="block h-1.5 w-1.5 rounded-full bg-ink-muted/50" />
                  )}
            </span>
            <span className="min-w-0">
              <span
                className={
                  step.state === "pending"
                    ? "text-ink-muted"
                    : "font-medium text-ink"
                }
              >
                {step.label}
              </span>
              {step.detail && (
                <span className="block text-xs text-ink-muted">
                  {step.detail}
                </span>
              )}
              <span className="sr-only">
                {step.state === "done"
                  ? " (done)"
                  : step.state === "active"
                    ? " (in progress)"
                    : " (pending)"}
              </span>
            </span>
          </li>
        ))}
      </ol>

      {terminal && (
        <p
          className={`mt-3 rounded-lg border px-3 py-2 text-sm ${
            terminal.kind === "completed"
              ? "border-success/40 bg-success/10 text-success"
              : terminal.kind === "incomplete"
                ? "border-warning/40 bg-warning/10 text-warning"
                : terminal.kind === "failed"
                  ? "border-danger/40 bg-danger/10 text-danger"
                  : "border-border bg-surface text-ink-muted"
          }`}
        >
          {terminal.kind === "failed" && terminal.category
            ? `${terminal.label} — ${ERROR_CATEGORY_LABELS[terminal.category] ?? terminal.category}`
            : terminal.label}
        </p>
      )}

      <div className="mt-3 flex items-center gap-2">
        {!terminal && (
          <button
            type="button"
            onClick={onCancel}
            disabled={!canCancel}
            className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-surface px-2.5 py-1 text-xs font-medium text-ink-muted hover:text-danger disabled:opacity-50"
          >
            {Icons.stop}
            {streamStatus === "reconnecting"
              ? "Reconnecting…"
              : "Stop research"}
          </button>
        )}
        {streamStatus === "error" && (
          <button
            type="button"
            onClick={onReconnect}
            className="inline-flex items-center gap-1.5 rounded-lg border border-accent/50 bg-accent-soft px-2.5 py-1 text-xs font-medium text-accent"
          >
            {Icons.refresh}
            Reconnect
          </button>
        )}
      </div>
    </section>
  );
}
