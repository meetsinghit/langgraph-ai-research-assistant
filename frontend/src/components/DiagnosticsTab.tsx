/**
 * Diagnostics tab: execution facts for the focused run, all numbers
 * and fixed-vocabulary labels (the backend contract is content-free —
 * this panel renders only what the API returns).
 *
 * Missing usage metadata (provider didn't report tokens, or no
 * diagnostics were persisted for an unfinished run) is shown
 * explicitly as "Not reported" — never fabricated.
 */
import type { DiagnosticsSnapshot, RunDetailOut } from "../api/types";
import {
  formatDuration,
  formatNumber,
  formatPercent,
} from "../lib/format";
import {
  ERROR_CATEGORY_LABELS,
  outcomeMessage,
  reviewLabel,
  statusLabel,
} from "../lib/messages";
import { Badge } from "./ui";

function Row({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1">
      <dt className="text-xs text-ink-muted">{label}</dt>
      <dd className="text-right text-sm font-medium text-ink" title={hint}>
        {value}
      </dd>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="border-b border-border px-4 py-3 last:border-b-0">
      <h3 className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-ink-muted">
        {title}
      </h3>
      {children}
    </section>
  );
}

/** Fixed-vocabulary setting label: "standard" -> "Standard". */
function settingLabel(value?: string | null): string {
  if (!value) return "—";
  return value.charAt(0).toUpperCase() + value.slice(1).replace(/_/g, " ");
}

export function DiagnosticsTab({ run }: { run: RunDetailOut }) {
  const diagnostics: DiagnosticsSnapshot | null | undefined =
    run.diagnostics;
  const usage = diagnostics?.token_usage;
  const totals = diagnostics?.totals;
  const counts = diagnostics?.counts;
  const security = diagnostics?.security;
  const retries = diagnostics?.retries ?? [];

  const hasUsageData =
    run.tokens_in > 0 || run.tokens_out > 0 || (usage?.coverage_ratio ?? 0) > 0;

  const coverageParts: string[] = [];
  if (usage) {
    coverageParts.push(`${usage.provider_reported_calls} reported`);
    coverageParts.push(`${usage.estimated_calls} estimated`);
    if (usage.unavailable_calls > 0) {
      coverageParts.push(`${usage.unavailable_calls} unavailable`);
    }
    if (
      usage.thinking_tokens != null &&
      usage.thinking_reported_calls != null
    ) {
      coverageParts.push(`${usage.thinking_reported_calls} with thinking`);
    }
  }

  const refusalEntries = Object.entries(security?.tool_refusals ?? {});

  // Retrieval counters: annotate failures and cache hits inline so
  // this panel and the live progress line use the SAME definitions
  // (WS-D: "searches" = executed web_search calls, including ones
  // served from the per-run cache; "fetches" = fetch attempts,
  // including failures — both now say so).
  const failedSearches = counts?.failed_searches ?? 0;
  const cacheHits = counts?.search_cache_hits ?? 0;
  const failedFetches = counts?.failed_fetches ?? 0;
  const searchNotes: string[] = [];
  if (failedSearches > 0) {
    searchNotes.push(`${formatNumber(failedSearches)} failed`);
  }
  if (cacheHits > 0) {
    searchNotes.push(`${formatNumber(cacheHits)} cached`);
  }
  const fetchNotes: string[] = [];
  if (failedFetches > 0) {
    fetchNotes.push(`${formatNumber(failedFetches)} failed`);
  }
  const annotated = (value: string, notes: string[]) =>
    notes.length > 0 ? `${value} (${notes.join(" · ")})` : value;
  const segments = diagnostics?.segments ?? 1;
  const resumed = segments > 1;

  return (
    <div aria-label="Run diagnostics">
      <Section title="Outcome">
        <div className="flex flex-wrap items-center gap-2 py-1">
          <Badge tone={run.status === "completed" ? "success" : run.status === "failed" ? "danger" : "neutral"}>
            {statusLabel(run.status)}
          </Badge>
          {run.review_status && reviewLabel(run.review_status) && (
            <Badge tone={run.review_status === "approved" ? "success" : "warning"}>
              {reviewLabel(run.review_status)}
            </Badge>
          )}
          {run.outcome_status && (
            <Badge tone="accent">{run.outcome_status}</Badge>
          )}
        </div>
        {run.error_category && (
          <p className="py-1 text-xs text-danger">
            {ERROR_CATEGORY_LABELS[run.error_category] ?? run.error_category}
          </p>
        )}
        {run.outcome_warnings.length > 0 && (
          <ul className="mt-1 list-inside list-disc text-xs text-warning">
            {run.outcome_warnings.map((warning) => (
              <li key={warning}>{warning.replace(/_/g, " ")}</li>
            ))}
          </ul>
        )}
        {run.outcome_status && outcomeMessage(run.outcome_status) && (
          <p className="mt-1 text-xs text-ink-muted">
            {outcomeMessage(run.outcome_status)}
          </p>
        )}
      </Section>

      <Section title="Run configuration">
        <dl>
          <Row
            label="Model"
            value={run.model_id ?? "Not recorded"}
            hint={
              run.model_id
                ? "Frozen when the run was created"
                : "Runs created before model selection existed do not record their model"
            }
          />
          <Row
            label="Thinking"
            value={
              run.thinking_level
                ? settingLabel(run.thinking_level)
                : "Not recorded"
            }
            hint="Frozen when the run was created"
          />
          <Row
            label="Research depth"
            value={settingLabel(run.research_depth)}
          />
          <Row
            label="Evidence policy"
            value={settingLabel(run.evidence_policy)}
          />
        </dl>
        <p className="pt-1 text-[11px] text-ink-muted">
          These settings were frozen when the run was created — later
          changes in the composer never alter a finished run.
        </p>
      </Section>

      <Section title="Execution">
        <dl>
          <Row label="Duration" value={formatDuration(run.total_execution_seconds)} />
          <Row
            label="Research iterations"
            value={
              run.iterations > 0 ? formatNumber(run.iterations) : "—"
            }
          />
          <Row
            label="Model calls"
            value={totals?.model_calls != null ? formatNumber(totals.model_calls) : "—"}
          />
          <Row
            label="Tool calls"
            value={
              totals?.tool_calls != null
                ? formatNumber(totals.tool_calls)
                : counts
                  ? formatNumber((counts.searches ?? 0) + (counts.fetches ?? 0))
                  : "—"
            }
          />
          {totals?.tool_latency_ms != null && totals.tool_latency_ms > 0 && (
            <Row
              label="Tool time"
              value={formatDuration(totals.tool_latency_ms / 1000)}
              hint="Combined search + fetch wall time, part of the run duration"
            />
          )}
        </dl>
      </Section>

      <Section title="Token usage">
        {hasUsageData ? (
          <dl>
            <Row label="Input tokens" value={formatNumber(run.tokens_in)} />
            <Row label="Output tokens" value={formatNumber(run.tokens_out)} />
            <Row
              label="Thinking tokens"
              value={
                usage?.thinking_tokens != null
                  ? formatNumber(usage.thinking_tokens)
                  : "Not reported"
              }
              hint="Reasoning tokens the provider reported (counted within output tokens); never fabricated"
            />
            <Row
              label="Measurement coverage"
              value={
                usage?.coverage_ratio != null
                  ? formatPercent(usage.coverage_ratio)
                  : "—"
              }
              hint={coverageParts.join(" · ")}
            />
            {coverageParts.length > 0 && (
              <p className="pt-1 text-[11px] text-ink-muted">
                {coverageParts.join(" · ")}
              </p>
            )}
            {usage?.scope === "since_resume" && (
              <p className="pt-1 text-[11px] text-ink-muted">
                Since the last resume — usage before the interruption
                was not recoverable.
              </p>
            )}
          </dl>
        ) : (
          <p className="py-1 text-xs text-ink-muted">
            Token usage was not reported by the provider for this run.
          </p>
        )}
      </Section>

      <Section title="Retrieval">
        {counts ? (
          <dl>
            <Row
              label="Web searches"
              value={annotated(formatNumber(counts.searches), searchNotes)}
              hint="web_search calls executed (failed attempts and cached repeats included, broken out)"
            />
            <Row
              label="Pages fetched"
              value={annotated(formatNumber(counts.fetches), fetchNotes)}
              hint="fetch attempts (failures included)"
            />
            <Row
              label="Failed tool calls"
              value={formatNumber(counts.failed_tool_calls)}
            />
          </dl>
        ) : (
          <p className="py-1 text-xs text-ink-muted">
            Retrieval counters are available once the run finishes.
          </p>
        )}
        {resumed && (
          <p className="pt-1 text-[11px] text-ink-muted">
            This run was resumed ({segments} segments): retrieval
            counters above cover the whole run — they are reconciled
            from the durable event log. Token usage covers the
            resumed segment only; usage before the interruption was
            not recoverable.
          </p>
        )}
      </Section>

      <Section title="Provider retries">
        {retries.length > 0 ? (
          <ul className="space-y-1 py-1">
            {retries.map((retry, index) => (
              <li key={index} className="text-xs text-ink-muted">
                <span className="font-medium text-ink">{retry.node}</span>
                {retry.iteration != null && retry.iteration > 0 && ` · pass ${retry.iteration}`} —{" "}
                {retry.reason_category.replace(/_/g, " ")}
              </li>
            ))}
          </ul>
        ) : (
          <p className="py-1 text-xs text-ink-muted">
            No provider retries during this run.
          </p>
        )}
      </Section>

      <Section title="Security summary">
        <dl>
          <Row
            label="Injection signals"
            value={formatNumber(security?.injection_signals?.total ?? 0)}
            hint="Content-free count of advisory signals in retrieved content"
          />
          <Row
            label="Answer redactions"
            value={formatNumber(security?.answer_redactions ?? 0)}
          />
        </dl>
        {refusalEntries.length > 0 && (
          <ul className="mt-1 list-inside list-disc text-xs text-warning">
            {refusalEntries.map(([category, count]) => (
              <li key={category}>
                {count}× {ERROR_CATEGORY_LABELS[category] ?? category}
              </li>
            ))}
          </ul>
        )}
      </Section>
    </div>
  );
}
