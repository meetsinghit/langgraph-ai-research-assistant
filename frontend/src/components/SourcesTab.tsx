/**
 * Sources tab: the focused run's cited sources, with per-run [S#]
 * identifiers, hostname, and a clear retrieved-vs-discovered split.
 * Selecting a citation in the answer highlights (and scrolls to) the
 * matching row here.
 */
import { useEffect, useRef } from "react";
import type { SourceOut } from "../api/types";
import { hostnameOf } from "../lib/format";
import { Badge } from "./ui";

export function SourcesTab({
  sources,
  selectedSid,
  onSelect,
}: {
  sources: SourceOut[];
  selectedSid?: string | null;
  onSelect?: (sid: string) => void;
}) {
  const selectedRef = useRef<HTMLLIElement | null>(null);

  useEffect(() => {
    if (selectedSid && selectedRef.current) {
      selectedRef.current.scrollIntoView({ block: "nearest" });
    }
  }, [selectedSid]);

  if (sources.length === 0) {
    return (
      <p className="p-4 text-sm text-ink-muted">
        No sources were cited in this run yet.
      </p>
    );
  }

  return (
    <ul className="space-y-2 p-3" aria-label="Cited sources">
      {sources.map((source) => {
        const selected = source.sid === selectedSid;
        return (
          <li
            key={source.sid}
            ref={selected ? selectedRef : undefined}
            className={`rounded-lg border p-2.5 transition-colors ${
              selected
                ? "border-accent bg-accent-soft ring-1 ring-accent"
                : "border-border bg-surface hover:bg-surface-2"
            }`}
          >
            <div className="flex items-center justify-between gap-2">
              <button
                type="button"
                onClick={() => onSelect?.(source.sid)}
                className="flex min-w-0 items-center gap-2 text-left"
                aria-label={`Citation ${source.sid}: ${source.host || "source"}`}
              >
                <span className="rounded border border-accent/40 bg-accent-soft px-1.5 py-0.5 text-[11px] font-semibold text-accent">
                  {source.sid}
                </span>
                <span className="truncate text-sm font-medium text-ink">
                  {source.host || hostnameOf(source.url)}
                </span>
              </button>
              {source.evidence === "retrieved" ? (
                <Badge tone="success">Verified</Badge>
              ) : (
                <Badge tone="neutral">Discovered</Badge>
              )}
            </div>
            <a
              href={source.url}
              target="_blank"
              rel="noopener noreferrer"
              className="mt-1 block truncate text-xs text-ink-muted underline-offset-2 hover:underline"
              title={source.url}
            >
              {source.url}
            </a>
            <p className="mt-1 text-[11px] text-ink-muted">
              {source.evidence === "retrieved"
                ? "Page fetched and evidence verified"
                : "Search result — page not fetched (snippet only)"}
            </p>
          </li>
        );
      })}
    </ul>
  );
}
