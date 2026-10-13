/** Right panel: Sources and Diagnostics tabs for the focused run. */
import { useEffect, useState } from "react";
import { useStore } from "../state/store";
import { SourcesTab } from "./SourcesTab";
import { DiagnosticsTab } from "./DiagnosticsTab";
import { Tabs } from "./ui";

export function RightPanel() {
  const store = useStore();
  const [tab, setTab] = useState<"sources" | "diagnostics">("sources");

  const focusedRun = store.focusedRunId
    ? Object.values(store.runsBySession)
        .flat()
        .find((run) => run.id === store.focusedRunId)
    : undefined;
  const detail = store.focusedRunId
    ? store.details[store.focusedRunId]
    : undefined;

  // A citation selection switches this panel to Sources.
  useEffect(() => {
    if (store.citation) setTab("sources");
  }, [store.citation]);

  return (
    <div className="flex h-full flex-col bg-surface">
      <Tabs
        ariaLabel="Sources and diagnostics"
        active={tab}
        onChange={(id) => setTab(id as "sources" | "diagnostics")}
        tabs={[
          { id: "sources", label: "Sources" },
          { id: "diagnostics", label: "Diagnostics" },
        ]}
      />
      <div className="min-h-0 flex-1 overflow-y-auto">
        {!focusedRun ? (
          <p className="p-4 text-sm text-ink-muted">
            Select a research run to inspect its sources and diagnostics.
          </p>
        ) : (
          <>
            <p className="border-b border-border px-4 py-2 text-[11px] text-ink-muted">
              For: <span className="font-medium text-ink">{focusedRun.question}</span>
            </p>
            {tab === "sources" ? (
              <SourcesTab
                sources={detail?.sources ?? []}
                selectedSid={
                  store.citation?.runId === focusedRun.id
                    ? store.citation.sid
                    : null
                }
                onSelect={(sid) => store.selectCitation(focusedRun.id, sid)}
              />
            ) : detail ? (
              <DiagnosticsTab run={detail} />
            ) : (
              <p className="p-4 text-sm text-ink-muted">Loading diagnostics…</p>
            )}
          </>
        )}
      </div>
    </div>
  );
}
