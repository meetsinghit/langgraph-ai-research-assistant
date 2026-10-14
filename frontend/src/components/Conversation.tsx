/**
 * Main conversation panel: ordered turns for the active session,
 * sticky question input, empty state, and scroll behavior that
 * follows live progress unless the user is reading history.
 */
import { useEffect, useRef, useState } from "react";
import { useStore } from "../state/store";
import { useAuth } from "../state/auth";
import { RunTurn } from "./RunTurn";
import { QuestionInput } from "./QuestionInput";
import { ResearchControls } from "./ResearchControls";

export function Conversation() {
  const store = useStore();
  const { config } = useAuth();
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const [pinned, setPinned] = useState(true);

  const runs = store.activeSessionId
    ? (store.runsBySession[store.activeSessionId] ?? [])
    : [];
  const session = store.sessions.find((s) => s.id === store.activeSessionId);

  const eventCount = runs.reduce(
    (total, run) => total + (store.eventsByRun[run.id]?.length ?? 0),
    0,
  );
  const lastAnswerLength =
    runs.length > 0
      ? (store.details[runs[runs.length - 1].id]?.answer?.length ?? 0)
      : 0;

  // Follow live progress only while the user is at (or near) the
  // bottom — manual reading is never disrupted.
  useEffect(() => {
    const node = scrollRef.current;
    if (!node || !pinned) return;
    node.scrollTop = node.scrollHeight;
  }, [runs.length, eventCount, lastAnswerLength, pinned]);

  const onScroll = () => {
    const node = scrollRef.current;
    if (!node) return;
    const distance = node.scrollHeight - node.scrollTop - node.clientHeight;
    setPinned(distance < 80);
  };

  const jumpToLatest = () => {
    const node = scrollRef.current;
    if (!node) return;
    setPinned(true);
    node.scrollTop = node.scrollHeight;
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <header className="flex items-center justify-between gap-2 border-b border-border bg-surface px-4 py-3">
        <div className="min-w-0">
          <h2 className="truncate text-sm font-semibold text-ink">
            {session?.title ?? "New research"}
          </h2>
          {session && (
            <p className="text-[11px] text-ink-muted">
              {runs.length} run{runs.length === 1 ? "" : "s"} in this session
            </p>
          )}
        </div>
      </header>

      <div
        ref={scrollRef}
        onScroll={onScroll}
        className="relative min-h-0 flex-1 overflow-y-auto px-4 py-4"
        data-testid="conversation-scroll"
      >
        {runs.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="mx-auto max-w-3xl space-y-4">
            {runs.map((run) => (
              <RunTurn
                key={run.id}
                run={run}
                detail={store.details[run.id]}
                events={store.eventsByRun[run.id] ?? []}
                isActive={store.watchedRunId === run.id}
                isSelected={store.focusedRunId === run.id}
                resumeEnabled={config?.resume_enabled ?? false}
                streamStatus={store.streamStatus[run.id]}
                onCancel={() => void store.cancelRun(run.id)}
                onResume={() => void store.resumeRun(run.id)}
                onReconnect={store.retryStream}
                onCite={(sid) => store.selectCitation(run.id, sid)}
                onFocus={() => store.focusRun(run.id)}
              />
            ))}
          </div>
        )}

        {!pinned && runs.length > 0 && (
          <button
            type="button"
            onClick={jumpToLatest}
            className="sticky bottom-2 left-1/2 z-10 -translate-x-1/2 rounded-full border border-border bg-surface px-3 py-1 text-xs font-medium text-accent shadow"
          >
            Jump to latest
          </button>
        )}
      </div>

      <footer className="border-t border-border bg-surface px-4 py-3">
        <div className="mx-auto max-w-3xl">
          {store.notice && (
            <div
              role="alert"
              className={`mb-2 flex items-start justify-between gap-2 rounded-lg border px-3 py-2 text-sm ${
                store.notice.kind === "error"
                  ? "border-danger/40 bg-danger/10 text-danger"
                  : "border-border bg-surface-2 text-ink"
              }`}
            >
              <span>{store.notice.message}</span>
              <button
                type="button"
                onClick={store.clearNotice}
                aria-label="Dismiss message"
                className="shrink-0 opacity-70 hover:opacity-100"
              >
                ×
              </button>
            </div>
          )}
          <QuestionInput
            onSubmit={(question) => void store.submitQuestion(question)}
            busy={store.busy}
            controls={<ResearchControls disabled={store.busy} />}
          />
        </div>
      </footer>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="mx-auto flex h-full max-w-xl flex-col items-center justify-center px-6 text-center">
      <div
        className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-accent text-lg font-bold text-white"
        aria-hidden="true"
      >
        AI
      </div>
      <h3 className="text-lg font-semibold text-ink">
        What would you like to research?
      </h3>
      <p className="mt-2 text-sm text-ink-muted">
        Ask a question below. The assistant searches the web, verifies
        sources, and answers with citations you can inspect — like{" "}
        <code className="rounded bg-surface-2 px-1">[S1]</code>.
      </p>
    </div>
  );
}
