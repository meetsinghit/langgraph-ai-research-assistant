/** Left sidebar: research history with create, select and delete. */
import { useState } from "react";
import { useStore } from "../state/store";
import { formatTimestamp } from "../lib/format";
import { Button, Icons, Modal } from "./ui";

export function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const store = useStore();
  const [pendingDelete, setPendingDelete] = useState<{
    id: string;
    title: string;
  } | null>(null);
  const [deleting, setDeleting] = useState(false);

  const confirmDelete = async () => {
    if (!pendingDelete) return;
    setDeleting(true);
    try {
      await store.deleteSession(pendingDelete.id);
      setPendingDelete(null);
      onNavigate?.();
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="flex h-full flex-col bg-surface">
      <div className="flex items-center justify-between gap-2 border-b border-border px-3 py-3">
        <h1 className="text-sm font-semibold text-ink">Research history</h1>
        <Button
          variant="primary"
          onClick={() => {
            store.startNewResearch();
            onNavigate?.();
          }}
          aria-label="Start a new research session"
        >
          {Icons.plus}
          New
        </Button>
      </div>

      <nav className="min-h-0 flex-1 overflow-y-auto p-2" aria-label="Research sessions">
        {store.sessions.length === 0 ? (
          <p className="p-3 text-xs text-ink-muted">
            No saved research yet. Ask a question to start your first
            session — it is stored locally in SQLite.
          </p>
        ) : (
          <ul className="space-y-1">
            {store.sessions.map((session) => {
              const active = session.id === store.activeSessionId;
              return (
                <li key={session.id}>
                  <div
                    className={`group relative rounded-lg border transition-colors ${
                      active
                        ? "border-accent/60 bg-accent-soft"
                        : "border-transparent hover:bg-surface-2"
                    }`}
                  >
                    <button
                      type="button"
                      onClick={() => {
                        void store.selectSession(session.id);
                        onNavigate?.();
                      }}
                      className="w-full rounded-lg px-3 py-2 pr-8 text-left"
                      aria-current={active ? "true" : undefined}
                    >
                      <span className="line-clamp-2 text-sm font-medium text-ink">
                        {session.title}
                      </span>
                      <span className="mt-0.5 block text-[11px] text-ink-muted">
                        {formatTimestamp(session.updated_at || session.created_at)}
                        {session.run_count != null &&
                          ` · ${session.run_count} run${session.run_count === 1 ? "" : "s"}`}
                      </span>
                    </button>
                    <button
                      type="button"
                      onClick={() =>
                        setPendingDelete({ id: session.id, title: session.title })
                      }
                      className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded p-1.5 text-ink-muted opacity-0 transition-opacity hover:text-danger focus-visible:opacity-100 group-hover:opacity-100"
                      aria-label={`Delete session: ${session.title}`}
                    >
                      {Icons.trash}
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </nav>

      <Modal
        open={pendingDelete !== null}
        title="Delete this research session?"
        onClose={() => setPendingDelete(null)}
        labelledBy="delete-session-title"
      >
        <p className="text-sm text-ink-muted">
          “{pendingDelete?.title}” and all of its runs, sources and progress
          history will be permanently removed from local storage. This cannot
          be undone.
        </p>
        <div className="mt-4 flex justify-end gap-2">
          <Button onClick={() => setPendingDelete(null)}>Keep it</Button>
          <Button
            variant="danger"
            onClick={() => void confirmDelete()}
            disabled={deleting}
          >
            {deleting ? "Deleting…" : "Delete session"}
          </Button>
        </div>
      </Modal>
    </div>
  );
}
