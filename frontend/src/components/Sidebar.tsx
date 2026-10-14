/** Left sidebar: research history with create, select, rename and delete. */
import { useRef, useState } from "react";
import { useStore } from "../state/store";
import { formatTimestamp } from "../lib/format";
import { Button, Icons, Modal } from "./ui";

const MAX_TITLE = 120;

export function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const store = useStore();
  const [pendingDelete, setPendingDelete] = useState<{
    id: string;
    title: string;
  } | null>(null);
  const [deleting, setDeleting] = useState(false);
  // inline rename (4E.3 WS-F): edit in place, same validation as the
  // backend (1-120 chars); failures keep edit mode open with a notice
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [renaming, setRenaming] = useState(false);
  const renameInputRef = useRef<HTMLInputElement | null>(null);

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

  const startRename = (id: string, title: string) => {
    setRenamingId(id);
    setRenameValue(title);
    // focus after the input mounts
    window.setTimeout(() => renameInputRef.current?.focus(), 0);
  };

  const saveRename = async () => {
    if (!renamingId) return;
    const title = renameValue.trim();
    if (!title || title.length > MAX_TITLE) return;
    setRenaming(true);
    try {
      const ok = await store.renameSession(renamingId, title);
      if (ok) setRenamingId(null);
      // on failure: keep editing so the user can correct the title;
      // the error notice is shown in the conversation area
    } finally {
      setRenaming(false);
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
              const isRenaming = renamingId === session.id;
              return (
                <li key={session.id}>
                  <div
                    className={`group relative rounded-lg border transition-colors ${
                      active
                        ? "border-accent/60 bg-accent-soft"
                        : "border-transparent hover:bg-surface-2"
                    }`}
                  >
                    {isRenaming ? (
                      <form
                        className="flex items-center gap-1.5 px-2 py-1.5"
                        onSubmit={(event) => {
                          event.preventDefault();
                          void saveRename();
                        }}
                      >
                        <label
                          htmlFor={`rename-${session.id}`}
                          className="sr-only"
                        >
                          New title for session: {session.title}
                        </label>
                        <input
                          id={`rename-${session.id}`}
                          ref={renameInputRef}
                          type="text"
                          value={renameValue}
                          maxLength={MAX_TITLE}
                          onChange={(event) => setRenameValue(event.target.value)}
                          onKeyDown={(event) => {
                            if (event.key === "Escape") {
                              event.preventDefault();
                              setRenamingId(null);
                            }
                          }}
                          className="min-w-0 flex-1 rounded-md border border-border bg-surface px-2 py-1 text-sm text-ink outline-none focus-visible:border-accent"
                        />
                        <button
                          type="submit"
                          disabled={
                            renaming ||
                            !renameValue.trim() ||
                            renameValue.trim().length > MAX_TITLE
                          }
                          className="rounded p-1.5 text-ink-muted hover:text-ink disabled:opacity-50"
                          aria-label={`Save title: ${session.title}`}
                        >
                          {Icons.check}
                        </button>
                        <button
                          type="button"
                          onClick={() => setRenamingId(null)}
                          className="rounded p-1.5 text-ink-muted hover:text-ink"
                          aria-label={`Cancel renaming: ${session.title}`}
                        >
                          {Icons.close}
                        </button>
                      </form>
                    ) : (
                      <>
                        <button
                          type="button"
                          onClick={() => {
                            void store.selectSession(session.id);
                            onNavigate?.();
                          }}
                          className="w-full rounded-lg px-3 py-2 pr-14 text-left"
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
                          onClick={() => startRename(session.id, session.title)}
                          className="absolute right-7 top-1/2 -translate-y-1/2 rounded p-1.5 text-ink-muted opacity-0 transition-opacity hover:text-ink focus-visible:opacity-100 group-hover:opacity-100"
                          aria-label={`Rename session: ${session.title}`}
                        >
                          {Icons.pencil}
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
                      </>
                    )}
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
