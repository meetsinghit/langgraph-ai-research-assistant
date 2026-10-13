/**
 * Connection gate: the app asks for the local APP_TOKEN once and keeps
 * it in memory only. Shown whenever the backend is unreachable or the
 * token was rejected — the user can always retry or reconnect here.
 */
import { useEffect, useRef, useState } from "react";
import { useAuth } from "../state/auth";
import { Button, Spinner } from "./ui";

export function ConnectionGate() {
  const { connection, connectionMessage, connect, hasToken } = useAuth();
  const [token, setToken] = useState("");
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (connection === "disconnected" || connection === "error") {
      inputRef.current?.focus();
    }
  }, [connection]);

  if (connection === "connected") return null;

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    const trimmed = token.trim();
    if (!trimmed) return;
    void connect(trimmed);
    setToken("");
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-canvas p-4">
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="connect-title"
        className="w-full max-w-md rounded-2xl border border-border bg-surface p-6 shadow-2xl"
      >
        <div className="mb-1 flex items-center gap-2">
          <span
            className="flex h-8 w-8 items-center justify-center rounded-xl bg-accent text-sm font-bold text-white"
            aria-hidden="true"
          >
            AI
          </span>
          <h1 id="connect-title" className="text-lg font-semibold text-ink">
            Connect to your research service
          </h1>
        </div>
        <p className="mt-1 text-sm text-ink-muted">
          Enter the local access token to connect. It was created on first
          start (see <code className="rounded bg-surface-2 px-1">data/app_token.txt</code>)
          or set via <code className="rounded bg-surface-2 px-1">APP_TOKEN</code>.
        </p>

        {connection === "error" && connectionMessage && (
          <p
            role="alert"
            className="mt-3 rounded-lg border border-danger/40 bg-danger/10 px-3 py-2 text-sm text-danger"
          >
            {connectionMessage}
          </p>
        )}

        <form onSubmit={submit} className="mt-4 space-y-3">
          <label htmlFor="token-input" className="block text-sm font-medium text-ink">
            Access token
          </label>
          <input
            id="token-input"
            ref={inputRef}
            type="password"
            autoComplete="off"
            value={token}
            onChange={(event) => setToken(event.target.value)}
            placeholder="Paste your local APP_TOKEN"
            className="w-full rounded-lg border border-border bg-surface-2 px-3 py-2 text-sm text-ink outline-none focus:border-accent"
          />
          <Button
            variant="primary"
            type="submit"
            disabled={token.trim().length === 0 || connection === "connecting"}
            className="w-full"
          >
            {connection === "connecting" ? (
              <>
                <Spinner /> Connecting…
              </>
            ) : (
              "Connect"
            )}
          </Button>
        </form>

        <p className="mt-4 text-xs text-ink-muted">
          For privacy the token stays in memory only — it is never written to
          browser storage or sent anywhere except this local service. After a
          page refresh you simply connect again.
        </p>
        {hasToken && connection === "error" && (
          <button
            type="button"
            onClick={() => setToken("")}
            className="mt-2 text-xs text-accent underline"
          >
            Use a different token
          </button>
        )}
      </div>
    </div>
  );
}
