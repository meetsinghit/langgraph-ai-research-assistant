/**
 * Application shell: three-region responsive layout (history |
 * conversation | sources/diagnostics), with the side regions becoming
 * drawers on narrow screens, and the connection gate overlaying
 * everything until the local backend is reachable.
 */
import { useState } from "react";
import { AuthProvider, useAuth } from "./state/auth";
import { StoreProvider } from "./state/store";
import { ConnectionGate } from "./components/ConnectionGate";
import { Sidebar } from "./components/Sidebar";
import { Conversation } from "./components/Conversation";
import { RightPanel } from "./components/RightPanel";
import { Drawer, Icons } from "./components/ui";
import { useTheme } from "./lib/theme";

export function App() {
  return (
    <AuthProvider>
      <StoreProvider>
        <Shell />
      </StoreProvider>
    </AuthProvider>
  );
}

function Shell() {
  const { connection, config, disconnect } = useAuth();
  const { theme, toggle } = useTheme();
  const [historyOpen, setHistoryOpen] = useState(false);
  const [panelOpen, setPanelOpen] = useState(false);

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between gap-2 border-b border-border bg-surface px-3 py-2">
        <div className="flex items-center gap-2">
          <button
            type="button"
            className="rounded-lg p-1.5 text-ink-muted hover:bg-surface-2 hover:text-ink lg:hidden"
            onClick={() => setHistoryOpen(true)}
            aria-label="Open research history"
          >
            {Icons.menu}
          </button>
          <span
            className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent text-xs font-bold text-white"
            aria-hidden="true"
          >
            AI
          </span>
          <div className="leading-tight">
            <p className="text-sm font-semibold text-ink">
              AI Research Assistant
            </p>
            {config && (
              <p className="text-[10px] text-ink-muted">
                {config.model}
                {config.thinking_level ? ` · thinking: ${config.thinking_level}` : ""}
              </p>
            )}
          </div>
        </div>
        <div className="flex items-center gap-1">
          <span
            className="mr-1 hidden items-center gap-1.5 text-[11px] text-ink-muted sm:flex"
            title="Connection to the local research service"
          >
            <span
              className={`h-2 w-2 rounded-full ${
                connection === "connected" ? "bg-success" : "bg-danger"
              }`}
              aria-hidden="true"
            />
            {connection === "connected" ? "Connected" : "Disconnected"}
          </span>
          <button
            type="button"
            onClick={toggle}
            className="rounded-lg p-1.5 text-ink-muted hover:bg-surface-2 hover:text-ink"
            aria-label={
              theme === "dark" ? "Switch to light theme" : "Switch to dark theme"
            }
          >
            {theme === "dark" ? Icons.sun : Icons.moon}
          </button>
          <button
            type="button"
            onClick={disconnect}
            className="hidden rounded-lg px-2 py-1 text-xs text-ink-muted hover:bg-surface-2 hover:text-ink sm:block"
          >
            Disconnect
          </button>
          <button
            type="button"
            className="rounded-lg p-1.5 text-ink-muted hover:bg-surface-2 hover:text-ink lg:hidden"
            onClick={() => setPanelOpen(true)}
            aria-label="Open sources and diagnostics"
          >
            {Icons.panel}
          </button>
        </div>
      </header>

      <div className="grid min-h-0 flex-1 lg:grid-cols-[280px_minmax(0,1fr)_360px]">
        <aside className="hidden min-h-0 border-r border-border lg:block">
          <Sidebar />
        </aside>
        <main className="flex min-h-0 flex-col">
          <Conversation />
        </main>
        <aside className="hidden min-h-0 border-l border-border lg:block">
          <RightPanel />
        </aside>
      </div>

      <Drawer
        open={historyOpen}
        side="left"
        label="Research history"
        onClose={() => setHistoryOpen(false)}
      >
        <Sidebar onNavigate={() => setHistoryOpen(false)} />
      </Drawer>
      <Drawer
        open={panelOpen}
        side="right"
        label="Sources and diagnostics"
        onClose={() => setPanelOpen(false)}
      >
        <RightPanel />
      </Drawer>

      {connection !== "connected" && <ConnectionGate />}
    </div>
  );
}
