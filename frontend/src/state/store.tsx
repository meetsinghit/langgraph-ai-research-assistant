/**
 * Application state: sessions, runs, per-run events and the single
 * attached run stream.
 *
 * Design notes:
 *  - Reopening history NEVER creates runs: rendering stored runs is
 *    read-only (GET). The only POST that creates research is the
 *    explicit submit action.
 *  - The SSE stream attaches to the active session's live run (if
 *    any) and resumes via Last-Event-ID; events are deduplicated by
 *    strictly-increasing sequence before they reach state.
 *  - When the stream reports a terminal event, the run detail is
 *    re-fetched from the API — the answer/sources/diagnostics shown
 *    are exactly what the backend persisted, never stream-derived.
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { ApiError } from "../api/client";
import { RunStream, type StreamStatus } from "../api/sse";
import {
  TERMINAL_STATUSES,
  type EvidencePolicy,
  type ModelOut,
  type ModelsOut,
  type ResearchDepth,
  type RunDetailOut,
  type RunEvent,
  type RunStatus,
  type RunSummaryOut,
  type SessionOut,
  type ThinkingLevel,
} from "../api/types";
import { submissionMessage } from "../lib/messages";
import { titleFromQuestion } from "../lib/format";
import { useAuth } from "./auth";

export interface Notice {
  kind: "error" | "info";
  message: string;
}

export interface CitationSelection {
  runId: string;
  sid: string;
}

/**
 * Composer research preferences (4E.1/4E.3). Each submit freezes the
 * current values into the run — changing them later never alters
 * existing runs. `modelId: null` means "configured default" (the
 * request omits the model), `thinkingLevel: null` means the model's
 * documented default.
 */
export interface RunPreferences {
  modelId: string | null;
  thinkingLevel: ThinkingLevel | null;
  depth: ResearchDepth;
  evidence: EvidencePolicy;
}

const DEFAULT_PREFERENCES: RunPreferences = {
  modelId: null,
  thinkingLevel: null,
  depth: "standard",
  evidence: "balanced",
};

interface StoreValue {
  ready: boolean;
  sessions: SessionOut[];
  activeSessionId: string | null;
  runsBySession: Record<string, RunSummaryOut[]>;
  details: Record<string, RunDetailOut>;
  eventsByRun: Record<string, RunEvent[]>;
  streamStatus: Record<string, StreamStatus>;
  watchedRunId: string | null;
  focusedRunId: string | null;
  notice: Notice | null;
  busy: boolean;
  citation: CitationSelection | null;
  /** catalogue entries from GET /api/models (registry-backed) */
  models: ModelOut[];
  modelsStatus: "idle" | "loading" | "ready" | "failed";
  discoveryStatus: ModelsOut["discovery"]["status"] | null;
  defaultModelId: string | null;
  defaultThinkingLevel: ThinkingLevel | null;
  preferences: RunPreferences;
  setPreference: (patch: Partial<RunPreferences>) => void;
  selectSession: (sessionId: string) => void;
  startNewResearch: () => void;
  submitQuestion: (question: string) => Promise<void>;
  cancelRun: (runId: string) => Promise<void>;
  resumeRun: (runId: string) => Promise<void>;
  deleteSession: (sessionId: string) => Promise<void>;
  /** returns true on success (the caller leaves edit mode) */
  renameSession: (sessionId: string, title: string) => Promise<boolean>;
  focusRun: (runId: string) => void;
  selectCitation: (runId: string, sid: string) => void;
  retryStream: () => void;
  clearNotice: () => void;
}

const StoreContext = createContext<StoreValue | null>(null);

function isLive(status: RunStatus | undefined): boolean {
  return (
    status === "queued" || status === "running" || status === "cancelling"
  );
}

export function StoreProvider({ children }: { children: ReactNode }) {
  const { client, getToken, markUnauthorized, connection } = useAuth();

  const [ready, setReady] = useState(false);
  const [sessions, setSessions] = useState<SessionOut[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [runsBySession, setRunsBySession] = useState<
    Record<string, RunSummaryOut[]>
  >({});
  const [details, setDetails] = useState<Record<string, RunDetailOut>>({});
  const [eventsByRun, setEventsByRun] = useState<Record<string, RunEvent[]>>({});
  const [streamStatus, setStreamStatus] = useState<Record<string, StreamStatus>>({});
  const [focusedRunId, setFocusedRunId] = useState<string | null>(null);
  const [notice, setNotice] = useState<Notice | null>(null);
  const [busy, setBusy] = useState(false);
  const [citation, setCitation] = useState<CitationSelection | null>(null);
  const [streamNonce, setStreamNonce] = useState(0);
  const [catalog, setCatalog] = useState<ModelsOut | null>(null);
  const [modelsStatus, setModelsStatus] = useState<
    "idle" | "loading" | "ready" | "failed"
  >("idle");
  const [preferences, setPreferences] = useState<RunPreferences>(
    DEFAULT_PREFERENCES,
  );
  const models = catalog?.models ?? [];

  // --- helpers -------------------------------------------------------------

  const handleAuthError = useCallback(
    (error: unknown): boolean => {
      if (error instanceof ApiError && error.category === "unauthorized") {
        markUnauthorized();
        return true;
      }
      return false;
    },
    [markUnauthorized],
  );

  const reportError = useCallback(
    (error: unknown, fallback: string) => {
      if (handleAuthError(error)) return;
      setNotice({
        kind: "error",
        message: error instanceof Error ? error.message : fallback,
      });
    },
    [handleAuthError],
  );

  const refreshSessions = useCallback(async () => {
    try {
      const list = await client.listSessions();
      setSessions(list);
      setReady(true);
    } catch (error) {
      reportError(error, "Could not load research sessions.");
    }
  }, [client, reportError]);

  const loadRuns = useCallback(
    async (sessionId: string): Promise<RunSummaryOut[]> => {
      const detail = await client.getSession(sessionId);
      setRunsBySession((prev) => ({ ...prev, [sessionId]: detail.runs }));
      setSessions((prev) =>
        prev.map((session) =>
          session.id === sessionId
            ? { ...session, ...detail.session }
            : session,
        ),
      );
      return detail.runs;
    },
    [client],
  );

  const loadDetail = useCallback(
    async (runId: string): Promise<RunDetailOut | null> => {
      try {
        const detail = await client.getRun(runId);
        setDetails((prev) => ({ ...prev, [runId]: detail }));
        return detail;
      } catch (error) {
        if (!handleAuthError(error)) {
          // Detail load is best-effort (e.g. transient race right
          // after a terminal event); retry once shortly after.
          window.setTimeout(() => {
            void client
              .getRun(runId)
              .then((next) =>
                setDetails((prev) => ({ ...prev, [runId]: next })),
              )
              .catch(() => undefined);
          }, 600);
        }
        return null;
      }
    },
    [client, handleAuthError],
  );

  // --- initial load --------------------------------------------------------

  // Sessions can only be loaded once authenticated (loading earlier
  // would just 401 on the protected endpoint).
  useEffect(() => {
    if (connection === "connected") void refreshSessions();
  }, [connection, refreshSessions]);

  // Model catalogue (4E.3): loaded once per connection. The backend
  // serves it from its maintained registry — discovery is off by
  // default, so this never triggers an external call. Failure is
  // non-fatal: the composer keeps working with the default model and
  // just cannot list alternatives.
  useEffect(() => {
    if (connection !== "connected") return;
    setModelsStatus((status) => (status === "ready" ? status : "loading"));
    client
      .listModels()
      .then((result) => {
        setCatalog(result);
        setModelsStatus("ready");
      })
      .catch(() => setModelsStatus("failed"));
  }, [connection, client]);

  // --- run stream attachment ----------------------------------------------

  const activeRuns = activeSessionId
    ? (runsBySession[activeSessionId] ?? [])
    : [];
  const liveRun = [...activeRuns]
    .reverse()
    .find((run) => isLive(run.status));
  const watchedRunId = liveRun ? liveRun.id : null;

  // Focus defaults to the live run of the active session.
  useEffect(() => {
    if (watchedRunId) {
      setFocusedRunId((current) => current ?? watchedRunId);
    }
  }, [watchedRunId]);

  // Load stored details for every visible non-live run — reopening
  // history shows the stored answer (read-only GETs, never a rerun).
  useEffect(() => {
    for (const run of activeRuns) {
      if (!details[run.id] && !isLive(run.status)) {
        void loadDetail(run.id);
      }
    }
  }, [activeRuns, details, loadDetail]);

  // The stream itself: one RunStream for the watched run.
  const finalizeRef = useRef<(runId: string) => void>(() => undefined);
  finalizeRef.current = (runId: string) => {
    void (async () => {
      await loadDetail(runId);
      // Refresh summary list + sidebar metadata (status/updated_at).
      if (activeSessionId) {
        try {
          await loadRuns(activeSessionId);
        } catch {
          /* reported via loadRuns' caller */
        }
      }
      void refreshSessions();
    })();
  };

  useEffect(() => {
    if (!watchedRunId) return undefined;
    const runId = watchedRunId;
    setStreamStatus((prev) => ({ ...prev, [runId]: "connecting" }));
    const stream = new RunStream(
      {
        onEvent: (event) => {
          setEventsByRun((prev) => {
            const existing = prev[runId] ?? [];
            // Strictly-increasing dedup (replay overlaps vanish).
            if (existing.length > 0 && event.seq <= existing[existing.length - 1].seq) {
              return prev;
            }
            return { ...prev, [runId]: [...existing, event] };
          });
          if (event.type === "run.finished" || event.type === "run.failed") {
            finalizeRef.current(runId);
          }
        },
        onStatus: (status) => {
          setStreamStatus((prev) => ({ ...prev, [runId]: status }));
          if (status === "error") {
            setNotice({
              kind: "error",
              message:
                "Lost the live connection to the research service. " +
                "Use “Reconnect” to resume watching — research is never resubmitted.",
            });
          }
        },
        onError: (info) => {
          if (!info.retryable) {
            setNotice({ kind: "error", message: info.message });
          }
        },
      },
      {
        url: client.eventsPath(runId),
        // Stream reads use the same in-memory token source.
        tokenProvider: getToken,
      },
    );
    stream.start();
    return () => {
      stream.stop();
      setStreamStatus((prev) => ({ ...prev, [runId]: "closed" }));
    };
    // streamNonce: manual reconnect remounts the stream.
  }, [watchedRunId, client, getToken, streamNonce, activeSessionId]);

  // --- actions -------------------------------------------------------------

  const startNewResearch = useCallback(() => {
    setActiveSessionId(null);
    setFocusedRunId(null);
    setCitation(null);
    setPreferences(DEFAULT_PREFERENCES);
  }, []);

  const selectSession = useCallback(
    async (sessionId: string) => {
      setActiveSessionId(sessionId);
      setFocusedRunId(null);
      setCitation(null);
      setNotice(null);
      try {
        const runs = await loadRuns(sessionId);
        const last = runs[runs.length - 1];
        if (last) setFocusedRunId(last.id);
        // Composer inherits this session's research settings from its
        // most recent run (WS-D: standard default, per-session
        // inheritance); a session without runs resets to defaults.
        setPreferences({
          modelId: last?.model_id ?? null,
          thinkingLevel: (last?.thinking_level as ThinkingLevel) ?? null,
          depth: (last?.research_depth as ResearchDepth) ?? "standard",
          evidence: (last?.evidence_policy as EvidencePolicy) ?? "balanced",
        });
      } catch (error) {
        reportError(error, "Could not load this session.");
      }
    },
    [loadRuns, reportError],
  );

  const setPreference = useCallback(
    (patch: Partial<RunPreferences>) => {
      setPreferences((prev) => {
        const next = { ...prev, ...patch };
        if ("modelId" in patch && next.thinkingLevel != null) {
          // Capability guard (4E.3 WS-B): a model without thinking
          // controls, or one that does not support the selected
          // level, falls back to its documented default — the UI
          // never offers a combination the backend must reject.
          // With "configured default" selected, the default model's
          // capabilities apply.
          const specId = next.modelId ?? catalog?.default_model_id ?? "";
          const spec = models.find((m) => m.model_id === specId);
          const supported = spec?.thinking;
          if (
            !supported ||
            supported.control === "none" ||
            !supported.levels.includes(next.thinkingLevel)
          ) {
            next.thinkingLevel = null;
          }
        }
        return next;
      });
    },
    [models, catalog],
  );

  const submitQuestion = useCallback(
    async (question: string) => {
      const trimmed = question.trim();
      if (!trimmed || busy) return;
      setBusy(true);
      setNotice(null);
      try {
        let sessionId = activeSessionId;
        if (!sessionId) {
          const session = await client.createSession(
            titleFromQuestion(trimmed),
          );
          setSessions((prev) => [session, ...prev]);
          setRunsBySession((prev) => ({ ...prev, [session.id]: [] }));
          sessionId = session.id;
          setActiveSessionId(sessionId);
        }
        const accepted = await client.createRun(sessionId, trimmed, {
          idempotencyKey: crypto.randomUUID(),
          // Per-run settings frozen at creation (4E.1/4E.3): null
          // model_id/thinking_level select the configured defaults.
          settings: {
            research_depth: preferences.depth,
            evidence_policy: preferences.evidence,
            model_id: preferences.modelId,
            thinking_level: preferences.thinkingLevel,
          },
        });
        const placeholder: RunSummaryOut = {
          id: accepted.run_id,
          session_id: sessionId,
          question: trimmed,
          status: accepted.status,
          has_answer: false,
          created_at: new Date().toISOString(),
        };
        setRunsBySession((prev) => ({
          ...prev,
          [sessionId]: [...(prev[sessionId] ?? []), placeholder],
        }));
        setFocusedRunId(accepted.run_id);
      } catch (error) {
        if (handleAuthError(error)) return;
        // Model/thinking selection errors and idempotency conflicts
        // get actionable copy (never raw exceptions — the backend
        // contract is category + fixed text).
        const friendly =
          error instanceof ApiError ? submissionMessage(error.category) : null;
        setNotice({
          kind: "error",
          message:
            friendly ??
            (error instanceof Error
              ? error.message
              : "Could not start the research run."),
        });
      } finally {
        setBusy(false);
      }
    },
    [activeSessionId, busy, client, handleAuthError, preferences],
  );

  const cancelRun = useCallback(
    async (runId: string) => {
      try {
        await client.cancelRun(runId);
        setRunsBySession((prev) => {
          const next: Record<string, RunSummaryOut[]> = {};
          for (const [key, runs] of Object.entries(prev)) {
            next[key] = runs.map((run) =>
              run.id === runId
                ? { ...run, status: "cancelling" as RunStatus }
                : run,
            );
          }
          return next;
        });
      } catch (error) {
        if (error instanceof ApiError && error.category === "conflict") {
          setNotice({ kind: "info", message: error.message });
          if (activeSessionId) void loadRuns(activeSessionId);
        } else {
          reportError(error, "Could not cancel this run.");
        }
      }
    },
    [activeSessionId, client, loadRuns, reportError],
  );

  const resumeRun = useCallback(
    async (runId: string) => {
      try {
        await client.resumeRun(runId);
        setRunsBySession((prev) => {
          const next: Record<string, RunSummaryOut[]> = {};
          for (const [key, runs] of Object.entries(prev)) {
            next[key] = runs.map((run) =>
              run.id === runId
                ? { ...run, status: "queued" as RunStatus }
                : run,
            );
          }
          return next;
        });
        setNotice(null);
      } catch (error) {
        reportError(error, "This run could not be resumed.");
      }
    },
    [client, reportError],
  );

  const deleteSession = useCallback(
    async (sessionId: string) => {
      try {
        await client.deleteSession(sessionId);
        setSessions((prev) => prev.filter((s) => s.id !== sessionId));
        setRunsBySession((prev) => {
          const next = { ...prev };
          delete next[sessionId];
          return next;
        });
        if (activeSessionId === sessionId) {
          setActiveSessionId(null);
          setFocusedRunId(null);
        }
        setCitation(null);
      } catch (error) {
        reportError(error, "Could not delete this session.");
      }
    },
    [activeSessionId, client, reportError],
  );

  const renameSession = useCallback(
    async (sessionId: string, title: string): Promise<boolean> => {
      try {
        const updated = await client.renameSession(sessionId, title);
        setSessions((prev) =>
          prev.map((session) =>
            session.id === sessionId ? { ...session, ...updated } : session,
          ),
        );
        setNotice(null);
        return true;
      } catch (error) {
        reportError(error, "Could not rename this session.");
        return false;
      }
    },
    [client, reportError],
  );

  const focusRun = useCallback((runId: string) => {
    setFocusedRunId(runId);
  }, []);

  const selectCitation = useCallback((runId: string, sid: string) => {
    setFocusedRunId(runId);
    setCitation({ runId, sid });
  }, []);

  const retryStream = useCallback(() => {
    setNotice(null);
    setStreamNonce((n) => n + 1);
  }, []);

  const clearNotice = useCallback(() => setNotice(null), []);

  const value = useMemo<StoreValue>(
    () => ({
      ready,
      sessions,
      activeSessionId,
      runsBySession,
      details,
      eventsByRun,
      streamStatus,
      watchedRunId,
      focusedRunId,
      notice,
      busy,
      citation,
      models,
      modelsStatus,
      discoveryStatus: catalog?.discovery.status ?? null,
      defaultModelId: catalog?.default_model_id ?? null,
      defaultThinkingLevel:
        (catalog?.default_thinking_level as ThinkingLevel) ?? null,
      preferences,
      setPreference,
      selectSession: (sessionId: string) => void selectSession(sessionId),
      startNewResearch,
      submitQuestion,
      cancelRun,
      resumeRun,
      deleteSession,
      renameSession,
      focusRun,
      selectCitation,
      retryStream,
      clearNotice,
    }),
    [
      ready,
      sessions,
      activeSessionId,
      runsBySession,
      details,
      eventsByRun,
      streamStatus,
      watchedRunId,
      focusedRunId,
      notice,
      busy,
      citation,
      models,
      modelsStatus,
      catalog,
      preferences,
      setPreference,
      selectSession,
      startNewResearch,
      submitQuestion,
      cancelRun,
      resumeRun,
      deleteSession,
      renameSession,
      focusRun,
      selectCitation,
      retryStream,
      clearNotice,
    ],
  );

  return (
    <StoreContext.Provider value={value}>{children}</StoreContext.Provider>
  );
}

export function useStore(): StoreValue {
  const value = useContext(StoreContext);
  if (!value) {
    throw new Error("useStore must be used inside StoreProvider");
  }
  return value;
}

export function runIsTerminal(status: RunStatus): boolean {
  return TERMINAL_STATUSES.has(status);
}
