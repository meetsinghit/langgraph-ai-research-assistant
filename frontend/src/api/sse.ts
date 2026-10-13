/**
 * Fetch-based SSE client for `/api/runs/{id}/events`.
 *
 * `EventSource` cannot send an `Authorization` header, so the stream
 * is read manually with fetch. Contract (from `app/api.py`):
 *  - frames are `id: <seq>`, `event: <type>`, `data: <json>` + blank line;
 *  - `: heartbeat` comments keep idle streams alive;
 *  - reconnecting with `Last-Event-ID: <seq>` replays exactly the
 *    events after that sequence (server-side guarantee);
 *  - the stream ends when the run reaches a terminal state.
 *
 * Client duties added here:
 *  - deduplicate by sequence (strictly increasing — replay overlaps
 *    are dropped, out-of-order junk is ignored);
 *  - bounded reconnect with backoff, resuming via Last-Event-ID so
 *    a reconnect can never duplicate visible progress;
 *  - read-only: this client NEVER creates runs, so a reconnect can
 *    never resubmit research;
 *  - hard cleanup via AbortController on stop().
 */
import type { RunEvent } from "./types";

export interface StreamErrorInfo {
  retryable: boolean;
  message: string;
}

export interface RunStreamHandlers {
  onEvent: (event: RunEvent) => void;
  onStatus?: (status: StreamStatus) => void;
  onError?: (info: StreamErrorInfo) => void;
}

export type StreamStatus =
  | "connecting"
  | "live"
  | "reconnecting"
  | "closed"
  | "error";

export interface RunStreamOptions {
  url: string;
  tokenProvider: () => string | null;
  fetchImpl?: typeof fetch;
  /** Delay before each reconnect attempt (ms). */
  backoffMs?: number[];
  /** Injectable for tests. */
  sleep?: (ms: number) => Promise<void>;
}

/** Event types after which the server closes the stream. */
const TERMINAL_TYPES = new Set(["run.finished", "run.failed"]);

const DEFAULT_BACKOFF = [500, 1000, 2000, 4000, 8000];

function defaultSleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export class RunStream {
  private readonly options: RunStreamOptions;
  private readonly handlers: RunStreamHandlers;
  private readonly fetchImpl: typeof fetch;
  private readonly backoff: number[];
  private readonly sleep: (ms: number) => Promise<void>;

  private controller: AbortController | null = null;
  private stopped = false;
  private lastSeq = -1;
  private attempt = 0;
  private sawTerminal = false;

  constructor(handlers: RunStreamHandlers, options: RunStreamOptions) {
    this.handlers = handlers;
    this.options = options;
    this.fetchImpl = options.fetchImpl ?? ((...args) => fetch(...args));
    this.backoff = options.backoffMs ?? DEFAULT_BACKOFF;
    this.sleep = options.sleep ?? defaultSleep;
  }

  /** Highest sequence delivered so far (for tests/diagnostics). */
  get lastEventId(): number {
    return this.lastSeq;
  }

  start(): void {
    void this.runLoop();
  }

  stop(): void {
    this.stopped = true;
    this.controller?.abort();
  }

  private emitStatus(status: StreamStatus): void {
    this.handlers.onStatus?.(status);
  }

  private async runLoop(): Promise<void> {
    while (!this.stopped && !this.sawTerminal) {
      this.emitStatus(this.attempt === 0 ? "connecting" : "reconnecting");
      this.controller = new AbortController();
      let terminal = false;
      try {
        const headers: Record<string, string> = { Accept: "text/event-stream" };
        const token = this.options.tokenProvider();
        if (token) {
          headers.Authorization = `Bearer ${token}`;
        }
        if (this.lastSeq >= 0) {
          headers["Last-Event-ID"] = String(this.lastSeq);
        }
        const response = await this.fetchImpl(this.options.url, {
          headers,
          signal: this.controller.signal,
          cache: "no-store",
        });
        if (response.status === 401 || response.status === 404) {
          this.handlers.onError?.({
            retryable: false,
            message:
              response.status === 401
                ? "Your access session expired. Reconnect to keep watching this run."
                : "This research run no longer exists.",
          });
          this.emitStatus("error");
          return;
        }
        if (!response.ok || !response.body) {
          throw new Error(`stream http ${response.status}`);
        }
        // Note: `attempt` is deliberately NOT reset on a successful
        // connect — the reconnect budget is bounded per stream
        // lifetime (a server that keeps replaying then dropping
        // could otherwise loop forever). The manual "Reconnect"
        // action starts a fresh stream with a fresh budget.
        this.emitStatus("live");
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          let split = buffer.indexOf("\n\n");
          while (split !== -1) {
            const rawFrame = buffer.slice(0, split);
            buffer = buffer.slice(split + 2);
            terminal = this.handleFrame(rawFrame);
            if (terminal) break; // never process frames after terminal
            split = buffer.indexOf("\n\n");
          }
          if (terminal) break;
        }
        if (!terminal) {
          // Server closed without a terminal event: reconnect (the
          // run may still be alive; bounded below).
          throw new Error("stream ended before terminal event");
        }
        // Terminal event delivered — the run is finished; never
        // reconnect after this point (read-only stream ends here).
        this.sawTerminal = true;
        this.emitStatus("closed");
        return;
      } catch (error) {
        if (this.stopped) return;
        if (terminal) {
          this.sawTerminal = true;
          this.emitStatus("closed");
          return;
        }
        const isAbort =
          error instanceof Error && error.name === "AbortError";
        if (isAbort) return;
        const delay = this.backoff[Math.min(this.attempt, this.backoff.length - 1)];
        this.attempt += 1;
        if (this.attempt > this.backoff.length) {
          this.handlers.onError?.({
            retryable: true,
            message:
              "Lost the live connection to the research service. " +
              "Reconnect to resume watching this run — no research will be resubmitted.",
          });
          this.emitStatus("error");
          return;
        }
        this.emitStatus("reconnecting");
        await this.sleep(delay);
        if (this.stopped) return;
      }
    }
    if (this.sawTerminal) {
      this.emitStatus("closed");
    }
  }

  /** Parse one SSE frame; returns true when a terminal event arrived. */
  private handleFrame(rawFrame: string): boolean {
    const lines = rawFrame.split("\n");
    let data = "";
    for (const line of lines) {
      if (line.startsWith(":")) continue; // heartbeat comment
      if (line.startsWith("data: ")) {
        data += line.slice("data: ".length);
      } else if (line.startsWith("data:")) {
        data += line.slice("data:".length);
      }
      // `id:`/`event:` lines are implicit in the JSON payload
      // (seq/type) — the backend sets both; the header form is kept
      // for EventSource compatibility only.
    }
    if (!data) return false;
    let parsed: unknown;
    try {
      parsed = JSON.parse(data);
    } catch {
      return false;
    }
    if (
      typeof parsed !== "object" ||
      parsed === null ||
      typeof (parsed as { seq?: unknown }).seq !== "number" ||
      typeof (parsed as { type?: unknown }).type !== "string"
    ) {
      return false;
    }
    const event = parsed as RunEvent;
    if (event.seq <= this.lastSeq) {
      // Deduplicate: replay overlaps and duplicate deliveries are
      // dropped; the visible timeline is strictly increasing.
      return false;
    }
    this.lastSeq = event.seq;
    this.handlers.onEvent(event);
    return TERMINAL_TYPES.has(event.type);
  }
}
