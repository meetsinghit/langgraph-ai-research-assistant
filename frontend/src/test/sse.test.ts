/**
 * RunStream (fetch-based SSE) contract:
 *  - frame parsing (id/event/data + heartbeat comments);
 *  - strictly-increasing sequence deduplication;
 *  - Last-Event-ID resume on reconnect (no duplicated delivery);
 *  - bounded reconnect with backoff, then a retryable error;
 *  - 401/404 surfaced as non-retryable;
 *  - terminal event ends the stream for good;
 *  - stop() hard-cancels everything.
 */
import { describe, expect, it, vi } from "vitest";
import { RunStream, type RunStreamOptions, type StreamStatus } from "../api/sse";
import type { RunEvent } from "../api/types";

function frame(seq: number, type: string, payload: object = {}): string {
  return `id: ${seq}\nevent: ${type}\ndata: ${JSON.stringify({ seq, type, ...payload })}\n\n`;
}

function responseFrom(frames: string[], status = 200): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const f of frames) controller.enqueue(encoder.encode(f));
      controller.close();
    },
  });
  if (status !== 200) return new Response(null, { status });
  return new Response(stream, {
    status: 200,
    headers: { "Content-Type": "text/event-stream" },
  });
}

interface Recorder {
  events: RunEvent[];
  statuses: StreamStatus[];
  errors: Array<{ retryable: boolean; message: string }>;
}

function recorder(): Recorder {
  return { events: [], statuses: [], errors: [] };
}

function makeStream(
  responses: Response[],
  recorder: Recorder,
  overrides: Partial<RunStreamOptions> = {},
): { stream: RunStream; calls: Array<{ url: string; headers: Headers }> } {
  const calls: Array<{ url: string; headers: Headers }> = [];
  let index = 0;
  const fetchImpl = vi.fn(async (url: string, init?: RequestInit) => {
    calls.push({ url, headers: new Headers(init?.headers) });
    const response = responses[Math.min(index, responses.length - 1)];
    index += 1;
    if (response instanceof Promise) return response;
    return response;
  }) as unknown as typeof fetch;

  const stream = new RunStream(
    {
      onEvent: (event) => recorder.events.push(event),
      onStatus: (status) => recorder.statuses.push(status),
      onError: (info) => recorder.errors.push(info),
    },
    {
      url: "/api/runs/r1/events",
      tokenProvider: () => "tok",
      fetchImpl,
      backoffMs: [1, 1, 1, 1, 1],
      sleep: async () => undefined,
      ...overrides,
    },
  );
  return { stream, calls };
}

const terminal = () =>
  frame(4, "run.finished", { outcome: "COMPLETED", review_status: "approved" });

describe("RunStream", () => {
  it("parses frames, ignores heartbeats and reports live status", async () => {
    const rec = recorder();
    const response = responseFrom([
      ": heartbeat\n\n",
      frame(1, "status.changed", { status: "running" }),
      frame(2, "stage.started", { node: "plan" }),
      terminal(),
    ]);
    const { stream, calls } = makeStream([response], rec);
    stream.start();
    await vi.waitFor(() => expect(rec.statuses).toContain("closed"));
    expect(rec.events.map((e) => e.seq)).toEqual([1, 2, 4]);
    expect(rec.events[1]).toMatchObject({ type: "stage.started", node: "plan" });
    expect(rec.errors).toHaveLength(0);
    expect(calls).toHaveLength(1);
    expect(calls[0].headers.get("Authorization")).toBe("Bearer tok");
    expect(calls[0].headers.get("Last-Event-ID")).toBeNull();
  });

  it("reconnects with Last-Event-ID and deduplicates replayed events", async () => {
    const rec = recorder();
    // First connection: seq 1-2 then the socket dies (no terminal).
    // Second connection: replays seq 2 (overlap) + 3 + terminal.
    const responses = [
      responseFrom([frame(1, "run.started"), frame(2, "stage.started", { node: "plan" })]),
      responseFrom([
        frame(2, "stage.started", { node: "plan" }),
        frame(3, "stage.finished", { node: "plan" }),
        terminal(),
      ]),
    ];
    const { stream, calls } = makeStream(responses, rec);
    stream.start();
    await vi.waitFor(() => expect(rec.statuses).toContain("closed"));

    expect(calls).toHaveLength(2);
    expect(calls[1].headers.get("Last-Event-ID")).toBe("2");
    // strictly increasing: overlap seq 2 dropped exactly once
    expect(rec.events.map((e) => e.seq)).toEqual([1, 2, 3, 4]);
    expect(rec.errors).toHaveLength(0);
  });

  it("does not reconnect after a terminal event", async () => {
    const rec = recorder();
    const { stream, calls } = makeStream(
      [responseFrom([frame(1, "run.failed", { error_category: "provider" }), terminal()])],
      rec,
    );
    // run.failed is terminal too — stream must stop at seq 1.
    stream.start();
    await vi.waitFor(() => expect(rec.statuses).toContain("closed"));
    expect(rec.events.map((e) => e.seq)).toEqual([1]);
    expect(calls).toHaveLength(1);
  });

  it("surfaces 401 as a non-retryable auth error", async () => {
    const rec = recorder();
    const { stream, calls } = makeStream(
      [responseFrom([], 401)],
      rec,
    );
    stream.start();
    await vi.waitFor(() => expect(rec.statuses).toContain("error"));
    expect(rec.errors).toEqual([
      { retryable: false, message: expect.stringMatching(/reconnect/i) },
    ]);
    expect(calls).toHaveLength(1);
  });

  it("surfaces 404 as a non-retryable 'run no longer exists' error", async () => {
    const rec = recorder();
    const { stream, calls } = makeStream([responseFrom([], 404)], rec);
    stream.start();
    await vi.waitFor(() => expect(rec.statuses).toContain("error"));
    expect(rec.errors[0]?.retryable).toBe(false);
    expect(rec.errors[0]?.message).toMatch(/no longer exists/i);
    expect(calls).toHaveLength(1);
  });

  it("gives up after the bounded reconnect budget with a friendly message", async () => {
    const rec = recorder();
    // 502 every time, backoff budget of 1 attempt
    const { stream, calls } = makeStream([responseFrom([], 502)], rec, {
      backoffMs: [1],
    });
    stream.start();
    await vi.waitFor(() => expect(rec.statuses).toContain("error"));
    expect(rec.errors[0]?.retryable).toBe(true);
    expect(rec.errors[0]?.message).toMatch(/never resubmitted|resubmit/i);
    expect(calls.length).toBeLessThanOrEqual(3);
    expect(calls.length).toBeGreaterThan(1);
  });

  it("counts replay-then-drop cycles toward the bounded budget", async () => {
    const rec = recorder();
    // Every connection accepts and immediately closes without a
    // terminal event — must terminate as an error, not loop forever.
    const drop = () => responseFrom([frame(1, "run.started")]);
    const { stream, calls } = makeStream(
      [drop(), drop(), drop(), drop(), drop(), drop(), drop()],
      rec,
      { backoffMs: [1, 1, 1, 1, 1] },
    );
    stream.start();
    await vi.waitFor(() => expect(rec.statuses).toContain("error"));
    expect(rec.errors[0]?.retryable).toBe(true);
    expect(calls.length).toBeLessThanOrEqual(7);
    // every retry after the first resumes via Last-Event-ID
    expect(
      calls.slice(1).every((c) => c.headers.get("Last-Event-ID") === "1"),
    ).toBe(true);
  });

  it("stop() aborts the connection and prevents further attempts", async () => {
    const rec = recorder();
    // A fetch that never yields data; it settles only when aborted.
    const fetchImpl = vi.fn(
      (_url: string, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => {
            const error = new Error("aborted");
            error.name = "AbortError";
            reject(error);
          });
        }),
    ) as unknown as typeof fetch;
    const stream = new RunStream(
      {
        onEvent: (e) => rec.events.push(e),
        onStatus: (s) => rec.statuses.push(s),
        onError: (i) => rec.errors.push(i),
      },
      {
        url: "/api/runs/r1/events",
        tokenProvider: () => "tok",
        fetchImpl,
        backoffMs: [1],
        sleep: async () => undefined,
      },
    );
    stream.start();
    await new Promise((r) => setTimeout(r, 5));
    stream.stop();
    await new Promise((r) => setTimeout(r, 20));
    // fetch was called once; no error, no reconnect after stop
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(rec.errors).toHaveLength(0);
    expect(rec.statuses).not.toContain("error");
  });
});
