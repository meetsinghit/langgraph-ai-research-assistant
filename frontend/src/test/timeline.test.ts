/** Timeline derivation from the real backend event contracts. */
import { describe, expect, it } from "vitest";
import { deriveTimeline } from "../lib/timeline";
import type { RunEvent } from "../api/types";

const e = (seq: number, type: string, payload: object = {}): RunEvent =>
  ({ seq, type, ...payload }) as RunEvent;

describe("deriveTimeline", () => {
  it("starts with every step pending and no terminal state", () => {
    const result = deriveTimeline([], "queued");
    expect(result.steps.map((s) => [s.id, s.state])).toEqual([
      ["plan", "pending"],
      ["search", "pending"],
      ["fetch", "pending"],
      ["synthesize", "pending"],
      ["review", "pending"],
    ]);
    expect(result.terminal).toBeNull();
    expect(result.running).toBe(true);
  });

  it("marks steps active/done as stage events arrive", () => {
    const result = deriveTimeline(
      [
        e(1, "run.started"),
        e(2, "stage.started", { node: "plan" }),
        e(3, "stage.finished", { node: "plan" }),
        e(4, "stage.started", { node: "research" }),
      ],
      "running",
    );
    expect(result.steps.find((s) => s.id === "plan")?.state).toBe("done");
    expect(result.steps.find((s) => s.id === "search")?.state).toBe("active");
    expect(result.steps.find((s) => s.id === "fetch")?.state).toBe("active");
    expect(result.steps.find((s) => s.id === "synthesize")?.state).toBe(
      "pending",
    );
    expect(result.terminal).toBeNull();
    expect(result.running).toBe(true);
  });

  it("surfaces live activity counters (no percentages)", () => {
    const result = deriveTimeline(
      [
        e(1, "stage.started", { node: "research" }),
        e(2, "tool.started", { tool: "web_search" }),
        e(3, "tool.finished", { tool: "web_search", ok: true, latency_ms: 10 }),
        e(4, "source.discovered", { url: "https://a.example/x" }),
        e(5, "tool.started", { tool: "fetch_page" }),
        e(6, "source.retrieved", { url: "https://a.example/x" }),
      ],
      "running",
    );
    const search = result.steps.find((s) => s.id === "search");
    const fetchStep = result.steps.find((s) => s.id === "fetch");
    expect(search?.detail).toMatch(/1 search/);
    expect(fetchStep?.detail).toMatch(/1 page verified/);
    expect(fetchStep?.detail).toMatch(/fetching/);
    // absolutely no % anywhere
    for (const step of result.steps) {
      expect(step.detail ?? "").not.toMatch(/%|percent/i);
    }
  });

  it("counts only exact tool names as searches and fetches (4E.2)", () => {
    const result = deriveTimeline(
      [
        e(1, "stage.started", { node: "research" }),
        // a failed fetch is still a fetch attempt
        e(2, "tool.finished", { tool: "fetch_page", ok: false, latency_ms: 5 }),
        // an unknown/future tool must not be miscounted as a fetch
        e(3, "tool.finished", { tool: "other_tool", ok: true, latency_ms: 5 }),
        e(4, "tool.finished", { tool: "web_search", ok: false, latency_ms: 5 }),
      ],
      "running",
    );
    const search = result.steps.find((s) => s.id === "search");
    const fetchStep = result.steps.find((s) => s.id === "fetch");
    expect(search?.detail).toMatch(/1 search/);
    expect(search?.detail).toMatch(/1 failed/);
    expect(fetchStep?.detail).toMatch(/1 fetch failed/);
    // only ONE fetch attempt — the unknown tool was not counted
    expect(fetchStep?.detail).not.toMatch(/2 fetch/);
  });

  it("renders a completed terminal state with review info", () => {
    const result = deriveTimeline(
      [
        e(1, "run.finished", { outcome: "COMPLETED", review_status: "approved" }),
        e(2, "status.changed", { status: "completed" }),
      ],
      "completed",
    );
    expect(result.terminal).toMatchObject({ kind: "completed" });
    expect((result.terminal as { outcome?: string }).outcome).toContain(
      "approved",
    );
    expect(result.running).toBe(false);
  });

  it("distinguishes an incomplete outcome from a clean completion", () => {
    const result = deriveTimeline(
      [
        e(1, "run.finished", { outcome: "INCOMPLETE", review_status: "needs_more_research" }),
        e(2, "status.changed", { status: "completed" }),
      ],
      "completed",
    );
    expect(result.terminal).toMatchObject({ kind: "incomplete" });
    expect(result.terminal?.label).toMatch(/incomplete/i);
  });

  it("renders failed with its error category", () => {
    const result = deriveTimeline(
      [
        e(1, "run.failed", { error_category: "rate_limit" }),
        e(2, "status.changed", { status: "failed" }),
      ],
      "failed",
    );
    expect(result.terminal).toMatchObject({
      kind: "failed",
      category: "rate_limit",
    });
  });

  it("renders cancelled and interrupted terminal states", () => {
    expect(
      deriveTimeline(
        [e(1, "status.changed", { status: "cancelled" })],
        "cancelled",
      ).terminal,
    ).toMatchObject({ kind: "cancelled" });
    expect(
      deriveTimeline(
        [e(1, "status.changed", { status: "interrupted" })],
        "interrupted",
      ).terminal,
    ).toMatchObject({ kind: "interrupted" });
  });

  it("counts failed tool attempts without failing the run", () => {
    const result = deriveTimeline(
      [
        e(1, "stage.started", { node: "research" }),
        e(2, "tool.started", { tool: "web_search" }),
        e(3, "tool.finished", {
          tool: "web_search",
          ok: false,
          error_category: "tls",
          latency_ms: 5,
        }),
      ],
      "running",
    );
    expect(result.steps.find((s) => s.id === "search")?.detail).toMatch(
      /1 failed/,
    );
    expect(result.running).toBe(true);
  });
});
