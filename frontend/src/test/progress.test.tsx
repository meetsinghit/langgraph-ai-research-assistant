/**
 * Live progress: real SSE transitions, cancellation, terminal outcome
 * banners, and reconnect behavior (dedup, no duplicate run creation).
 */
import { beforeEach, describe, expect, it } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { connectApp } from "./harness";
import {
  backend,
  pushEvent,
  resetBackend,
  seedRun,
  seedSession,
} from "./mockBackend";

beforeEach(() => {
  resetBackend();
});

function seedLiveRun(prefix = "p1") {
  seedSession({ id: `${prefix}-s1`, title: "Live session", run_count: 1 });
  seedRun(
    {
      id: `${prefix}-r1`,
      session_id: `${prefix}-s1`,
      question: "Live question",
      status: "running",
      has_answer: false,
    },
    {
      answer: null,
      status: "running",
      review_status: null,
      outcome_status: null,
      sources: [],
    },
    [
      { seq: 1, type: "status.changed", status: "running" },
      { seq: 2, type: "run.started" },
      { seq: 3, type: "stage.started", node: "plan" },
      { seq: 4, type: "stage.finished", node: "plan" },
      { seq: 5, type: "stage.started", node: "research" },
    ],
  );
}

async function finishLiveRun(runId: string, firstSeq = 6) {
  pushEvent(runId, { seq: firstSeq, type: "stage.finished", node: "research" });
  pushEvent(runId, { seq: firstSeq + 1, type: "stage.started", node: "synthesize" });
  pushEvent(runId, { seq: firstSeq + 2, type: "stage.finished", node: "synthesize" });
  pushEvent(runId, { seq: firstSeq + 3, type: "stage.started", node: "review" });
  pushEvent(runId, { seq: firstSeq + 4, type: "stage.finished", node: "review" });
  pushEvent(runId, {
    seq: firstSeq + 5,
    type: "run.finished",
    outcome: "COMPLETED",
    review_status: "approved",
  });
  pushEvent(runId, { seq: firstSeq + 6, type: "status.changed", status: "completed" });
}

describe("live progress", () => {
  it("streams stage transitions into the timeline and completes with an answer", async () => {
    seedLiveRun("a");
    const user = await connectApp();
    await user.click(await screen.findByText("Live session"));

    // replayed progress: plan done, research active — no fabricated %
    expect(await screen.findByText("Planning the research")).toBeInTheDocument();
    expect(screen.getByText("Searching the web")).toBeInTheDocument();
    expect(screen.queryByText(/\d+%/)).toBeNull();
    // no answer yet
    expect(screen.queryByText(/Mock answer/)).toBeNull();

    // live continuation
    pushEvent("a-r1", { seq: 6, type: "tool.started", tool: "web_search" });
    pushEvent("a-r1", {
      seq: 7,
      type: "tool.finished",
      tool: "web_search",
      ok: true,
      latency_ms: 50,
    });
    expect(await screen.findByText(/1 search/)).toBeInTheDocument();

    await finishLiveRun("a-r1");
    expect(
      await screen.findByText("Mock answer for: Live question", {}, { timeout: 4000 }),
    ).toBeInTheDocument();
    // terminal state is visible
    expect(screen.getByText("Research completed")).toBeInTheDocument();
    expect(screen.getByText("Approved by review")).toBeInTheDocument();
  });

  it("cancels an active run (POST only, status becomes cancelling)", async () => {
    seedLiveRun("b");
    const user = await connectApp();
    await user.click(await screen.findByText("Live session"));

    await user.click(await screen.findByRole("button", { name: /stop research/i }));
    await waitFor(() => {
      expect(
        backend.requests.some(
          (entry) => entry.method === "POST" && entry.path === "/api/runs/b-r1/cancel",
        ),
      ).toBe(true);
    });
    // no duplicate run creation from any UI action here
    expect(
      backend.requests.filter((entry) => entry.path.endsWith("/runs") && entry.method === "POST"),
    ).toHaveLength(0);
  });

  it("shows friendly text for a failed run (never a raw trace)", async () => {
    seedSession({ id: "f-s1", title: "Failed session", run_count: 1 });
    seedRun(
      {
        id: "f-r1",
        session_id: "f-s1",
        question: "Doomed question",
        status: "failed",
        has_answer: false,
      },
      {
        answer: null,
        status: "failed",
        error_category: "rate_limit",
        outcome_status: "FAILED",
        review_status: null,
        sources: [],
      },
      [
        { seq: 1, type: "status.changed", status: "running" },
        { seq: 2, type: "run.started" },
        { seq: 3, type: "run.failed", error_category: "rate_limit" },
        { seq: 4, type: "status.changed", status: "failed" },
      ],
    );
    const user = await connectApp();
    await user.click(await screen.findByText("Failed session"));

    const banner = await screen.findByRole("alert");
    expect(banner.textContent).toMatch(/rate limiting/i);
    expect(document.body.textContent).not.toMatch(/Traceback|Exception|File "/);
    expect(screen.getByText("Research failed — Rate limited")).toBeInTheDocument();
  });

  it("surfaces an incomplete outcome with its warning", async () => {
    seedSession({ id: "i-s1", title: "Partial session", run_count: 1 });
    seedRun(
      {
        id: "i-r1",
        session_id: "i-s1",
        question: "Partial question",
        status: "completed",
        has_answer: true,
      },
      {
        answer: "A partial answer [S1].",
        status: "completed",
        outcome_status: "INCOMPLETE",
        review_status: "needs_more_research",
        outcome_warnings: ["review_wanted_more_research"],
        sources: [
          {
            sid: "S1",
            url: "https://example.com/p",
            host: "example.com",
            evidence: "snippet",
            position: 0,
          },
        ],
      },
      [
        { seq: 1, type: "run.finished", outcome: "INCOMPLETE", review_status: "needs_more_research" },
        { seq: 2, type: "status.changed", status: "completed" },
      ],
    );
    const user = await connectApp();
    await user.click(await screen.findByText("Partial session"));

    expect(await screen.findByText(/A partial answer/)).toBeInTheDocument();
    expect(screen.getByText("Incomplete")).toBeInTheDocument();
    expect(screen.getByText("Needs more research")).toBeInTheDocument();
    expect(screen.getByText(/quality review found it incomplete/i)).toBeInTheDocument();
    // outcome warnings are content-free diagnostics — visible on the Diagnostics tab
    await user.click(screen.getByRole("tab", { name: "Diagnostics" }));
    expect(screen.getByText("review wanted more research")).toBeInTheDocument();
  });

  it("reconnects after a dropped stream without duplicating progress", async () => {
    seedSession({ id: "r-s1", title: "Flaky session", run_count: 1 });
    seedRun(
      {
        id: "r-r1",
        session_id: "r-s1",
        question: "Flaky question",
        status: "running",
        has_answer: false,
      },
      {
        answer: null,
        status: "running",
        review_status: null,
        outcome_status: null,
        sources: [],
      },
      [
        { seq: 1, type: "status.changed", status: "running" },
        { seq: 2, type: "run.started" },
        { seq: 3, type: "stage.started", node: "plan" },
        { seq: 4, type: "stage.finished", node: "plan" },
        { seq: 5, type: "stage.started", node: "research" },
      ],
    );
    backend.runs["r-r1"].dropAfter = 2; // first connection dies early
    const user = await connectApp();
    await user.click(await screen.findByText("Flaky session"));

    // progress resumes via Last-Event-ID replay and completes exactly once
    await finishLiveRun("r-r1");
    expect(
      await screen.findByText("Mock answer for: Flaky question", {}, { timeout: 6000 }),
    ).toBeInTheDocument();
    // no duplicated steps or answers after the reconnect
    expect(screen.getAllByText("Planning the research")).toHaveLength(1);
    expect(screen.getAllByText("Research completed")).toHaveLength(1);
    expect(
      screen.getAllByText("Mock answer for: Flaky question"),
    ).toHaveLength(1);
    // still exactly one run in the session — reconnect never creates runs
    expect(
      backend.requests.filter((entry) => entry.method === "POST" && entry.path.endsWith("/runs")),
    ).toHaveLength(0);
  });

  it("shows a cancelled state once the backend confirms cancellation", async () => {
    seedLiveRun("c");
    const user = await connectApp();
    await user.click(await screen.findByText("Live session"));
    await user.click(await screen.findByRole("button", { name: /stop research/i }));

    pushEvent("c-r1", { seq: 6, type: "run.finished", outcome: "CANCELLED" });
    pushEvent("c-r1", { seq: 7, type: "status.changed", status: "cancelled" });

    expect(
      await screen.findByText("Research cancelled", {}, { timeout: 4000 }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/cancelled before an answer/i),
    ).toBeInTheDocument();
  });
});
