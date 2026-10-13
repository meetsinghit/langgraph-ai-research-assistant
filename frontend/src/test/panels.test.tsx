/**
 * Right panel: per-run sources with citation highlighting, and
 * diagnostics including the missing-usage-metadata case.
 */
import { beforeEach, describe, expect, it } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { connectApp } from "./harness";
import {
  makeDetail,
  resetBackend,
  seedRun,
  seedSession,
} from "./mockBackend";

beforeEach(() => {
  resetBackend();
});

function seedCitedRun(prefix = "p") {
  seedSession({ id: `${prefix}-s1`, title: "Cited session", run_count: 1 });
  seedRun(
    {
      id: `${prefix}-r1`,
      session_id: `${prefix}-s1`,
      question: "Cited question",
      status: "completed",
      has_answer: true,
    },
    {
      answer: "Claim one [S1] and claim two [S2].",
      status: "completed",
      sources: [
        {
          sid: "S1",
          url: "https://official.example.gov/report",
          host: "official.example.gov",
          evidence: "retrieved",
          position: 0,
        },
        {
          sid: "S2",
          url: "https://blog.example.net/notes",
          host: "blog.example.net",
          evidence: "snippet",
          position: 1,
        },
      ],
    },
    [
      { seq: 1, type: "run.finished", outcome: "COMPLETED", review_status: "approved" },
      { seq: 2, type: "status.changed", status: "completed" },
    ],
  );
}

describe("sources panel", () => {
  it("lists per-run sources with hostname and retrieved/discovered status", async () => {
    seedCitedRun("a");
    const user = await connectApp();
    await user.click(await screen.findByText("Cited session"));

    expect(await screen.findByText("official.example.gov")).toBeInTheDocument();
    expect(screen.getByText("blog.example.net")).toBeInTheDocument();
    expect(screen.getByText("Verified")).toBeInTheDocument();
    expect(screen.getByText("Discovered")).toBeInTheDocument();
    // discovered sources explain they were not fetched
    expect(
      screen.getByText(/page not fetched/i),
    ).toBeInTheDocument();
  });

  it("clicking an inline citation highlights and scrolls to its source", async () => {
    seedCitedRun("b");
    const user = await connectApp();
    await user.click(await screen.findByText("Cited session"));
    await screen.findByText(/Claim one/);

    await user.click(screen.getByRole("button", { name: "Show source S1" }));
    const row = screen
      .getByRole("button", { name: /Citation S1: official\.example\.gov/ })
      .closest("li");
    await waitFor(() => expect(row).toHaveClass("ring-1"));
    expect(row).toHaveClass("ring-accent");
  });

  it("external source links open safely", async () => {
    seedCitedRun("c");
    const user = await connectApp();
    await user.click(await screen.findByText("Cited session"));
    const link = await screen.findByRole("link", {
      name: "https://official.example.gov/report",
    });
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
    expect(link).toHaveAttribute("target", "_blank");
  });
});

describe("diagnostics panel", () => {
  it("shows duration, tokens, coverage, counts and review state", async () => {
    seedCitedRun("d");
    const user = await connectApp();
    await user.click(await screen.findByText("Cited session"));
    await user.click(await screen.findByRole("tab", { name: "Diagnostics" }));

    expect(screen.getByText("Duration")).toBeInTheDocument();
    expect(screen.getByText("12.5 s")).toBeInTheDocument();
    expect(screen.getByText("Input tokens")).toBeInTheDocument();
    expect(screen.getByText("1,200")).toBeInTheDocument();
    expect(screen.getByText("Output tokens")).toBeInTheDocument();
    expect(screen.getByText("80")).toBeInTheDocument();
    expect(screen.getByText("100%")).toBeInTheDocument();
    expect(screen.getByText("Web searches")).toBeInTheDocument();
    expect(screen.getByText("No provider retries during this run.")).toBeInTheDocument();
    // review badge appears both in the answer header and diagnostics
    expect(screen.getAllByText("Approved by review").length).toBeGreaterThan(0);
    expect(screen.getByText("Injection signals")).toBeInTheDocument();
  });

  it("lists provider retries with categories", async () => {
    seedSession({ id: "e-s1", title: "Retry session", run_count: 1 });
    const detail = makeDetail({ id: "e-r1", session_id: "e-s1" });
    detail.diagnostics = {
      ...detail.diagnostics!,
      totals: { ...detail.diagnostics!.totals!, retries: 2 },
      retries: [
        { node: "research", iteration: 1, reason_category: "rate_limit" },
        { node: "synthesize", iteration: null, reason_category: "provider" },
      ],
    };
    seedRun(
      {
        id: "e-r1",
        session_id: "e-s1",
        question: "Retry question",
        status: "completed",
      },
      detail,
      [
        { seq: 1, type: "run.finished", outcome: "COMPLETED", review_status: "approved" },
        { seq: 2, type: "status.changed", status: "completed" },
      ],
    );
    const user = await connectApp();
    await user.click(await screen.findByText("Retry session"));
    await user.click(await screen.findByRole("tab", { name: "Diagnostics" }));

    expect(screen.getByText(/rate limit/i)).toBeInTheDocument();
    expect(screen.getByText(/synthesiz/)).toBeInTheDocument();
  });

  it("handles missing usage metadata without fabricating numbers", async () => {
    seedSession({ id: "f-s1", title: "No usage session", run_count: 1 });
    const detail = makeDetail({
      id: "f-r1",
      session_id: "f-s1",
      tokens_in: 0,
      tokens_out: 0,
      iterations: 1,
      diagnostics: null,
    });
    seedRun(
      {
        id: "f-r1",
        session_id: "f-s1",
        question: "No usage question",
        status: "completed",
      },
      detail,
      [
        { seq: 1, type: "run.finished", outcome: "COMPLETED", review_status: "approved" },
        { seq: 2, type: "status.changed", status: "completed" },
      ],
    );
    const user = await connectApp();
    await user.click(await screen.findByText("No usage session"));
    await screen.findByText(/An answer/);
    await user.click(screen.getByRole("tab", { name: "Diagnostics" }));

    expect(
      screen.getByText(/not reported by the provider/i),
    ).toBeInTheDocument();
    expect(screen.getByText("Retrieval counters are available once the run finishes.")).toBeInTheDocument();
    // model calls are NOT fabricated as 0
    const modelCallsRow = screen.getByText("Model calls").nextElementSibling;
    expect(modelCallsRow).toHaveTextContent("—");
  });
});
