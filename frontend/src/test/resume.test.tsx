/**
 * Interrupted-run recovery: the resume card appears ONLY when the
 * backend reports the run as interrupted and resume is enabled, with
 * an explicit repetition caveat.
 */
import { beforeEach, describe, expect, it } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { connectApp } from "./harness";
import {
  backend,
  defaultConfig,
  resetBackend,
  seedRun,
  seedSession,
  server,
} from "./mockBackend";

function seedInterrupted(prefix = "v") {
  seedSession({ id: `${prefix}-s1`, title: "Interrupted session", run_count: 1 });
  seedRun(
    {
      id: `${prefix}-r1`,
      session_id: `${prefix}-s1`,
      question: "Interrupted question",
      status: "interrupted",
      has_answer: false,
    },
    {
      answer: null,
      status: "interrupted",
      review_status: null,
      outcome_status: null,
      sources: [],
      diagnostics: null,
    },
    [],
  );
}

beforeEach(() => {
  resetBackend();
});

describe("interrupted run recovery", () => {
  it("offers resume with the repetition caveat when enabled", async () => {
    seedInterrupted("a");
    const user = await connectApp();
    await user.click(await screen.findByText("Interrupted session"));

    expect(
      await screen.findByText(/interrupted — resumable/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/may repeat/i),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Resume research" }));
    await waitFor(() => {
      expect(
        backend.requests.some(
          (entry) =>
            entry.method === "POST" && entry.path === "/api/runs/a-r1/resume",
        ),
      ).toBe(true);
    });
  });

  it("never offers resume when the backend disables it", async () => {
    server.use(
      http.get("/api/config", () =>
        HttpResponse.json({ ...defaultConfig, resume_enabled: false }),
      ),
    );
    seedInterrupted("b");
    const user = await connectApp();
    await user.click(await screen.findByText("Interrupted session"));

    expect(
      await screen.findByText(/cannot be resumed/i),
    ).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Resume research" })).toBeNull();
  });

  it("never offers resume for non-interrupted runs", async () => {
    seedSession({ id: "c-s1", title: "Done session", run_count: 1 });
    seedRun(
      {
        id: "c-r1",
        session_id: "c-s1",
        question: "Done question",
        status: "completed",
      },
      { answer: "Done answer." },
      [
        { seq: 1, type: "run.finished", outcome: "COMPLETED", review_status: "approved" },
        { seq: 2, type: "status.changed", status: "completed" },
      ],
    );
    const user = await connectApp();
    await user.click(await screen.findByText("Done session"));
    await screen.findByText("Done answer.");
    expect(screen.queryByRole("button", { name: "Resume research" })).toBeNull();
    expect(screen.queryByText(/resumable/i)).toBeNull();
  });

  it("surfaces the backend's 409 conflict when resume is refused", async () => {
    server.use(
      http.post("/api/runs/:id/resume", () =>
        HttpResponse.json(
          {
            error: {
              category: "conflict",
              message: "resume unavailable: not_interrupted",
            },
          },
          { status: 409 },
        ),
      ),
    );
    seedInterrupted("d");
    const user = await connectApp();
    await user.click(await screen.findByText("Interrupted session"));
    await user.click(await screen.findByRole("button", { name: "Resume research" }));

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toMatch(/not_interrupted|resume unavailable/i);
  });
});
