/** Sessions, submitting questions, follow-ups, history navigation. */
import { beforeEach, describe, expect, it } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import { connectApp } from "./harness";
import { backend, resetBackend, seedRun, seedSession } from "./mockBackend";

beforeEach(() => {
  resetBackend();
});

describe("conversation lifecycle", () => {
  it("submits a question, creates a titled session and shows the run", async () => {
    const user = await connectApp();
    await user.type(
      screen.getByLabelText("Research question"),
      "What is quantum supremacy?",
    );
    await user.click(screen.getByRole("button", { name: "Start research" }));

    // session created with a title derived from the question
    await waitFor(() => {
      expect(
        backend.requests.some(
          (entry) => entry.method === "POST" && entry.path === "/api/sessions",
        ),
      ).toBe(true);
    });
    expect(backend.sessions[0].title).toContain("quantum supremacy");

    // the question appears as the user turn immediately (paragraph)
    const matches = await screen.findAllByText("What is quantum supremacy?");
    expect(matches.some((el) => el.tagName === "P")).toBe(true);
  });

  it("renders the completed answer with per-run citations and sources", async () => {
    seedSession({ id: "s1", title: "Quantum", run_count: 1 });
    seedRun(
      { id: "r1", session_id: "s1", question: "What is X?", status: "completed" },
      { answer: "X is a thing [S1]. See also [S9].", status: "completed" },
    );
    const user = await connectApp();
    await user.click(await screen.findByText("Quantum"));

    expect(await screen.findByText(/X is a thing/)).toBeInTheDocument();
    // known citation is a button; unknown renders as plain text
    expect(
      screen.getByRole("button", { name: "Show source S1" }),
    ).toBeInTheDocument();
    expect(screen.getByText("[S9]")).toBeInTheDocument();
  });

  it("follow-up questions become new runs in the SAME session", async () => {
    seedSession({ id: "s1", title: "Quantum", run_count: 1 });
    seedRun(
      { id: "r1", session_id: "s1", question: "First question", status: "completed" },
      { answer: "First answer." },
    );
    const user = await connectApp();
    await user.click(await screen.findByText("Quantum"));

    await user.type(
      screen.getByLabelText("Research question"),
      "And what about Y?",
    );
    await user.click(screen.getByRole("button", { name: "Start research" }));

    await waitFor(() => {
      const runPosts = backend.requests.filter(
        (entry) => entry.method === "POST" && entry.path === "/api/sessions/s1/runs",
      );
      expect(runPosts).toHaveLength(1);
    });
    // no second session was created
    const sessionPosts = backend.requests.filter(
      (entry) => entry.method === "POST" && entry.path === "/api/sessions",
    );
    expect(sessionPosts).toHaveLength(0);
    // both turns are visible, in order (paragraph question turns only)
    const newQuestion = await screen.findAllByText("And what about Y?");
    expect(newQuestion.some((el) => el.tagName === "P")).toBe(true);
    const questions = screen
      .getAllByText(/^(First question|And what about Y\?)$/)
      .filter((el) => el.tagName === "P");
    expect(questions).toHaveLength(2);
  });

  it("sends an Idempotency-Key so double submits cannot duplicate research", async () => {
    const user = await connectApp();
    await user.type(screen.getByLabelText("Research question"), "Question?");
    await user.click(screen.getByRole("button", { name: "Start research" }));
    await waitFor(() => {
      const runPost = backend.requests.find(
        (entry) => entry.method === "POST" && entry.path.endsWith("/runs"),
      );
      expect(runPost?.headers.get("Idempotency-Key")).toBeTruthy();
    });
  });

  it("reopening a completed run never resubmits it (read-only GETs)", async () => {
    seedSession({ id: "s1", title: "Old research", run_count: 1 });
    seedRun(
      { id: "r1", session_id: "s1", question: "Old question", status: "completed" },
      { answer: "Stored answer." },
    );
    const user = await connectApp();
    await user.click(await screen.findByText("Old research"));
    expect(await screen.findByText("Stored answer.")).toBeInTheDocument();

    const writes = backend.requests.filter(
      (entry) => entry.method === "POST" || entry.method === "DELETE",
    );
    expect(writes).toHaveLength(0);
  });

  it("navigating history loads runs in stored order", async () => {
    seedSession({ id: "s1", title: "First session", run_count: 2 });
    seedRun(
      { id: "r1", session_id: "s1", question: "Earlier question", status: "completed" },
      { answer: "Earlier answer." },
    );
    seedRun(
      { id: "r2", session_id: "s1", question: "Later question", status: "completed" },
      { answer: "Later answer." },
    );
    const user = await connectApp();
    await user.click(await screen.findByText("First session"));
    await screen.findByText("Earlier answer.");
    expect(screen.getByText("Later answer.")).toBeInTheDocument();
    const questions = screen
      .getAllByText(/^(Earlier|Later) question$/)
      .filter((el) => el.tagName === "P");
    expect(questions[0]).toHaveTextContent("Earlier question");
    expect(questions[1]).toHaveTextContent("Later question");
  });

  it("deleting a session requires confirmation and removes it", async () => {
    seedSession({ id: "s1", title: "Doomed", run_count: 0 });
    const user = await connectApp();
    await user.click(
      await screen.findByRole("button", { name: /delete session: doomed/i }),
    );
    const dialog = await screen.findByRole("dialog", {
      name: /delete this research session/i,
    });
    expect(within(dialog).getByText(/cannot be undone/i)).toBeInTheDocument();

    // cancel path
    await user.click(within(dialog).getByRole("button", { name: "Keep it" }));
    expect(screen.getByText("Doomed")).toBeInTheDocument();

    // confirm path
    await user.click(
      screen.getByRole("button", { name: /delete session: doomed/i }),
    );
    const dialog2 = await screen.findByRole("dialog", {
      name: /delete this research session/i,
    });
    await user.click(
      within(dialog2).getByRole("button", { name: "Delete session" }),
    );
    await waitFor(() => {
      expect(screen.queryByText("Doomed")).not.toBeInTheDocument();
    });
    expect(
      backend.requests.some((entry) => entry.method === "DELETE"),
    ).toBe(true);
  });

  it("shows an empty state before any research", async () => {
    await connectApp();
    expect(
      screen.getByRole("heading", { name: /what would you like to research/i }),
    ).toBeInTheDocument();
    expect(screen.getByLabelText("Research question")).toBeInTheDocument();
  });

  it("disables the submit button for empty input", async () => {
    const user = await connectApp();
    expect(screen.getByRole("button", { name: "Start research" })).toBeDisabled();
    await user.type(screen.getByLabelText("Research question"), "   ");
    expect(screen.getByRole("button", { name: "Start research" })).toBeDisabled();
  });
});
