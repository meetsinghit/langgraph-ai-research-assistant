/**
 * Accessibility smoke tests and responsive drawer behavior.
 */
import { beforeEach, describe, expect, it } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import { connectApp } from "./harness";
import { backend, resetBackend, seedRun, seedSession } from "./mockBackend";

beforeEach(() => {
  resetBackend();
});

describe("accessibility", () => {
  it("exposes landmark roles and labelled controls", async () => {
    const user = await connectApp();
    // navigation landmark with sessions
    expect(screen.getByRole("navigation", { name: "Research sessions" })).toBeInTheDocument();
    // main question control is labelled
    expect(screen.getByRole("textbox", { name: /research question/i })).toBeInTheDocument();
    // right panel tabs
    const tabs = screen.getByRole("tablist", { name: /sources and diagnostics/i });
    expect(within(tabs).getByRole("tab", { name: "Sources" })).toBeInTheDocument();
    expect(within(tabs).getByRole("tab", { name: "Diagnostics" })).toBeInTheDocument();
    // header actions are labelled
    expect(screen.getByRole("button", { name: "Start a new research session" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /switch to (light|dark) theme/i })).toBeInTheDocument();
    expect(user).toBeDefined();
  });

  it("marks the active session with aria-current", async () => {
    seedSession({ id: "s1", title: "Active session", run_count: 0 });
    const user = await connectApp();
    await user.click(await screen.findByText("Active session"));
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /^Active session/ }),
      ).toHaveAttribute("aria-current", "true");
    });
  });

  it("submits with Enter and keeps Shift+Enter as a newline", async () => {
    seedSession({ id: "s1", title: "Keyboard session", run_count: 0 });
    const user = await connectApp();
    await user.click(await screen.findByText("Keyboard session"));
    const input = screen.getByRole("textbox", { name: /research question/i });

    // Shift+Enter inserts a newline without submitting
    await user.click(input);
    await user.keyboard("line one{Shift>}{Enter}{/Shift}line two");
    expect(input).toHaveValue("line one\nline two");
    expect(
      backend.requests.filter((entry) => entry.method === "POST" && entry.path.endsWith("/runs")),
    ).toHaveLength(0);

    // Enter submits
    await user.clear(input);
    await user.keyboard("keyboard question{Enter}");
    await waitFor(() => {
      expect(
        backend.requests.filter(
          (entry) => entry.method === "POST" && entry.path === "/api/sessions/s1/runs",
        ),
      ).toHaveLength(1);
    });
  });

  it("announces a character counter near the limit", async () => {
    const user = await connectApp();
    const input = screen.getByRole("textbox", { name: /research question/i });
    await user.click(input);
    await user.paste("x".repeat(1850));
    expect(screen.getByText(/1850\/2000/)).toBeInTheDocument();
  });
});

describe("responsive drawers", () => {
  it("opens the history drawer and closes it after navigation", async () => {
    seedSession({ id: "s1", title: "Drawer session", run_count: 0 });
    const user = await connectApp();

    await user.click(screen.getByRole("button", { name: "Open research history" }));
    const drawer = await screen.findByRole("dialog", { name: "Research history" });
    expect(
      within(drawer).getByRole("button", { name: "Start a new research session" }),
    ).toBeInTheDocument();

    await user.click(within(drawer).getByText("Drawer session"));
    await waitFor(() => {
      expect(
        screen.queryByRole("dialog", { name: "Research history" }),
      ).toBeNull();
    });
  });

  it("opens the sources drawer on narrow screens", async () => {
    seedSession({ id: "s1", title: "Panel session", run_count: 1 });
    seedRun(
      { id: "r1", session_id: "s1", question: "Q", status: "completed" },
      { answer: "Answer." },
      [
        { seq: 1, type: "run.finished", outcome: "COMPLETED", review_status: "approved" },
        { seq: 2, type: "status.changed", status: "completed" },
      ],
    );
    const user = await connectApp();
    await user.click(
      await screen.findByText("Panel session", {}, { timeout: 3000 }),
    );
    await screen.findByText("Answer.");

    await user.click(
      screen.getByRole("button", { name: "Open sources and diagnostics" }),
    );
    const drawer = await screen.findByRole("dialog", {
      name: "Sources and diagnostics",
    });
    expect(within(drawer).getByRole("tab", { name: "Sources" })).toBeInTheDocument();
  });
});
