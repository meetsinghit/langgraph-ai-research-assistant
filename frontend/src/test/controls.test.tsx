/**
 * Composer research settings (4E.3): model/thinking/depth/evidence
 * selectors with capability-aware thinking levels, frozen per-run
 * payloads, session inheritance, session rename, and the diagnostics
 * run-configuration + thinking-token display.
 *
 * All offline against the MSW mock backend (registry-shaped catalogue,
 * discovery disabled — mirroring the real default).
 */
import { beforeEach, describe, expect, it } from "vitest";
import { screen, waitFor, within } from "@testing-library/react";
import { connectApp } from "./harness";
import { backend, resetBackend, seedRun, seedSession } from "./mockBackend";

beforeEach(() => {
  resetBackend();
});

function lastRunCreateBody(): Record<string, unknown> {
  const posts = backend.requests.filter(
    (entry) => entry.method === "POST" && entry.path.endsWith("/runs"),
  );
  expect(posts.length).toBeGreaterThan(0);
  return posts[posts.length - 1].body as Record<string, unknown>;
}

describe("research settings composer", () => {
  it("offers only selectable models with the configured default first", async () => {
    await connectApp();
    const model = (await screen.findByLabelText("Model")) as HTMLSelectElement;
    const options = [...model.options].map((option) => option.value);
    // default is the null option
    expect(options[0]).toBe("");
    expect(model.options[0].textContent).toContain("Gemini 3.8 Flash");
    // selectable alternatives are listed…
    expect(options).toContain("gemini-3.6-flash");
    expect(options).toContain("gemini-2.5-flash");
    // …and nothing else: paid, unknown-eligibility and retired
    // models never appear (the backend catalogue is the authority)
    expect(options).not.toContain("gemini-4-argon");
    expect(options).not.toContain("gemini-3.7-flash");
    expect(options).not.toContain("gemini-2.0-flash");
    // depth and evidence selectors exist with working defaults
    expect((screen.getByLabelText("Depth") as HTMLSelectElement).value).toBe(
      "standard",
    );
    expect(
      (screen.getByLabelText("Evidence") as HTMLSelectElement).value,
    ).toBe("balanced");
  });

  it("capability-filters thinking levels per selected model", async () => {
    const user = await connectApp();
    const thinking = (await screen.findByLabelText(
      "Thinking",
    )) as HTMLSelectElement;
    const model = screen.getByLabelText("Model") as HTMLSelectElement;

    // default model (3.8-flash): auto/low/medium/high — no Minimal
    await waitFor(() => expect(thinking.disabled).toBe(false));
    expect([...thinking.options].map((o) => o.value)).toEqual([
      "",
      "auto",
      "low",
      "medium",
      "high",
    ]);

    // budget-family model (2.5-flash): Minimal becomes available
    await user.selectOptions(model, "gemini-2.5-flash");
    await waitFor(() =>
      expect([...thinking.options].map((o) => o.value)).toEqual([
        "",
        "auto",
        "minimal",
        "low",
        "medium",
        "high",
      ]),
    );

    // a level-controlled model without minimal: Minimal disappears
    await user.selectOptions(model, "gemini-3.6-flash");
    await waitFor(() =>
      expect(
        [...thinking.options].map((o) => o.value),
      ).not.toContain("minimal"),
    );
  });

  it("falls back to the model default when switching to a model that does not support the selected level", async () => {
    const user = await connectApp();
    const model = await screen.findByLabelText("Model");
    const thinking = screen.getByLabelText("Thinking") as HTMLSelectElement;

    await user.selectOptions(model, "gemini-2.5-flash");
    await user.selectOptions(thinking, "minimal");
    expect(thinking.value).toBe("minimal");

    // switching to a model without minimal resets to its default
    await user.selectOptions(model, "gemini-3.6-flash");
    await waitFor(() => expect(thinking.value).toBe(""));
  });

  it("submits the frozen selections with the question", async () => {
    const user = await connectApp();
    await user.selectOptions(await screen.findByLabelText("Model"), "gemini-2.5-flash");
    await user.selectOptions(screen.getByLabelText("Thinking"), "minimal");
    await user.selectOptions(screen.getByLabelText("Depth"), "deep");
    await user.selectOptions(screen.getByLabelText("Evidence"), "verified");
    await user.type(
      screen.getByLabelText("Research question"),
      "What changed in the registry?",
    );
    await user.click(screen.getByRole("button", { name: "Start research" }));

    await waitFor(() => {
      expect(lastRunCreateBody()).toEqual({
        question: "What changed in the registry?",
        research_depth: "deep",
        evidence_policy: "verified",
        model_id: "gemini-2.5-flash",
        thinking_level: "minimal",
      });
    });
    // the mock backend froze the same values onto the stored run
    const run = Object.values(backend.runs)[0];
    expect(run.summary.model_id).toBe("gemini-2.5-flash");
    expect(run.detail.thinking_level).toBe("minimal");
    expect(run.detail.diagnostics?.run_config).toEqual({
      research_depth: "deep",
      evidence_policy: "verified",
      model_id: "gemini-2.5-flash",
      thinking_level: "minimal",
    });
  });

  it("default selections submit explicit nulls for model and thinking", async () => {
    const user = await connectApp();
    await user.type(
      screen.getByLabelText("Research question"),
      "Plain default question",
    );
    await user.click(screen.getByRole("button", { name: "Start research" }));

    await waitFor(() => {
      expect(lastRunCreateBody()).toEqual({
        question: "Plain default question",
        research_depth: "standard",
        evidence_policy: "balanced",
        model_id: null,
        thinking_level: null,
      });
    });
    // null resolves to the configured default on the stored run
    const run = Object.values(backend.runs)[0];
    expect(run.summary.model_id).toBe("gemini-3.8-flash");
    expect(run.summary.thinking_level).toBe("medium");
  });

  it("inherits the session's most recent run settings and corrects stale ones", async () => {
    seedSession({ id: "s1", title: "Registry", run_count: 1 });
    seedRun(
      {
        id: "r1",
        session_id: "s1",
        question: "Earlier question",
        model_id: "gemini-3.6-flash",
        thinking_level: "high",
        research_depth: "deep",
        evidence_policy: "verified",
      },
      { model_id: "gemini-3.6-flash", thinking_level: "high" },
    );
    const user = await connectApp();
    await user.click(await screen.findByText("Registry"));

    // the composer adopts the session's settings
    await waitFor(() => {
      expect(
        (screen.getByLabelText("Model") as HTMLSelectElement).value,
      ).toBe("gemini-3.6-flash");
      expect(
        (screen.getByLabelText("Thinking") as HTMLSelectElement).value,
      ).toBe("high");
      expect((screen.getByLabelText("Depth") as HTMLSelectElement).value).toBe(
        "deep",
      );
      expect(
        (screen.getByLabelText("Evidence") as HTMLSelectElement).value,
      ).toBe("verified");
    });
  });

  it("resets an inherited model that is no longer selectable", async () => {
    seedSession({ id: "s1", title: "Stale", run_count: 1 });
    // a run created with a model that has since been retired
    seedRun({
      id: "r1",
      session_id: "s1",
      question: "Old question",
      model_id: "gemini-4-argon",
      thinking_level: "high",
    });
    const user = await connectApp();
    await user.click(await screen.findByText("Stale"));

    // the composer sanitizes to the configured default (never submits
    // an unselectable model); the inherited thinking level survives
    // because "high" is still valid for the default model
    await waitFor(() => {
      expect(
        (screen.getByLabelText("Model") as HTMLSelectElement).value,
      ).toBe("");
    });
    expect(
      (screen.getByLabelText("Thinking") as HTMLSelectElement).value,
    ).toBe("high");
  });

  it("disables the settings while a run is being submitted", async () => {
    const user = await connectApp();
    backend.delayNextCreateMs = 60;
    await user.type(screen.getByLabelText("Research question"), "Busy work");
    await user.click(screen.getByRole("button", { name: "Start research" }));

    const depth = (await screen.findByLabelText("Depth")) as HTMLSelectElement;
    await waitFor(() => expect(depth.disabled).toBe(true));
    await waitFor(() => expect(depth.disabled).toBe(false));
  });

  it("shows friendly copy when the backend rejects the selection", async () => {
    const user = await connectApp();
    backend.rejectNextCreate = {
      status: 400,
      category: "model_not_selectable",
      message: "'gemini-4-argon' is not selectable (free tier)",
    };
    await user.type(screen.getByLabelText("Research question"), "Bad pick");
    await user.click(screen.getByRole("button", { name: "Start research" }));

    const notice = await screen.findByRole("alert");
    expect(notice.textContent).toContain(
      "That model or thinking selection is no longer valid.",
    );
    expect(notice.textContent).not.toContain("gemini-4-argon' is not");
    // no run was created and no raw exception text leaked
    expect(Object.keys(backend.runs)).toHaveLength(0);
  });
});

describe("session rename (WS-F)", () => {
  it("renames inline with the PATCH contract and updates the list", async () => {
    seedSession({ id: "s1", title: "Original title" });
    const user = await connectApp();
    await user.click(
      await screen.findByRole("button", { name: "Rename session: Original title" }),
    );
    const input = await screen.findByLabelText(
      "New title for session: Original title",
    );
    await user.clear(input);
    await user.type(input, "Renamed research{Enter}");

    await waitFor(() => {
      const patch = backend.requests.find(
        (entry) => entry.method === "PATCH" && entry.path === "/api/sessions/s1",
      );
      expect(patch).toBeTruthy();
      expect(patch?.body).toEqual({ title: "Renamed research" });
    });
    expect(
      await screen.findByText("Renamed research"),
    ).toBeInTheDocument();
    // renaming does not reorder or reload anything else
    expect(backend.sessions[0].title).toBe("Renamed research");
  });

  it("cancels with Escape and blocks empty titles", async () => {
    seedSession({ id: "s1", title: "Careful" });
    const user = await connectApp();
    const trigger = await screen.findByRole("button", {
      name: "Rename session: Careful",
    });
    await user.click(trigger);
    const input = await screen.findByLabelText("New title for session: Careful");
    const save = screen.getByRole("button", { name: "Save title: Careful" });
    // the input starts prefilled with the current title
    expect((save as HTMLButtonElement).disabled).toBe(false);
    // an empty title can never be saved
    await user.clear(input);
    expect((save as HTMLButtonElement).disabled).toBe(true);
    await user.type(input, "x");
    expect((save as HTMLButtonElement).disabled).toBe(false);
    await user.keyboard("{Escape}");
    expect(
      screen.queryByLabelText("New title for session: Careful"),
    ).not.toBeInTheDocument();
    expect(backend.sessions[0].title).toBe("Careful");
  });
});

describe("diagnostics run configuration (WS-G)", () => {
  it("shows the frozen model/thinking/depth/evidence of a run", async () => {
    seedSession({ id: "s1", title: "Configured", run_count: 1 });
    seedRun(
      {
        id: "r1",
        session_id: "s1",
        question: "Question",
        model_id: "gemini-2.5-flash",
        thinking_level: "low",
        research_depth: "deep",
        evidence_policy: "verified",
      },
      {
        model_id: "gemini-2.5-flash",
        thinking_level: "low",
        research_depth: "deep",
        evidence_policy: "verified",
      },
    );
    const user = await connectApp();
    await user.click(await screen.findByText("Configured"));
    await screen.findAllByText("Question");
    await user.click(await screen.findByRole("tab", { name: "Diagnostics" }));

    const panel = screen.getByLabelText("Run diagnostics");
    await waitFor(() => {
      expect(within(panel).getByText("Run configuration")).toBeInTheDocument();
      expect(within(panel).getByText("gemini-2.5-flash")).toBeInTheDocument();
      expect(within(panel).getByText("Deep")).toBeInTheDocument();
      expect(within(panel).getByText("Verified")).toBeInTheDocument();
    });
  });

  it("says Not recorded for legacy runs without a stored config", async () => {
    seedSession({ id: "s1", title: "Legacy", run_count: 1 });
    seedRun(
      { id: "r1", session_id: "s1", question: "Old question" },
      {
        model_id: null,
        thinking_level: null,
        research_depth: undefined,
        evidence_policy: undefined,
      },
    );
    const user = await connectApp();
    await user.click(await screen.findByText("Legacy"));
    await screen.findAllByText("Old question");
    await user.click(await screen.findByRole("tab", { name: "Diagnostics" }));

    const panel = screen.getByLabelText("Run diagnostics");
    await waitFor(() => {
      const rows = within(panel).getAllByText("Not recorded");
      expect(rows.length).toBeGreaterThanOrEqual(2); // model + thinking
    });
    // never a fabricated default for an old run
    expect(within(panel).queryByText("gemini-3.8-flash")).toBeNull();
  });

  it("shows provider-reported thinking tokens and honest absence", async () => {
    // session A: provider reported a thinking breakdown
    seedSession({ id: "s1", title: "With thinking", run_count: 1 });
    seedRun(
      { id: "r1", session_id: "s1", question: "Thinking question" },
      {
        diagnostics: {
          outcome: { status: "COMPLETED", warnings: [] },
          token_usage: {
            input_tokens: 100,
            output_tokens: 50,
            provider_reported_calls: 2,
            estimated_calls: 0,
            unavailable_calls: 0,
            coverage_ratio: 1,
            thinking_tokens: 42,
            thinking_reported_calls: 2,
          },
        },
      },
    );
    // session B: provider reported no thinking breakdown
    seedSession({ id: "s2", title: "No thinking", run_count: 1 });
    seedRun(
      { id: "r2", session_id: "s2", question: "Silent question" },
      {
        diagnostics: {
          outcome: { status: "COMPLETED", warnings: [] },
          token_usage: {
            input_tokens: 100,
            output_tokens: 50,
            provider_reported_calls: 2,
            estimated_calls: 0,
            unavailable_calls: 0,
            coverage_ratio: 1,
            thinking_tokens: null,
            thinking_reported_calls: 0,
          },
        },
      },
    );
    const user = await connectApp();
    await user.click(await screen.findByText("With thinking"));
    await screen.findAllByText("Thinking question");
    await user.click(await screen.findByRole("tab", { name: "Diagnostics" }));

    const panel = screen.getByLabelText("Run diagnostics");
    expect(await within(panel).findByText("42")).toBeInTheDocument();
    // never fabricated: sessions without a breakdown say so
    await user.click(await screen.findByText("No thinking"));
    await screen.findAllByText("Silent question");
    const secondPanel = screen.getByLabelText("Run diagnostics");
    expect(
      await within(secondPanel).findByText("Not reported"),
    ).toBeInTheDocument();
  });
});
