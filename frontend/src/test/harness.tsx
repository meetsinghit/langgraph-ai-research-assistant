/** Shared test harness: renders the real <App/> against the MSW
 *  mock backend and connects with the in-memory token. */
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { App } from "../App";
import { TOKEN } from "./mockBackend";

export async function connectApp(user = userEvent.setup()) {
  render(<App />);
  const input = await screen.findByLabelText("Access token");
  await user.type(input, TOKEN);
  await user.click(screen.getByRole("button", { name: "Connect" }));
  await screen.findByText("What would you like to research?");
  return user;
}

export function renderConnectedApp() {
  const utils = render(<App />);
  return { user: userEvent.setup(), ...utils };
}

export function event(
  seq: number,
  type: string,
  payload: Record<string, unknown> = {},
): Record<string, unknown> {
  return { seq, type, ...payload };
}

/** A full scripted happy-path run: plan → research (search+fetch) →
 *  synthesize → review → finish. */
export function happyPathEvents(): Array<Record<string, unknown>> {
  return [
    event(1, "status.changed", { status: "running" }),
    event(2, "run.started"),
    event(3, "stage.started", { node: "plan" }),
    event(4, "stage.finished", { node: "plan" }),
    event(5, "stage.started", { node: "research" }),
    event(6, "tool.started", { tool: "web_search" }),
    event(7, "tool.finished", { tool: "web_search", ok: true, latency_ms: 120 }),
    event(8, "source.discovered", {
      url: "https://example.com/doc",
      title: "Doc",
    }),
    event(9, "tool.started", { tool: "fetch_page" }),
    event(10, "tool.finished", { tool: "fetch_page", ok: true, latency_ms: 300 }),
    event(11, "source.retrieved", { url: "https://example.com/doc" }),
    event(12, "stage.finished", { node: "research" }),
    event(13, "stage.started", { node: "synthesize" }),
    event(14, "stage.finished", { node: "synthesize" }),
    event(15, "stage.started", { node: "review" }),
    event(16, "stage.finished", { node: "review" }),
    event(17, "run.finished", { outcome: "COMPLETED", review_status: "approved" }),
    event(18, "status.changed", { status: "completed" }),
  ];
}
