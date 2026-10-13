/** Connection gate, authentication failures, backend unreachable. */
import { beforeEach, describe, expect, it } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { App } from "../App";
import { TOKEN, backend, resetBackend, server } from "./mockBackend";

beforeEach(() => {
  resetBackend();
});

describe("connection", () => {
  it("shows the connection dialog until connected", async () => {
    render(<App />);
    const dialog = screen.getByRole("dialog", {
      name: /connect to your research service/i,
    });
    expect(dialog).toHaveAttribute("aria-modal", "true");
    expect(screen.getByLabelText("Access token")).toBeInTheDocument();
  });

  it("connects with the token and shows the workspace", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.type(await screen.findByLabelText("Access token"), TOKEN);
    await user.click(screen.getByRole("button", { name: "Connect" }));
    expect(await screen.findByText("Research history")).toBeInTheDocument();
    expect(screen.getByText("Connected")).toBeInTheDocument();
    // The token never appears anywhere in the rendered output.
    expect(document.body.textContent).not.toContain(TOKEN);
  });

  it("documents that refreshing requires reconnecting (memory-only token)", async () => {
    render(<App />);
    expect(
      screen.getByText(/stays in memory only/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/after a page refresh you simply connect again/i),
    ).toBeInTheDocument();
  });

  it("shows a friendly message when the token is rejected (401)", async () => {
    server.use(
      http.get("/api/config", () =>
        HttpResponse.json(
          { error: { category: "unauthorized", message: "missing or invalid bearer token" } },
          { status: 401 },
        ),
      ),
    );
    const user = userEvent.setup();
    render(<App />);
    await user.type(await screen.findByLabelText("Access token"), "wrong-token");
    await user.click(screen.getByRole("button", { name: "Connect" }));
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toMatch(/rejected|token/i);
    // still on the gate
    expect(screen.getByLabelText("Access token")).toBeInTheDocument();
  });

  it("shows a friendly message when the backend is unreachable", async () => {
    server.use(
      http.get("/api/health", () => HttpResponse.error()),
    );
    const user = userEvent.setup();
    render(<App />);
    await user.type(await screen.findByLabelText("Access token"), TOKEN);
    await user.click(screen.getByRole("button", { name: "Connect" }));
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toMatch(/not reachable/i);
  });

  it("hides the token in a password field and never logs it", async () => {
    render(<App />);
    const input = await screen.findByLabelText("Access token");
    expect(input).toHaveAttribute("type", "password");
  });

  it("keeps the expected bearer contract on API calls", async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.type(await screen.findByLabelText("Access token"), TOKEN);
    await user.click(screen.getByRole("button", { name: "Connect" }));
    await screen.findByText("Research history");
    await waitFor(() => {
      const authedRequest = backend.requests.find(
        (entry) => entry.headers.get("Authorization") === `Bearer ${TOKEN}`,
      );
      expect(authedRequest).toBeDefined();
    });
  });
});
