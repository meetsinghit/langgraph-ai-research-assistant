import { test, expect } from "@playwright/test";
import {
  assertUiBuilt,
  connect,
  newDataDir,
  startServer,
  submitQuestion,
  type ServerHandle,
} from "../helpers";

let server: ServerHandle;

test.beforeAll(async () => {
  assertUiBuilt();
  server = await startServer(newDataDir());
});

test.afterAll(async () => {
  await server?.stop();
});

test("full research run: progress, answer, code block, outcome", async ({
  page,
}) => {
  await connect(page);
  await submitQuestion(page, "What is the capital of France?");

  // live progress is visible while the run executes
  await expect(page.getByLabel("Research progress")).toBeVisible();
  await expect(page.getByRole("button", { name: "Stop research" })).toBeVisible();

  // answer arrives with markdown formatting intact
  await expect(page.getByText("Paris is the capital of France")).toBeVisible({
    timeout: 40_000,
  });
  await expect(page.locator("pre code").first()).toContainText("hello research");
  // clean outcome
  await expect(page.getByText("Research completed").first()).toBeVisible();
});

test("citations link to sources and show verification badges", async ({
  page,
}) => {
  await connect(page);
  await submitQuestion(page, "What is the capital of France?");
  await expect(page.getByText("Paris is the capital of France")).toBeVisible({
    timeout: 40_000,
  });

  // [S1] citation chip in the answer (rendered as an internal button)
  const citation = page.getByRole("button", { name: "Show source S1" });
  await expect(citation).toBeVisible();
  await citation.click();

  // sources panel: S1 highlighted, verified/discovered differentiation
  const sources = page.getByRole("list", { name: "Cited sources" });
  const retrieved = sources.locator("li").filter({ hasText: "example.com/paris" });
  await expect(retrieved).toBeVisible();
  await expect(retrieved).toHaveClass(/ring-accent/); // click-to-highlight
  await expect(sources.getByText("Verified").first()).toBeVisible();
  const snippet = sources.locator("li").filter({ hasText: "example.org/insee" });
  await expect(snippet.getByText("Discovered")).toBeVisible();
});

test("diagnostics panel reports duration, tokens and model calls", async ({
  page,
}) => {
  await connect(page);
  await submitQuestion(page, "What is the capital of France?");
  await expect(page.getByText("Research completed").first()).toBeVisible({
    timeout: 40_000,
  });

  await page.getByRole("tab", { name: "Diagnostics" }).click();
  const diag = page.getByLabel("Run diagnostics");
  await expect(diag).toBeVisible();
  await expect(diag.getByText("Duration")).toBeVisible();
  await expect(diag.getByText("Model calls")).toBeVisible();
  await expect(diag.getByText("Input tokens")).toBeVisible();
  // the fake model reports real (synthesized) usage - not a fabricated 0
  const values = await diag.locator("dd").allTextContents();
  expect(values.some((v) => /[1-9]\d*/.test(v))).toBeTruthy();
});

test("follow-up questions stay in the same session", async ({ page }) => {
  await connect(page);
  await submitQuestion(page, "What is the capital of France?");
  await expect(page.getByText("Research completed").first()).toBeVisible({
    timeout: 40_000,
  });
  await submitQuestion(page, "And what is its population?");
  // both runs landed in the SAME session: its header says so, and
  // exactly one session in the sidebar has two runs (earlier tests in
  // this file own the others)
  await expect(page.getByText(/2 runs? in this session/)).toBeVisible({
    timeout: 40_000,
  });
  const sessions = page.getByRole("navigation", { name: "Research sessions" });
  await expect(sessions.getByText(/2 runs/)).toHaveCount(1);
});

test("model/thinking/depth/evidence selections are frozen into the run", async ({
  page,
}) => {
  await connect(page);
  // capability-aware catalogue: wait until the registry-backed model
  // list has loaded before picking (discovery is off server-side)
  const modelSelect = page.getByLabel("Model");
  await expect(
    modelSelect.locator('option[value="gemini-2.5-flash"]'),
  ).toHaveCount(1, { timeout: 15_000 });
  await modelSelect.selectOption("gemini-2.5-flash");
  // 2.5-flash is a budget-family model: Minimal becomes available
  const thinkingSelect = page.getByLabel("Thinking");
  await expect(
    thinkingSelect.locator('option[value="minimal"]'),
  ).toHaveCount(1,
  );
  await thinkingSelect.selectOption("minimal");
  await page.getByLabel("Depth").selectOption("deep");
  await page.getByLabel("Evidence").selectOption("verified");

  await submitQuestion(page, "Settings freeze: what is the capital of France?");
  await expect(page.getByText("Paris is the capital of France")).toBeVisible({
    timeout: 40_000,
  });

  // the diagnostics panel shows exactly what the run used
  await page.getByRole("tab", { name: "Diagnostics" }).click();
  const diag = page.getByLabel("Run diagnostics");
  await expect(diag.getByText("Run configuration")).toBeVisible();
  await expect(diag.getByText("gemini-2.5-flash")).toBeVisible();
  await expect(diag.getByText("Minimal", { exact: true })).toBeVisible();
  await expect(diag.getByText("Deep", { exact: true })).toBeVisible();
  await expect(diag.getByText("Verified", { exact: true })).toBeVisible();
  // thinking tokens: honest provider-reported value or "Not reported"
  await expect(diag.getByText("Thinking tokens")).toBeVisible();
});
