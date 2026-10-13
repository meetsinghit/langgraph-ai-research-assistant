import { test, expect } from "@playwright/test";
import {
  assertUiBuilt,
  connect,
  newDataDir,
  startServer,
  startServerWithRetry,
  submitQuestion,
  type ServerHandle,
} from "../helpers";

// Every test owns its own server + data dir: recovery/restart tests
// must be deterministic and must not depend on cross-test state.

test.beforeAll(() => assertUiBuilt());

test("a running research can be cancelled", async ({ page }) => {
  const server = await startServer(newDataDir());
  try {
    await connect(page);
    await submitQuestion(page, "SLOWCANCEL question about long research");
    await expect(
      page.getByRole("button", { name: "Stop research" }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Stop research" }).click();
    await expect(page.getByText("Research cancelled").first()).toBeVisible({
      timeout: 25_000,
    });
  } finally {
    await server.stop();
  }
});

test("crash + restart: interrupted run is recovered and resumable", async ({
  page,
}) => {
  const dataDir = newDataDir();
  let server = await startServer(dataDir);
  try {
    await connect(page);
    await submitQuestion(page, "CRASHDEATH question about caches");
    // the scripted model "dies" mid-run; the backend stays up but the
    // run is left non-terminal
    await page.waitForTimeout(2_000);

    // hard-kill the backend: live progress is lost, honestly reported
    // (the stream client exhausts its bounded reconnect budget first)
    await server.stop({ graceful: false });
    await expect(page.getByText(/Live progress was lost/)).toBeVisible({
      timeout: 40_000,
    });

    // restart on the SAME data dir -> recovery marks the run interrupted
    server = await startServerWithRetry(dataDir);
    await page.reload();
    await connect(page);

    const sessions = page.getByRole("navigation", {
      name: "Research sessions",
    });
    await expect(sessions.locator("li")).toHaveCount(1);
    await sessions.getByRole("button").first().click();

    // resume affordance with its repetition caveat
    await expect(page.getByText(/Research interrupted/).first()).toBeVisible();
    await expect(page.getByText(/may repeat/)).toBeVisible();
    await page.getByRole("button", { name: "Resume research" }).click();

    // the resumed run finishes and stores its answer. (The synthetic
    // keyword question trips the deterministic topic-coverage gate,
    // so the honest outcome here is a completed run flagged
    // incomplete -- recovery + resume are what this test proves.)
    await expect(page.getByText("Paris is the capital of France")).toBeVisible({
      timeout: 40_000,
    });
    await expect(
      page.getByText(/quality review found it incomplete/),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Resume research" }),
    ).toHaveCount(0);
  } finally {
    await server.stop();
  }
});

test("backend down: clear error state, then a working reconnect", async ({
  page,
}) => {
  const dataDir = newDataDir();
  let server = await startServer(dataDir);
  try {
    await connect(page);
    await submitQuestion(page, "What is the capital of France?");
    await expect(page.getByText("Research completed").first()).toBeVisible({
      timeout: 40_000,
    });

    // go to the gate first (deterministic), THEN take the backend down
    await page.getByRole("button", { name: "Disconnect" }).click();
    await expect(
      page.getByRole("heading", { name: "Connect to your research service" }),
    ).toBeVisible();
    await server.stop({ graceful: false });

    // realistic user path: click the (auto-focused) input, type, Enter
    const input = page.getByPlaceholder("Paste your local APP_TOKEN");
    await input.click();
    await input.pressSequentially("e2e-token-not-a-real-credential");
    await page.keyboard.press("Enter");
    await expect(
      page.getByText(/The research service is not reachable/),
    ).toBeVisible();

    server = await startServerWithRetry(dataDir);
    const input2 = page.getByPlaceholder("Paste your local APP_TOKEN");
    await input2.click();
    await input2.pressSequentially("e2e-token-not-a-real-credential");
    await page.keyboard.press("Enter");
    await expect(page.getByPlaceholder(/Ask a research question/)).toBeVisible({
      timeout: 20_000,
    });
    // history survived the outage
    const sessions = page.getByRole("navigation", {
      name: "Research sessions",
    });
    await expect(sessions.locator("li")).toHaveCount(1);
    await expect(sessions.getByText(/capital of France/)).toBeVisible();
  } finally {
    await server.stop();
  }
});

test("unsafe markdown links and raw HTML are neutralized", async ({ page }) => {
  const server = await startServer(newDataDir());
  try {
    await connect(page);
    await submitQuestion(page, "UNSAFELINK question for rendering safety");
    await expect(page.getByText(/Click/).first()).toBeVisible({
      timeout: 40_000,
    });
    // no javascript:/data: anchors, no injected elements
    await expect(page.locator('a[href^="javascript:"]')).toHaveCount(0);
    await expect(page.locator('a[href^="data:"]')).toHaveCount(0);
    await expect(page.locator("img[onerror]")).toHaveCount(0);
    // raw HTML is displayed as text, not markup
    await expect(page.locator("b", { hasText: "bold?" })).toHaveCount(0);
    await expect(page.getByText(/bold\?/)).toBeVisible();
  } finally {
    await server.stop();
  }
});
