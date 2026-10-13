import { test, expect } from "@playwright/test";
import {
  assertUiBuilt,
  composer,
  connect,
  newDataDir,
  startServer,
  submitQuestion,
  TOKEN,
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

test("launch shows the token screen and explains where the token comes from", async ({
  page,
}) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Connect to your research service" }),
  ).toBeVisible();
  // explains the bootstrap location without revealing the token
  await expect(page.getByText("data/app_token.txt")).toBeVisible();
  await expect(page.getByPlaceholder("Paste your local APP_TOKEN")).toBeVisible();
  await expect(page.getByRole("button", { name: "Connect", exact: true })).toBeVisible();
});

test("a wrong token is rejected with a friendly message", async ({ page }) => {
  await page.goto("/");
  await page.getByPlaceholder("Paste your local APP_TOKEN").fill("definitely-wrong");
  await page.getByRole("button", { name: "Connect", exact: true }).click();
  await expect(
    page.getByText("The access token was rejected. Reconnect to continue."),
  ).toBeVisible();
  // still on the gate, able to retry
  await expect(
    page.getByRole("heading", { name: "Connect to your research service" }),
  ).toBeVisible();
});

test("the correct token connects and shows the workspace", async ({ page }) => {
  await connect(page);
  await expect(composer(page)).toBeVisible();
  await expect(page.getByText("Connected")).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Start a new research session" }),
  ).toBeVisible();
});

test("keyboard: Shift+Enter adds a line, Enter submits", async ({ page }) => {
  await connect(page);
  const input = composer(page);
  await input.click();
  await page.keyboard.type("First line of the question");
  await page.keyboard.press("Shift+Enter");
  await page.keyboard.type("Second line");
  await page.keyboard.press("Enter");
  // the run card quotes the multi-line question
  await expect(page.getByText("First line of the question").first()).toBeVisible();
  await expect(page.getByText("Second line").first()).toBeVisible();
  // and it runs to an answer
  await expect(page.getByText("Paris is the capital of France")).toBeVisible({
    timeout: 40_000,
  });
});
