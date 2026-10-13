import { test, expect } from "@playwright/test";
import {
  assertUiBuilt,
  connect,
  newDataDir,
  startServer,
  submitQuestion,
  composer,
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

test("narrow layout uses drawers for history and the panel", async ({
  page,
}) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await connect(page);
  await expect(composer(page)).toBeVisible();

  // history becomes a drawer (closed via its backdrop; Escape is not
  // a Drawer affordance in this app)
  await page.getByRole("button", { name: "Open research history" }).click();
  await expect(
    page.getByRole("heading", { name: "Research history" }),
  ).toBeVisible();
  await page.mouse.click(380, 600); // backdrop to the right of the drawer
  await expect(
    page.getByRole("heading", { name: "Research history" }),
  ).toBeHidden();

  // sources/diagnostics become a drawer with real tabs
  await page.getByRole("button", { name: "Open sources and diagnostics" }).click();
  await expect(page.getByRole("tab", { name: "Sources" })).toBeVisible();
  await expect(page.getByRole("tab", { name: "Diagnostics" })).toBeVisible();
  await page.mouse.click(10, 600); // backdrop to the left of the drawer
  await expect(page.getByRole("tab", { name: "Sources" })).toBeHidden();
});

test("theme toggle applies and persists the dark theme", async ({ page }) => {
  await connect(page);
  await page.getByRole("button", { name: "Switch to dark theme" }).click();
  await expect(page.locator("html")).toHaveClass(/dark/);
  await page.reload();
  await expect(page.locator("html")).toHaveClass(/dark/);
});

test("keyboard-only: tab to the composer and submit", async ({ page }) => {
  await connect(page);

  // Tab until the composer has focus (header + sidebar buttons come first)
  let focused = false;
  for (let i = 0; i < 12; i++) {
    await page.keyboard.press("Tab");
    focused = await page.evaluate(() => {
      const el = document.activeElement as HTMLTextAreaElement | null;
      return Boolean(el && el.placeholder?.includes("Ask a research question"));
    });
    if (focused) break;
  }
  expect(focused).toBeTruthy();

  await page.keyboard.type("Typed with the keyboard only");
  await page.keyboard.press("Enter");
  await expect(page.getByText("Paris is the capital of France")).toBeVisible({
    timeout: 40_000,
  });
});
