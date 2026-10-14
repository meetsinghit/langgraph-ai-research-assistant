import { test, expect } from "@playwright/test";
import {
  assertUiBuilt,
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

test("saved history reopens after a reload without re-running", async ({
  page,
}) => {
  await connect(page);
  await submitQuestion(page, "What is the capital of France?");
  await expect(page.getByText("Research completed").first()).toBeVisible({
    timeout: 40_000,
  });

  // fresh page load: the memory-only token means reconnecting, as the
  // UI explains on the gate
  const posts: string[] = [];
  page.on("request", (req) => {
    if (req.method() === "POST" && req.url().includes("/runs")) {
      posts.push(req.url());
    }
  });
  await page.reload();
  await connect(page);

  // the saved session is listed and opens with its stored answer
  const sessions = page.getByRole("navigation", { name: "Research sessions" });
  await expect(sessions.locator("li")).toHaveCount(1);
  await sessions.getByRole("button").first().click();
  await expect(page.getByText("Paris is the capital of France")).toBeVisible();
  // ...and nothing was re-run to show it
  expect(posts).toHaveLength(0);
});

test("a new session is created from the sidebar button", async ({ page }) => {
  await connect(page);
  await submitQuestion(page, "Session seed one: capital of France?");
  await expect(page.getByText("Research completed").first()).toBeVisible({
    timeout: 40_000,
  });

  await page
    .getByRole("button", { name: "Start a new research session" })
    .click();
  await submitQuestion(page, "What is the tallest mountain?");
  // the conversation now focuses the NEW session (its own run), and
  // the sidebar lists both of this test's sessions (title counts stay
  // robust no matter what earlier tests in this file created)
  const sessions = page.getByRole("navigation", { name: "Research sessions" });
  await expect(sessions.getByText(/Session seed one/)).toHaveCount(1);
  await expect(sessions.getByText("What is the tallest mountain?")).toHaveCount(
    1,
  );
});

test("sessions can be renamed inline and the title persists", async ({
  page,
}) => {
  await connect(page);
  await submitQuestion(page, "Rename me: what is the capital of France?");
  await expect(page.getByText("Research completed").first()).toBeVisible({
    timeout: 40_000,
  });

  const sessions = page.getByRole("navigation", { name: "Research sessions" });
  const target = sessions
    .locator("li")
    .filter({ hasText: "Rename me:" })
    .first();
  await target
    .getByRole("button", { name: /^Rename session:/ })
    .click();

  // inline edit: prefilled with the current title, saved with Enter
  const input = target.locator("input[type='text']");
  await expect(input).toBeVisible();
  await input.fill("Renamed in E2E");
  await input.press("Enter");
  await expect(sessions.getByText("Renamed in E2E")).toBeVisible();

  // the rename round-trips through the authenticated PATCH and is
  // persisted in SQLite: a reload still shows the new title
  await page.reload();
  await connect(page);
  await expect(
    page.getByRole("navigation", { name: "Research sessions" }).getByText("Renamed in E2E"),
  ).toBeVisible();
});

test("deleting a session asks for confirmation", async ({ page }) => {
  await connect(page);
  await submitQuestion(page, "Delete me: what is the capital of France?");
  await expect(page.getByText("Research completed").first()).toBeVisible({
    timeout: 40_000,
  });

  // scope to THIS test's session (earlier tests own the others)
  const sessions = page.getByRole("navigation", { name: "Research sessions" });
  const target = sessions
    .locator("li")
    .filter({ hasText: "Delete me:" })
    .first();
  const deleteButton = () =>
    target.getByRole("button", { name: /^Delete session:/ });

  await deleteButton().click();
  await expect(page.getByText("Delete this research session?")).toBeVisible();
  // cancel keeps it
  await page.getByRole("button", { name: "Keep it" }).click();
  await expect(target).toBeVisible();

  // confirm removes it
  await deleteButton().click();
  await page
    .getByRole("button", { name: "Delete session", exact: true })
    .click();
  await expect(target).toHaveCount(0);
});
