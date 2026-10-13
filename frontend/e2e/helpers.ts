import { spawn, type ChildProcess } from "node:child_process";
import { existsSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { expect, type Page } from "@playwright/test";

const ROOT = resolve(__dirname, "..", "..");
const PYTHON = join(ROOT, "ai_poc", "Scripts", "python.exe");
const SERVER = join(ROOT, "e2e", "server.py");
const UI_INDEX = join(ROOT, "frontend", "dist", "index.html");

export const TOKEN = "e2e-token-not-a-real-credential";
export const PORT = 8123;
export const BASE_URL = `http://127.0.0.1:${PORT}`;

export interface ServerHandle {
  dataDir: string;
  stop(opts?: { graceful?: boolean }): Promise<void>;
}

export function assertUiBuilt(): void {
  if (!existsSync(UI_INDEX)) {
    throw new Error(
      "frontend/dist is missing - run `npm run build` in frontend/ " +
        "before the E2E suite.",
    );
  }
}

export function newDataDir(): string {
  return mkdtempSync(join(tmpdir(), "aipoc-e2e-"));
}

async function waitForHealth(proc: ChildProcess, stderr: () => string) {
  const deadline = Date.now() + 45_000;
  for (;;) {
    if (proc.exitCode !== null) {
      throw new Error(
        `E2E server exited early (code ${proc.exitCode}):\n${stderr()}`,
      );
    }
    try {
      const res = await fetch(`${BASE_URL}/api/health`);
      if (res.status === 200) return;
    } catch {
      /* not listening yet */
    }
    if (Date.now() > deadline) {
      proc.kill();
      throw new Error(`E2E server not healthy in 45s:\n${stderr()}`);
    }
    await new Promise((r) => setTimeout(r, 250));
  }
}

export async function startServer(dataDir: string): Promise<ServerHandle> {
  const proc = spawn(
    PYTHON,
    [SERVER, "--port", String(PORT), "--data-dir", dataDir, "--token", TOKEN],
    { cwd: ROOT, stdio: ["ignore", "pipe", "pipe"] },
  );
  let stderrBuf = "";
  proc.stderr?.on("data", (chunk) => {
    stderrBuf += String(chunk);
  });
  const exited = new Promise<void>((res) => proc.once("exit", () => res()));

  await waitForHealth(proc, () => stderrBuf);

  return {
    dataDir,
    async stop({ graceful = true } = {}) {
      if (proc.exitCode !== null) return;
      if (graceful) {
        try {
          await fetch(`${BASE_URL}/__e2e__/shutdown`, {
            method: "POST",
            headers: { Authorization: `Bearer ${TOKEN}` },
          });
          const stopped = await Promise.race([
            exited.then(() => true),
            new Promise((r) => setTimeout(() => r(false), 4_000)),
          ]);
          if (stopped) return;
        } catch {
          /* fall through to kill */
        }
      }
      proc.kill();
      await exited;
    },
  };
}

/** Start, retrying once after a short pause (port release after stop). */
export async function startServerWithRetry(
  dataDir: string,
): Promise<ServerHandle> {
  try {
    return await startServer(dataDir);
  } catch {
    await new Promise((r) => setTimeout(r, 1_500));
    return startServer(dataDir);
  }
}

export async function connect(page: Page, token = TOKEN): Promise<void> {
  await page.goto("/");
  await page.getByPlaceholder("Paste your local APP_TOKEN").fill(token);
  await page.getByRole("button", { name: "Connect", exact: true }).click();
  await expect(page.getByPlaceholder(/Ask a research question/)).toBeVisible({
    timeout: 20_000,
  });
}

export async function submitQuestion(
  page: Page,
  question: string,
): Promise<void> {
  const input = page.getByPlaceholder(/Ask a research question/);
  await input.fill(question);
  await input.press("Enter");
}

export const composer = (page: Page) =>
  page.getByPlaceholder(/Ask a research question/);
