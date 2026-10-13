import { defineConfig } from "@playwright/test";

/**
 * Real-browser E2E (Phase 4D.3). Drives the REAL production path:
 * the FastAPI server (real auth, real SSE, real SQLite) serving the
 * built UI from frontend/dist, in the system Microsoft Edge.
 *
 * Prerequisites:
 *   1. `npm run build` in frontend/   (creates frontend/dist)
 *   2. `npm install` in frontend/e2e  (this package)
 *
 * Servers are managed per spec file by helpers.ts (not webServer),
 * so tests can stop/restart the backend to exercise recovery.
 */
export default defineConfig({
  testDir: "./specs",
  workers: 1, // one shared local port; deterministic sequencing
  fullyParallel: false,
  timeout: 90_000,
  expect: { timeout: 12_000 },
  hookTimeout: 90_000,
  reporter: [["list"]],
  use: {
    baseURL: "http://127.0.0.1:8123",
    channel: "msedge",
    viewport: { width: 1280, height: 800 },
    trace: "retain-on-failure",
  },
});
