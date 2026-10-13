import "@testing-library/jest-dom/vitest";
import { afterEach, afterAll, beforeAll, vi } from "vitest";
import { cleanup } from "@testing-library/react";
import { server } from "./mockBackend";

// jsdom lacks matchMedia (theme detection) — provide a minimal stub.
if (typeof window.matchMedia !== "function") {
  window.matchMedia = ((query: string) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: () => undefined,
    removeListener: () => undefined,
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
    dispatchEvent: () => false,
  })) as unknown as typeof window.matchMedia;
}

// jsdom lacks scrollIntoView (citation highlight).
Element.prototype.scrollIntoView = Element.prototype.scrollIntoView ?? (() => undefined);

// Clipboard stub for the copy-answer button.
Object.assign(navigator, {
  clipboard: { writeText: vi.fn().mockResolvedValue(undefined) },
});

// jsdom's AbortController produces signals that Node's undici fetch
// rejects (realm mismatch — real browsers are unaffected, and the
// API client never passes signals). We wrap fetch AFTER MSW patches
// it (inside beforeAll below) to strip `signal` so SSE streaming
// works under test; aborts are best-effort there (tests that need
// real abort behavior inject their own fetch stub).
beforeAll(() => {
  server.listen({ onUnhandledRequest: "error" });
  const mswFetch = globalThis.fetch.bind(globalThis);
  globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
    if (init?.signal) {
      const { signal: _ignored, ...rest } = init;
      return mswFetch(input, rest);
    }
    return mswFetch(input, init);
  }) as typeof fetch;
});
afterEach(() => {
  cleanup();
  server.resetHandlers();
});
afterAll(() => server.close());
