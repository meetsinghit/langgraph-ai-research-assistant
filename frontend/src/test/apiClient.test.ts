/** ApiClient contract: auth header, error mapping, unreachable backend. */
import { describe, expect, it, vi } from "vitest";
import { ApiClient, ApiError } from "../api/client";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("ApiClient", () => {
  it("injects the Bearer token from the in-memory provider", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse({ ok: true }));
    const client = new ApiClient({
      tokenProvider: () => "secret-token",
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    await client.health();
    const [, init] = fetchImpl.mock.calls[0] as unknown as [string, RequestInit];
    expect(new Headers(init.headers).get("Authorization")).toBe(
      "Bearer secret-token",
    );
  });

  it("omits Authorization when no token is present", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse({ ok: true }));
    const client = new ApiClient({
      tokenProvider: () => null,
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    await client.health();
    const [, init] = fetchImpl.mock.calls[0] as unknown as [string, RequestInit];
    expect(new Headers(init.headers).get("Authorization")).toBeNull();
  });

  it("maps 401 to a typed unauthorized error", async () => {
    const client = new ApiClient({
      tokenProvider: () => "bad",
      fetchImpl: (async () =>
        jsonResponse(
          { error: { category: "unauthorized", message: "invalid token" } },
          401,
        )) as unknown as typeof fetch,
    });
    await expect(client.config()).rejects.toMatchObject({
      category: "unauthorized",
      status: 401,
    });
  });

  it("passes through the backend's fixed error contract", async () => {
    const client = new ApiClient({
      tokenProvider: () => "t",
      fetchImpl: (async () =>
        jsonResponse(
          {
            error: {
              category: "conflict",
              message: "resume unavailable: not_interrupted",
            },
          },
          409,
        )) as unknown as typeof fetch,
    });
    await expect(client.resumeRun("r1")).rejects.toMatchObject({
      category: "conflict",
      message: "resume unavailable: not_interrupted",
    });
  });

  it("reports connection failure as backend_unreachable (no raw error)", async () => {
    const client = new ApiClient({
      tokenProvider: () => "t",
      fetchImpl: (async () => {
        throw new TypeError("fetch failed");
      }) as unknown as typeof fetch,
    });
    const error = await client.listSessions().catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).category).toBe("backend_unreachable");
    expect((error as ApiError).message).toMatch(/not reachable/i);
    expect((error as ApiError).message).not.toContain("fetch failed");
  });

  it("sends the run create request with the question contract", async () => {
    const fetchImpl = vi.fn(async () =>
      jsonResponse(
        { run_id: "r1", status: "queued", events_url: "/api/runs/r1/events" },
        202,
      ),
    );
    const client = new ApiClient({
      tokenProvider: () => "t",
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    const out = await client.createRun("s1", "my question", "idem-1");
    expect(out.run_id).toBe("r1");
    const [url, init] = fetchImpl.mock.calls[0] as unknown as [
      string,
      RequestInit,
    ];
    expect(url).toBe("/api/sessions/s1/runs");
    expect(new Headers(init.headers).get("Idempotency-Key")).toBe("idem-1");
    expect(JSON.parse(String(init.body))).toEqual({ question: "my question" });
  });

  it("returns undefined on 204 responses", async () => {
    const client = new ApiClient({
      tokenProvider: () => "t",
      fetchImpl: (async () => new Response(null, { status: 204 })) as unknown as typeof fetch,
    });
    await expect(client.deleteSession("s1")).resolves.toBeUndefined();
  });
});
