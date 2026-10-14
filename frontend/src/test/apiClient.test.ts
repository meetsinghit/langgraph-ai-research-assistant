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
    const out = await client.createRun("s1", "my question", {
      idempotencyKey: "idem-1",
    });
    expect(out.run_id).toBe("r1");
    const [url, init] = fetchImpl.mock.calls[0] as unknown as [
      string,
      RequestInit,
    ];
    expect(url).toBe("/api/sessions/s1/runs");
    expect(new Headers(init.headers).get("Idempotency-Key")).toBe("idem-1");
    expect(JSON.parse(String(init.body))).toEqual({ question: "my question" });
  });

  it("sends frozen per-run settings when provided (4E.3)", async () => {
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
    await client.createRun("s1", "my question", {
      idempotencyKey: "idem-2",
      settings: {
        research_depth: "deep",
        evidence_policy: "verified",
        model_id: "gemini-3.6-flash",
        thinking_level: "high",
      },
    });
    const [, init] = fetchImpl.mock.calls[0] as unknown as [string, RequestInit];
    expect(JSON.parse(String(init.body))).toEqual({
      question: "my question",
      research_depth: "deep",
      evidence_policy: "verified",
      model_id: "gemini-3.6-flash",
      thinking_level: "high",
    });
    // and a "configured default" selection sends explicit nulls
    await client.createRun("s1", "second question", {
      settings: {
        research_depth: "standard",
        evidence_policy: "balanced",
        model_id: null,
        thinking_level: null,
      },
    });
    const [, secondInit] = fetchImpl.mock.calls[1] as unknown as [
      string,
      RequestInit,
    ];
    expect(JSON.parse(String(secondInit.body))).toEqual({
      question: "second question",
      research_depth: "standard",
      evidence_policy: "balanced",
      model_id: null,
      thinking_level: null,
    });
  });

  it("maps selection and idempotency categories (4E.3)", async () => {
    const cases: Array<[string, string, number, string]> = [
      ["unknown_model", "not a known selectable model", 400, "invalid_selection"],
      ["model_not_selectable", "not selectable", 400, "invalid_selection"],
      [
        "unsupported_thinking_level",
        "not supported",
        400,
        "invalid_selection",
      ],
      ["model_unavailable", "model unavailable", 400, "model_unavailable"],
      ["idempotency_conflict", "already used", 409, "conflict"],
    ];
    for (const [contract, message, status, expected] of cases) {
      const client = new ApiClient({
        tokenProvider: () => "t",
        fetchImpl: (async () =>
          jsonResponse({ error: { category: contract, message } }, status)) as unknown as typeof fetch,
      });
      await expect(client.listSessions()).rejects.toMatchObject({
        category: expected,
        message,
      });
    }
  });

  it("GET /api/models returns the catalogue", async () => {
    const catalog = {
      default_model_id: "gemini-3.5-flash-lite",
      default_thinking_level: "medium",
      discovery: { status: "disabled" },
      models: [
        {
          model_id: "gemini-3.5-flash-lite",
          display_name: "Gemini 3.8 Flash",
          selectable: true,
        },
      ],
      unregistered: [],
    };
    const client = new ApiClient({
      tokenProvider: () => "t",
      fetchImpl: (async () => jsonResponse(catalog, 200)) as unknown as typeof fetch,
    });
    await expect(client.listModels()).resolves.toEqual(catalog);
  });

  it("PATCH /api/sessions renames with the fixed method", async () => {
    const fetchImpl = vi.fn(async () =>
      jsonResponse(
        {
          id: "s1",
          title: "Renamed",
          created_at: "2026-10-14T10:00:00Z",
          updated_at: "2026-10-14T10:00:00Z",
        },
        200,
      ),
    );
    const client = new ApiClient({
      tokenProvider: () => "t",
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    const out = await client.renameSession("s1", "Renamed");
    expect(out.title).toBe("Renamed");
    const [url, init] = fetchImpl.mock.calls[0] as unknown as [
      string,
      RequestInit,
    ];
    expect(url).toBe("/api/sessions/s1");
    expect(init.method).toBe("PATCH");
    expect(JSON.parse(String(init.body))).toEqual({ title: "Renamed" });
  });

  it("returns undefined on 204 responses", async () => {
    const client = new ApiClient({
      tokenProvider: () => "t",
      fetchImpl: (async () => new Response(null, { status: 204 })) as unknown as typeof fetch,
    });
    await expect(client.deleteSession("s1")).resolves.toBeUndefined();
  });
});
