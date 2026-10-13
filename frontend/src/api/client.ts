/**
 * Authenticated fetch client for the local FastAPI backend.
 *
 * - injects the Bearer token (memory-only, passed per call site via a
 *   token provider — never stored here, never logged);
 * - parses the backend's fixed `{error:{category,message}}` contract
 *   into typed errors so the UI shows friendly text, never raw
 *   exceptions;
 * - distinguishes 401 (need to reconnect) from unreachable backend.
 */
import type {
  CancelOut,
  ConfigOut,
  HealthOut,
  ResumeOut,
  RunAcceptedOut,
  RunDetailOut,
  SessionDetailOut,
  SessionOut,
} from "./types";

export type ApiErrorCategory =
  | "unauthorized"
  | "not_found"
  | "conflict"
  | "validation"
  | "backend_unreachable"
  | "http"
  | "network";

export class ApiError extends Error {
  readonly category: ApiErrorCategory;
  readonly status: number;

  constructor(category: ApiErrorCategory, message: string, status = 0) {
    super(message);
    this.name = "ApiError";
    this.category = category;
    this.status = status;
  }
}

export type TokenProvider = () => string | null;

export interface ClientOptions {
  tokenProvider: TokenProvider;
  fetchImpl?: typeof fetch;
}

export class ApiClient {
  private readonly tokenProvider: TokenProvider;
  private readonly fetchImpl: typeof fetch;

  constructor(options: ClientOptions) {
    this.tokenProvider = options.tokenProvider;
    this.fetchImpl =
      options.fetchImpl ?? ((...args) => fetch(...args));
  }

  async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const token = this.tokenProvider();
    const headers = new Headers(init.headers);
    if (token) {
      headers.set("Authorization", `Bearer ${token}`);
    }
    let response: Response;
    try {
      response = await this.fetchImpl(path, { ...init, headers });
    } catch {
      // fetch rejects on connection failure (backend not running,
      // proxy target down) — never surface the raw error object.
      throw new ApiError(
        "backend_unreachable",
        "The research service is not reachable. Make sure it is running locally, then retry.",
      );
    }
    if (response.status === 401) {
      throw new ApiError(
        "unauthorized",
        "The access token was rejected. Reconnect to continue.",
        401,
      );
    }
    if (response.status === 204) {
      return undefined as T;
    }
    const text = await response.text();
    let payload: unknown = null;
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch {
        payload = null;
      }
    }
    if (!response.ok) {
      const contract = (payload as { error?: { category?: string; message?: string } } | null)
        ?.error;
      const category = mapErrorCategory(contract?.category, response.status);
      throw new ApiError(
        category,
        contract?.message ?? "The request could not be completed.",
        response.status,
      );
    }
    return payload as T;
  }

  // ---- endpoints ---------------------------------------------------------

  health(): Promise<HealthOut> {
    return this.request<HealthOut>("/api/health");
  }

  config(): Promise<ConfigOut> {
    return this.request<ConfigOut>("/api/config");
  }

  listSessions(): Promise<SessionOut[]> {
    return this.request<SessionOut[]>("/api/sessions");
  }

  createSession(title: string): Promise<SessionOut> {
    return this.request<SessionOut>("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    });
  }

  getSession(sessionId: string): Promise<SessionDetailOut> {
    return this.request<SessionDetailOut>(
      `/api/sessions/${encodeURIComponent(sessionId)}`,
    );
  }

  deleteSession(sessionId: string): Promise<void> {
    return this.request<void>(
      `/api/sessions/${encodeURIComponent(sessionId)}`,
      { method: "DELETE" },
    );
  }

  createRun(
    sessionId: string,
    question: string,
    idempotencyKey?: string,
  ): Promise<RunAcceptedOut> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    };
    if (idempotencyKey) {
      headers["Idempotency-Key"] = idempotencyKey;
    }
    return this.request<RunAcceptedOut>(
      `/api/sessions/${encodeURIComponent(sessionId)}/runs`,
      {
        method: "POST",
        headers,
        body: JSON.stringify({ question }),
      },
    );
  }

  getRun(runId: string): Promise<RunDetailOut> {
    return this.request<RunDetailOut>(
      `/api/runs/${encodeURIComponent(runId)}`,
    );
  }

  cancelRun(runId: string): Promise<CancelOut> {
    return this.request<CancelOut>(
      `/api/runs/${encodeURIComponent(runId)}/cancel`,
      { method: "POST" },
    );
  }

  resumeRun(runId: string): Promise<ResumeOut> {
    return this.request<ResumeOut>(
      `/api/runs/${encodeURIComponent(runId)}/resume`,
      { method: "POST" },
    );
  }

  eventsPath(runId: string): string {
    return `/api/runs/${encodeURIComponent(runId)}/events`;
  }
}

function mapErrorCategory(
  contractCategory: string | undefined,
  status: number,
): ApiErrorCategory {
  switch (contractCategory) {
    case "not_found":
      return "not_found";
    case "conflict":
      return "conflict";
    case "validation":
      return "validation";
    default:
      break;
  }
  if (status >= 500) return "http";
  return "http";
}
