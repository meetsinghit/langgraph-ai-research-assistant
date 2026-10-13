/**
 * In-memory authentication for the local backend.
 *
 * The APP_TOKEN lives ONLY in this context (a React ref mirrored to
 * state) — never in localStorage/sessionStorage, URLs, cookies or
 * any external service. Consequence (documented in the UI and the
 * README): refreshing the page requires reconnecting.
 *
 * The token value is never rendered — only "connected" state is.
 */
import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { ApiClient } from "../api/client";
import type { ConfigOut } from "../api/types";

export type ConnectionState =
  | "disconnected"
  | "connecting"
  | "connected"
  | "error";

interface AuthContextValue {
  client: ApiClient;
  /** In-memory token accessor for request/stream layers. */
  getToken: () => string | null;
  connection: ConnectionState;
  connectionMessage: string | null;
  config: ConfigOut | null;
  /** True only for UI affordances like "reconnect" hints. */
  hasToken: boolean;
  connect: (token: string) => Promise<boolean>;
  reconnect: () => Promise<boolean>;
  disconnect: () => void;
  markUnauthorized: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const tokenRef = useRef<string | null>(null);
  const [connection, setConnection] = useState<ConnectionState>("disconnected");
  const [connectionMessage, setConnectionMessage] = useState<string | null>(null);
  const [config, setConfig] = useState<ConfigOut | null>(null);
  const [hasToken, setHasToken] = useState(false);

  const client = useMemo(
    () => new ApiClient({ tokenProvider: () => tokenRef.current }),
    [],
  );

  const attemptConnect = useCallback(
    async (token: string): Promise<boolean> => {
      tokenRef.current = token;
      setHasToken(true);
      setConnection("connecting");
      setConnectionMessage(null);
      try {
        await client.health();
        const next = await client.config();
        setConfig(next);
        setConnection("connected");
        return true;
      } catch (error) {
        tokenRef.current = null;
        setHasToken(false);
        setConfig(null);
        setConnection("error");
        setConnectionMessage(
          error instanceof Error
            ? error.message
            : "Could not connect to the research service.",
        );
        return false;
      }
    },
    [client],
  );

  const connect = useCallback(
    (token: string) => attemptConnect(token),
    [attemptConnect],
  );

  const reconnect = useCallback(async (): Promise<boolean> => {
    const token = tokenRef.current;
    if (!token) return false;
    return attemptConnect(token);
  }, [attemptConnect]);

  const disconnect = useCallback(() => {
    tokenRef.current = null;
    setHasToken(false);
    setConfig(null);
    setConnection("disconnected");
    setConnectionMessage(null);
  }, []);

  const markUnauthorized = useCallback(() => {
    tokenRef.current = null;
    setHasToken(false);
    setConfig(null);
    setConnection("disconnected");
    setConnectionMessage(
      "Your access token was rejected. Reconnect with the current token to continue.",
    );
  }, []);

  const getToken = useCallback(() => tokenRef.current, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      client,
      getToken,
      connection,
      connectionMessage,
      config,
      hasToken,
      connect,
      reconnect,
      disconnect,
      markUnauthorized,
    }),
    [
      client,
      getToken,
      connection,
      connectionMessage,
      config,
      hasToken,
      connect,
      reconnect,
      disconnect,
      markUnauthorized,
    ],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const value = useContext(AuthContext);
  if (!value) {
    throw new Error("useAuth must be used inside AuthProvider");
  }
  return value;
}
