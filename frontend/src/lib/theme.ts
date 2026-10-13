/** Light/dark theme with persisted (cosmetic-only) preference. */
import { useCallback, useEffect, useState } from "react";

const STORAGE_KEY = "theme";

type Theme = "light" | "dark";

function initialTheme(): Theme {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    if (stored === "light" || stored === "dark") return stored;
  } catch {
    /* storage unavailable (private mode) — fall through */
  }
  const prefersDark =
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-color-scheme: dark)").matches;
  return prefersDark ? "dark" : "light";
}

export function useTheme(): { theme: Theme; toggle: () => void } {
  const [theme, setTheme] = useState<Theme>(initialTheme);

  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    // Persist the cosmetic preference so a reload keeps it (the
    // access token is deliberately NOT stored - only the theme).
    try {
      window.localStorage.setItem(STORAGE_KEY, theme);
    } catch {
      /* storage unavailable: theme stays session-only */
    }
  }, [theme]);

  const toggle = useCallback(() => {
    setTheme((current) => (current === "dark" ? "light" : "dark"));
  }, []);

  return { theme, toggle };
}
