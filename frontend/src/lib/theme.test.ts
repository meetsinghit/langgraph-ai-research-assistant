/** Theme hook: dark-class application + persistence (4D.3 defect fix). */
import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useTheme } from "./theme";

describe("useTheme", () => {
  it("toggling applies the dark class and persists the choice", () => {
    const { result } = renderHook(() => useTheme());
    expect(document.documentElement.classList.contains("dark")).toBe(false);

    act(() => result.current.toggle());

    expect(result.current.theme).toBe("dark");
    expect(document.documentElement.classList.contains("dark")).toBe(true);
    // persisted (cosmetic only; the access token is never stored)
    expect(window.localStorage.getItem("theme")).toBe("dark");
  });

  it("restores a persisted theme on load", () => {
    window.localStorage.setItem("theme", "dark");
    const { result } = renderHook(() => useTheme());
    expect(result.current.theme).toBe("dark");
    expect(document.documentElement.classList.contains("dark")).toBe(true);
  });

  it("falls back to light without a stored preference", () => {
    const { result } = renderHook(() => useTheme());
    expect(result.current.theme).toBe("light");
    expect(window.localStorage.getItem("theme")).toBe("light");
  });
});
