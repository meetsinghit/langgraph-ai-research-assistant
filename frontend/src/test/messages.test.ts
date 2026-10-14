/**
 * Error-category copy (4E.3 review item 4): a refused model access
 * must read clearly — fixed wording only, and it must never suggest
 * that another model ran instead.
 */
import { describe, expect, it } from "vitest";
import { categoryMessage, ERROR_CATEGORY_LABELS } from "../lib/messages";

describe("run error categories", () => {
  it("explains a refused model access without promising a fallback", () => {
    const message = categoryMessage("model_access");
    expect(message).toContain("could not be accessed");
    expect(message).toContain("No other model is used automatically");
    expect(ERROR_CATEGORY_LABELS.model_access).toBe("Model access failed");
  });

  it("keeps every category on fixed, non-technical copy", () => {
    for (const category of Object.keys(ERROR_CATEGORY_LABELS)) {
      expect(categoryMessage(category)).toBeTruthy();
    }
    // unknown/future categories fall back to the generic message —
    // the UI never renders a raw category string as prose
    expect(categoryMessage("some_brand_new_category")).toBe(
      categoryMessage("unknown"),
    );
    expect(categoryMessage(null)).toBe(categoryMessage("unknown"));
  });
});
