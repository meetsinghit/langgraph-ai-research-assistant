/** Safe Markdown rendering: protocol filtering, no raw HTML, citations. */
import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { SafeMarkdown, safeUrlTransform } from "../components/SafeMarkdown";
import type { SourceOut } from "../api/types";

const sources: SourceOut[] = [
  {
    sid: "S1",
    url: "https://example.com/a",
    host: "example.com",
    evidence: "retrieved",
    position: 0,
  },
  {
    sid: "S2",
    url: "https://other.org/b",
    host: "other.org",
    evidence: "snippet",
    position: 1,
  },
];

describe("safeUrlTransform", () => {
  it("allows http(s) and mailto only", () => {
    expect(safeUrlTransform("https://example.com")).toBe("https://example.com");
    expect(safeUrlTransform("http://example.com")).toBe("http://example.com");
    expect(safeUrlTransform("mailto:a@b.c")).toBe("mailto:a@b.c");
  });

  it("strips javascript:, data: and other unsafe protocols", () => {
    expect(safeUrlTransform("javascript:alert(1)")).toBe("");
    expect(safeUrlTransform("JAVASCRIPT:alert(1)")).toBe("");
    expect(safeUrlTransform("data:text/html,<script>x</script>")).toBe("");
    expect(safeUrlTransform("vbscript:msgbox")).toBe("");
    expect(safeUrlTransform("file:///etc/passwd")).toBe("");
  });
});

describe("SafeMarkdown", () => {
  it("renders headings, lists, tables and code blocks as content", () => {
    const { container } = render(
      <SafeMarkdown
        markdown={`# Title

- item one
- item two

| a | b |
| - | - |
| 1 | 2 |

` + "```python\nprint('hi')\n```"}
        sources={[]}
      />,
    );
    expect(container.querySelector("h1")).toHaveTextContent("Title");
    expect(container.querySelectorAll("li")).toHaveLength(2);
    expect(container.querySelector("table")).not.toBeNull();
    const code = container.querySelector("pre code");
    expect(code).toHaveTextContent("print('hi')");
  });

  it("does not render raw HTML as HTML", () => {
    const { container } = render(
      <SafeMarkdown
        markdown={'before <img src="x" onerror="alert(1)"> after <script>alert(2)</script>'}
        sources={[]}
      />,
    );
    expect(container.querySelector("script")).toBeNull();
    expect(container.querySelector("img")).toBeNull();
    // the literal text is preserved (escaped) instead
    expect(container.textContent).toContain("after");
  });

  it("neutralizes javascript: links (label stays, link goes)", () => {
    const { container } = render(
      <SafeMarkdown markdown="[click me](javascript:alert(1))" sources={[]} />,
    );
    expect(container.querySelector("a")).toBeNull();
    expect(container.textContent).toContain("click me");
  });

  it("strips unsafe image sources entirely", () => {
    const { container } = render(
      <SafeMarkdown markdown={'![x](data:text/html;base64,AAAA)'} sources={[]} />,
    );
    expect(container.querySelector("img")).toBeNull();
  });

  it("opens safe external links with noopener noreferrer", () => {
    const { container } = render(
      <SafeMarkdown markdown="[doc](https://example.com/doc)" sources={[]} />,
    );
    const anchor = container.querySelector("a");
    expect(anchor).toHaveAttribute("href", "https://example.com/doc");
    expect(anchor).toHaveAttribute("target", "_blank");
    expect(anchor).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("renders known [S#] citations as buttons and unknown ones as plain text", async () => {
    const onCiteSelect = vi.fn();
    const user = userEvent.setup();
    render(
      <SafeMarkdown
        markdown={"Claim A [S1] and B [S2], plus an unknown [S9]."}
        sources={sources}
        onCiteSelect={onCiteSelect}
      />,
    );
    const s1 = screen.getByRole("button", { name: "Show source S1" });
    const s2 = screen.getByRole("button", { name: "Show source S2" });
    expect(s1).toHaveTextContent("S1");
    expect(screen.getByText("[S9]")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Show source S9" })).toBeNull();

    await user.click(s1);
    expect(onCiteSelect).toHaveBeenCalledWith("S1");
    await user.click(s2);
    expect(onCiteSelect).toHaveBeenCalledWith("S2");
  });

  it("citations inside list items also become interactive", () => {
    render(
      <SafeMarkdown markdown={"- fact [S1]"} sources={sources} onCiteSelect={() => undefined} />,
    );
    expect(
      screen.getByRole("button", { name: "Show source S1" }),
    ).toBeInTheDocument();
  });

  it("never renders dangerous HTML embedded in tables (GFM)", () => {
    const { container } = render(
      <SafeMarkdown
        markdown={"| c |\n| - |\n| <b>bold</b> |\n| <iframe src=x></iframe> |"}
        sources={[]}
      />,
    );
    expect(container.querySelector("iframe")).toBeNull();
    expect(container.querySelector("b")).toBeNull();
  });
});
