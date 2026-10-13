/**
 * Safe markdown rendering for research answers.
 *
 * Safety rules (Workstream 5):
 *  - no raw-HTML rendering (react-markdown without rehype-raw —
 *    HTML in the source renders as literal text);
 *  - link protocols limited to http(s)/mailto; javascript:, data:
 *    and friends are stripped (href removed entirely);
 *  - external links open with rel="noopener noreferrer";
 *  - code renders as code (pre/code elements, never interpreted);
 *  - inline [S#] citation tokens are rewritten by a remark plugin
 *    into INTERNAL `cite:S#` pseudo-links (never emitted as hrefs):
 *    known sources of THIS run become highlight buttons, unknown ids
 *    degrade to plain muted text — and code blocks are never touched.
 *  - `src` attributes are stripped outright (no remote asset loads
 *    from answer content).
 */
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Components } from "react-markdown";
import type { SourceOut } from "../api/types";

const SAFE_URL = /^(https?:\/\/|mailto:)/i;
const CITE_PREFIX = "cite:";
const CITE_TOKEN = /\[(S\d{1,3})\]/g;

export function safeUrlTransform(url: string): string {
  const trimmed = url.trim();
  // `cite:` URLs only ever exist inside our own rendered tree (the
  // anchor component swaps them for buttons/spans) and are never
  // written to the DOM as links.
  if (trimmed.toLowerCase().startsWith(CITE_PREFIX)) return trimmed;
  return SAFE_URL.test(trimmed) ? trimmed : "";
}

// ---- remark plugin: [S#] -> internal cite links ---------------------------

interface HastNode {
  type: string;
  value?: string;
  children?: HastNode[];
  [key: string]: unknown;
}

function remarkCitations(): (tree: HastNode) => void {
  return (tree) => {
    if (tree.children) {
      tree.children = transformChildren(tree.children);
    }
  };
}

function transformChildren(children: HastNode[]): HastNode[] {
  const out: HastNode[] = [];
  for (const child of children) {
    // code stays literal — a [S1] inside code is text, not a citation
    if (
      child.type === "code" ||
      child.type === "inlineCode" ||
      child.type === "pre"
    ) {
      out.push(child);
      continue;
    }
    if (
      child.type === "text" &&
      typeof child.value === "string" &&
      child.value.includes("[")
    ) {
      out.push(...splitTextNode(child));
      continue;
    }
    if (child.children) {
      child.children = transformChildren(child.children);
    }
    out.push(child);
  }
  return out;
}

function splitTextNode(node: HastNode): HastNode[] {
  const text = String(node.value);
  const out: HastNode[] = [];
  const pattern = new RegExp(CITE_TOKEN.source, "g");
  let match: RegExpExecArray | null;
  let lastIndex = 0;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      out.push({ type: "text", value: text.slice(lastIndex, match.index) });
    }
    out.push({
      type: "link",
      url: `${CITE_PREFIX}${match[1]}`,
      title: null,
      children: [{ type: "text", value: match[1] }],
    });
    lastIndex = match.index + match[0].length;
  }
  if (lastIndex < text.length) {
    out.push({ type: "text", value: text.slice(lastIndex) });
  }
  return out;
}

// ---- component ------------------------------------------------------------

export function SafeMarkdown({
  markdown,
  sources,
  onCiteSelect,
}: {
  markdown: string;
  sources: SourceOut[];
  onCiteSelect?: (sid: string) => void;
}) {
  const components: Components = {
    a: ({ href, children }) => {
      if (href && href.startsWith(CITE_PREFIX)) {
        const sid = href.slice(CITE_PREFIX.length);
        const known = sources.some((source) => source.sid === sid);
        if (!known) {
          return (
            <span
              className="mx-0.5 text-ink-muted/70"
              title="This citation is not in this run's sources"
            >
              [{sid}]
            </span>
          );
        }
        return (
          <button
            type="button"
            onClick={() => onCiteSelect?.(sid)}
            aria-label={`Show source ${sid}`}
            className="mx-0.5 inline-flex items-center rounded border border-accent/40 bg-accent-soft px-1 align-baseline text-[11px] font-semibold text-accent hover:bg-accent hover:text-white"
          >
            {sid}
          </button>
        );
      }
      if (!href || !SAFE_URL.test(href)) {
        // Unsafe protocol: render the label as plain text, no link.
        return <>{children}</>;
      }
      return (
        <a href={href} target="_blank" rel="noopener noreferrer">
          {children}
        </a>
      );
    },
    img: () => null,
  };

  return (
    <div className="prose-assistant">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkCitations]}
        urlTransform={safeUrlTransform}
        components={components}
      >
        {markdown}
      </ReactMarkdown>
    </div>
  );
}
