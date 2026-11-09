import type { Emphasis, Root, Link, Text } from "mdast";
import { toString } from "mdast-util-to-string";
import remarkGfm from "remark-gfm";
import remarkParse from "remark-parse";
import { unified } from "unified";
import { visit } from "unist-util-visit";
import { splitUnsupported } from "./grounding";

export function remarkCitations() {
  return (tree: Root) => {
    visit(tree, "text", (node, index, parent) => {
      if (
        index === undefined ||
        !parent ||
        parent.type === "link" ||
        parent.type === "linkReference"
      )
        return;
      const parts: (Text | Link)[] = [];
      let start = 0;
      for (const match of node.value.matchAll(/\[(C\d+)\]/g)) {
        const offset = match.index;
        if (offset > start)
          parts.push({ type: "text", value: node.value.slice(start, offset) });
        parts.push({
          type: "link",
          url: `#citation-${match[1]}`,
          children: [{ type: "text", value: match[1] }],
        });
        start = offset + match[0].length;
      }
      if (!parts.length) return;
      if (start < node.value.length)
        parts.push({ type: "text", value: node.value.slice(start) });
      parent.children.splice(index, 1, ...parts);
      return index + parts.length;
    });
  };
}

/** Wraps verifier-reported unsupported numbers in `<mark>` without touching links or code. */
export function remarkUnsupported(numbers: string[]) {
  return () => (tree: Root) => {
    if (!numbers.length) return;
    visit(tree, "text", (node, index, parent) => {
      if (
        index === undefined ||
        !parent ||
        parent.type === "link" ||
        parent.type === "linkReference" ||
        (parent as { data?: { hName?: string } }).data?.hName === "mark"
      )
        return;
      const segments = splitUnsupported(node.value, numbers);
      if (!segments.some((segment) => segment.unsupported)) return;
      const parts: (Text | Emphasis)[] = segments.map((segment) =>
        segment.unsupported
          ? {
              type: "emphasis",
              data: {
                hName: "mark",
                hProperties: { className: ["unsupported-number"] },
              },
              children: [{ type: "text", value: segment.text }],
            }
          : { type: "text", value: segment.text },
      );
      parent.children.splice(index, 1, ...parts);
      return index + parts.length;
    });
  };
}

export type Artifact =
  | {
      id: string;
      kind: "table";
      title: string;
      rows: string[][];
      markdown: string;
    }
  | {
      id: string;
      kind: "code";
      title: string;
      language: string;
      content: string;
      markdown: string;
    };

const parser = unified().use(remarkParse).use(remarkGfm);

export function extractArtifacts(markdown: string): Artifact[] {
  const tree = parser.parse(markdown);
  const artifacts: Artifact[] = [];
  let heading = "Evidence";
  visit(tree, (node) => {
    if (node.type === "heading") heading = toString(node);
    if (artifacts.length >= 12) return;
    if (node.type === "table") {
      artifacts.push({
        id: `artifact-${artifacts.length + 1}`,
        kind: "table",
        title: `${heading} table`,
        rows: node.children.map((row) =>
          row.children.map((cell) => toString(cell)),
        ),
        markdown: markdown.slice(
          node.position?.start.offset,
          node.position?.end.offset,
        ),
      });
    }
    if (node.type === "code") {
      artifacts.push({
        id: `artifact-${artifacts.length + 1}`,
        kind: "code",
        title: `${node.lang || "Plain text"} excerpt`,
        language: node.lang || "text",
        content: node.value,
        markdown: markdown.slice(
          node.position?.start.offset,
          node.position?.end.offset,
        ),
      });
    }
  });
  return artifacts;
}

export function safeExternalUrl(value: string): string | undefined {
  try {
    const url = new URL(value);
    return ["https:", "http:"].includes(url.protocol) &&
      !url.username &&
      !url.password
      ? url.href
      : undefined;
  } catch {
    return undefined;
  }
}

export function artifactDownload(artifact: Artifact): {
  content: string;
  extension: string;
  type: string;
} {
  if (artifact.kind === "code")
    return { content: artifact.content, extension: "txt", type: "text/plain" };
  const content = artifact.rows
    .map((row) =>
      row
        .map((cell) => {
          const safe =
            /^[\s]*[=+\-@]/.test(cell) || /^[\t\r]/.test(cell)
              ? `'${cell}`
              : cell;
          return `"${safe.replaceAll('"', '""')}"`;
        })
        .join(","),
    )
    .join("\r\n");
  return { content, extension: "csv", type: "text/csv" };
}
