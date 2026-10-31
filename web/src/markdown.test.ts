import { describe, expect, it } from "vitest";
import { unified } from "unified";
import remarkParse from "remark-parse";
import {
  artifactDownload,
  extractArtifacts,
  remarkCitations,
  remarkUnsupported,
  safeExternalUrl,
} from "./markdown";
import { visit } from "unist-util-visit";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import type { Root } from "mdast";
import { Markdown } from "./components/Markdown";

describe("safe artifacts and references", () => {
  it("only transforms citation text, never code blocks or existing link labels", () => {
    const tree = unified()
      .use(remarkParse)
      .parse(
        "Text [C1]. `[C2]` [label [C3]](https://example.org)\n\n```\n[C4]\n```",
      );
    remarkCitations()(tree);
    const links: string[] = [];
    visit(tree, "link", (node) => {
      links.push(node.url);
    });
    expect(links).toEqual(["#citation-C1", "https://example.org"]);
  });
  it("extracts structured tables and inert code using the Markdown AST", () => {
    const artifacts = extractArtifacts(
      "### Revenue\n\n| FY | USD |\n|---|---|\n|2025|150|\n\n```html\n<script>alert(1)</script>\n```",
    );
    expect(artifacts).toHaveLength(2);
    expect(artifacts[0]).toMatchObject({
      kind: "table",
      title: "Revenue table",
      rows: [
        ["FY", "USD"],
        ["2025", "150"],
      ],
    });
    expect(artifacts[1]).toMatchObject({
      kind: "code",
      language: "html",
      content: "<script>alert(1)</script>",
    });
  });
  it("does not permit unsafe, credentialed or relative outbound URLs", () => {
    for (const value of [
      "javascript:alert(1)",
      "data:text/html,bad",
      "//evil.test",
      "/v1/private",
      "https://key@example.org",
      "urn:ledger:synthetic",
    ]) {
      expect(safeExternalUrl(value)).toBeUndefined();
    }
    expect(safeExternalUrl("https://www.sec.gov/")).toBe(
      "https://www.sec.gov/",
    );
  });
  it("neutralizes spreadsheet formulas and quotes when exporting CSV", () => {
    const file = artifactDownload({
      id: "1",
      kind: "table",
      title: "Data",
      markdown: "",
      rows: [['=HYPERLINK("bad")', " +42", "-2", "@sum", "normal,cell"]],
    });
    expect(file.content).toBe(
      `"'=HYPERLINK(""bad"")","' +42","'-2","'@sum","normal,cell"`,
    );
  });
  it("caps artifact count", () => {
    expect(extractArtifacts("```\na\n```\n\n".repeat(20))).toHaveLength(12);
  });
});

describe("unsupported-number marks in rendered Markdown", () => {
  const render = (text: string, unsupported: string[]) =>
    renderToStaticMarkup(
      createElement(Markdown, {
        text,
        citations: [
          {
            id: "C1",
            document_id: "d",
            title: "T",
            source: "urn:x",
            published_at: "2026-01-01",
            quote: "q",
          },
        ],
        onCitation: () => undefined,
        unsupported,
      }),
    );

  it("marks numbers inside formatting and tables but not links or code", () => {
    const html = render(
      "Revenue grew **31%** to 150. [C1]\n\n| A | B |\n|---|---|\n| 31% | `31%` |\n\n[31%](https://example.com)",
      ["31%"],
    );
    expect(html.match(/<mark class="unsupported-number"/g)).toHaveLength(2);
    expect(html).toContain("<strong><mark");
    expect(html).toContain("<code>31%</code>");
    expect(html).toContain('href="https://example.com/"');
    expect(html).toContain('aria-label="View citation C1: T"');
  });

  it("renders unchanged Markdown when nothing is unsupported", () => {
    expect(render("Revenue was 150. [C1]", [])).not.toContain("<mark");
  });

  it("works on the mdast level next to citations", () => {
    const tree = unified()
      .use(remarkParse)
      .use(remarkCitations)
      .use(remarkUnsupported(["C1", "12"]))
      .runSync(unified().use(remarkParse).parse("Up 12 points [C1]")) as Root;
    const marks: string[] = [];
    visit(tree, "emphasis", (node) => {
      if (node.data?.hName === "mark")
        marks.push((node.children[0] as { value: string }).value);
    });
    expect(marks).toEqual(["12"]);
  });
});
