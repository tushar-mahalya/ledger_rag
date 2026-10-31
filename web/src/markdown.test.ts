import { describe, expect, it } from "vitest";
import { unified } from "unified";
import remarkParse from "remark-parse";
import {
  artifactDownload,
  extractArtifacts,
  remarkCitations,
  safeExternalUrl,
} from "./markdown";
import { visit } from "unist-util-visit";

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
