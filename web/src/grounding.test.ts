import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { answerSchema } from "./schemas";
import {
  describeReason,
  finalResult,
  formatDerivation,
  formatValue,
  groupDecimal,
  highlightSpan,
  shiftDecimal,
  splitUnsupported,
  timeline,
} from "./grounding";

const fixtures = JSON.parse(
  readFileSync(new URL("./fixtures/answers.json", import.meta.url), "utf8"),
) as Record<string, unknown>;
const calculation = answerSchema.parse(fixtures.verified).calculation!;

describe("decimal formatting", () => {
  it("shifts and groups exact decimal strings", () => {
    expect(shiftDecimal("30000000", 6)).toBe("30");
    expect(shiftDecimal("1234500", 6)).toBe("1.2345");
    expect(shiftDecimal("-45", 3)).toBe("-0.045");
    expect(shiftDecimal("0", 6)).toBe("0");
    expect(groupDecimal("1234567.891")).toBe("1,234,567.891");
    expect(groupDecimal("0.333333333333", 4)).toBe("0.3333");
    expect(groupDecimal("9.99996", 4)).toBe("10");
    expect(groupDecimal("-1000")).toBe("-1,000");
  });

  it("formats values by result kind", () => {
    expect(formatValue("25", "percent")).toBe("25%");
    expect(formatValue("0.25", "ratio")).toBe("0.25");
    expect(formatValue("1", "boolean")).toBe("true");
    expect(
      formatValue("30000000", "currency", {
        scale: "million",
        currency: "USD",
      }),
    ).toBe("USD 30 million");
    expect(formatValue("1250", "amount")).toBe("1,250");
    expect(formatValue("-2000000", "currency", { currency: "$" })).toBe(
      "-$2,000,000",
    );
  });
});

describe("derivation formatting", () => {
  it("renders each step with evidence quotes, step refs and constants", () => {
    expect(formatDerivation(calculation, 0)).toBe(
      "subtract(“USD 150 million” [C1], “USD 120 million” [C2]) = USD 30 million",
    );
    expect(formatDerivation(calculation, 1)).toBe(
      "divide(#0 (USD 30 million), “USD 120 million” [C2]) = 0.25",
    );
    expect(formatDerivation(calculation, 2)).toBe(
      "multiply(#1 (0.25), 100) = 25%",
    );
  });

  it("reports the final result and its unit", () => {
    expect(finalResult(calculation)).toEqual({ text: "25%", kind: "percent" });
  });
});

describe("unsupported-number highlighting", () => {
  const marked = (text: string, numbers: string[]) =>
    splitUnsupported(text, numbers)
      .filter((segment) => segment.unsupported)
      .map((segment) => segment.text);

  it("marks whole mentions and keeps the text intact", () => {
    const text =
      "Revenue rose to USD 150 million, a 31% increase over USD 115 million.";
    const segments = splitUnsupported(text, ["31%", "USD 115 million"]);
    expect(segments.map((segment) => segment.text).join("")).toBe(text);
    expect(marked(text, ["31%", "USD 115 million"])).toEqual([
      "31%",
      "USD 115 million",
    ]);
  });

  it("does not match inside larger numbers or words", () => {
    expect(marked("Margin was 131% and 31.5% and 31%.", ["31%"])).toEqual([
      "31%",
    ]);
    expect(marked("FY2031 and 2031", ["2031"])).toEqual(["2031"]);
    expect(marked("12km versus 12", ["12"])).toEqual(["12"]);
  });

  it("prefers the longest mention and ignores empty input", () => {
    expect(marked("$1,234m total", ["1,234", "$1,234m"])).toEqual(["$1,234m"]);
    expect(splitUnsupported("No numbers", [])).toEqual([
      { text: "No numbers", unsupported: false },
    ]);
    expect(marked("abc", ["", "  "])).toEqual([]);
  });

  it("highlights an operand span inside a source quote", () => {
    expect(highlightSpan("Revenue was USD 150 million.", "USD 150 million"))
      .toMatchInlineSnapshot(`
        [
          {
            "text": "Revenue was ",
            "unsupported": false,
          },
          {
            "text": "USD 150 million",
            "unsupported": true,
          },
          {
            "text": ".",
            "unsupported": false,
          },
        ]
      `);
    expect(highlightSpan("abc", "zzz")).toEqual([
      { text: "abc", unsupported: false },
    ]);
  });
});

describe("reasons and timeline", () => {
  it("describes known and unknown reasons", () => {
    expect(describeReason("verification_failed")).toMatch(/withheld/);
    expect(describeReason("new_backend_reason")).toBe("New backend reason.");
  });

  it("folds started/completed pairs including planning and validation", () => {
    const rows = timeline([
      { stage: "retrieval", status: "started", label: "Retrieving evidence" },
      { stage: "retrieval", status: "completed", label: "Retrieving evidence" },
      { stage: "planning", status: "started", label: "Planning query" },
      { stage: "planning", status: "completed", label: "Planning query" },
      {
        stage: "tool",
        status: "started",
        label: "Calculating",
        tool: "calculator",
      },
      { stage: "validation", status: "started", label: "Validating citations" },
      {
        stage: "validation",
        status: "completed",
        label: "Validating citations",
      },
      { stage: "validation", status: "started", label: "Verifying grounding" },
    ]);
    expect(rows.map((row) => [row.label, row.done])).toEqual([
      ["Retrieving evidence", true],
      ["Planning query", true],
      ["Calculating", false],
      ["Validating citations", true],
      ["Verifying grounding", false],
    ]);
  });
});
