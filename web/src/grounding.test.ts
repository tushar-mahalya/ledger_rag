import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { answerSchema } from "./schemas";
import type {
  Activity,
  Calculation,
  CalculationUnit,
  PlanSummary,
} from "./contracts";
import {
  activityKind,
  describePlan,
  describeReason,
  describeUnit,
  finalResult,
  formatDerivation,
  formatUnit,
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

const unit = (
  kind: CalculationUnit["kind"],
  extra: Partial<CalculationUnit> = {},
): CalculationUnit => ({
  kind,
  currency: null,
  scale: null,
  display:
    kind === "percent" ? "percent" : kind === "ratio" ? "ratio" : "amount",
  ...extra,
});

describe("unit formatting", () => {
  it("renders amounts in the evidence currency and scale", () => {
    const usd = unit("currency", { currency: "USD", scale: "million" });
    expect(formatUnit("30000000", usd)).toBe("USD 30 million");
    expect(formatUnit("-2500000", usd)).toBe("-USD 2.5 million");
    expect(formatUnit("0", usd)).toBe("USD 0 million");
    expect(formatUnit("1234", unit("currency", { currency: "EUR" }))).toBe(
      "EUR 1,234",
    );
    // An ambiguous symbol such as C$ arrives without an ISO code; never guess one.
    expect(formatUnit("5000000", unit("currency", { scale: "million" }))).toBe(
      "5 million",
    );
  });

  it("renders percent, ratio, per-share, count and boolean results", () => {
    expect(formatUnit("25", unit("percent"))).toBe("25%");
    expect(formatUnit("-3.456", unit("percent"))).toBe("-3.46%");
    expect(formatUnit("0.25", unit("ratio"))).toBe("0.25");
    expect(formatUnit("0.333333", unit("ratio"))).toBe("0.3333");
    expect(formatUnit("0.25", unit("ratio", { display: "percent" }))).toBe(
      "25%",
    );
    expect(formatUnit("1.5", unit("per_share", { currency: "USD" }))).toBe(
      "USD 1.5 per share",
    );
    expect(formatUnit("12000", unit("count", { scale: "thousand" }))).toBe(
      "12 thousand",
    );
    expect(formatUnit("1", unit("unitless", { display: "boolean" }))).toBe(
      "true",
    );
    expect(formatUnit("0", unit("unitless", { display: "boolean" }))).toBe(
      "false",
    );
  });

  it("describes units for the trace", () => {
    expect(
      describeUnit(unit("currency", { currency: "USD", scale: "million" })),
    ).toBe("Currency · USD million");
    expect(describeUnit(unit("currency"))).toBe(
      "Currency · currency unspecified",
    );
    expect(describeUnit(unit("ratio", { display: "percent" }))).toBe("Percent");
    expect(describeUnit(unit("count"))).toBe("Count");
    expect(describeUnit(unit("unitless", { display: "boolean" }))).toBe(
      "Yes / no",
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
    expect(finalResult(calculation)).toEqual({
      text: "25%",
      kind: "percent",
      label: "Percent",
    });
  });

  it("prefers server units over the step-kind guess", () => {
    // The legacy guess reads currency and scale from the operands: USD million.
    const steps = calculation.steps.map((step) => ({
      ...step,
      unit: unit("currency", { currency: "EUR", scale: "thousand" }),
    }));
    const eur: Calculation = { ...calculation, steps, unit: steps[2].unit };
    expect(formatDerivation(eur, 0)).toBe(
      "subtract(“USD 150 million” [C1], “USD 120 million” [C2]) = EUR 30,000 thousand",
    );
    expect(finalResult(eur).label).toBe("Currency · EUR thousand");
  });

  it("falls back to the legacy guess for payloads without units", () => {
    const legacy: Calculation = {
      ...calculation,
      unit: undefined,
      steps: calculation.steps.map((step) => ({ ...step, unit: undefined })),
    };
    expect(formatDerivation(legacy, 0)).toBe(formatDerivation(calculation, 0));
    expect(finalResult(legacy)).toEqual({
      text: "25%",
      kind: "percent",
      label: "Percent",
    });
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

  it("explains an out-of-scope refusal", () => {
    expect(describeReason("out_of_scope")).toMatch(
      /outside what the disclosures/,
    );
  });

  it("keys rows on kind, not label, and merges the completed plan", () => {
    const plan: PlanSummary = {
      intent: "percent_change",
      route: "retrieve",
      subquestion_count: 1,
      subquestions: ["AURORA revenue 2025"],
      companies: ["AURORA"],
      periods: ["2025"],
    };
    const rows = timeline([
      {
        stage: "planning",
        status: "started",
        label: "Planning query",
        kind: "planning",
      },
      {
        stage: "validation",
        status: "started",
        label: "Display text A",
        kind: "citation_validation",
      },
      {
        stage: "validation",
        status: "started",
        label: "Display text A",
        kind: "grounding_verification",
      },
      {
        stage: "validation",
        status: "completed",
        label: "Display text A",
        kind: "citation_validation",
      },
      {
        stage: "planning",
        status: "completed",
        label: "Planning query",
        kind: "planning",
        plan,
      },
    ]);
    expect(rows.map((row) => [row.kind, row.done])).toEqual([
      ["planning", true],
      ["citation_validation", true],
      ["grounding_verification", false],
    ]);
    expect(rows[0].plan).toEqual(plan);
    expect(new Set(rows.map((row) => row.key)).size).toBe(3);
  });

  it("derives kinds for activities from older servers", () => {
    const legacy = (activity: Omit<Activity, "status">) =>
      activityKind({ ...activity, status: "started" });
    expect(legacy({ stage: "validation", label: "Verifying grounding" })).toBe(
      "grounding_verification",
    );
    expect(legacy({ stage: "validation", label: "Validating citations" })).toBe(
      "citation_validation",
    );
    expect(legacy({ stage: "tool", label: "x", tool: "calculator" })).toBe(
      "calculator",
    );
    expect(legacy({ stage: "tool", label: "x", tool: "search_evidence" })).toBe(
      "evidence_search",
    );
    expect(legacy({ stage: "tool", label: "Checking tool request" })).toBe(
      "tool_check",
    );
  });

  it("describes plans in plain language", () => {
    expect(
      describePlan({
        intent: "compare",
        route: "retrieve",
        subquestion_count: 6,
        subquestions: ["a", "b", "c", "d"],
        companies: ["AURORA"],
        periods: [],
      }),
    ).toEqual([
      { term: "Intent", values: ["Compare figures"] },
      { term: "Route", values: ["Search the disclosures"] },
      { term: "Companies", values: ["AURORA"] },
      { term: "Searches (6)", values: ["a", "b", "c", "d", "and 2 more"] },
    ]);
    expect(
      describePlan({
        intent: "lookup",
        route: "refuse_out_of_scope",
        subquestion_count: 0,
        subquestions: [],
        companies: [],
        periods: ["2027"],
      }).map((fact) => fact.term),
    ).toEqual(["Intent", "Route", "Periods"]);
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
