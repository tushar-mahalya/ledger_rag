import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { activitySchema, answerSchema, evidenceSchema } from "./schemas";

// Generated from the pydantic models (see `_generated_by` in the fixture).
const fixtures = JSON.parse(
  readFileSync(
    new URL("../../../contracts/answers.json", import.meta.url),
    "utf8",
  ),
) as Record<string, Record<string, unknown>> & {
  activities: Record<string, unknown[]>;
  evidence: Record<string, unknown>;
};
const scenarios = ["verified", "unverified", "abstained", "refused"];

describe("answer contract", () => {
  it.each(scenarios)(
    "parses the %s backend payload without dropping fields",
    (name) => {
      // zod strips unknown keys, so equality proves the schema mirrors every Python field.
      expect(answerSchema.parse(fixtures[name])).toEqual(fixtures[name]);
    },
  );

  it("keeps parsing pre-grounding payloads with additive defaults", () => {
    const parsed = answerSchema.parse(fixtures.legacy);
    expect(parsed.calculation).toBeNull();
    expect(parsed.calculations).toEqual([]);
    expect(parsed.verification).toBeNull();
    expect(parsed.abstention).toBeNull();
  });

  it("exposes provenance, verification and uncalibrated abstention", () => {
    const verified = answerSchema.parse(fixtures.verified);
    const [first] = verified.calculation!.steps;
    expect(first.operands.map((operand) => operand.kind)).toEqual([
      "evidence",
      "evidence",
    ]);
    expect(first.operands[0]).toMatchObject({
      citation_id: "C1",
      quote: "USD 150 million",
    });
    expect(verified.calculation!.steps[2].operands[1]).toMatchObject({
      kind: "constant",
      value: "100",
    });
    expect(verified.verification?.status).toBe("verified");
    const unverified = answerSchema.parse(fixtures.unverified);
    expect(unverified.verification?.unsupported_numbers).toEqual([
      "31%",
      "USD 115 million",
    ]);
    const abstained = answerSchema.parse(fixtures.abstained);
    expect(abstained.abstained).toBe(true);
    expect(abstained.abstention).toMatchObject({
      abstained: true,
      calibrated: false,
      reasons: ["verification_failed"],
      features: { unsupported_number_count: 2 },
    });
    // Enforce mode withholds the draft: no mentions, citations or calculations.
    expect(abstained.verification?.unsupported_numbers).toEqual([]);
    expect(abstained.citations).toEqual([]);
    expect(abstained.calculation).toBeNull();
    expect(abstained.calculations).toEqual([]);
    const refused = answerSchema.parse(fixtures.refused);
    expect(refused.abstention?.reasons).toEqual(["out_of_scope"]);
  });

  it("carries the calculator's unit metadata", () => {
    const { calculation } = answerSchema.parse(fixtures.verified);
    expect(calculation!.steps.map((step) => step.unit)).toEqual([
      {
        kind: "currency",
        currency: "USD",
        scale: "million",
        display: "amount",
      },
      { kind: "ratio", currency: null, scale: null, display: "ratio" },
      { kind: "percent", currency: null, scale: null, display: "percent" },
    ]);
    expect(calculation!.unit).toEqual(calculation!.steps[2].unit);
  });

  it("reads the legacy unsupported_numbers feature count", () => {
    const abstained = fixtures.abstained;
    const { unsupported_number_count: count, ...features } = (
      abstained.abstention as { features: Record<string, unknown> }
    ).features;
    const legacy = answerSchema.parse({
      ...abstained,
      abstention: {
        ...(abstained.abstention as object),
        features: { ...features, unsupported_numbers: count },
      },
    });
    expect(legacy.abstention?.features.unsupported_number_count).toBe(2);
  });

  it("never references a citation the payload does not carry", () => {
    for (const name of scenarios) {
      const answer = answerSchema.parse(fixtures[name]);
      const ids = new Set(answer.citations.map((citation) => citation.id));
      const programs = [answer.calculation, ...answer.calculations];
      const referenced = programs.flatMap((calculation) =>
        !calculation
          ? []
          : [
              ...calculation.citation_ids,
              ...calculation.steps.flatMap((step) => [
                ...step.citation_ids,
                ...step.operands.flatMap((operand) =>
                  operand.citation_id ? [operand.citation_id] : [],
                ),
              ]),
            ],
      );
      for (const id of referenced) expect(ids, `${name}: ${id}`).toContain(id);
    }
  });

  it("rejects malformed grounding fields", () => {
    const verified = fixtures.verified as Record<string, unknown>;
    expect(() =>
      answerSchema.parse({
        ...verified,
        verification: { status: "maybe", mode: "flag" },
      }),
    ).toThrow();
    expect(() =>
      answerSchema.parse({
        ...verified,
        calculation: { steps: [{ op: "add" }], result: "1" },
      }),
    ).toThrow();
  });
});

describe("activity contract", () => {
  it.each(scenarios)("round-trips the %s activity stream", (name) => {
    for (const item of fixtures.activities[name])
      expect(activitySchema.parse(item)).toEqual(item);
  });

  it("separates the two validation operations by kind", () => {
    const parsed = fixtures.activities.verified.map((item) =>
      activitySchema.parse(item),
    );
    const validation = parsed.filter((item) => item.stage === "validation");
    expect(validation.map((item) => item.kind)).toEqual([
      "citation_validation",
      "citation_validation",
      "grounding_verification",
      "grounding_verification",
    ]);
    expect(parsed.find((item) => item.tool)?.kind).toBe("calculator");
  });

  it("sends a bounded plan summary only with completed planning", () => {
    const parsed = fixtures.activities.verified.map((item) =>
      activitySchema.parse(item),
    );
    const planned = parsed.filter((item) => item.plan);
    expect(planned).toHaveLength(1);
    expect(planned[0]).toMatchObject({ kind: "planning", status: "completed" });
    expect(planned[0].plan).toEqual({
      intent: "percent_change",
      route: "retrieve",
      subquestion_count: 3,
      subquestions: [
        "What was the percentage change in Aurora Devices revenue from 2024 to 2025?",
        "AURORA revenue 2024",
        "AURORA revenue 2025",
      ],
      companies: ["AURORA"],
      periods: ["2024", "2025"],
    });
    expect(() =>
      activitySchema.parse({
        ...planned[0],
        plan: { ...planned[0].plan, subquestions: Array(5).fill("q") },
      }),
    ).toThrow();
  });

  it("parses activities from servers that predate kind and plan", () => {
    const legacy = activitySchema.parse({
      stage: "validation",
      status: "started",
      label: "Verifying grounding",
    });
    expect(legacy.kind).toBeUndefined();
    expect(legacy.plan).toBeUndefined();
  });

  it.each(scenarios)("parses the %s evidence event", (name) => {
    expect(evidenceSchema.parse(fixtures.evidence[name])).toEqual(
      fixtures.evidence[name],
    );
  });
});
