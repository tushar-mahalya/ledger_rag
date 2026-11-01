import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { activitySchema, answerSchema } from "./schemas";

// Generated from the pydantic models (see `_generated_by` in the fixture).
const fixtures = JSON.parse(
  readFileSync(new URL("./fixtures/answers.json", import.meta.url), "utf8"),
) as Record<string, unknown> & { activities: unknown[] };

describe("answer contract", () => {
  it.each(["verified", "unverified", "abstained"])(
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
    });
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
  it("parses planning, tool and grounding-verification activities", () => {
    const parsed = fixtures.activities.map((item) =>
      activitySchema.parse(item),
    );
    expect(parsed.map((item) => item.label)).toEqual([
      "Planning query",
      "Planning query",
      "Calculating",
      "Verifying grounding",
      "Verifying grounding",
    ]);
    expect(parsed[0].tool).toBeUndefined();
    expect(parsed[2].tool).toBe("calculator");
  });
});
