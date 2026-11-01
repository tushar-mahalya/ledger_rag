import { describe, expect, it } from "vitest";
import fixtures from "./fixtures/answers.json";
import { demoTurn, exampleQuestions } from "./demo";
import { timeline } from "./grounding";
import { activitySchema, answerSchema, citationSchema } from "./schemas";

describe("point-in-time external evidence", () => {
  it("replays recorded provider payloads as a labeled offline demo", () => {
    const turn = demoTurn(exampleQuestions[4]);
    expect(turn.demo).toBe(true);
    expect(turn.answer).toEqual(answerSchema.parse(fixtures.external));
    expect(turn.answer?.citations.map((item) => item.source_kind)).toEqual([
      "sec_xbrl",
      "fred",
    ]);
    expect(
      turn.answer?.citations.every(
        (item) => item.provenance?.as_of === "2024-01-01",
      ),
    ).toBe(true);
    expect(
      timeline(turn.activities)
        .filter((row) => row.outcome)
        .map((row) => row.outcome),
    ).toEqual(["ok", "ok"]);
  });

  it("keeps timeout and unavailable outcomes when folding activity pairs", () => {
    const started = activitySchema.parse({
      stage: "tool",
      status: "started",
      label: "Looking up macro data",
      tool: "lookup_macro_series",
      kind: "macro_series_lookup",
    });
    for (const outcome of ["timeout", "unavailable", "error"] as const) {
      const rows = timeline([
        started,
        { ...started, status: "completed", outcome },
      ]);
      expect(rows).toHaveLength(1);
      expect(rows[0]).toMatchObject({ done: true, outcome });
    }
  });

  it("bounds provenance and rejects mismatched or credential-bearing sources", () => {
    const valid = fixtures.external.citations[0];
    expect(citationSchema.safeParse(valid).success).toBe(true);
    for (const value of ["1E+20", "999999999999999999999999999999", "1E-20"]) {
      expect(
        citationSchema.safeParse({
          ...valid,
          provenance: { ...valid.provenance, value },
        }).success,
      ).toBe(true);
    }
    for (const changes of [
      { source_kind: "corpus" },
      { source_kind: "fred" },
      { document_id: "other-id" },
      { source: "https://evil.example" },
      { provenance: null },
      { provenance: { ...valid.provenance, entity: "a".repeat(301) } },
      { provenance: { ...valid.provenance, value: "NaN" } },
      { provenance: { ...valid.provenance, value: "1E-21" } },
      {
        provenance: {
          ...valid.provenance,
          value: "1000000000000000000000000000000",
        },
      },
      {
        provenance: { ...valid.provenance, source_url: "https://evil.example" },
      },
      {
        provenance: {
          ...valid.provenance,
          source_url: "https://www.sec.gov/?api_key=redacted",
        },
      },
      { provenance: { ...valid.provenance, published_at: "2025-01-01" } },
    ]) {
      expect(citationSchema.safeParse({ ...valid, ...changes }).success).toBe(
        false,
      );
    }
    expect(
      activitySchema.safeParse({
        stage: "tool",
        status: "completed",
        label: "Looking up macro data",
        tool: "lookup_macro_series",
        kind: "macro_series_lookup",
        outcome: "raw failure",
      }).success,
    ).toBe(false);
    expect(
      activitySchema.safeParse({
        stage: "tool",
        status: "started",
        label: "Looking up macro data",
        tool: "lookup_macro_series",
        kind: "macro_series_lookup",
        outcome: "ok",
      }).success,
    ).toBe(false);
  });
});
