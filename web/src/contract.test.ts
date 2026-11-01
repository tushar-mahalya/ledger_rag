import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { z } from "zod";
import { demoTurn, exampleQuestions } from "./demo";
import { REASONS } from "./grounding";
import {
  activitySchema,
  answerSchema,
  evidenceSchema,
  MAX_RESPONSE_JSON_BYTES,
} from "./schemas";

type Node = Record<string, unknown>;
type Facts = Record<string, unknown>;

// Pydantic JSON Schema of the response models, written by scripts/dump_contract_fixtures.py.
const defs = (
  JSON.parse(
    readFileSync(
      new URL("./fixtures/contract.schema.json", import.meta.url),
      "utf8",
    ),
  ) as { $defs: Record<string, Node> }
).$defs;
const fixtures = JSON.parse(
  readFileSync(new URL("./fixtures/answers.json", import.meta.url), "utf8"),
) as Record<string, Node> & { activities: Record<string, Node[]> };

const BOUNDS = [
  "type",
  "maxLength",
  "minLength",
  "maxItems",
  "minItems",
  "minimum",
  "maximum",
  "exclusiveMinimum",
  "exclusiveMaximum",
  "pattern",
];

function resolve(node: Node): Node {
  const ref = node.$ref;
  return typeof ref === "string"
    ? resolve(defs[ref.replace("#/$defs/", "")])
    : node;
}

/**
 * Flattens a JSON Schema into path -> bound facts. Required-ness, defaults and titles are
 * ignored: zod keeps new fields optional so older payloads still parse.
 */
function flatten(raw: Node, path = "$", out: Record<string, Facts> = {}) {
  let node = resolve(raw);
  let nullable = false;
  if (Array.isArray(node.anyOf)) {
    const options = (node.anyOf as Node[]).map(resolve);
    const present = options.filter((option) => option.type !== "null");
    nullable = present.length < options.length;
    expect(present, `${path} is a single type or null`).toHaveLength(1);
    node = present[0];
  }
  const facts: Facts = { nullable };
  for (const key of BOUNDS) if (key in node) facts[key] = node[key];
  // zod writes a nullable primitive as `type: [T, "null"]` rather than anyOf.
  if (Array.isArray(facts.type) && facts.type.includes("null")) {
    const types = facts.type.filter((type) => type !== "null");
    expect(types, `${path} is a single type or null`).toHaveLength(1);
    facts.type = types[0];
    facts.nullable = true;
  }
  // zod adds the safe-integer range to every .int(); Pydantic leaves it implicit.
  if (facts.maximum === Number.MAX_SAFE_INTEGER) delete facts.maximum;
  if (facts.minimum === Number.MIN_SAFE_INTEGER) delete facts.minimum;
  if (node.format === "date") facts.pattern = "^\\d{4}-\\d{2}-\\d{2}$";
  if (Array.isArray(node.enum)) facts.enum = [...node.enum].sort();
  const properties = node.properties as Record<string, Node> | undefined;
  if (properties) facts.keys = Object.keys(properties).sort();
  out[path] = facts;
  for (const [key, child] of Object.entries(properties ?? {}))
    flatten(child, `${path}.${key}`, out);
  if (node.items) flatten(node.items as Node, `${path}[]`, out);
  return out;
}

const zodFacts = (schema: z.ZodType) =>
  flatten(
    z.toJSONSchema(schema, { io: "output", unrepresentable: "any" }) as Node,
  );

// Python closes these to an enum; zod keeps a bounded string so a new server value is
// still displayed. Each Python value must fit the zod bound.
const OPEN_ENUMS = new Set([
  "$.label",
  "$.verification.reasons[]",
  "$.abstention.reasons[]",
]);

function compare(model: string, schema: z.ZodType) {
  const python = flatten(defs[model]);
  const client = zodFacts(schema);
  expect(Object.keys(client).sort()).toEqual(Object.keys(python).sort());
  const drift: Record<string, { python: Facts; zod: Facts }> = {};
  for (const [path, facts] of Object.entries(python)) {
    const ok = OPEN_ENUMS.has(path)
      ? openEnumFits(facts, client[path])
      : JSON.stringify(sorted(client[path])) === JSON.stringify(sorted(facts));
    if (!ok) drift[path] = { python: facts, zod: client[path] };
  }
  expect(drift).toEqual({});
}

function openEnumFits(python: Facts, client: Facts) {
  const { enum: values, type, ...rest } = python;
  return (
    type === "string" &&
    client.type === "string" &&
    !("enum" in client) &&
    Object.entries(rest).every(([key, value]) => client[key] === value) &&
    (values as string[]).every(
      (value) => value.length <= (client.maxLength as number),
    )
  );
}

const sorted = (facts: Facts) =>
  Object.fromEntries(
    Object.entries(facts).sort(([a], [b]) => (a < b ? -1 : 1)),
  );

describe("bound parity with the Pydantic response models", () => {
  it.each([
    ["Answer", answerSchema],
    ["Activity", activitySchema],
    ["Evidence", evidenceSchema],
  ] as const)("%s matches field for field", (model, schema) => {
    compare(model, schema);
  });

  it("detects a drifted bound", () => {
    const loose = answerSchema.safeExtend({ answer: z.string().max(100_001) });
    expect(() => compare("Answer", loose)).toThrow(/\$\.answer/);
    const missing = z.object(answerSchema.shape).omit({ tool_calls: true });
    expect(() => compare("Answer", missing)).toThrow();
  });

  it("matches the aggregate serialized response budget", () => {
    for (const model of ["Answer", "Evidence"]) {
      expect(defs[model]["x-max-json-bytes"]).toBe(MAX_RESPONSE_JSON_BYTES);
    }
  });

  it("describes every server reason code", () => {
    const answer = flatten(defs.Answer);
    const codes = [
      answer["$.verification.reasons[]"].enum,
      answer["$.abstention.reasons[]"].enum,
    ].flat() as string[];
    expect(codes).toContain("out_of_scope");
    for (const code of codes) expect(REASONS, code).toHaveProperty(code);
  });
});

describe("demo payloads mirror the backend", () => {
  // stage and label per kind, as the server's ACTIVITY_KINDS sends them.
  const kinds = new Map<unknown, string>();
  for (const activity of Object.values(fixtures.activities).flat())
    kinds.set(activity.kind, `${activity.stage}/${activity.label}`);

  it.each(exampleQuestions)("%s parses and uses server activity kinds", (q) => {
    const turn = demoTurn(q);
    expect(answerSchema.parse(turn.answer)).toEqual(turn.answer);
    for (const activity of turn.activities) {
      expect(activitySchema.parse(activity)).toEqual(activity);
      const expected = kinds.get(activity.kind);
      if (expected)
        expect(`${activity.stage}/${activity.label}`).toBe(expected);
    }
  });

  it("refuses forecasts with the backend's reason and message", () => {
    const refused = answerSchema.parse(fixtures.refused);
    const demo = demoTurn(exampleQuestions[3]);
    expect(demo.answer?.answer).toBe(refused.answer);
    expect(demo.answer?.abstention?.reasons).toEqual(
      refused.abstention?.reasons,
    );
    expect(demo.answer?.abstention?.features).toMatchObject({
      hit_count: refused.abstention?.features.hit_count,
      top_score: null,
    });
    const plan = (turn: Node[]) =>
      turn.find((activity) => activity.plan)?.plan as Node;
    expect(plan(demo.activities as unknown as Node[]).route).toBe(
      plan(fixtures.activities.refused).route,
    );
  });
});
