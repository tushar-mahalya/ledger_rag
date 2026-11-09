import { z } from "zod";

export const MAX_RESPONSE_JSON_BYTES = 900_000;

function boundedResponse(value: unknown, context: z.RefinementCtx) {
  if (
    new TextEncoder().encode(JSON.stringify(value)).byteLength >
    MAX_RESPONSE_JSON_BYTES
  ) {
    context.addIssue({
      code: "custom",
      message: "The response JSON exceeded the size limit.",
    });
  }
}

const dateText = z.string().regex(/^\d{4}-\d{2}-\d{2}$/);
export const provenanceSchema = z
  .object({
    source_kind: z.enum(["sec_xbrl", "fred"]),
    provider: z.enum(["SEC EDGAR", "FRED"]),
    evidence_id: z
      .string()
      .min(5)
      .max(261)
      .regex(/^(?:sec|fred):[A-Za-z0-9._:/-]{1,257}$/),
    identifier: z.string().min(1).max(64),
    entity: z.string().min(1).max(300),
    concept: z.string().min(1).max(300),
    form: z.string().max(16).nullable().optional(),
    fiscal_year: z.number().int().min(1900).max(2200).nullable().optional(),
    fiscal_period: z.string().max(8).nullable().optional(),
    period_label: z.string().min(1).max(40),
    period_start: dateText.nullable().optional(),
    period_end: dateText,
    unit: z.string().min(1).max(120),
    scale: z.number().int().min(0).max(12).default(0),
    value: z.string(),
    published_at: dateText,
    retrieved_at: z.string(),
    as_of: dateText,
    source_url: z.string().min(1).max(2048),
    freshness: z.enum(["current", "stale", "superseded"]),
    amended: z.boolean().default(false),
    superseded_by: z.string().max(64).nullable().optional(),
    revised_before_as_of: z.boolean().default(false),
  })
  .strict()
  .superRefine((value, ctx) => {
    let validUrl = false;
    try {
      const url = new URL(value.source_url);
      validUrl =
        url.protocol === "https:" &&
        ["www.sec.gov", "fred.stlouisfed.org"].includes(url.hostname) &&
        !url.port &&
        !url.username &&
        !url.password &&
        !url.hash &&
        ![...url.searchParams.keys()].some((key) =>
          /key|token|secret/i.test(key),
        );
    } catch {
      /* Rejected below. */
    }
    if (!validUrl)
      ctx.addIssue({ code: "custom", message: "Invalid provenance URL" });
    const decimal = /^[-+]?(\d+)(?:\.(\d*))?(?:[eE]([-+]?\d+))?$/.exec(
      value.value,
    );
    const digits = decimal
      ? `${decimal[1]}${decimal[2] ?? ""}`.replace(/^0+/, "")
      : "";
    const exponent = decimal
      ? Number(decimal[3] ?? 0) - (decimal[2]?.length ?? 0)
      : NaN;
    if (
      !decimal ||
      !Number.isFinite(exponent) ||
      exponent < -20 ||
      (digits.length > 0 && digits.length + exponent > 30)
    )
      ctx.addIssue({ code: "custom", message: "Invalid provenance number" });
    for (const day of [
      value.period_start,
      value.period_end,
      value.published_at,
      value.as_of,
    ]) {
      if (day != null && !dateText.safeParse(day).success)
        ctx.addIssue({ code: "custom", message: "Invalid provenance date" });
    }
    if (value.published_at > value.as_of)
      ctx.addIssue({
        code: "custom",
        message: "Evidence published after cutoff",
      });
  });

export const citationSchema = z
  .object({
    id: z
      .string()
      .max(16)
      .regex(/^C\d+$/),
    document_id: z.string().max(261),
    title: z.string().max(500),
    source: z.string().max(2048),
    published_at: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
    quote: z.string().max(100_000),
    source_kind: z.enum(["corpus", "sec_xbrl", "fred"]).optional(),
    provenance: provenanceSchema.nullable().optional(),
  })
  .superRefine((value, ctx) => {
    if (
      (value.source_kind ?? "corpus") === "corpus" &&
      value.document_id.length > 200
    )
      ctx.addIssue({
        code: "custom",
        message: "Corpus document identity exceeds 200 characters",
      });
    if (
      ((value.source_kind ?? "corpus") === "corpus") !==
        (value.provenance == null) ||
      (value.provenance != null &&
        value.provenance.source_kind !== value.source_kind)
    )
      ctx.addIssue({
        code: "custom",
        message: "Citation provenance does not match source kind",
      });
    if (
      value.provenance &&
      (value.document_id !== value.provenance.evidence_id ||
        value.source !== value.provenance.source_url ||
        value.published_at !== value.provenance.published_at)
    )
      ctx.addIssue({
        code: "custom",
        message: "Citation identity does not match provenance",
      });
  });

// Mirrors ledger_rag.core.models; contract.test.ts asserts the bounds match the Pydantic
// JSON Schema in contracts/contract.schema.json. Decimal values are strings so they
// round-trip exactly.
const decimalString = z.string().max(64);
const shortText = z.string().max(500);
const count = z.number().int().nonnegative();

export const calculationOperandSchema = z.object({
  kind: z.enum(["evidence", "step", "constant"]),
  value: decimalString,
  citation_id: z.string().max(200).nullable().optional(),
  quote: z.string().max(10_000).nullable().optional(),
  source_value: shortText.nullable().optional(),
  scale: z.string().max(64).nullable().optional(),
  unit: z.string().max(64).nullable().optional(),
  step: count.nullable().optional(),
});

export const calculationKindSchema = z.enum([
  "currency",
  "amount",
  "ratio",
  "percent",
  "boolean",
]);

// Deterministic result unit from the program DSL. Absent from older servers, where the
// client falls back to guessing from the step kind and operands.
export const unitKindSchema = z.enum([
  "currency",
  "percent",
  "ratio",
  "count",
  "per_share",
  "unitless",
]);

export const calculationUnitSchema = z.object({
  kind: unitKindSchema,
  currency: z
    .string()
    .regex(/^[A-Z]{3}$/)
    .nullable()
    .optional(),
  scale: z
    .enum(["thousand", "million", "billion", "trillion"])
    .nullable()
    .optional(),
  display: z.enum(["amount", "percent", "ratio", "boolean"]),
});

export const calculationStepSchema = z.object({
  index: count,
  op: z.string().max(64),
  operands: z.array(calculationOperandSchema).max(64),
  result: decimalString,
  kind: calculationKindSchema.nullable().optional(),
  citation_ids: z.array(z.string().max(200)).max(64).default([]),
  evidence_backed: z.boolean().default(false),
  unit: calculationUnitSchema.nullable().optional(),
});

export const calculationSchema = z.object({
  steps: z.array(calculationStepSchema).max(16),
  result: decimalString,
  citation_ids: z.array(z.string().max(200)).max(64),
  unit: calculationUnitSchema.nullable().optional(),
});

export const verificationStatusSchema = z.enum([
  "verified",
  "unverified",
  "failed",
  "skipped",
]);

export const verificationSchema = z.object({
  status: verificationStatusSchema,
  mode: z.enum(["off", "flag", "enforce"]),
  checked_numbers: count.default(0),
  // Empty on abstained answers: the rejected draft is never sent.
  unsupported_numbers: z.array(z.string().max(200)).max(64).default([]),
  reasons: z.array(shortText).max(32).default([]),
  verifier: shortText.nullable().optional(),
  verifier_score: z.number().finite().nullable().optional(),
});

// Older servers sent the count as `unsupported_numbers`; it is read as the new field.
function legacyFeatures(raw: unknown): unknown {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) return raw;
  const { unsupported_numbers: legacy, ...rest } = raw as Record<
    string,
    unknown
  >;
  return "unsupported_number_count" in rest || typeof legacy !== "number"
    ? rest
    : { ...rest, unsupported_number_count: legacy };
}

export const abstentionFeaturesSchema = z.preprocess(
  legacyFeatures,
  z.object({
    hit_count: count,
    top_score: z.number().finite().nullable().optional(),
    mean_score: z.number().finite().nullable().optional(),
    citation_coverage: z.number().finite().nullable().optional(),
    checked_numbers: count.default(0),
    unsupported_number_count: count.default(0),
    verification_status: verificationStatusSchema.default("skipped"),
    verifier_score: z.number().finite().nullable().optional(),
    tool_calls: count.default(0),
    tool_errors: count.default(0),
  }),
);

export const abstentionDecisionSchema = z.object({
  abstained: z.boolean(),
  reasons: z.array(shortText).max(32).default([]),
  features: abstentionFeaturesSchema,
  calibrated: z.boolean().default(false),
});

export const answerSchema = z
  .object({
    answer: z.string().max(100_000),
    citations: z.array(citationSchema).max(12),
    abstained: z.boolean(),
    profile: z.enum(["chat", "voice"]),
    retrieval_ms: z.number().nonnegative().finite(),
    generation_ms: z.number().nonnegative().finite(),
    request_id: z.string().min(1).max(200),
    model: z.string().max(500),
    tool_calls: count.default(0),
    calculation: calculationSchema.nullable().default(null),
    calculations: z.array(calculationSchema).max(16).default([]),
    verification: verificationSchema.nullable().default(null),
    abstention: abstentionDecisionSchema.nullable().default(null),
  })
  .superRefine(boundedResponse);

export const activityKindSchema = z.enum([
  "planning",
  "retrieval",
  "generation",
  "calculator",
  "evidence_search",
  "company_facts_lookup",
  "macro_series_lookup",
  "tool_check",
  "citation_validation",
  "grounding_verification",
]);

// Client-safe plan digest on the completed planning activity: an allowlist, never prompts.
export const planSummarySchema = z.object({
  intent: z.enum([
    "lookup",
    "compare",
    "ratio",
    "change",
    "percent_change",
    "aggregate",
  ]),
  route: z.enum([
    "retrieve",
    "direct_answer",
    "clarify",
    "refuse_out_of_scope",
  ]),
  subquestion_count: count.max(64),
  subquestions: z.array(z.string().max(200)).max(4).default([]),
  companies: z.array(z.string().max(64)).max(8).default([]),
  periods: z.array(z.string().max(64)).max(8).default([]),
});

export const activitySchema = z
  .object({
    stage: z.enum([
      "planning",
      "retrieval",
      "generation",
      "tool",
      "validation",
    ]),
    status: z.enum(["started", "completed"]),
    // The server sends a fixed set of labels; display text only, never a key.
    label: z.string().max(160),
    tool: z
      .enum([
        "calculator",
        "search_evidence",
        "lookup_company_facts",
        "lookup_macro_series",
      ])
      .nullable()
      .optional(),
    // Absent from older servers; see activityKind() in grounding.ts.
    kind: activityKindSchema.optional(),
    plan: planSummarySchema.nullable().optional(),
    outcome: z
      .enum(["ok", "error", "timeout", "unavailable"])
      .nullable()
      .optional(),
  })
  .superRefine((value, ctx) => {
    const externalTool =
      value.kind === "company_facts_lookup"
        ? "lookup_company_facts"
        : value.kind === "macro_series_lookup"
          ? "lookup_macro_series"
          : null;
    if (externalTool && (value.stage !== "tool" || value.tool !== externalTool))
      ctx.addIssue({
        code: "custom",
        message: "External activity kind does not match tool",
      });
    if (
      value.outcome != null &&
      (value.status !== "completed" ||
        value.stage !== "tool" ||
        !["lookup_company_facts", "lookup_macro_series"].includes(
          value.tool ?? "",
        ))
    )
      ctx.addIssue({
        code: "custom",
        message: "Only completed external lookups carry outcomes",
      });
  });

export const evidenceSchema = z
  .object({
    request_id: z.string().max(200),
    retrieval_ms: z.number().nonnegative().finite(),
    sources: z.array(z.string().max(2048)).max(12),
    citations: z.array(citationSchema).max(12).optional(),
  })
  .superRefine(boundedResponse);
