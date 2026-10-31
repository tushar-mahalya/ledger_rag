import { z } from "zod";

export const citationSchema = z.object({
  id: z.string().regex(/^C\d+$/),
  document_id: z.string().max(200),
  title: z.string().max(500),
  source: z.string().max(2048),
  published_at: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  quote: z.string().max(100_000),
});

// Mirrors ledger_rag.core.models. Decimal values are strings so they round-trip exactly.
const decimalString = z.string().max(64);
const shortText = z.string().max(500);

export const calculationOperandSchema = z.object({
  kind: z.enum(["evidence", "step", "constant"]),
  value: decimalString,
  citation_id: z.string().max(200).nullable().optional(),
  quote: z.string().max(10_000).nullable().optional(),
  source_value: shortText.nullable().optional(),
  scale: z.string().max(64).nullable().optional(),
  unit: z.string().max(64).nullable().optional(),
  step: z.number().int().nonnegative().nullable().optional(),
});

export const calculationKindSchema = z.enum([
  "currency",
  "amount",
  "ratio",
  "percent",
  "boolean",
]);

export const calculationStepSchema = z.object({
  index: z.number().int().nonnegative(),
  op: z.string().max(64),
  operands: z.array(calculationOperandSchema).max(64),
  result: decimalString,
  kind: calculationKindSchema.nullable().optional(),
  citation_ids: z.array(z.string().max(200)).max(64).default([]),
  evidence_backed: z.boolean().default(false),
});

export const calculationSchema = z.object({
  steps: z.array(calculationStepSchema).max(16),
  result: decimalString,
  citation_ids: z.array(z.string().max(200)).max(64),
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
  checked_numbers: z.number().int().nonnegative().default(0),
  unsupported_numbers: z.array(z.string().max(200)).max(64).default([]),
  reasons: z.array(shortText).max(32).default([]),
  verifier: shortText.nullable().optional(),
  verifier_score: z.number().finite().nullable().optional(),
});

export const abstentionFeaturesSchema = z.object({
  hit_count: z.number().int().nonnegative(),
  top_score: z.number().finite().nullable().optional(),
  mean_score: z.number().finite().nullable().optional(),
  citation_coverage: z.number().finite().nullable().optional(),
  checked_numbers: z.number().int().nonnegative().default(0),
  unsupported_numbers: z.number().int().nonnegative().default(0),
  verification_status: verificationStatusSchema.default("skipped"),
  verifier_score: z.number().finite().nullable().optional(),
  tool_calls: z.number().int().nonnegative().default(0),
  tool_errors: z.number().int().nonnegative().default(0),
});

export const abstentionDecisionSchema = z.object({
  abstained: z.boolean(),
  reasons: z.array(shortText).max(32).default([]),
  features: abstentionFeaturesSchema,
  calibrated: z.boolean().default(false),
});

export const answerSchema = z.object({
  answer: z.string().max(100_000),
  citations: z.array(citationSchema).max(12),
  abstained: z.boolean(),
  profile: z.enum(["chat", "voice"]),
  retrieval_ms: z.number().nonnegative().finite(),
  generation_ms: z.number().nonnegative().finite(),
  request_id: z.string().min(1).max(200),
  model: z.string().max(500),
  tool_calls: z.number().int().nonnegative().default(0),
  calculation: calculationSchema.nullable().default(null),
  calculations: z.array(calculationSchema).max(16).default([]),
  verification: verificationSchema.nullable().default(null),
  abstention: abstentionDecisionSchema.nullable().default(null),
});

export const activitySchema = z.object({
  stage: z.enum(["planning", "retrieval", "generation", "tool", "validation"]),
  status: z.enum(["started", "completed"]),
  label: z.string().max(160),
  tool: z.enum(["calculator", "search_evidence"]).nullable().optional(),
});

export const evidenceSchema = z.object({
  request_id: z.string().max(200),
  retrieval_ms: z.number().nonnegative().finite(),
  sources: z.array(z.string().max(2048)).max(12),
  citations: z.array(citationSchema).max(12).optional(),
});
