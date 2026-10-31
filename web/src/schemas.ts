import { z } from "zod";

export const citationSchema = z.object({
  id: z.string().regex(/^C\d+$/),
  document_id: z.string().max(200),
  title: z.string().max(500),
  source: z.string().max(2048),
  published_at: z.string().regex(/^\d{4}-\d{2}-\d{2}$/),
  quote: z.string().max(100_000),
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
});

export const activitySchema = z.object({
  stage: z.enum(["planning", "retrieval", "generation", "tool", "validation"]),
  status: z.enum(["started", "completed"]),
  label: z.string().max(160),
  tool: z.enum(["calculator", "search_evidence"]).optional(),
});

export const evidenceSchema = z.object({
  request_id: z.string().max(200),
  retrieval_ms: z.number().nonnegative().finite(),
  sources: z.array(z.string().max(2048)).max(12),
  citations: z.array(citationSchema).max(12).optional(),
});
