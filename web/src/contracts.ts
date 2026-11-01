import type { z } from "zod";
import type {
  abstentionDecisionSchema,
  abstentionFeaturesSchema,
  activityKindSchema,
  activitySchema,
  answerSchema,
  calculationKindSchema,
  calculationOperandSchema,
  calculationSchema,
  calculationStepSchema,
  calculationUnitSchema,
  citationSchema,
  evidenceSchema,
  planSummarySchema,
  verificationSchema,
  verificationStatusSchema,
} from "./schemas";

export type Citation = z.infer<typeof citationSchema>;
export type Answer = z.infer<typeof answerSchema>;
export type Activity = z.infer<typeof activitySchema>;
export type ActivityKind = z.infer<typeof activityKindSchema>;
export type PlanSummary = z.infer<typeof planSummarySchema>;
export type Evidence = z.infer<typeof evidenceSchema>;
export type CalculationOperand = z.infer<typeof calculationOperandSchema>;
export type CalculationKind = z.infer<typeof calculationKindSchema>;
export type CalculationStep = z.infer<typeof calculationStepSchema>;
export type CalculationUnit = z.infer<typeof calculationUnitSchema>;
export type Calculation = z.infer<typeof calculationSchema>;
export type VerificationStatus = z.infer<typeof verificationStatusSchema>;
export type Verification = z.infer<typeof verificationSchema>;
export type AbstentionFeatures = z.infer<typeof abstentionFeaturesSchema>;
export type AbstentionDecision = z.infer<typeof abstentionDecisionSchema>;
export type HistoryMessage = { role: "user" | "assistant"; content: string };
export type Turn = {
  id: string;
  question: string;
  status: "pending" | "complete" | "error" | "cancelled";
  activities: Activity[];
  evidence?: Evidence;
  answer?: Answer;
  error?: string;
  roundTripMs?: number;
  demo?: boolean;
};
export type Conversation = { id: string; title: string; turns: Turn[] };

export function boundedHistory(turns: Turn[]): HistoryMessage[] {
  const messages: HistoryMessage[] = [];
  let total = 0;
  for (const turn of [...turns].reverse()) {
    if (turn.status !== "complete" || !turn.answer || turn.demo) continue;
    const pair: HistoryMessage[] = [
      { role: "user", content: turn.question },
      { role: "assistant", content: turn.answer.answer },
    ];
    const size = pair.reduce((sum, item) => sum + item.content.length, 0);
    if (messages.length + 2 > 12 || total + size > 32_000) break;
    if (pair.some((item) => item.content.length > 8000)) break;
    messages.unshift(...pair);
    total += size;
  }
  return messages;
}
