import type { z } from "zod";
import type {
  activitySchema,
  answerSchema,
  citationSchema,
  evidenceSchema,
} from "./schemas";

export type Citation = z.infer<typeof citationSchema>;
export type Answer = z.infer<typeof answerSchema>;
export type Activity = z.infer<typeof activitySchema>;
export type Evidence = z.infer<typeof evidenceSchema>;
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
