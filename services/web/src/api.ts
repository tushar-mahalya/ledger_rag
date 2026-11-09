import { z } from "zod";
import { activitySchema, answerSchema, evidenceSchema } from "./schemas";
import type { Activity, Answer, Evidence, HistoryMessage } from "./contracts";
import { ChatError } from "./errors";

type StreamEvent = { event: string; data: string };
const MAX_EVENT = 1_000_000;

export class EventParser {
  private buffer = "";
  private event = "message";
  private data: string[] = [];
  private size = 0;

  push(chunk: string): StreamEvent[] {
    this.buffer += chunk;
    const events: StreamEvent[] = [];
    let end: number;
    while ((end = this.buffer.indexOf("\n")) !== -1) {
      const line = this.buffer.slice(0, end).replace(/\r$/, "");
      this.buffer = this.buffer.slice(end + 1);
      this.size += line.length;
      if (this.size > MAX_EVENT)
        throw new Error("The server event exceeded the size limit.");
      if (!line) {
        if (this.data.length)
          events.push({ event: this.event, data: this.data.join("\n") });
        this.data = [];
        this.event = "message";
        this.size = 0;
      } else if (!line.startsWith(":")) {
        const colon = line.indexOf(":");
        const field = colon === -1 ? line : line.slice(0, colon);
        const value =
          colon === -1 ? "" : line.slice(colon + 1).replace(/^ /, "");
        if (field === "event") this.event = value;
        if (field === "data") this.data.push(value);
      }
    }
    if (this.buffer.length + this.size > MAX_EVENT) {
      throw new Error("The server event exceeded the size limit.");
    }
    return events;
  }
}

type StreamOptions = {
  token: string;
  question: string;
  asOf: string;
  history: HistoryMessage[];
  signal: AbortSignal;
  onActivity: (activity: Activity) => void;
  onEvidence: (evidence: Evidence) => void;
};

export async function streamAnswer(options: StreamOptions): Promise<Answer> {
  const response = await fetch("/v1/query/stream", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${options.token}`,
    },
    body: JSON.stringify({
      question: options.question,
      profile: "chat",
      top_k: 5,
      history: options.history,
      ...(options.asOf ? { as_of: options.asOf } : {}),
    }),
    signal: options.signal,
    credentials: "omit",
    redirect: "error",
    cache: "no-store",
  });
  if (!response.ok) {
    const messages: Record<number, string> = {
      401: "Your access token is missing or expired. Reconnect to continue.",
      403: "This identity does not have permission to query the workspace.",
      422: "The question or conversation exceeds the API's input limits. Start a new chat.",
      429: "The workspace rate limit was reached. Wait before trying again.",
      503: "The service is at capacity. Try again shortly.",
      504: "The query deadline was exceeded. Try a narrower question.",
    };
    throw new ChatError(
      messages[response.status] ??
        `The service returned HTTP ${response.status}. No answer was accepted.`,
    );
  }
  if (
    !response.body ||
    !response.headers.get("content-type")?.includes("text/event-stream")
  ) {
    throw new ChatError("The service did not return an event stream.");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: true });
  const parser = new EventParser();
  let answer: Answer | undefined;
  let complete = false;
  let count = 0;
  let bytes = 0;
  try {
    while (!complete) {
      const result = await reader.read();
      if (result.done) break;
      bytes += result.value.byteLength;
      if (bytes > 4_000_000)
        throw new ChatError("The response exceeded the workspace size limit.");
      for (const event of parser.push(
        decoder.decode(result.value, { stream: true }),
      )) {
        if (++count > 256)
          throw new ChatError("The response exceeded the event limit.");
        if (
          !["activity", "evidence", "answer", "done", "error"].includes(
            event.event,
          )
        )
          continue;
        const payload: unknown = JSON.parse(event.data);
        if (event.event === "error") {
          const error = z
            .object({ request_id: z.string().max(200).optional() })
            .parse(payload);
          throw new ChatError(
            "The run failed. Partial evidence is not a final answer.",
            error.request_id,
          );
        }
        if (complete)
          throw new ChatError("The server sent events after completion.");
        if (event.event === "activity")
          options.onActivity(activitySchema.parse(payload));
        if (event.event === "evidence")
          options.onEvidence(evidenceSchema.parse(payload));
        if (event.event === "answer") {
          if (answer)
            throw new ChatError("The server returned multiple final answers.");
          answer = answerSchema.parse(payload);
        }
        if (event.event === "done") complete = true;
      }
    }
    if (!complete || !answer)
      throw new ChatError(
        "The connection ended before a complete answer. Please retry.",
      );
    return answer;
  } catch (error) {
    if (error instanceof z.ZodError || error instanceof SyntaxError) {
      throw new ChatError(
        "The service returned an invalid event. No answer was accepted.",
      );
    }
    throw error;
  } finally {
    await reader.cancel();
    reader.releaseLock();
  }
}
