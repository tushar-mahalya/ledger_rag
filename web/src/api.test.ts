import { afterEach, describe, expect, it, vi } from "vitest";
import { EventParser, streamAnswer } from "./api";
import { demoTurn, exampleQuestions } from "./demo";
import { boundedHistory } from "./contracts";

afterEach(() => vi.unstubAllGlobals());
const answer = demoTurn(exampleQuestions[0]).answer!;
const options = () => ({
  token: "local-test-token",
  question: "Revenue?",
  history: [],
  asOf: "",
  signal: new AbortController().signal,
  onEvidence: vi.fn(),
  onActivity: vi.fn(),
});
const sse = (event: string, data: unknown) =>
  `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
function mockStream(body: string) {
  vi.stubGlobal(
    "fetch",
    vi.fn(
      async () =>
        new Response(body, {
          headers: { "Content-Type": "text/event-stream" },
        }),
    ),
  );
}

describe("SSE contract", () => {
  it("handles every split boundary, CRLF, comments and multiline data", () => {
    const input =
      ': keepalive\r\nevent: answer\r\ndata: {"first":1,\r\ndata: "second":2}\r\n\r\n';
    for (let offset = 0; offset < input.length; offset++) {
      const parser = new EventParser();
      const events = [
        ...parser.push(input.slice(0, offset)),
        ...parser.push(input.slice(offset)),
      ];
      expect(events).toEqual([
        { event: "answer", data: '{"first":1,\n"second":2}' },
      ]);
    }
  });
  it("requires both a validated answer and done", async () => {
    mockStream(sse("answer", answer) + sse("done", {}));
    await expect(streamAnswer(options())).resolves.toEqual(answer);
    mockStream(sse("answer", answer));
    await expect(streamAnswer(options())).rejects.toThrow(
      "before a complete answer",
    );
    mockStream(sse("done", {}));
    await expect(streamAnswer(options())).rejects.toThrow(
      "before a complete answer",
    );
  });
  it("does not accept an answer followed by an error", async () => {
    mockStream(
      sse("answer", answer) +
        sse("error", { request_id: "failure-id", detail: "PRIVATE" }),
    );
    await expect(streamAnswer(options())).rejects.toThrow(
      "Partial evidence is not a final answer",
    );
  });
  it("validates wire payloads and never renders upstream error bodies", async () => {
    mockStream(
      sse("answer", { answer: "<script>bad</script>" }) + sse("done", {}),
    );
    await expect(streamAnswer(options())).rejects.toThrow("invalid event");
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => new Response("secret", { status: 401 })),
    );
    await expect(streamAnswer(options())).rejects.toThrow(
      "token is missing or expired",
    );
  });
  it("validates actual activity and evidence while awaiting a final answer", async () => {
    const activity = {
      stage: "tool",
      status: "completed",
      label: "Calculator completed",
      tool: "calculator",
    };
    const evidence = {
      request_id: "id",
      retrieval_ms: 5,
      sources: [],
      citations: [],
    };
    mockStream(
      sse("activity", activity) +
        sse("evidence", evidence) +
        sse("answer", answer) +
        sse("done", {}),
    );
    const callbacks = options();
    await streamAnswer(callbacks);
    expect(callbacks.onActivity).toHaveBeenCalledExactlyOnceWith(activity);
    expect(callbacks.onEvidence).toHaveBeenCalledExactlyOnceWith(evidence);
    const call = vi.mocked(fetch).mock.calls[0];
    expect(call[0]).toBe("/v1/query/stream");
    expect(call[1]?.redirect).toBe("error");
    expect(call[1]?.credentials).toBe("omit");
  });
  it("bounds event sizes even when the stream contains no line break", () => {
    expect(() => new EventParser().push("x".repeat(1_000_001))).toThrow(
      "size limit",
    );
    expect(() =>
      new EventParser().push("data: " + "x".repeat(1_000_001) + "\n"),
    ).toThrow("size limit");
  });
  it("rejects duplicate answers", async () => {
    mockStream(sse("answer", answer) + sse("answer", answer) + sse("done", {}));
    await expect(streamAnswer(options())).rejects.toThrow(
      "multiple final answers",
    );
  });
  it("preserves UTF-8 across byte boundaries", async () => {
    const expected = { ...answer, answer: "Currency: € · 日本語 [C1]" };
    const bytes = new TextEncoder().encode(
      sse("answer", expected) + sse("done", {}),
    );
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(
            new ReadableStream({
              start(controller) {
                for (const byte of bytes)
                  controller.enqueue(new Uint8Array([byte]));
                controller.close();
              },
            }),
            { headers: { "Content-Type": "text/event-stream" } },
          ),
      ),
    );
    await expect(streamAnswer(options())).resolves.toEqual(expected);
  });
});

describe("conversation limits", () => {
  it("excludes demo, failed, cancelled and unfinished turns from model history", () => {
    const base = demoTurn(exampleQuestions[0]);
    expect(
      boundedHistory([base, { ...base, status: "error", demo: false }]),
    ).toEqual([]);
    const live = { ...base, demo: false };
    expect(boundedHistory([live])).toEqual([
      { role: "user", content: live.question },
      { role: "assistant", content: live.answer!.answer },
    ]);
  });
  it("bounds both turns and characters without truncating evidence into malformed messages", () => {
    const base = { ...demoTurn(exampleQuestions[0]), demo: false };
    expect(boundedHistory(Array.from({ length: 20 }, () => base))).toHaveLength(
      12,
    );
    const large = {
      ...base,
      question: "q".repeat(4000),
      answer: { ...answer, answer: "a".repeat(8000) },
    };
    expect(boundedHistory([large, large, large])).toHaveLength(4);
    expect(
      boundedHistory([
        { ...large, answer: { ...answer, answer: "a".repeat(8001) } },
      ]),
    ).toEqual([]);
  });
});
