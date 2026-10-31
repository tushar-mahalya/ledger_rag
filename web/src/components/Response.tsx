import { lazy, Suspense, useMemo } from "react";
import {
  Activity as ActivityIcon,
  ArrowUpRight,
  Braces,
  Check,
  ChevronDown,
  Copy,
  FileText,
  RotateCcw,
  Table2,
} from "lucide-react";
import type { Citation, Turn } from "../contracts";
import { extractArtifacts, type Artifact } from "../markdown";
const Markdown = lazy(() =>
  import("./Markdown").then((module) => ({ default: module.Markdown })),
);

export function Response({
  turn,
  onCitation,
  onArtifact,
  onRun,
  onRetry,
  onNotice,
}: {
  turn: Turn;
  onCitation: (citation: Citation) => void;
  onArtifact: (artifact: Artifact) => void;
  onRun: () => void;
  onRetry: () => void;
  onNotice: (message: string) => void;
}) {
  const artifacts = useMemo(
    () => extractArtifacts(turn.answer?.answer ?? ""),
    [turn.answer?.answer],
  );
  const pending = turn.status === "pending";
  return (
    <article className="turn" aria-label="Research exchange">
      <div className="question-row">
        <span className="user-avatar">Y</span>
        <div>
          <div className="message-author">You</div>
          <p>{turn.question}</p>
        </div>
      </div>
      <div className="answer-row">
        <div className="answer-avatar">
          <img src="/ledger.svg" alt="" />
        </div>
        <div className="answer-content">
          <div className="answer-byline">
            <span className="message-author">Ledger</span>
            <span className="subtle">Research assistant</span>
            {turn.demo && (
              <span className="example-badge">CURATED EXAMPLE</span>
            )}
          </div>
          {(turn.activities.length > 0 || pending) && (
            <details className="activity-trace" open={pending}>
              <summary>
                <ActivityIcon size={14} />
                <span>
                  {pending
                    ? "Working with your evidence"
                    : "Execution activity"}
                </span>
                {!pending && <span>{turn.activities.length} events</span>}
                <ChevronDown size={14} />
              </summary>
              <ol>
                {!turn.activities.length && (
                  <li>
                    <span className="activity-dot active" />
                    Request sent. Waiting for the service.
                  </li>
                )}
                {turn.activities.map((activity, index) => (
                  <li key={index}>
                    <span
                      className={`activity-dot ${activity.status === "started" && pending ? "active" : ""}`}
                    />
                    {activity.label}
                    <span className="activity-status">{activity.status}</span>
                  </li>
                ))}
              </ol>
              <p>
                Server execution events, not the model&apos;s private reasoning.
              </p>
            </details>
          )}
          {pending && (
            <div className="working" role="status">
              <span className="loading-dots">
                <i />
                <i />
                <i />
              </span>
              Waiting for a citation-checked answer
            </div>
          )}
          {turn.answer && (
            <Suspense
              fallback={
                <p className="subtle">Preparing the formatted answer...</p>
              }
            >
              <Markdown
                text={turn.answer.answer}
                citations={turn.answer.citations}
                onCitation={onCitation}
              />
            </Suspense>
          )}
          {(turn.status === "error" || turn.status === "cancelled") && (
            <div
              className="turn-error"
              role={turn.status === "error" ? "alert" : "status"}
            >
              <strong>
                {turn.status === "cancelled"
                  ? "Response stopped"
                  : "This run could not finish"}
              </strong>
              <p>
                {turn.error ??
                  "No partial response was accepted. You can try this question again."}
              </p>
              <button className="text-button" onClick={onRetry}>
                <RotateCcw size={14} />
                Retry question
              </button>
            </div>
          )}
          {artifacts.length > 0 && (
            <div className="artifact-cards">
              {artifacts.map((artifact) => (
                <button
                  className="artifact-card"
                  key={artifact.id}
                  onClick={() => onArtifact(artifact)}
                >
                  <span className="artifact-icon">
                    {artifact.kind === "table" ? (
                      <Table2 size={20} />
                    ) : (
                      <Braces size={20} />
                    )}
                  </span>
                  <span>
                    <strong>{artifact.title}</strong>
                    <small>
                      {artifact.kind === "table"
                        ? "Data table · inspect & export"
                        : "Read-only code · inspect & export"}
                    </small>
                  </span>
                  <ArrowUpRight size={17} />
                </button>
              ))}
            </div>
          )}
          {turn.answer && (
            <div className="answer-footer">
              <button
                className="evidence-count"
                onClick={() =>
                  turn.answer?.citations[0] &&
                  onCitation(turn.answer.citations[0])
                }
                disabled={!turn.answer.citations.length}
              >
                <FileText size={14} />
                {turn.answer.citations.length} sources
              </button>
              <span
                className={`answer-outcome ${turn.answer.abstained ? "abstained" : ""}`}
              >
                <Check size={13} />
                {turn.demo
                  ? "Fictional data"
                  : turn.answer.abstained
                    ? "Abstained"
                    : "References checked"}
              </span>
              <span className="footer-spacer" />
              <button
                className="icon-button"
                aria-label="Copy answer"
                onClick={async () => {
                  try {
                    await navigator.clipboard.writeText(
                      turn.answer?.answer ?? "",
                    );
                    onNotice("Answer copied.");
                  } catch {
                    onNotice(
                      "Clipboard access was blocked. Select the answer text to copy it.",
                    );
                  }
                }}
              >
                <Copy size={15} />
              </button>
              <button
                className="icon-button"
                onClick={onRun}
                aria-label="Inspect run details"
              >
                <ActivityIcon size={15} />
              </button>
            </div>
          )}
        </div>
      </div>
    </article>
  );
}
