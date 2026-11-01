import { lazy, Suspense, useMemo } from "react";
import {
  Activity as ActivityIcon,
  ArrowUpRight,
  Braces,
  Calculator,
  Check,
  ChevronDown,
  Compass,
  Copy,
  FileSearch,
  FileText,
  PenLine,
  Quote,
  RotateCcw,
  ShieldCheck,
  Table2,
  Wrench,
  type LucideIcon,
} from "lucide-react";
import type { ActivityKind, Citation, PlanSummary, Turn } from "../contracts";
import { describePlan, timeline } from "../grounding";
import { extractArtifacts, type Artifact } from "../markdown";
import {
  AbstentionNotice,
  CalculationTrace,
  VerificationBadge,
} from "./Grounding";
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
  onCitation: (citation: Citation, quote?: string) => void;
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
  const rows = useMemo(() => timeline(turn.activities), [turn.activities]);
  const answer = turn.answer;
  const verification = answer?.verification;
  // Inline marks exist only in flag mode: enforce failures abstain and send no draft.
  const unsupported =
    answer && !answer.abstained && verification?.status !== "verified"
      ? (verification?.unsupported_numbers ?? [])
      : [];
  const calculations = answer?.calculations.length
    ? answer.calculations
    : answer?.calculation
      ? [answer.calculation]
      : [];
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
                    : turn.demo
                      ? "Example activity · not executed"
                      : "Execution activity"}
                </span>
                {!pending && (
                  <span>
                    {rows.length} {rows.length === 1 ? "step" : "steps"}
                  </span>
                )}
                <ChevronDown size={14} />
              </summary>
              <ol>
                {!turn.activities.length && (
                  <li>
                    <span className="activity-dot active" />
                    Request sent. Waiting for the service.
                  </li>
                )}
                {rows.map((row) => {
                  const running = !row.done && pending;
                  return (
                    <li
                      key={row.key}
                      className={`timeline-row ${row.stage} ${row.kind} ${row.plan ? "has-plan" : ""} ${running ? "running" : ""}`}
                    >
                      <span className="timeline-icon">
                        <KindIcon kind={row.kind} />
                      </span>
                      <div className="timeline-body">
                        <span>{row.label}</span>
                        {row.plan && <PlanFacts plan={row.plan} />}
                      </div>
                      <span className="activity-status">
                        {row.done ? (
                          <>
                            {(!row.outcome || row.outcome === "ok") && (
                              <Check size={11} />
                            )}
                            {row.outcome && row.outcome !== "ok"
                              ? row.outcome
                              : "done"}
                          </>
                        ) : running ? (
                          "in progress"
                        ) : (
                          "not completed"
                        )}
                      </span>
                    </li>
                  );
                })}
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
          {answer?.abstained && answer.abstention ? (
            <AbstentionNotice
              message={answer.answer}
              abstention={answer.abstention}
            />
          ) : (
            answer && (
              <Suspense
                fallback={
                  <p className="subtle">Preparing the formatted answer...</p>
                }
              >
                <Markdown
                  text={answer.answer}
                  citations={answer.citations}
                  onCitation={onCitation}
                  unsupported={unsupported}
                />
              </Suspense>
            )
          )}
          {answer && !answer.abstained && calculations.length > 0 && (
            <CalculationTrace
              calculations={calculations}
              citations={answer.citations}
              onCitation={onCitation}
            />
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
              {verification && (
                <VerificationBadge
                  verification={verification}
                  unsupportedCount={
                    turn.answer.abstention?.features.unsupported_number_count
                  }
                />
              )}
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

const KIND_ICONS: Record<ActivityKind, LucideIcon> = {
  planning: Compass,
  retrieval: FileSearch,
  generation: PenLine,
  calculator: Calculator,
  evidence_search: FileSearch,
  company_facts_lookup: FileText,
  macro_series_lookup: Table2,
  tool_check: Wrench,
  citation_validation: Quote,
  grounding_verification: ShieldCheck,
};

function KindIcon({ kind }: { kind: ActivityKind }) {
  const Icon = KIND_ICONS[kind];
  return <Icon size={12} />;
}

function PlanFacts({ plan }: { plan: PlanSummary }) {
  return (
    <div className="plan-facts" role="group" aria-label="Query plan">
      <dl>
        {describePlan(plan).map((fact) => (
          <div key={fact.term}>
            <dt>{fact.term}</dt>
            {fact.values.map((value, index) => (
              <dd key={index}>{value}</dd>
            ))}
          </div>
        ))}
      </dl>
    </div>
  );
}
