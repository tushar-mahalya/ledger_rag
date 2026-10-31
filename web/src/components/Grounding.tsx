import { useEffect, useId, useRef, useState } from "react";
import {
  AlertTriangle,
  Calculator,
  ChevronDown,
  CircleSlash,
  CornerDownRight,
  Info,
  ShieldCheck,
  ShieldMinus,
  ShieldX,
} from "lucide-react";
import type {
  AbstentionDecision,
  Calculation,
  Citation,
  Verification,
} from "../contracts";
import {
  describeReason,
  finalResult,
  formatDerivation,
  formatValue,
  presentations,
  stepRef,
  unitLabel,
} from "../grounding";

type OnCitation = (citation: Citation, quote?: string) => void;

export function CalculationTrace({
  calculations,
  citations,
  onCitation,
}: {
  calculations: Calculation[];
  citations: Citation[];
  onCitation: OnCitation;
}) {
  const prefix = useId();
  const last = calculations.at(-1);
  if (!last?.steps.length) return null;
  const result = finalResult(last);
  const steps = calculations.reduce((sum, item) => sum + item.steps.length, 0);
  return (
    <details className="calc-trace">
      <summary>
        <Calculator size={14} />
        <span>Calculation trace</span>
        <span className="calc-summary">
          {steps} {steps === 1 ? "step" : "steps"} · ={" "}
          <strong>{result.text}</strong>
        </span>
        <ChevronDown size={14} />
      </summary>
      <div className="calc-body">
        {calculations.map((calculation, program) => {
          const shown = presentations(calculation);
          const final = finalResult(calculation);
          const id = (index: number) => `${prefix}-p${program}-s${index}`;
          return (
            <section
              key={program}
              className="calc-program"
              aria-label={`Program ${program + 1} of ${calculations.length}`}
            >
              {calculations.length > 1 && (
                <div className="calc-program-title">
                  Program {program + 1}
                  {program === calculations.length - 1 && " · final"}
                </div>
              )}
              <ol className="calc-steps">
                {calculation.steps.map((step, index) => (
                  <li key={step.index} id={id(index)} tabIndex={-1}>
                    <span
                      className="calc-index"
                      title={formatDerivation(calculation, index)}
                    >
                      <span className="sr-only">Step </span>
                      {stepRef(step.index)}
                    </span>
                    <span className="calc-line">
                      <span className="calc-expression">
                        <code className="calc-op">{step.op}</code>
                        <span className="calc-paren">(</span>
                        {step.operands.map((operand, position) => (
                          <span className="calc-arg" key={position}>
                            {operand.kind === "evidence" ? (
                              <EvidenceChip
                                quote={operand.quote ?? operand.value}
                                citation={citations.find(
                                  (item) => item.id === operand.citation_id,
                                )}
                                citationId={operand.citation_id}
                                onCitation={onCitation}
                              />
                            ) : operand.kind === "step" ? (
                              <button
                                type="button"
                                className="operand-chip step"
                                aria-label={`Result of step ${stepRef(operand.step ?? 0)}: ${formatValue(
                                  calculation.steps[operand.step ?? 0]
                                    ?.result ?? operand.value,
                                  calculation.steps[operand.step ?? 0]?.kind,
                                  shown[operand.step ?? 0],
                                )}. Go to step`}
                                onClick={() => {
                                  const target = document.getElementById(
                                    id(operand.step ?? 0),
                                  );
                                  target?.focus();
                                  target?.classList.remove("flash");
                                  void target?.offsetWidth;
                                  target?.classList.add("flash");
                                }}
                              >
                                <CornerDownRight size={11} />
                                {stepRef(operand.step ?? 0)}
                                <span className="chip-value">
                                  {formatValue(
                                    calculation.steps[operand.step ?? 0]
                                      ?.result ?? operand.value,
                                    calculation.steps[operand.step ?? 0]?.kind,
                                    shown[operand.step ?? 0],
                                  )}
                                </span>
                              </button>
                            ) : (
                              <span
                                className="operand-chip constant"
                                title="Allow-listed constant, not evidence"
                              >
                                <span className="sr-only">constant </span>
                                {operand.value}
                              </span>
                            )}
                            {position < step.operands.length - 1 && (
                              <span className="calc-comma">,</span>
                            )}
                          </span>
                        ))}
                        <span className="calc-paren">)</span>
                      </span>
                      <span className="calc-outcome">
                        <span className="calc-equals">=</span>
                        <strong className="calc-result">
                          {formatValue(step.result, step.kind, shown[index])}
                        </strong>
                        <span className="calc-kind">
                          {unitLabel(step.kind)}
                        </span>
                      </span>
                    </span>
                  </li>
                ))}
              </ol>
              <div className="calc-final">
                <span>Result</span>
                <strong>{final.text}</strong>
                <span className="calc-unit">{unitLabel(final.kind)}</span>
                <span className="calc-exact" title="Exact backend value">
                  exact {calculation.result}
                </span>
              </div>
            </section>
          );
        })}
        <p className="calc-note">
          Inputs are exact spans of cited evidence. Constants are limited to
          scaling factors. Amounts are shown in the evidence scale.
        </p>
      </div>
    </details>
  );
}

function EvidenceChip({
  quote,
  citation,
  citationId,
  onCitation,
}: {
  quote: string;
  citation?: Citation;
  citationId?: string | null;
  onCitation: OnCitation;
}) {
  const content = (
    <>
      <span className="chip-quote">“{quote}”</span>
      <span className="chip-cite">{citationId ?? "?"}</span>
    </>
  );
  if (!citation)
    return (
      <span
        className="operand-chip evidence missing"
        title="This source was not returned with the answer"
      >
        {content}
      </span>
    );
  return (
    <button
      type="button"
      className="operand-chip evidence"
      aria-label={`Evidence “${quote}” from citation ${citation.id}: ${citation.title}. Show source`}
      onClick={() => onCitation(citation, quote)}
    >
      {content}
    </button>
  );
}

const STATUS_TEXT = {
  verified: "Verified",
  unverified: "Unverified",
  failed: "Failed",
  skipped: "Skipped",
} as const;

export function VerificationBadge({
  verification,
}: {
  verification: Verification;
}) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLSpanElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const id = useId();
  const { status, unsupported_numbers: unsupported } = verification;
  const score = verification.verifier_score;
  useEffect(() => {
    if (!open) return;
    const close = (event: MouseEvent | FocusEvent) => {
      if (!wrapRef.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    document.addEventListener("focusin", close);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("focusin", close);
    };
  }, [open]);
  const Icon =
    status === "verified"
      ? ShieldCheck
      : status === "unverified"
        ? AlertTriangle
        : status === "failed"
          ? ShieldX
          : ShieldMinus;
  const summary = `Numeric verification: ${STATUS_TEXT[status].toLowerCase()}. ${verification.checked_numbers} ${
    verification.checked_numbers === 1 ? "number" : "numbers"
  } checked, ${unsupported.length || "none"} unsupported.${
    score != null ? ` Groundedness score ${score.toFixed(2)}.` : ""
  }`;
  return (
    <span
      className="verify-wrap"
      ref={wrapRef}
      onKeyDown={(event) => {
        if (event.key === "Escape" && open) {
          event.stopPropagation();
          setOpen(false);
          buttonRef.current?.focus();
        }
      }}
    >
      <button
        ref={buttonRef}
        type="button"
        className={`verify-pill ${status}`}
        aria-expanded={open}
        aria-controls={id}
        aria-label={`${summary} Show details`}
        onClick={() => setOpen((value) => !value)}
      >
        <Icon size={12} />
        {STATUS_TEXT[status]}
        {score != null && (
          <span className="verify-score">{score.toFixed(2)}</span>
        )}
      </button>
      {open && (
        <div
          className="verify-popover"
          id={id}
          role="group"
          aria-label="Verification details"
        >
          <dl>
            <dt>Status</dt>
            <dd>{STATUS_TEXT[status]}</dd>
            <dt>Mode</dt>
            <dd>{verification.mode}</dd>
            <dt>Numbers checked</dt>
            <dd>{verification.checked_numbers}</dd>
            {score != null && (
              <>
                <dt>Groundedness</dt>
                <dd>
                  {score.toFixed(2)}
                  {verification.verifier && ` · ${verification.verifier}`}
                </dd>
              </>
            )}
          </dl>
          <div className="verify-section">
            <span>Unsupported numbers</span>
            {unsupported.length ? (
              <ul className="verify-numbers">
                {unsupported.map((item) => (
                  <li key={item}>
                    <mark className="unsupported-number">{item}</mark>
                  </li>
                ))}
              </ul>
            ) : (
              <p>None. Every checked number matched cited evidence.</p>
            )}
          </div>
          {verification.reasons.length > 0 && (
            <ul className="verify-reasons">
              {verification.reasons.map((reason) => (
                <li key={reason}>{describeReason(reason)}</li>
              ))}
            </ul>
          )}
          <p className="verify-note">
            Checks numbers against cited evidence and bound calculations. It
            does not prove every statement is true.
          </p>
        </div>
      )}
    </span>
  );
}

export function AbstentionNotice({
  message,
  abstention,
}: {
  message: string;
  abstention: AbstentionDecision;
}) {
  const id = useId();
  return (
    <section className="abstain-card" aria-labelledby={id}>
      <div className="abstain-heading">
        <span className="abstain-icon">
          <CircleSlash size={16} />
        </span>
        <h3 id={id}>Ledger held back an answer</h3>
      </div>
      <p className="abstain-message">{message}</p>
      {abstention.reasons.length > 0 && (
        <ul
          className="abstain-reasons"
          aria-label="Why the answer was withheld"
        >
          {abstention.reasons.map((reason) => (
            <li key={reason}>{describeReason(reason)}</li>
          ))}
        </ul>
      )}
      {!abstention.calibrated && (
        <p className="calibration-note">
          <Info size={13} />
          <span>
            <strong>Thresholds uncalibrated.</strong> These abstention rules are
            configured defaults, not yet fitted to measured error rates.
          </span>
        </p>
      )}
      <div className="rephrase">
        <span className="section-eyebrow">TRY REPHRASING</span>
        <ul>
          <li>Name the company and the fiscal period you mean.</li>
          <li>
            Ask for a figure a disclosure reports, not a forecast or
            recommendation.
          </li>
          <li>Split a multi-part question into smaller ones.</li>
        </ul>
      </div>
    </section>
  );
}
