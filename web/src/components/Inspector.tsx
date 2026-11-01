import {
  Braces,
  Check,
  Download,
  ExternalLink,
  FileText,
  Fingerprint,
  Layers,
  Table2,
} from "lucide-react";
import { useEffect, useId, useRef } from "react";
import type { Citation, Turn } from "../contracts";
import { highlightSpan } from "../grounding";
import { artifactDownload, safeExternalUrl, type Artifact } from "../markdown";

export type InspectorTab = "sources" | "artifacts" | "run";
export type Selection = {
  turnId: string;
  citationId?: string;
  artifactId?: string;
  quote?: string;
};

function download(content: string, filename: string, type: string) {
  const url = URL.createObjectURL(
    new Blob([content], { type: `${type};charset=utf-8` }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function Inspector({
  turn,
  artifacts,
  tab,
  onTab,
  selection,
  onSelect,
}: {
  turn?: Turn;
  artifacts: Artifact[];
  tab: InspectorTab;
  onTab: (tab: InspectorTab) => void;
  selection?: Selection;
  onSelect: (citation: Citation) => void;
}) {
  const tabs: InspectorTab[] = ["sources", "artifacts", "run"];
  const prefix = useId();
  const citations = turn?.answer?.citations ?? turn?.evidence?.citations ?? [];
  const artifact =
    artifacts.find((item) => item.id === selection?.artifactId) ?? artifacts[0];
  const bodyRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!selection?.citationId || tab !== "sources") return;
    const reduce = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    bodyRef.current
      ?.querySelector(`[data-citation="${selection.citationId}"]`)
      ?.scrollIntoView({
        behavior: reduce ? "auto" : "smooth",
        block: "nearest",
      });
  }, [selection?.citationId, selection?.quote, tab]);
  return (
    <div className="inspector-content">
      <div className="inspector-heading">
        <Layers size={17} />
        <h2 data-inspector-heading tabIndex={-1}>
          Research context
        </h2>
        <span className="tiny-badge">
          {turn?.demo ? "PREVIEW" : "THIS TURN"}
        </span>
      </div>
      <div
        className="inspector-tabs"
        role="tablist"
        aria-label="Research context"
      >
        {tabs.map((item, index) => (
          <button
            key={item}
            role="tab"
            id={`${prefix}-tab-${item}`}
            aria-selected={tab === item}
            tabIndex={tab === item ? 0 : -1}
            aria-controls={`${prefix}-panel-${item}`}
            onClick={() => onTab(item)}
            onKeyDown={(event) => {
              const offset =
                event.key === "ArrowRight"
                  ? 1
                  : event.key === "ArrowLeft"
                    ? -1
                    : 0;
              if (!offset && !["Home", "End"].includes(event.key)) return;
              event.preventDefault();
              const next =
                event.key === "Home"
                  ? 0
                  : event.key === "End"
                    ? 2
                    : (index + offset + 3) % 3;
              onTab(tabs[next]);
              const buttons =
                event.currentTarget.parentElement?.querySelectorAll<HTMLButtonElement>(
                  '[role="tab"]',
                );
              buttons?.[next].focus();
            }}
          >
            {item === "run"
              ? "Run details"
              : item === "sources"
                ? "Sources"
                : "Artifacts"}
            {item === "sources" && <span>{citations.length}</span>}
            {item === "artifacts" && <span>{artifacts.length}</span>}
          </button>
        ))}
      </div>
      <div
        className="inspector-body"
        ref={bodyRef}
        role="tabpanel"
        id={`${prefix}-panel-${tab}`}
        aria-labelledby={`${prefix}-tab-${tab}`}
      >
        {tab === "sources" && (
          <>
            <div className="section-eyebrow">
              THE EVIDENCE BEHIND THE ANSWER
            </div>
            <p className="context-intro">
              Read the original excerpt. Keep the reporting period and
              publication date in view.
            </p>
            {!citations.length && (
              <Empty
                title="Evidence will appear here"
                text="Select a cited answer, or wait for authorized retrieval to complete."
              />
            )}
            {citations.map((citation) => {
              const selected =
                selection?.citationId === citation.id ||
                (!selection?.citationId && citation === citations[0]);
              const href = safeExternalUrl(citation.source);
              return (
                <section
                  key={citation.id}
                  data-citation={citation.id}
                  className={`source-card ${selected ? "selected" : ""}`}
                >
                  <button
                    className="source-heading"
                    onClick={() => onSelect(citation)}
                    aria-expanded={selected}
                  >
                    <span className="source-number">
                      {citation.id.slice(1)}
                    </span>
                    <span>
                      <strong>{citation.title}</strong>
                      <small>Published {citation.published_at}</small>
                      {citation.source_kind &&
                        citation.source_kind !== "corpus" && (
                          <span className="tiny-badge">
                            {citation.source_kind === "sec_xbrl"
                              ? "SEC XBRL"
                              : "FRED / ALFRED"}
                          </span>
                        )}
                    </span>
                    <FileText size={17} />
                  </button>
                  {selected && (
                    <div className="source-content">
                      <span className="source-type">
                        {citation.provenance
                          ? turn?.demo
                            ? "RECORDED EXTERNAL EVIDENCE"
                            : "OFFICIAL EXTERNAL EVIDENCE"
                          : turn?.demo ||
                              citation.source.startsWith(
                                "urn:ledger:synthetic:",
                              )
                            ? "FICTIONAL DISCLOSURE"
                            : "RETRIEVED EXCERPT"}
                      </span>
                      <blockquote>
                        {highlightSpan(
                          citation.quote,
                          selection?.citationId === citation.id
                            ? selection.quote
                            : undefined,
                        ).map((segment, index) =>
                          segment.unsupported ? (
                            <mark key={index} className="operand-highlight">
                              {segment.text}
                            </mark>
                          ) : (
                            segment.text
                          ),
                        )}
                      </blockquote>
                      {citation.provenance && (
                        <dl
                          className="provenance-panel"
                          aria-label="Evidence provenance"
                        >
                          {Object.entries({
                            Provider: citation.provenance.provider,
                            Entity: citation.provenance.entity,
                            Concept: citation.provenance.concept,
                            Identifier: citation.provenance.identifier,
                            Form: citation.provenance.form,
                            Period: citation.provenance.period_label,
                            Value: citation.provenance.value,
                            Unit: citation.provenance.unit,
                            "Base-unit scale": `10^${citation.provenance.scale}`,
                            "As of": citation.provenance.as_of,
                            "Filed / vintage": citation.provenance.published_at,
                            Retrieved: citation.provenance.retrieved_at,
                            Freshness: citation.provenance.freshness,
                            "Later revision": citation.provenance.superseded_by,
                            "Restated before cutoff": citation.provenance
                              .revised_before_as_of
                              ? "Yes"
                              : "No",
                          })
                            .filter(([, value]) => value != null)
                            .map(([label, value]) => (
                              <div key={label}>
                                <dt>{label}</dt>
                                <dd>{value}</dd>
                              </div>
                            ))}
                        </dl>
                      )}
                      <div className="source-identifier">
                        <span>Document ID</span>
                        <code>{citation.document_id}</code>
                      </div>
                      {href ? (
                        <a
                          className="source-link"
                          href={href}
                          target="_blank"
                          rel="noopener noreferrer"
                        >
                          Open original source <ExternalLink size={13} />
                        </a>
                      ) : (
                        <div className="source-identifier">
                          <span>Source identifier</span>
                          <code>{citation.source}</code>
                        </div>
                      )}
                    </div>
                  )}
                </section>
              );
            })}
            <div className="context-note">
              <Check size={15} />
              <p>
                Citation checks validate references, not semantic truth. Review
                source evidence before relying on an answer.
              </p>
            </div>
          </>
        )}
        {tab === "artifacts" && (
          <>
            <div className="section-eyebrow">A CLOSER LOOK</div>
            <p className="context-intro">
              Tables and code extracted from this answer. No generated code is
              executed.
            </p>
            {!artifact && (
              <Empty
                title="No artifacts in this answer"
                text="Tables and fenced code blocks become inspectable, downloadable artifacts."
              />
            )}
            {artifacts.map((item) => (
              <a
                className="artifact-jump"
                key={item.id}
                href={`#${item.id}`}
                onClick={(event) => {
                  event.preventDefault();
                  document
                    .getElementById(`${prefix}-detail-${item.id}`)
                    ?.scrollIntoView({ behavior: "auto", block: "start" });
                }}
              >
                {item.kind === "table" ? (
                  <Table2 size={16} />
                ) : (
                  <Braces size={16} />
                )}
                {item.title}
              </a>
            ))}
            {artifacts.map((item) => (
              <section
                id={`${prefix}-detail-${item.id}`}
                key={item.id}
                className={`artifact-detail ${item.id === artifact?.id ? "selected" : ""}`}
              >
                <div className="artifact-title">
                  <strong>{item.title}</strong>
                  <button
                    className="icon-button"
                    aria-label={`Download ${item.title}`}
                    onClick={() => {
                      const result = artifactDownload(item);
                      download(
                        result.content,
                        `ledger-${item.id}.${result.extension}`,
                        result.type,
                      );
                    }}
                  >
                    <Download size={16} />
                  </button>
                </div>
                {item.kind === "table" ? (
                  <div
                    className="table-scroll"
                    tabIndex={0}
                    role="region"
                    aria-label="Artifact table"
                  >
                    <table>
                      <thead>
                        <tr>
                          {item.rows[0]?.map((cell, i) => (
                            <th key={i}>{cell}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {item.rows.slice(1).map((row, i) => (
                          <tr key={i}>
                            {row.map((cell, j) => (
                              <td key={j}>{cell}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <pre tabIndex={0}>
                    <code>{item.content}</code>
                  </pre>
                )}
              </section>
            ))}
          </>
        )}
        {tab === "run" && (
          <>
            <div className="section-eyebrow">OBSERVABLE, NOT OPAQUE</div>
            <p className="context-intro">
              Measured execution metadata, without private reasoning or hidden
              confidence scores.
            </p>
            {turn?.demo ? (
              <Empty
                title="Design preview, not a benchmark"
                text="No model, retrieval or tool was run for this example. Connect the API to see actual request IDs and timings."
              />
            ) : (
              turn && (
                <>
                  <dl className="run-metrics">
                    <dt>State</dt>
                    <dd>{turn.status}</dd>
                    <dt>Request ID</dt>
                    <dd className="mono">
                      {turn.answer?.request_id ??
                        turn.evidence?.request_id ??
                        "Not received"}
                    </dd>
                    <dt>Model</dt>
                    <dd>{turn.answer?.model ?? "Not received"}</dd>
                    <dt>Retrieval</dt>
                    <dd>
                      {formatMs(
                        turn.answer?.retrieval_ms ??
                          turn.evidence?.retrieval_ms,
                      )}
                    </dd>
                    <dt>Generation + tools</dt>
                    <dd>{formatMs(turn.answer?.generation_ms)}</dd>
                    <dt>Browser round trip</dt>
                    <dd>{formatMs(turn.roundTripMs)}</dd>
                    <dt>Tool calls</dt>
                    <dd>{turn.answer?.tool_calls ?? "Not received"}</dd>
                    <dt>Outcome</dt>
                    <dd>
                      {turn.answer
                        ? turn.answer.abstained
                          ? "Abstained"
                          : "Cited answer"
                        : "No final answer"}
                    </dd>
                    <dt>Verification</dt>
                    <dd>
                      {turn.answer?.verification
                        ? `${turn.answer.verification.status} · ${turn.answer.verification.mode} mode`
                        : "Not received"}
                    </dd>
                    <dt>Calculations</dt>
                    <dd>
                      {turn.answer?.calculations.length ?? "Not received"}
                    </dd>
                    {turn.answer?.abstention && (
                      <>
                        <dt>Abstention policy</dt>
                        <dd>
                          {turn.answer.abstention.calibrated
                            ? "Calibrated"
                            : "Uncalibrated thresholds"}
                        </dd>
                      </>
                    )}
                  </dl>
                  <p className="context-intro">
                    Timings describe this request, not p95 latency or a
                    production SLO. Request IDs correlate with server logs. No
                    prompt content is exported by this interface.
                  </p>
                </>
              )
            )}
            {!turn && (
              <Empty
                title="No run selected"
                text="Start a query to inspect its execution metadata."
              />
            )}
          </>
        )}
      </div>
      <div className="inspector-footer">
        <Fingerprint size={14} />
        <span>Evidence first. Always.</span>
      </div>
    </div>
  );
}

function formatMs(ms: number | undefined): string {
  if (ms === undefined) return "Not received";
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
}

function Empty({ title, text }: { title: string; text: string }) {
  return (
    <div className="context-empty">
      <FileText size={24} />
      <h3>{title}</h3>
      <p>{text}</p>
    </div>
  );
}
