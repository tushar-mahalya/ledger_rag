import { memo, useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  remarkCitations,
  remarkUnsupported,
  safeExternalUrl,
} from "../markdown";
import type { Citation } from "../contracts";

export const Markdown = memo(function Markdown({
  text,
  citations,
  onCitation,
  unsupported,
}: {
  text: string;
  citations: Citation[];
  onCitation: (citation: Citation) => void;
  unsupported?: string[];
}) {
  const key = unsupported?.join("\u0000") ?? "";
  const plugins = useMemo(
    () => [
      remarkGfm,
      remarkCitations,
      remarkUnsupported(key ? key.split("\u0000") : []),
    ],
    [key],
  );
  return (
    <div className="markdown">
      <ReactMarkdown
        remarkPlugins={plugins}
        skipHtml
        urlTransform={(url) =>
          /^#citation-C\d+$/.test(url) ? url : (safeExternalUrl(url) ?? "")
        }
        components={{
          img: ({ alt }) => (
            <span className="blocked-image">
              [Image not loaded{alt ? `: ${alt}` : ""}]
            </span>
          ),
          a: ({ href, children }) => {
            const id = href?.match(/^#citation-(C\d+)$/)?.[1];
            if (id) {
              const citation = citations.find((item) => item.id === id);
              return citation ? (
                <button
                  className="citation-inline"
                  onClick={() => onCitation(citation)}
                  aria-label={`View citation ${id}: ${citation.title}`}
                >
                  {id.slice(1)}
                </button>
              ) : (
                <span
                  className="missing-citation"
                  title="Reference not supplied"
                >
                  {children}
                </span>
              );
            }
            return href ? (
              <a href={href} target="_blank" rel="noopener noreferrer">
                {children}
              </a>
            ) : (
              <span>{children}</span>
            );
          },
          table: ({ children }) => (
            <div
              className="table-scroll"
              role="region"
              aria-label="Answer table"
              tabIndex={0}
            >
              <table>{children}</table>
            </div>
          ),
          mark: ({ children }) => (
            <mark
              className="unsupported-number"
              title="Not found in the cited evidence"
            >
              {children}
              <span className="sr-only">
                {" "}
                (not supported by cited evidence)
              </span>
            </mark>
          ),
          th: ({ children }) => <th>{children}</th>,
          td: ({ children }) => <td>{children}</td>,
          pre: ({ children }) => (
            <div className="code-block">
              <span className="code-label">
                Read-only excerpt · never executed
              </span>
              <pre tabIndex={0}>{children}</pre>
            </div>
          ),
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
});
