import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
} from "react";
import {
  ArrowDown,
  ArrowRight,
  ArrowUp,
  BookOpen,
  Check,
  ChevronDown,
  CircleHelp,
  FileText,
  FlaskConical,
  History,
  Layers,
  Link2,
  LockKeyhole,
  Menu,
  MessageSquare,
  Plus,
  Search,
  Settings2,
  ShieldCheck,
  Square,
  Trash2,
  X,
} from "lucide-react";
import { ChatError } from "./errors";
import {
  boundedHistory,
  type Citation,
  type Conversation,
  type Turn,
} from "./contracts";
import { demoTurn, exampleQuestions, initialConversation } from "./demo";
import { extractArtifacts, type Artifact } from "./markdown";
import {
  Inspector,
  type InspectorTab,
  type Selection,
} from "./components/Inspector";
import { Modal } from "./components/Modal";
import { Response } from "./components/Response";

const MAX_CONVERSATIONS = 20;
const MAX_TURNS = 40;

export default function App() {
  const [conversations, setConversations] = useState<Conversation[]>(() => [
    initialConversation(),
  ]);
  const [activeId, setActiveId] = useState<string | undefined>();
  const [question, setQuestion] = useState("");
  const [filter, setFilter] = useState("");
  const [token, setToken] = useState("");
  const [asOf, setAsOf] = useState("");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [helpOpen, setHelpOpen] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [contextOpen, setContextOpen] = useState(false);
  const [selection, setSelection] = useState<Selection>();
  const [tab, setTab] = useState<InspectorTab>("sources");
  const [nearBottom, setNearBottom] = useState(true);
  const abortRef = useRef<AbortController | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const composerRef = useRef<HTMLTextAreaElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const active =
    conversations.find((item) => item.id === activeId) ?? conversations[0];
  const priorTurnsRef = useRef(active.turns);
  const selectedTurn =
    active.turns.find((turn) => turn.id === selection?.turnId) ??
    active.turns.at(-1);
  const artifacts = useMemo(
    () => extractArtifacts(selectedTurn?.answer?.answer ?? ""),
    [selectedTurn?.answer?.answer],
  );
  const mode = token ? "live" : "preview";

  useEffect(() => () => abortRef.current?.abort(), []);
  useEffect(() => {
    if (!notice) return;
    const timeout = setTimeout(() => setNotice(""), 6500);
    return () => clearTimeout(timeout);
  }, [notice]);
  useEffect(() => {
    const handle = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        if (window.matchMedia("(max-width: 760px)").matches) setMenuOpen(true);
        else searchRef.current?.focus();
      }
    };
    document.addEventListener("keydown", handle);
    return () => document.removeEventListener("keydown", handle);
  }, []);
  useEffect(() => {
    if (nearBottom && priorTurnsRef.current !== active.turns) {
      scrollRef.current?.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: "auto",
      });
    }
    priorTurnsRef.current = active.turns;
  }, [active.turns, nearBottom]);

  function stop() {
    abortRef.current?.abort();
  }
  function resetWorkspace(nextToken: string) {
    stop();
    setToken(nextToken);
    const next: Conversation = {
      id: crypto.randomUUID(),
      title: "New research",
      turns: [],
    };
    setConversations([next]);
    setActiveId(next.id);
    setSelection(undefined);
    setQuestion("");
    setAsOf("");
    setSettingsOpen(false);
    setNotice(
      nextToken
        ? "Live connection configured. Your token stays in this tab's memory."
        : "Session cleared. You are in preview mode.",
    );
  }
  function newConversation() {
    if (conversations.length >= MAX_CONVERSATIONS) {
      setNotice(
        "This session has 20 conversations. Clear the session in connection settings to begin again.",
      );
      return;
    }
    stop();
    const next: Conversation = {
      id: crypto.randomUUID(),
      title: "New research",
      turns: [],
    };
    setConversations((items) => [next, ...items]);
    setActiveId(next.id);
    setSelection(undefined);
    setQuestion("");
    setMenuOpen(false);
    setNearBottom(true);
    composerRef.current?.focus();
  }
  function inspect(
    nextTab: InspectorTab,
    turn: Turn,
    citation?: Citation,
    artifact?: Artifact,
  ) {
    setSelection({
      turnId: turn.id,
      citationId: citation?.id,
      artifactId: artifact?.id,
    });
    setTab(nextTab);
    if (window.matchMedia("(max-width: 1120px)").matches) setContextOpen(true);
    else
      document
        .querySelector<HTMLElement>("[data-inspector-heading]")
        ?.focus({ preventScroll: true });
  }
  const submit = useCallback(
    async (event?: FormEvent, proposed?: string) => {
      event?.preventDefault();
      const text = (proposed ?? question).trim();
      if (!text || busy || abortRef.current) return;
      if (text.length > 4000) {
        setNotice("Questions must contain at most 4,000 characters.");
        return;
      }
      if (active.turns.length >= MAX_TURNS) {
        setNotice("This chat has reached 40 turns. Start a new research chat.");
        return;
      }
      const turn: Turn = token
        ? {
            id: crypto.randomUUID(),
            question: text,
            status: "pending",
            activities: [],
          }
        : demoTurn(text);
      const conversationId = active.id;
      const history = boundedHistory(active.turns);
      function update(patch: Partial<Turn> | ((turn: Turn) => Partial<Turn>)) {
        setConversations((items) =>
          items.map((item) =>
            item.id !== conversationId
              ? item
              : {
                  ...item,
                  turns: item.turns.map((current) =>
                    current.id !== turn.id
                      ? current
                      : {
                          ...current,
                          ...(typeof patch === "function"
                            ? patch(current)
                            : patch),
                        },
                  ),
                },
          ),
        );
      }
      setConversations((items) =>
        items.map((item) =>
          item.id !== conversationId
            ? item
            : {
                ...item,
                title: item.turns.length ? item.title : text.slice(0, 55),
                turns: [...item.turns, turn],
              },
        ),
      );
      setQuestion("");
      setSelection({ turnId: turn.id });
      setNearBottom(true);
      if (!token) return;
      const controller = new AbortController();
      abortRef.current = controller;
      setBusy(true);
      const timeout = setTimeout(() => controller.abort("timeout"), 120_000);
      const start = performance.now();
      try {
        const { streamAnswer } = await import("./api");
        const answer = await streamAnswer({
          token,
          question: text,
          asOf,
          history,
          signal: controller.signal,
          onActivity: (activity) =>
            update((current) => ({
              activities: [...current.activities, activity].slice(-128),
            })),
          onEvidence: (evidence) => update({ evidence }),
        });
        if (controller.signal.aborted)
          throw new DOMException("Stopped", "AbortError");
        update({
          answer,
          status: "complete",
          roundTripMs: performance.now() - start,
        });
      } catch (error) {
        const cancelled =
          controller.signal.aborted && controller.signal.reason !== "timeout";
        const message =
          controller.signal.reason === "timeout"
            ? "The browser's 120-second deadline was exceeded. No answer was accepted."
            : cancelled
              ? undefined
              : error instanceof ChatError
                ? `${error.message}${error.requestId ? ` Request: ${error.requestId}` : ""}`
                : "The connection failed. Check the API connection and try again.";
        update({
          status: cancelled ? "cancelled" : "error",
          error: message,
          roundTripMs: performance.now() - start,
        });
      } finally {
        clearTimeout(timeout);
        if (abortRef.current === controller) {
          abortRef.current = null;
          setBusy(false);
        }
      }
    },
    [active, asOf, busy, question, token],
  );

  const sidebar = (
    <>
      <a className="brand" href="/" aria-label="Ledger home">
        <img src="/ledger.svg" alt="" />
        <span>
          ledger<span className="brand-period">.</span>
        </span>
        <span className="brand-tag">RESEARCH</span>
      </a>
      <button className="new-chat" onClick={newConversation}>
        <Plus size={17} />
        New research
      </button>
      <label className="search-box">
        <Search size={15} />
        <input
          ref={searchRef}
          aria-label="Search conversations"
          placeholder="Search this session"
          value={filter}
          onChange={(event) => setFilter(event.target.value)}
        />
        <kbd>⌘ K</kbd>
      </label>
      <nav className="workspace-nav" aria-label="Workspace">
        <button
          className="active"
          onClick={() => {
            setMenuOpen(false);
            composerRef.current?.focus();
          }}
        >
          <MessageSquare size={17} />
          Research workspace
          <span className="nav-dot" />
        </button>
        <button
          onClick={() => {
            setMenuOpen(false);
            if (selectedTurn) inspect("sources", selectedTurn);
            else setNotice("Start a research question to collect sources.");
          }}
        >
          <BookOpen size={17} />
          Source library
          <span className="nav-count">
            {selectedTurn?.answer?.citations.length ?? 0}
          </span>
        </button>
      </nav>
      <div className="sidebar-label">
        THIS SESSION <History size={13} />
      </div>
      <div className="conversation-list">
        {conversations
          .filter((item) =>
            item.title.toLowerCase().includes(filter.toLowerCase()),
          )
          .map((item) => (
            <button
              className={`conversation-link ${item.id === active.id ? "selected" : ""}`}
              key={item.id}
              onClick={() => {
                stop();
                setActiveId(item.id);
                setSelection(undefined);
                setMenuOpen(false);
                setNearBottom(true);
              }}
              aria-current={item.id === active.id ? "page" : undefined}
            >
              <MessageSquare size={14} />
              <span>{item.title}</span>
            </button>
          ))}
        {!conversations.some((item) =>
          item.title.toLowerCase().includes(filter.toLowerCase()),
        ) && <p className="empty-search">No conversations match.</p>}
      </div>
      <div className="sidebar-bottom">
        <div className="workspace-note">
          <ShieldCheck size={18} />
          <strong>Built on evidence</strong>
          <p>
            Every answer deserves a source.
            <br />
            Every source deserves a closer look.
          </p>
        </div>
        <button className="sidebar-help" onClick={() => setHelpOpen(true)}>
          <CircleHelp size={16} />
          How Ledger works
          <ArrowUpRightIcon />
        </button>
        <button
          className="workspace-account"
          onClick={() => setSettingsOpen(true)}
        >
          <span className="account-avatar">L</span>
          <span>
            <strong>{token ? "Connected workspace" : "Demo workspace"}</strong>
            <small>
              {token ? "Token held in memory" : "Fictional disclosures"}
            </small>
          </span>
          <Settings2 size={16} />
        </button>
      </div>
    </>
  );
  const inspector = (
    <Inspector
      turn={selectedTurn}
      artifacts={artifacts}
      tab={tab}
      onTab={setTab}
      selection={selection}
      onSelect={(citation) =>
        selectedTurn &&
        setSelection({ turnId: selectedTurn.id, citationId: citation.id })
      }
    />
  );

  return (
    <div className="workspace">
      <a className="skip-link" href="#composer">
        Skip to question
      </a>
      <aside className="sidebar">{sidebar}</aside>
      <div className="workspace-main">
        <header className="topbar">
          <button
            className="icon-button mobile-menu"
            aria-label="Open navigation"
            onClick={() => setMenuOpen(true)}
          >
            <Menu size={20} />
          </button>
          <div className="breadcrumb">
            <span>Workspace</span>
            <span>/</span>
            <strong>Financial research</strong>
          </div>
          <span className="topbar-space" />
          <button
            className={`connection-status ${token ? "live" : ""}`}
            onClick={() => setSettingsOpen(true)}
          >
            <span />
            {token ? "API configured" : "Design preview"}
            <ChevronDown size={13} />
          </button>
          <button
            className="icon-button context-toggle"
            aria-label="Open research context"
            onClick={() => setContextOpen(true)}
          >
            <Layers size={18} />
          </button>
        </header>
        <div className="research-layout">
          <main className="chat-column">
            <div className="chat-heading">
              <div>
                <span className="section-eyebrow">THE RESEARCH DESK</span>
                <h1>
                  {active.turns.length
                    ? "A clearer view of the numbers."
                    : "Good questions. Grounded answers."}
                </h1>
              </div>
              <span className="private-label">
                <LockKeyhole size={12} />
                Tab-only session
              </span>
            </div>
            <div
              className="conversation-scroll"
              ref={scrollRef}
              onScroll={(event) => {
                const el = event.currentTarget;
                setNearBottom(
                  el.scrollHeight - el.scrollTop - el.clientHeight < 100,
                );
              }}
            >
              {mode === "preview" && (
                <div className="preview-banner">
                  <FlaskConical size={16} />
                  <p>
                    <strong>A working design preview.</strong> Fictional
                    filings, curated answers. No live model or benchmark
                    results.
                  </p>
                  <button onClick={() => setSettingsOpen(true)}>
                    Connect API <ArrowRight size={14} />
                  </button>
                </div>
              )}
              {!active.turns.length && (
                <div className="welcome">
                  <div className="welcome-mark">
                    <img src="/ledger.svg" alt="" />
                  </div>
                  <h2>
                    Go from disclosure
                    <br />
                    to understanding.
                  </h2>
                  <p>
                    Compare reporting periods, inspect the evidence,
                    <br className="desktop-break" /> and keep every conclusion
                    connected to its source.
                  </p>
                  <div className="suggestion-list">
                    {exampleQuestions.map((item, index) => (
                      <button
                        key={item}
                        onClick={() => void submit(undefined, item)}
                      >
                        <span>0{index + 1}</span>
                        {item}
                        <ArrowUp size={16} />
                      </button>
                    ))}
                  </div>
                </div>
              )}
              {active.turns.map((turn) => (
                <Response
                  key={turn.id}
                  turn={turn}
                  onCitation={(citation) => inspect("sources", turn, citation)}
                  onArtifact={(artifact) =>
                    inspect("artifacts", turn, undefined, artifact)
                  }
                  onRun={() => inspect("run", turn)}
                  onRetry={() => {
                    setQuestion(turn.question);
                    composerRef.current?.focus();
                  }}
                  onNotice={setNotice}
                />
              ))}
              {!busy && active.turns.length > 0 && mode === "preview" && (
                <div className="followups">
                  <span>EXPLORE THE EXAMPLE</span>
                  {exampleQuestions.slice(1).map((item) => (
                    <button
                      key={item}
                      onClick={() => void submit(undefined, item)}
                    >
                      {item}
                      <ArrowRight size={14} />
                    </button>
                  ))}
                </div>
              )}
            </div>
            <div className="composer-region">
              {!nearBottom && (
                <button
                  className="jump-latest"
                  onClick={() => {
                    scrollRef.current?.scrollTo({
                      top: scrollRef.current.scrollHeight,
                      behavior: "auto",
                    });
                    setNearBottom(true);
                  }}
                >
                  <ArrowDown size={14} />
                  Latest response
                </button>
              )}
              <form
                className="composer"
                onSubmit={(event) => void submit(event)}
              >
                <label className="sr-only" htmlFor="composer">
                  Ask a financial research question
                </label>
                <textarea
                  id="composer"
                  ref={composerRef}
                  placeholder="Ask a question. Follow the evidence."
                  value={question}
                  maxLength={4000}
                  rows={2}
                  onChange={(event) => setQuestion(event.target.value)}
                  onKeyDown={(event) => {
                    if (
                      event.key === "Enter" &&
                      !event.shiftKey &&
                      !event.nativeEvent.isComposing
                    ) {
                      event.preventDefault();
                      void submit();
                    }
                  }}
                />
                <div className="composer-controls">
                  <span className="corpus-label">
                    <FileText size={14} />
                    {token ? "Authorized corpus" : "Aurora demo filings"}
                  </span>
                  <label
                    className="date-filter"
                    title="Disclosure publication cutoff, not fiscal year"
                  >
                    <History size={14} />
                    <span className="sr-only">Publication cutoff</span>
                    <input
                      aria-label="Publication cutoff"
                      type="date"
                      value={asOf}
                      disabled={!token || busy}
                      onChange={(event) => setAsOf(event.target.value)}
                    />
                  </label>
                  <span className="composer-spacer" />
                  <span className="character-count">
                    {question.length ? `${question.length}/4000` : "↵"}
                  </span>
                  {busy ? (
                    <button
                      type="button"
                      className="send-button stop"
                      aria-label="Stop response"
                      onClick={stop}
                    >
                      <Square size={15} fill="currentColor" />
                    </button>
                  ) : (
                    <button
                      className="send-button"
                      disabled={!question.trim()}
                      aria-label="Send question"
                    >
                      <ArrowUp size={20} />
                    </button>
                  )}
                </div>
                <div
                  className="sr-only"
                  role="status"
                  aria-live="polite"
                  aria-atomic="true"
                >
                  {busy
                    ? "Research is in progress."
                    : active.turns.at(-1)?.answer
                      ? mode === "preview"
                        ? "Curated preview displayed."
                        : `Response complete. ${active.turns.at(-1)?.answer?.citations.length} source references available.`
                      : "Ready for a research question."}
                </div>
              </form>
              <p className="composer-disclaimer">
                <ShieldCheck size={12} />
                {token
                  ? "Cited, not infallible. Verify important figures against original sources."
                  : "Preview data is fictional. Connect your API for live, authorized research."}
                <span>Shift + Enter for a new line</span>
              </p>
            </div>
          </main>
          <aside className="inspector-dock" aria-label="Research context">
            {inspector}
          </aside>
        </div>
      </div>
      {notice && (
        <div className="toast" role="status">
          <Check size={16} />
          <span>{notice}</span>
          <button
            className="icon-button"
            aria-label="Dismiss notification"
            onClick={() => setNotice("")}
          >
            <X size={15} />
          </button>
        </div>
      )}
      {menuOpen && (
        <Modal
          title="Navigation"
          onClose={() => setMenuOpen(false)}
          className="navigation-modal"
        >
          {sidebar}
        </Modal>
      )}
      {contextOpen && (
        <Modal
          title="Research context"
          onClose={() => setContextOpen(false)}
          className="context-modal"
        >
          {inspector}
        </Modal>
      )}
      {settingsOpen && (
        <ConnectionDialog
          live={!!token}
          onClose={() => setSettingsOpen(false)}
          onConnect={resetWorkspace}
        />
      )}
      {helpOpen && (
        <Modal
          title="Evidence-first, by design"
          onClose={() => setHelpOpen(false)}
        >
          <div className="help-content">
            <p>
              Ledger connects a question to authorized financial evidence,
              bounded tools and a citation-checked answer.
            </p>
            <ol>
              <li>
                <strong>Retrieve.</strong> Tenant, group and publication-date
                filters are enforced by the backend.
              </li>
              <li>
                <strong>Investigate.</strong> A bounded LangGraph chat workflow
                can search again or calculate. The activity view shows real
                stage events, never private model reasoning.
              </li>
              <li>
                <strong>Inspect.</strong> Open citations, compare tables and
                inspect request timings. Citation checks do not prove that every
                statement is true.
              </li>
            </ol>
            <p>
              Conversations and access tokens remain in this tab&apos;s memory
              and disappear on refresh. No third-party analytics, remote images
              or code execution. Live queries are sent to your configured
              backend and its model providers.
            </p>
            <p>
              There is no file upload, persistent conversation store or sign-in
              flow yet. Production uses your OIDC access token and same-origin
              API routing.
            </p>
          </div>
        </Modal>
      )}
    </div>
  );
}

function ArrowUpRightIcon() {
  return <ArrowRight size={14} className="arrow-diagonal" />;
}

function ConnectionDialog({
  live,
  onClose,
  onConnect,
}: {
  live: boolean;
  onClose: () => void;
  onConnect: (token: string) => void;
}) {
  const [value, setValue] = useState("");
  return (
    <Modal title="Connect your research backend" onClose={onClose}>
      <form
        className="connection-form"
        onSubmit={(event) => {
          event.preventDefault();
          if (value.trim()) onConnect(value.trim());
        }}
      >
        <div className="connection-icon">
          <Link2 size={23} />
        </div>
        <p>
          Use a Ledger <strong>OIDC access token</strong>, or the local demo API
          token. Never enter an LLM, Qdrant or Modal API key here.
        </p>
        <label>
          Ledger access token
          <input
            type="password"
            value={value}
            onChange={(event) => setValue(event.target.value)}
            required
            autoComplete="off"
            spellCheck={false}
            maxLength={16_000}
            placeholder="Held only in this tab's memory"
          />
        </label>
        <div className="endpoint-info">
          <LockKeyhole size={14} />
          <span>
            Same-origin endpoint <code>/v1/query/stream</code>
          </span>
        </div>
        <p className="connection-caveat">
          Connecting replaces this tab&apos;s current conversations. Live access
          is checked on your first query. Configure the upstream API on the
          server, not in the browser.
        </p>
        <button className="primary-button" disabled={!value.trim()}>
          Connect & start fresh <ArrowRight size={16} />
        </button>
        <button
          className="clear-session"
          type="button"
          onClick={() => onConnect("")}
        >
          <Trash2 size={14} />
          {live ? "Disconnect & clear session" : "Clear preview conversations"}
        </button>
      </form>
    </Modal>
  );
}
