import { Component, type ReactNode } from "react";

export class ErrorBoundary extends Component<
  { children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidCatch() {
    // Keep model text, credentials and component state out of render-failure logs.
    console.error("Ledger interface rendering failed.");
  }
  render() {
    return this.state.failed ? (
      <main className="fatal-error">
        <h1>The workspace could not render.</h1>
        <p>
          Reload to start a clean session. In-memory conversations will be
          cleared.
        </p>
        <button onClick={() => location.reload()}>Reload workspace</button>
      </main>
    ) : (
      this.props.children
    );
  }
}
