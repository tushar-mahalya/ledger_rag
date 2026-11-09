# Research workspace (web)

React/TypeScript chat client for the engine: rich answers, source and artifact
inspectors, live execution activity and grounding surfaces. Static build served by
Nginx, which proxies `/v1/` to the engine. Details, design research and checks:
[docs/CHAT_WORKSPACE.md](../../docs/CHAT_WORKSPACE.md).

```bash
npm ci --prefix services/web --ignore-scripts
npm --prefix services/web run dev          # http://127.0.0.1:5173
npm --prefix services/web test             # unit; contract tests read ../../contracts
npm --prefix services/web run test:e2e     # Playwright, desktop and mobile
```

`src/fixtures/external.json` is the demo's recorded external-evidence turn; the
engine contract the tests check lives in [`contracts/`](../../contracts/).
