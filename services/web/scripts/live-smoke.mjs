import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { fileURLToPath } from "node:url";
import { chromium } from "@playwright/test";

const root = fileURLToPath(new URL("../../../", import.meta.url));
const web = fileURLToPath(new URL("../", import.meta.url));
const token = randomBytes(32).toString("hex");
const children = [];
let browser;

function start(command, args, cwd, env) {
  const child = spawn(command, args, {
    cwd,
    env: { ...process.env, ...env },
    stdio: ["ignore", "ignore", "pipe"],
  });
  let diagnostic = "";
  child.stderr.on("data", (chunk) => {
    diagnostic = (diagnostic + chunk.toString()).slice(-4000);
  });
  child.on("error", () => {
    diagnostic = "Failed to launch required executable";
  });
  children.push(child);
  return { child, diagnostic: () => diagnostic };
}

async function ready(url, processInfo) {
  for (let attempt = 0; attempt < 100; attempt++) {
    if (processInfo.child.exitCode !== null)
      throw new Error(`Local test server exited: ${processInfo.diagnostic()}`);
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(1000) });
      if (response.ok) return;
    } catch (error) {
      if (!(error instanceof TypeError) && !(error instanceof DOMException))
        throw error;
    }
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  throw new Error("Local server readiness deadline exceeded");
}

try {
  const api = start(
    `${root}.venv/bin/python`,
    [
      "-m",
      "uvicorn",
      "ledger_rag.api.app:app",
      "--host",
      "127.0.0.1",
      "--port",
      "18081",
      "--no-access-log",
    ],
    root,
    {
      LEDGER_ENVIRONMENT: "development",
      LEDGER_AUTH_MODE: "demo",
      LEDGER_DEMO_TOKEN: token,
      LEDGER_PROVIDER: "extractive",
      LEDGER_RETRIEVAL_BACKEND: "memory",
      LEDGER_SEED_PATH: `${root}data/examples.jsonl`,
      LEDGER_OTLP_ENDPOINT: "",
      LEDGER_VOICE_ENABLED: "false",
    },
  );
  const ui = start(
    process.execPath,
    [
      "node_modules/vite/bin/vite.js",
      "--host",
      "127.0.0.1",
      "--port",
      "15173",
      "--strictPort",
    ],
    web,
    { LEDGER_API_ORIGIN: "http://127.0.0.1:18081" },
  );
  await Promise.all([
    ready("http://127.0.0.1:18081/health/ready", api),
    ready("http://127.0.0.1:15173", ui),
  ]);
  browser = await chromium.launch();
  const page = await browser.newPage({
    viewport: { width: 1440, height: 1000 },
  });
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.name));
  await page.goto("http://127.0.0.1:15173");
  await page
    .getByRole("button", { name: "Design preview", exact: true })
    .click();
  await page.getByLabel("Ledger access token", { exact: true }).fill(token);
  await page.getByRole("button", { name: "Connect & start fresh" }).click();
  const question = page.getByRole("textbox", {
    name: "Ask a financial research question",
  });
  await question.fill("What was Aurora revenue in fiscal year 2025?");
  const responsePromise = page.waitForResponse((response) =>
    response.url().endsWith("/v1/query/stream"),
  );
  await page.getByRole("button", { name: "Send question" }).click();
  const response = await responsePromise;
  assert.equal(response.status(), 200);
  assert.match(response.headers()["content-type"], /text\/event-stream/);
  await page.getByText("References checked", { exact: true }).waitFor();
  assert.match(await page.locator(".markdown").innerText(), /150 million/);
  assert.ok((await page.locator(".activity-trace li").count()) >= 4);
  await page.getByRole("button", { name: "Inspect run details" }).click();
  await page.getByText("extractive-demo", { exact: true }).waitFor();
  assert.match(
    await page.locator(".run-metrics .mono").innerText(),
    /^[a-zA-Z0-9-]{16,}$/,
  );
  assert.deepEqual(errors, []);
  console.log(
    "Live local integration passed: browser -> same-origin proxy -> authenticated API -> LangGraph -> citation-checked answer, actual activity/metadata, no external model.",
  );
} finally {
  await browser?.close();
  await Promise.all(
    children.map(
      (child) =>
        new Promise((resolve) => {
          if (child.exitCode !== null) return resolve();
          child.once("exit", resolve);
          child.kill("SIGTERM");
          const timeout = setTimeout(() => child.kill("SIGKILL"), 5000);
          timeout.unref();
          child.once("exit", () => clearTimeout(timeout));
        }),
    ),
  );
}
