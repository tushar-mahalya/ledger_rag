import { expect, test, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

async function connect(page: Page) {
  await page
    .getByRole("button", { name: "Design preview", exact: true })
    .click();
  await page
    .getByLabel("Ledger access token", { exact: true })
    .fill("local-browser-test-token");
  await page.getByRole("button", { name: "Connect & start fresh" }).click();
}
const answer = {
  answer:
    "Revenue was **USD 150 million**. [C1]\n\n| Year | Revenue |\n|---|---|\n|2025|150|\n\n<script>window.hacked=true</script>\n\n![beacon](https://tracker.invalid/pixel)\n\n[Unsafe](javascript:alert(1))",
  citations: [
    {
      id: "C1",
      document_id: "doc",
      source: "urn:ledger:synthetic:test",
      title: "Authorized report",
      published_at: "2026-02-01",
      quote: "Revenue was USD 150 million.",
    },
  ],
  abstained: false,
  profile: "chat",
  retrieval_ms: 12,
  generation_ms: 80,
  request_id: "browser-contract-test",
  model: "test-model",
  tool_calls: 1,
};
const sse = (event: string, data: unknown) =>
  `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;

test("preview is honest, citation inspector and artifact download work", async ({
  page,
  isMobile,
}) => {
  await page.goto("/");
  await expect(
    page.getByText("No live model or benchmark results.", { exact: false }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: /View citation C1/ })
    .first()
    .click();
  const context = isMobile
    ? page.getByRole("dialog", { name: "Research context" })
    : page.getByRole("complementary", { name: "Research context" });
  await expect(
    context.getByText("urn:ledger:synthetic:aurora:2025"),
  ).toBeVisible();
  if (isMobile)
    await page
      .getByRole("button", { name: "Close Research context", exact: true })
      .click();
  await page
    .getByRole("button", { name: /Year-over-year performance table/ })
    .click();
  await expect(context.getByRole("tab", { name: /Artifacts/ })).toHaveAttribute(
    "aria-selected",
    "true",
  );
  const download = page.waitForEvent("download");
  await context
    .getByRole("button", { name: /Download Year-over-year/ })
    .click();
  expect((await download).suggestedFilename()).toMatch(/\.csv$/);
});

test("live SSE carries history, citations, actual run metadata and no secret persistence", async ({
  page,
  isMobile,
}) => {
  const requests: Record<string, unknown>[] = [];
  await page.route("**/v1/query/stream", async (route) => {
    expect(route.request().headers().authorization).toBe(
      "Bearer local-browser-test-token",
    );
    requests.push(route.request().postDataJSON() as Record<string, unknown>);
    await route.fulfill({
      contentType: "text/event-stream",
      body:
        sse("activity", {
          stage: "retrieval",
          status: "completed",
          label: "Retrieved authorized evidence",
        }) +
        sse("evidence", {
          request_id: "browser-contract-test",
          retrieval_ms: 12,
          sources: [answer.citations[0].source],
          citations: answer.citations,
        }) +
        sse("answer", answer) +
        sse("done", {}),
    });
  });
  const external: string[] = [];
  page.on("request", (request) => {
    if (request.url().includes("tracker.invalid")) external.push(request.url());
  });
  await page.goto("/");
  await connect(page);
  await page
    .getByRole("textbox", { name: "Ask a financial research question" })
    .fill("What was revenue?");
  await page.getByRole("button", { name: "Send question" }).click();
  await expect(
    page.getByText("References checked", { exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Inspect run details" }).click();
  const context = isMobile
    ? page.getByRole("dialog", { name: "Research context" })
    : page.getByRole("complementary", { name: "Research context" });
  await expect(
    context.getByText("browser-contract-test", { exact: true }),
  ).toBeVisible();
  await expect(context.getByText("12 ms", { exact: true })).toBeVisible();
  await expect(context.getByText("80 ms", { exact: true })).toBeVisible();
  if (isMobile)
    await page
      .getByRole("button", { name: "Close Research context", exact: true })
      .click();
  await page
    .getByRole("textbox", { name: "Ask a financial research question" })
    .fill("Compare with last year");
  await page.getByRole("button", { name: "Send question" }).click();
  await expect(
    page.getByText("References checked", { exact: true }),
  ).toHaveCount(2);
  expect(requests[0].history).toEqual([]);
  expect(requests[1].history).toHaveLength(2);
  expect(external).toEqual([]);
  expect(await page.locator(".markdown script").count()).toBe(0);
  expect(await page.locator('.markdown a[href^="javascript:"]').count()).toBe(
    0,
  );
  expect(
    await page.evaluate(() => ({
      local: localStorage.length,
      session: sessionStorage.length,
      cookies: document.cookie,
    })),
  ).toEqual({ local: 0, session: 0, cookies: "" });
  await page.reload();
  await expect(
    page.getByRole("button", { name: "Design preview", exact: true }),
  ).toBeVisible();
});

test("truncated stream is an error and does not display unconfirmed answer", async ({
  page,
}) => {
  await page.route("**/v1/query/stream", (route) =>
    route.fulfill({
      contentType: "text/event-stream",
      body: sse("answer", answer),
    }),
  );
  await page.goto("/");
  await connect(page);
  await page
    .getByRole("textbox", { name: "Ask a financial research question" })
    .fill("Revenue?");
  await page.getByRole("button", { name: "Send question" }).click();
  await expect(page.getByRole("alert")).toContainText(
    "before a complete answer",
  );
  await expect(
    page.getByText("References checked", { exact: true }),
  ).toHaveCount(0);
  await page.getByRole("button", { name: "Retry question" }).click();
  await expect(
    page.getByRole("textbox", { name: "Ask a financial research question" }),
  ).toHaveValue("Revenue?");
});

test("cancel is explicit and stale responses cannot enter a new conversation", async ({
  page,
  isMobile,
}) => {
  let release: (() => void) | undefined;
  let finish: (() => void) | undefined;
  const hold = new Promise<void>((resolve) => {
    release = resolve;
  });
  const finished = new Promise<void>((resolve) => {
    finish = resolve;
  });
  await page.route("**/v1/query/stream", async (route) => {
    await hold;
    await route
      .fulfill({
        contentType: "text/event-stream",
        body: sse("answer", answer) + sse("done", {}),
      })
      .catch(() => undefined);
    finish?.();
  });
  await page.goto("/");
  await connect(page);
  await page
    .getByRole("textbox", { name: "Ask a financial research question" })
    .fill("Slow question");
  const sent = page.waitForRequest("**/v1/query/stream");
  await page.getByRole("button", { name: "Send question" }).click();
  await sent;
  await page.getByRole("button", { name: "Stop response" }).click();
  await expect(
    page.getByText("Response stopped", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("References checked", { exact: true }),
  ).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "Send question" }),
  ).toBeVisible();
  if (isMobile) {
    await page.getByRole("button", { name: "Open navigation" }).click();
  }
  await page.getByRole("button", { name: "New research", exact: true }).click();
  release?.();
  await finished;
  await expect(page.locator(".turn")).toHaveCount(0);
  await expect(
    page.getByRole("heading", { name: "Good questions. Grounded answers." }),
  ).toBeVisible();
});

test("keyboard focus, reduced motion, accessibility and layout", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await expect(
    page.getByRole("textbox", { name: "Ask a financial research question" }),
  ).toBeVisible();
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .analyze();
  expect(results.violations).toEqual([]);
  await page
    .getByRole("button", { name: "Design preview", exact: true })
    .click();
  await expect(page.getByRole("dialog")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(
    page.getByRole("button", { name: "Design preview", exact: true }),
  ).toBeFocused();
  expect(
    await page
      .locator(".turn")
      .first()
      .evaluate((el) => getComputedStyle(el).animationName),
  ).toBe("none");
});
