import { readFileSync } from "node:fs";
import { expect, test, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

// Payloads generated from the backend pydantic models; see src/fixtures/answers.json.
const fixtures = JSON.parse(
  readFileSync(
    new URL("../src/fixtures/answers.json", import.meta.url),
    "utf8",
  ),
) as Record<string, Record<string, unknown>> & {
  activities: Record<string, Record<string, unknown>[]>;
  evidence: Record<string, unknown>;
};
const shots = process.env.SCREENSHOT_DIR;
const sse = (event: string, data: unknown) =>
  `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;

async function shot(page: Page, name: string, project: string) {
  if (!shots) return;
  await page.evaluate(() =>
    Promise.all(
      document
        .getAnimations()
        .filter((a) => a.effect?.getTiming().iterations !== Infinity)
        .map((a) => a.finished),
    ),
  );
  await page.screenshot({
    path: `${shots}/${project}-${name}.png`,
    fullPage: false,
  });
}

// Replays a recorded backend run: retrieval and planning, evidence, then generation.
async function live(page: Page, name: string) {
  const activities = fixtures.activities[name];
  const split = activities.findIndex((item) => item.stage === "generation");
  const before = split < 0 ? activities : activities.slice(0, split);
  const after = split < 0 ? [] : activities.slice(split);
  const events = (items: unknown[]) =>
    items.map((item) => sse("activity", item)).join("");
  await page.route("**/v1/query/stream", (route) =>
    route.fulfill({
      contentType: "text/event-stream",
      body:
        events(before) +
        sse("evidence", fixtures.evidence[name]) +
        events(after) +
        sse("answer", fixtures[name]) +
        sse("done", {}),
    }),
  );
  await page.goto("/");
  await page
    .getByRole("button", { name: "Design preview", exact: true })
    .click();
  await page
    .getByLabel("Ledger access token", { exact: true })
    .fill("local-browser-test-token");
  await page.getByRole("button", { name: "Connect & start fresh" }).click();
  await page
    .getByRole("textbox", { name: "Ask a financial research question" })
    .fill("How fast did revenue grow?");
  await page.getByRole("button", { name: "Send question" }).click();
}

test("external evidence shows provenance and bounded lookup activities", async ({
  page,
  isMobile,
}) => {
  await live(page, "external");
  const turn = page.locator(".turn").last();
  await expect(turn).toContainText("5.33%");
  await turn.locator("details.activity-trace > summary").click();
  await expect(turn.locator(".company_facts_lookup")).toContainText(
    "Looking up SEC filings",
  );
  await expect(turn.locator(".macro_series_lookup")).toContainText(
    "Looking up macro data",
  );
  await turn
    .getByRole("button", { name: /View citation C2:/ })
    .first()
    .click();
  const context = isMobile
    ? page.getByRole("dialog", { name: "Research context" })
    : page.getByRole("complementary", { name: "Research context" });
  await expect(
    context.getByText("FRED / ALFRED", { exact: true }),
  ).toBeVisible();
  const provenance = context.getByLabel("Evidence provenance");
  await expect(provenance).toContainText("2024-01-01");
  await expect(provenance).toContainText("FEDFUNDS");
  await expect(provenance).toContainText("5.33");
  await expect(context.locator("a.source-link").last()).toHaveAttribute(
    "href",
    /https:\/\/fred\.stlouisfed\.org\//,
  );
});

test("demo shows a traced calculation, verification and an abstention", async ({
  page,
  isMobile,
}, info) => {
  await page.goto("/");
  await page
    .getByRole("button", {
      name: "Show the calculation for Aurora's revenue growth.",
    })
    .click();
  const turn = page.locator(".turn").last();
  const pill = turn.getByRole("button", {
    name: /Numeric verification: verified\. 4 numbers checked, none unsupported\. Groundedness score 0\.94/,
  });
  await expect(pill).toBeVisible();
  await pill.click();
  const details = turn.getByRole("group", { name: "Verification details" });
  await expect(details).toContainText("Every checked number matched");
  await page.keyboard.press("Escape");
  await expect(details).toHaveCount(0);
  await expect(pill).toBeFocused();

  await turn.getByText("Example activity · not executed").click();
  await expect(turn.locator(".timeline-row")).toHaveText([
    /Retrieving evidence/,
    /Planning query/,
    /Generating answer/,
    /Calculating/,
    /Generating answer/,
    /Validating citations/,
    /Verifying grounding/,
  ]);
  const plan = turn.getByRole("group", { name: "Query plan" });
  await expect(plan).toContainText("Measure a change");
  await expect(plan).toContainText("AURORA");

  const trace = turn.locator("details.calc-trace");
  await expect(trace.locator("summary")).toContainText("3 steps");
  await expect(trace.locator("summary")).toContainText("25%");
  await trace.locator("summary").focus();
  await page.keyboard.press("Enter");
  await expect(trace).toHaveAttribute("open", "");
  const steps = trace.locator(".calc-steps li");
  await expect(steps).toHaveCount(3);
  await expect(steps.nth(0)).toContainText("subtract");
  await expect(steps.nth(0)).toContainText("USD 30 million");
  await expect(steps.nth(2).locator(".operand-chip.constant")).toHaveText(
    /100/,
  );
  await expect(trace.locator(".calc-final")).toContainText("25%");
  await expect(trace.locator(".calc-final")).toContainText("Percent");
  await steps
    .nth(1)
    .getByRole("button", { name: /Result of step #0: USD 30 million/ })
    .click();
  await expect(steps.nth(0)).toBeFocused();
  await shot(page, "calculation-trace", info.project.name);

  await steps
    .nth(0)
    .getByRole("button", {
      name: /Evidence “USD 150 million” from citation C1/,
    })
    .click();
  const context = isMobile
    ? page.getByRole("dialog", { name: "Research context" })
    : page.getByRole("complementary", { name: "Research context" });
  await expect(context.locator("mark.operand-highlight")).toHaveText(
    "USD 150 million",
  );
  await shot(page, "evidence-highlight", info.project.name);
  if (isMobile)
    await page
      .getByRole("button", { name: "Close Research context", exact: true })
      .click();

  await page
    .getByRole("button", {
      name: "Forecast the share price for Aurora in 2027.",
    })
    .click();
  const abstained = page.locator(".turn").last();
  await expect(
    abstained.getByRole("heading", { name: "Ledger held back an answer" }),
  ).toBeVisible();
  await expect(abstained).toContainText("Thresholds uncalibrated.");
  await expect(abstained).toContainText(
    "outside what the disclosures can support",
  );
  await expect(abstained.locator("details.calc-trace")).toHaveCount(0);
  await abstained.scrollIntoViewIfNeeded();
  await shot(page, "abstention", info.project.name);
});

test("unverified live answer marks unsupported numbers inline", async ({
  page,
}, info) => {
  await live(page, "unverified");
  const turn = page.locator(".turn").last();
  await expect(turn.locator("mark.unsupported-number")).toHaveText([
    /31%/,
    /USD 115 million/,
  ]);
  await expect(
    turn.getByRole("button", { name: /View citation C1/ }),
  ).toHaveCount(1);
  const pill = turn.getByRole("button", {
    name: /Numeric verification: unverified\. \d+ numbers checked, 2 unsupported/,
  });
  await pill.click();
  const details = turn.getByRole("group", { name: "Verification details" });
  await expect(details.locator("mark")).toHaveText(["31%", "USD 115 million"]);
  await expect(details).toContainText("flag");
  await shot(page, "unverified", info.project.name);
  await page.mouse.click(5, 5);
  await expect(details).toHaveCount(0);
  await turn.getByText("Execution activity").click();
  // Flag mode has no groundedness model configured, so only citation validation runs.
  await expect(turn.locator(".timeline-row")).toHaveText([
    /Retrieving evidence\s*done/,
    /Planning query.*done/,
    /Generating answer\s*done/,
    /Calculating\s*done/,
    /Generating answer\s*done/,
    /Validating citations\s*done/,
  ]);
  const plan = turn.getByRole("group", { name: "Query plan" });
  await expect(plan).toContainText("Measure a percentage change");
  await expect(plan).toContainText("Searches (3)");
  await expect(plan).toContainText("AURORA revenue 2025");
  await expect(plan).toContainText("2024");
  await turn.locator("details.calc-trace summary").click();
  const final = turn.locator(".calc-final");
  await expect(final).toContainText("25%");
  await expect(final).toContainText("Percent");
  await expect(turn.locator(".calc-steps li").first()).toContainText(
    "Currency · USD million",
  );
});

test("verified live answer separates the two validation steps", async ({
  page,
}) => {
  await live(page, "verified");
  const turn = page.locator(".turn").last();
  await expect(
    turn.getByRole("button", {
      name: /Numeric verification: verified\. 3 numbers checked, none unsupported\. Groundedness score 0\.93/,
    }),
  ).toBeVisible();
  await turn.getByText("Execution activity").click();
  await expect(turn.locator(".timeline-row.citation_validation")).toHaveText(
    /Validating citations\s*done/,
  );
  await expect(turn.locator(".timeline-row.grounding_verification")).toHaveText(
    /Verifying grounding\s*done/,
  );
});

test("abstained live answer explains reasons calmly", async ({ page }) => {
  await live(page, "abstained");
  const turn = page.locator(".turn").last();
  await expect(
    turn.getByRole("heading", { name: "Ledger held back an answer" }),
  ).toBeVisible();
  await expect(turn).toContainText(
    "Numeric verification failed, so the answer was withheld. 2 numbers in the draft did not match cited evidence.",
  );
  await expect(turn).toContainText("Thresholds uncalibrated.");
  // Enforce mode never sends the rejected draft, so nothing is highlighted inline.
  await expect(turn.locator("mark.unsupported-number")).toHaveCount(0);
  await expect(turn.locator("details.calc-trace")).toHaveCount(0);
  const pill = turn.getByRole("button", {
    name: /Numeric verification: failed\. 3 numbers checked, 2 unsupported/,
  });
  await pill.click();
  const details = turn.getByRole("group", { name: "Verification details" });
  await expect(details).toContainText("2 numbers did not match cited evidence");
  await expect(details.locator("mark")).toHaveCount(0);
  await expect(turn.getByText("Abstained", { exact: true })).toBeVisible();
});

test("refused live answer shows the out-of-scope plan and reason", async ({
  page,
}) => {
  await live(page, "refused");
  const turn = page.locator(".turn").last();
  await expect(turn).toContainText("outside what the disclosures can support");
  await turn.getByText("Execution activity").click();
  const plan = turn.getByRole("group", { name: "Query plan" });
  await expect(plan).toContainText("Decline: outside the disclosures' scope");
  await expect(plan).toContainText("2027");
  await expect(plan).not.toContainText("Searches");
});

test("grounding UI passes accessibility checks with reduced motion", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await page
    .getByRole("button", {
      name: "Show the calculation for Aurora's revenue growth.",
    })
    .click();
  const turn = page.locator(".turn").last();
  await turn.locator("details.calc-trace summary").click();
  await turn.getByRole("button", { name: /Numeric verification/ }).click();
  expect(
    await turn
      .locator(".calc-body")
      .evaluate((el) => getComputedStyle(el).animationName),
  ).toBe("none");
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .analyze();
  expect(results.violations).toEqual([]);
  await page
    .getByRole("button", {
      name: "Forecast the share price for Aurora in 2027.",
    })
    .click();
  await page
    .locator(".turn")
    .last()
    .getByText("Example activity · not executed")
    .click();
  const after = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .analyze();
  expect(after.violations).toEqual([]);
});
