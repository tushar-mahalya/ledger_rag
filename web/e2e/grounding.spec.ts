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
  activities: Record<string, unknown>[];
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

async function live(page: Page, answer: unknown) {
  await page.route("**/v1/query/stream", (route) =>
    route.fulfill({
      contentType: "text/event-stream",
      body:
        sse("activity", {
          stage: "retrieval",
          status: "started",
          label: "Retrieving evidence",
        }) +
        sse("activity", {
          stage: "retrieval",
          status: "completed",
          label: "Retrieving evidence",
        }) +
        fixtures.activities.map((item) => sse("activity", item)).join("") +
        sse("activity", {
          stage: "tool",
          status: "completed",
          label: "Calculating",
          tool: "calculator",
        }) +
        sse("answer", answer) +
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
      name: "Will Aurora's revenue keep growing in 2027?",
    })
    .click();
  const abstained = page.locator(".turn").last();
  await expect(
    abstained.getByRole("heading", { name: "Ledger held back an answer" }),
  ).toBeVisible();
  await expect(abstained).toContainText("Thresholds uncalibrated.");
  await expect(abstained).toContainText("No answer passed citation validation");
  await expect(abstained.locator("details.calc-trace")).toHaveCount(0);
  await abstained.scrollIntoViewIfNeeded();
  await shot(page, "abstention", info.project.name);
});

test("unverified live answer marks unsupported numbers inline", async ({
  page,
}, info) => {
  await live(page, fixtures.unverified);
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
  await expect(turn.locator(".timeline-row")).toHaveText([
    /Retrieving evidence\s*done/,
    /Planning query\s*done/,
    /Calculating\s*done/,
    /Verifying grounding\s*done/,
  ]);
});

test("abstained live answer explains reasons calmly", async ({ page }) => {
  await live(page, fixtures.abstained);
  const turn = page.locator(".turn").last();
  await expect(
    turn.getByRole("heading", { name: "Ledger held back an answer" }),
  ).toBeVisible();
  await expect(turn).toContainText("Numeric verification failed");
  await expect(turn).toContainText("Thresholds uncalibrated.");
  await expect(
    turn.getByRole("button", { name: /Numeric verification: failed/ }),
  ).toBeVisible();
  await expect(turn.getByText("Abstained", { exact: true })).toBeVisible();
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
      name: "Will Aurora's revenue keep growing in 2027?",
    })
    .click();
  const after = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .analyze();
  expect(after.violations).toEqual([]);
});
