import type {
  AbstentionDecision,
  Activity,
  ActivityKind,
  Answer,
  Calculation,
  CalculationOperand,
  CalculationUnit,
  Citation,
  Conversation,
  PlanSummary,
  Turn,
  Verification,
} from "./contracts";
import externalFixture from "./fixtures/external.json";

const citations: Citation[] = [
  {
    id: "C1",
    document_id: "aurora-2025",
    title: "Aurora Devices · FY2025",
    source: "urn:ledger:synthetic:aurora:2025",
    published_at: "2026-02-01",
    quote:
      "In fiscal year 2025, Aurora Devices reported revenue of USD 150 million, compared with USD 120 million in fiscal year 2024. Revenue increased by 25 percent.\n\nOperating income was USD 24 million in fiscal year 2025. Operating margin was 16 percent.\n\nLiquidity: cash and cash equivalents totaled USD 42 million at December 31, 2025. Total debt was USD 10 million.",
  },
  {
    id: "C2",
    document_id: "aurora-2024",
    title: "Aurora Devices · FY2024",
    source: "urn:ledger:synthetic:aurora:2024",
    published_at: "2025-02-01",
    quote:
      "In fiscal year 2024, Aurora Devices reported revenue of USD 120 million, compared with USD 100 million in fiscal year 2023. Revenue increased by 20 percent.\n\nOperating income was USD 18 million in fiscal year 2024. Operating margin was 15 percent.\n\nLiquidity: cash and cash equivalents totaled USD 30 million at December 31, 2024. Total debt was USD 12 million.",
  },
];

export const exampleQuestions = [
  "Compare Aurora's FY2024 and FY2025 performance.",
  "How did Aurora's liquidity change?",
  "Show the calculation for Aurora's revenue growth.",
  // Matches the planner's high-precision out-of-scope rule, so the live API refuses it too.
  "Forecast the share price for Aurora in 2027.",
  "Show recorded SEC and FRED point-in-time evidence.",
];

// Curated grounding examples. Shapes mirror the backend; nothing below was executed.
const evidence = (
  citation_id: string,
  quote: string,
  value: string,
): CalculationOperand => ({
  kind: "evidence",
  value,
  citation_id,
  quote,
  source_value: quote,
  scale: "million",
  unit: "currency",
  step: null,
});
const ref = (step: number, value: string): CalculationOperand => ({
  kind: "step",
  value,
  citation_id: null,
  quote: null,
  source_value: null,
  scale: null,
  unit: null,
  step,
});
const usdMillion: CalculationUnit = {
  kind: "currency",
  currency: "USD",
  scale: "million",
  display: "amount",
};
const ratio: CalculationUnit = {
  kind: "ratio",
  currency: null,
  scale: null,
  display: "ratio",
};
const percent: CalculationUnit = {
  kind: "percent",
  currency: null,
  scale: null,
  display: "percent",
};
const constant = (value: string): CalculationOperand => ({
  kind: "constant",
  value,
  citation_id: null,
  quote: null,
  source_value: null,
  scale: null,
  unit: null,
  step: null,
});

const growth: Calculation = {
  steps: [
    {
      index: 0,
      op: "subtract",
      operands: [
        evidence("C1", "USD 150 million", "150000000"),
        evidence("C2", "USD 120 million", "120000000"),
      ],
      result: "30000000",
      kind: "currency",
      unit: usdMillion,
      citation_ids: ["C1", "C2"],
      evidence_backed: true,
    },
    {
      index: 1,
      op: "divide",
      operands: [
        ref(0, "30000000"),
        evidence("C2", "USD 120 million", "120000000"),
      ],
      result: "0.25",
      kind: "ratio",
      unit: ratio,
      citation_ids: ["C1", "C2"],
      evidence_backed: true,
    },
    {
      index: 2,
      op: "multiply",
      operands: [ref(1, "0.25"), constant("100")],
      result: "25",
      kind: "percent",
      unit: percent,
      citation_ids: ["C1", "C2"],
      evidence_backed: true,
    },
  ],
  result: "25",
  unit: percent,
  citation_ids: ["C1", "C2"],
};

const netCash = (
  citation: string,
  cash: string,
  debt: string,
  result: string,
): Calculation => ({
  steps: [
    {
      index: 0,
      op: "subtract",
      operands: [
        evidence(citation, `USD ${cash} million`, `${cash}000000`),
        evidence(citation, `USD ${debt} million`, `${debt}000000`),
      ],
      result: `${result}000000`,
      kind: "currency",
      unit: usdMillion,
      citation_ids: [citation],
      evidence_backed: true,
    },
  ],
  result: `${result}000000`,
  unit: usdMillion,
  citation_ids: [citation],
});

function verified(checked: number, score?: number): Verification {
  return {
    status: "verified",
    mode: "enforce",
    checked_numbers: checked,
    unsupported_numbers: [],
    reasons: [],
    verifier: score === undefined ? null : "lettucedetect",
    verifier_score: score ?? null,
  };
}

function decision(
  verification: Verification,
  reasons: AbstentionDecision["reasons"],
  toolCalls: number,
): AbstentionDecision {
  // A planner refusal happens before retrieval, so no evidence features exist.
  const refused = reasons.includes("out_of_scope");
  return {
    abstained: reasons.length > 0,
    reasons,
    calibrated: false,
    features: {
      hit_count: refused ? 0 : 2,
      top_score: refused ? null : 0.82,
      mean_score: refused ? null : 0.79,
      citation_coverage: reasons.length ? null : 1,
      checked_numbers: verification.checked_numbers,
      unsupported_number_count: verification.unsupported_numbers.length,
      verification_status: verification.status,
      verifier_score: verification.verifier_score ?? null,
      tool_calls: toolCalls,
      tool_errors: 0,
    },
  };
}

// Stage, label and tool for each kind, as the backend's ACTIVITY_KINDS sends them.
const KINDS: Record<
  ActivityKind,
  Pick<Activity, "stage" | "label"> & { tool?: Activity["tool"] }
> = {
  planning: { stage: "planning", label: "Planning query" },
  retrieval: { stage: "retrieval", label: "Retrieving evidence" },
  generation: { stage: "generation", label: "Generating answer" },
  calculator: { stage: "tool", label: "Calculating", tool: "calculator" },
  evidence_search: {
    stage: "tool",
    label: "Searching evidence",
    tool: "search_evidence",
  },
  company_facts_lookup: {
    stage: "tool",
    label: "Looking up SEC filings",
    tool: "lookup_company_facts",
  },
  macro_series_lookup: {
    stage: "tool",
    label: "Looking up macro data",
    tool: "lookup_macro_series",
  },
  tool_check: { stage: "tool", label: "Checking tool request" },
  citation_validation: { stage: "validation", label: "Validating citations" },
  grounding_verification: {
    stage: "validation",
    label: "Verifying grounding",
  },
};

const pair = (kind: ActivityKind, plan?: PlanSummary): Activity[] => [
  { ...KINDS[kind], kind, status: "started" },
  { ...KINDS[kind], kind, status: "completed", ...(plan ? { plan } : {}) },
];

// Summaries the deterministic planner produces for the example questions.
const plan = (
  intent: PlanSummary["intent"],
  periods: string[],
  subquestions: string[],
  route: PlanSummary["route"] = "retrieve",
): PlanSummary => ({
  intent,
  route,
  subquestion_count: subquestions.length,
  subquestions,
  companies: ["AURORA"],
  periods,
});

const comparison = `## Growth, with stronger fundamentals

Aurora Devices' revenue grew **25% to USD 150 million** in FY2025. Operating income increased from USD 18 million to USD 24 million, while the operating margin expanded from **15% to 16%**. [C1] [C2]

### Year-over-year performance

| Metric | FY2024 | FY2025 | Sources |
| :--- | ---: | ---: | :--- |
| Revenue | $120m | $150m | [C2] [C1] |
| Operating income | $18m | $24m | [C2] [C1] |
| Operating margin | 15% | 16% | [C2] [C1] |
| Cash & equivalents | $30m | $42m | [C2] [C1] |
| Total debt | $12m | $10m | [C2] [C1] |

*USD millions, except margin. Figures are taken from the two fictional disclosures.* [C1] [C2]

### The takeaway

Cash rose from **USD 30 million to USD 42 million**, while total debt fell from USD 12 million to USD 10 million. These disclosures show improved liquidity; they do not establish a future outlook or an investment recommendation. [C1] [C2]`;

// Counts produced by the backend's check_numbers on these exact texts.
const LIQUIDITY_CHECKED = 12;
const COMPARISON_CHECKED = 20;
const ABSTENTION =
  "I do not have sufficient authorized evidence to answer this question.";

export function demoTurn(question: string): Turn {
  if (question === exampleQuestions[4]) {
    // Generated by the backend; contract tests validate the recorded wire payload.
    const recorded = externalFixture as unknown as {
      answer: Answer;
      activities: Activity[];
    };
    return {
      id: crypto.randomUUID(),
      question,
      status: "complete",
      demo: true,
      answer: recorded.answer,
      activities: recorded.activities,
    };
  }
  const supported = exampleQuestions.includes(question);
  const calculation = /calculation|calculate/.test(question.toLowerCase());
  const liquidity = /liquidity/.test(question.toLowerCase());
  const forecast = question === exampleQuestions[3];
  const comparing = question === exampleQuestions[0];
  const base = {
    profile: "chat" as const,
    retrieval_ms: 0,
    generation_ms: 0,
    request_id: "design-preview",
    model: "Curated preview · no model call",
  };
  const activities: Activity[] = !supported
    ? []
    : forecast
      ? [
          ...pair("retrieval"),
          ...pair(
            "planning",
            plan("lookup", ["2027"], [], "refuse_out_of_scope"),
          ),
          ...pair("citation_validation"),
        ]
      : [
          ...pair("retrieval"),
          ...pair(
            "planning",
            comparing
              ? plan(
                  "compare",
                  ["2024", "2025"],
                  [
                    question,
                    `AURORA ${question} 2024`,
                    `AURORA ${question} 2025`,
                  ],
                )
              : plan("change", [], [question]),
          ),
          ...pair("generation"),
          ...(calculation || liquidity
            ? [...pair("calculator"), ...pair("generation")]
            : []),
          ...pair("citation_validation"),
          ...(calculation ? pair("grounding_verification") : []),
        ];
  let answer: Answer;
  if (!supported) {
    answer = {
      ...base,
      answer:
        "This design preview contains curated examples, not a live language model. Choose an example or connect your Ledger API to research a new question.",
      citations: [],
      abstained: true,
      tool_calls: 0,
      calculation: null,
      calculations: [],
      verification: null,
      abstention: null,
    };
  } else if (forecast) {
    const verification: Verification = {
      status: "skipped",
      mode: "enforce",
      checked_numbers: 0,
      unsupported_numbers: [],
      reasons: [],
      verifier: null,
      verifier_score: null,
    };
    answer = {
      ...base,
      answer: ABSTENTION,
      citations: [],
      abstained: true,
      tool_calls: 0,
      calculation: null,
      calculations: [],
      verification,
      abstention: decision(verification, ["out_of_scope"], 0),
    };
  } else if (calculation) {
    const verification = verified(4, 0.94);
    answer = {
      ...base,
      answer: `## Revenue growth, made explicit\n\nRevenue rose from USD 120 million to USD 150 million, an increase of USD 30 million. [C1] [C2]\n\nThat is growth of **25%** year over year. These are fictional source figures. [C1] [C2]\n\nOpen the calculation trace to follow each step back to its source. This curated example shows the format; no tool was executed.`,
      citations,
      abstained: false,
      tool_calls: 1,
      calculation: growth,
      calculations: [growth],
      verification,
      abstention: decision(verification, [], 1),
    };
  } else if (liquidity) {
    const programs = [
      netCash("C2", "30", "12", "18"),
      netCash("C1", "42", "10", "32"),
    ];
    const verification = verified(LIQUIDITY_CHECKED);
    answer = {
      ...base,
      answer: `## A stronger cash position\n\nCash increased from **USD 30 million to USD 42 million**, while debt decreased from **USD 12 million to USD 10 million**. Net cash therefore increased from USD 18 million to USD 32 million. [C1] [C2]\n\n| Metric | FY2024 | FY2025 | Sources |\n| :--- | ---: | ---: | :--- |\n| Cash | $30m | $42m | [C2] [C1] |\n| Debt | $12m | $10m | [C2] [C1] |\n| Net cash (derived) | $18m | $32m | [C2] [C1] |\n\nNet cash here means cash less total debt, not free cash flow. [C1] [C2]`,
      citations,
      abstained: false,
      tool_calls: 2,
      calculation: programs[1],
      calculations: programs,
      verification,
      abstention: decision(verification, [], 2),
    };
  } else {
    const verification = verified(COMPARISON_CHECKED);
    answer = {
      ...base,
      answer: comparison,
      citations,
      abstained: false,
      tool_calls: 0,
      calculation: null,
      calculations: [],
      verification,
      abstention: decision(verification, [], 0),
    };
  }
  return {
    id: crypto.randomUUID(),
    question,
    status: "complete",
    activities,
    demo: true,
    answer,
  };
}

export function initialConversation(): Conversation {
  return {
    id: crypto.randomUUID(),
    title: "Aurora · performance review",
    turns: [demoTurn(exampleQuestions[0])],
  };
}
