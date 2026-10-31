import type { Citation, Conversation, Turn } from "./contracts";

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
];

const comparison = `## Growth, with stronger fundamentals

Aurora Devices' revenue grew **25% to USD 150 million** in FY2025. Operating income increased from USD 18 million to USD 24 million, while the operating margin expanded by **1 percentage point**. [C1] [C2]

### Year-over-year performance

| Metric | FY2024 | FY2025 |
| :--- | ---: | ---: |
| Revenue | $120m | $150m |
| Operating income | $18m | $24m |
| Operating margin | 15% | 16% |
| Cash & equivalents | $30m | $42m |
| Total debt | $12m | $10m |

*USD millions, except margin. Figures are taken from the two fictional disclosures.* [C1] [C2]

### The takeaway

Cash increased by **USD 12 million**, while total debt fell by **USD 2 million**. These disclosures show improved liquidity; they do not establish a future outlook or an investment recommendation. [C1] [C2]`;

export function demoTurn(question: string): Turn {
  const calculation = /calculation|calculate/.test(question.toLowerCase());
  const liquidity = /liquidity/.test(question.toLowerCase());
  const supported = exampleQuestions.includes(question);
  const content = !supported
    ? "This design preview contains three curated examples, not a live language model. Choose an example or connect your Ledger API to research a new question."
    : calculation
      ? `## Revenue growth, made explicit\n\nRevenue rose from USD 120 million to USD 150 million. The change is **25%**. These are fictional source figures. [C1] [C2]\n\n\`\`\`text\n(150 - 120) / 120 * 100 = 25%\n\`\`\`\n\nThe displayed calculation is a curated example, not a tool execution. [C1] [C2]`
      : liquidity
        ? `## A stronger cash position\n\nCash increased from **USD 30 million to USD 42 million**, while debt decreased from **USD 12 million to USD 10 million**. Net cash therefore increased from USD 18 million to USD 32 million. [C1] [C2]\n\n| USD millions | FY2024 | FY2025 |\n| :--- | ---: | ---: |\n| Cash | 30 | 42 |\n| Debt | 12 | 10 |\n| Net cash (derived) | 18 | 32 |\n\nNet cash here means cash less total debt, not free cash flow. [C1] [C2]`
        : comparison;
  return {
    id: crypto.randomUUID(),
    question,
    status: "complete",
    activities: [],
    demo: true,
    answer: {
      answer: content,
      citations: supported ? citations : [],
      abstained: !supported,
      profile: "chat",
      retrieval_ms: 0,
      generation_ms: 0,
      request_id: "design-preview",
      model: "Curated preview · no model call",
      tool_calls: 0,
      calculation: null,
      calculations: [],
      verification: null,
      abstention: null,
    },
  };
}

export function initialConversation(): Conversation {
  return {
    id: crypto.randomUUID(),
    title: "Aurora · performance review",
    turns: [demoTurn(exampleQuestions[0])],
  };
}
