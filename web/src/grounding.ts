import type {
  Activity,
  ActivityKind,
  Calculation,
  CalculationKind,
  CalculationOperand,
  CalculationStep,
  CalculationUnit,
  PlanSummary,
  VerificationStatus,
} from "./contracts";

// The backend normalises scale words (k, mn, bn, ...) to these names.
const SCALES: Record<string, number> = {
  thousand: 3,
  million: 6,
  billion: 9,
  trillion: 12,
};

/** Moves the decimal point of an exact decimal string without float rounding. */
export function shiftDecimal(value: string, places: number): string {
  const match = /^(-?)(\d*)(?:\.(\d*))?$/.exec(value.trim());
  if (!match) return value;
  const [, sign, int = "", frac = ""] = match;
  let digits = `${int}${frac}`;
  let point = int.length - places;
  if (point <= 0) {
    digits = "0".repeat(1 - point) + digits;
    point = 1;
  }
  if (point > digits.length) digits = digits.padEnd(point, "0");
  const whole = digits.slice(0, point).replace(/^0+(?=\d)/, "");
  const rest = digits.slice(point).replace(/0+$/, "");
  const text = rest ? `${whole}.${rest}` : whole;
  return text === "0" ? "0" : `${sign}${text}`;
}

/** Groups thousands and trims to `maxFraction` decimals (half-up on the digit string). */
export function groupDecimal(value: string, maxFraction = 4): string {
  const match = /^(-?)(\d+)(?:\.(\d+))?$/.exec(value.trim());
  if (!match) return value;
  const [, sign, int, frac = ""] = match;
  let whole = int;
  let fraction = frac;
  if (fraction.length > maxFraction) {
    const roundUp = Number(fraction[maxFraction]) >= 5;
    fraction = fraction.slice(0, maxFraction);
    if (roundUp) {
      const bumped = (BigInt(`${whole}${fraction}`) + 1n)
        .toString()
        .padStart(whole.length + fraction.length, "0");
      whole = bumped.slice(0, bumped.length - fraction.length) || "0";
      fraction = bumped.slice(bumped.length - fraction.length);
    }
  }
  fraction = fraction.replace(/0+$/, "");
  const grouped = whole.replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  const text = fraction ? `${grouped}.${fraction}` : grouped;
  return /^[0.,]+$/.test(text) ? text : `${sign}${text}`;
}

type Presentation = { scale?: string; currency?: string };

function currencyPrefix(source: string | null | undefined): string | undefined {
  return /^[\s(\-−]*(US\$|USD|EUR|GBP|JPY|[$€£¥])/i
    .exec(source ?? "")?.[1]
    ?.toUpperCase();
}

function common<T>(values: (T | undefined)[]): T | undefined {
  const defined = values.filter((value): value is T => value !== undefined);
  return defined.length && defined.every((value) => value === defined[0])
    ? defined[0]
    : undefined;
}

/**
 * Legacy fallback for payloads without `unit`: guesses the scale word and currency
 * shared by every evidence operand a step depends on.
 */
export function presentations(calculation: Calculation): Presentation[] {
  const result: Presentation[] = [];
  for (const step of calculation.steps) {
    const parts = step.operands
      .filter((operand) => operand.kind !== "constant")
      .map((operand): Presentation => {
        if (operand.kind === "step") return result[operand.step ?? -1] ?? {};
        const scale = operand.scale?.toLowerCase();
        return {
          scale: scale && scale in SCALES ? scale : undefined,
          currency:
            operand.unit === "currency"
              ? currencyPrefix(operand.source_value ?? operand.quote)
              : undefined,
        };
      });
    result.push({
      scale: parts.length
        ? common(parts.map((part) => part.scale ?? "-"))
        : undefined,
      currency: parts.length
        ? common(parts.map((part) => part.currency ?? "-"))
        : undefined,
    });
  }
  return result.map((item) => ({
    scale: item.scale === "-" ? undefined : item.scale,
    currency: item.currency === "-" ? undefined : item.currency,
  }));
}

/** Human-readable value. Results are in base units; percent-family evidence is a fraction. */
export function formatValue(
  value: string,
  kind: CalculationKind | null | undefined,
  presentation: Presentation = {},
): string {
  if (kind === "percent") return `${groupDecimal(value, 2)}%`;
  if (kind === "ratio") return groupDecimal(value, 4);
  if (kind === "boolean") return value === "1" ? "true" : "false";
  if (kind === "currency" || kind === "amount") {
    const places = presentation.scale ? SCALES[presentation.scale] : 0;
    const shown = groupDecimal(shiftDecimal(value, places), 2);
    const scaleWord = presentation.scale ? ` ${presentation.scale}` : "";
    const currency =
      kind === "currency" && presentation.currency
        ? `${presentation.currency}${/^[A-Z]{3}$/.test(presentation.currency) ? " " : ""}`
        : "";
    const sign = shown.startsWith("-") ? "-" : "";
    return `${sign}${currency}${shown.slice(sign.length)}${scaleWord}`;
  }
  return groupDecimal(value, 4);
}

export const KIND_LABELS: Record<CalculationKind, string> = {
  currency: "Currency",
  amount: "Amount",
  ratio: "Ratio",
  percent: "Percent",
  boolean: "Yes / no",
};

export function unitLabel(kind: CalculationKind | null | undefined): string {
  return kind ? KIND_LABELS[kind] : "Unspecified unit";
}

/** Formats a base-unit value with the backend's unit metadata; no guessing. */
export function formatUnit(value: string, unit: CalculationUnit): string {
  if (unit.display === "boolean") return value === "1" ? "true" : "false";
  if (unit.kind === "percent") return `${groupDecimal(value, 2)}%`;
  if (unit.kind === "ratio")
    return unit.display === "percent"
      ? `${groupDecimal(shiftDecimal(value, -2), 2)}%`
      : groupDecimal(value, 4);
  const places = unit.scale ? SCALES[unit.scale] : 0;
  const shown = groupDecimal(shiftDecimal(value, places), 2);
  const sign = shown.startsWith("-") ? "-" : "";
  const currency =
    unit.kind === "currency" || unit.kind === "per_share"
      ? unit.currency
        ? `${unit.currency} `
        : ""
      : "";
  const scale = unit.scale ? ` ${unit.scale}` : "";
  const perShare = unit.kind === "per_share" ? " per share" : "";
  return `${sign}${currency}${shown.slice(sign.length)}${scale}${perShare}`;
}

const UNIT_LABELS: Record<CalculationUnit["kind"], string> = {
  currency: "Currency",
  percent: "Percent",
  ratio: "Ratio",
  count: "Count",
  per_share: "Per share",
  unitless: "Number",
};

/** e.g. `Currency · USD million`; an ambiguous currency symbol is called out. */
export function describeUnit(unit: CalculationUnit): string {
  if (unit.display === "boolean") return KIND_LABELS.boolean;
  if (unit.kind === "ratio" && unit.display === "percent")
    return UNIT_LABELS.percent;
  const money = unit.kind === "currency" || unit.kind === "per_share";
  const detail = [
    money ? (unit.currency ?? "currency unspecified") : undefined,
    unit.scale ?? undefined,
  ].filter(Boolean);
  return detail.length
    ? `${UNIT_LABELS[unit.kind]} · ${detail.join(" ")}`
    : UNIT_LABELS[unit.kind];
}

/** How a step result is shown: backend units when present, the legacy guess otherwise. */
export type Shown = {
  kind: CalculationKind | null | undefined;
  unit: CalculationUnit | null | undefined;
  legacy: Presentation;
};

export function stepDisplays(calculation: Calculation): Shown[] {
  const legacy = calculation.steps.some((step) => !step.unit)
    ? presentations(calculation)
    : [];
  return calculation.steps.map((step, index) => ({
    kind: step.kind,
    unit: step.unit,
    legacy: legacy[index] ?? {},
  }));
}

export function formatShown(value: string, shown: Shown | undefined): string {
  return shown?.unit
    ? formatUnit(value, shown.unit)
    : formatValue(value, shown?.kind, shown?.legacy);
}

export function shownLabel(shown: Shown | undefined): string {
  return shown?.unit ? describeUnit(shown.unit) : unitLabel(shown?.kind);
}

export function stepRef(index: number): string {
  return `#${index}`;
}

export function formatOperand(
  operand: CalculationOperand,
  steps: CalculationStep[],
  shown: Shown[],
): string {
  if (operand.kind === "evidence")
    return `“${operand.quote ?? operand.source_value ?? operand.value}” [${operand.citation_id ?? "?"}]`;
  if (operand.kind === "step") {
    const index = operand.step ?? -1;
    const step = steps[index];
    return step
      ? `${stepRef(index)} (${formatShown(step.result, shown[index])})`
      : stepRef(index);
  }
  return operand.value;
}

/** e.g. `subtract(“USD 150 million” [C1], “USD 120 million” [C2]) = USD 30 million` */
export function formatDerivation(
  calculation: Calculation,
  index: number,
): string {
  const shown = stepDisplays(calculation);
  const step = calculation.steps[index];
  const args = step.operands
    .map((operand) => formatOperand(operand, calculation.steps, shown))
    .join(", ");
  return `${step.op}(${args}) = ${formatShown(step.result, shown[index])}`;
}

export function finalResult(calculation: Calculation): {
  text: string;
  kind: CalculationKind | null | undefined;
  label: string;
} {
  const shown = stepDisplays(calculation);
  const last = shown.at(-1);
  // The calculation-level unit is authoritative; it equals the last step's unit.
  const final: Shown | undefined = last && {
    ...last,
    unit: calculation.unit ?? last.unit,
  };
  return {
    text: formatShown(calculation.result, final),
    kind: final?.kind,
    label: shownLabel(final),
  };
}

export type Segment = { text: string; unsupported: boolean };

/**
 * Splits text around unsupported number mentions reported by the verifier.
 * Matches only whole mentions: `31%` never matches inside `131%` or `31.5%`.
 */
export function splitUnsupported(text: string, numbers: string[]): Segment[] {
  const needles = [...new Set(numbers.map((item) => item.trim()))]
    .filter((item) => item.length > 0 && /\d/.test(item))
    .sort((a, b) => b.length - a.length);
  if (!needles.length || !text) return [{ text, unsupported: false }];
  const segments: Segment[] = [];
  let start = 0;
  let plain = 0;
  while (start < text.length) {
    let hit: string | undefined;
    for (const needle of needles) {
      if (!text.startsWith(needle, start)) continue;
      const before = text[start - 1] ?? "";
      const after = text.slice(
        start + needle.length,
        start + needle.length + 2,
      );
      const leftOk = !/[\w.,]/.test(before) || !/^[\w]/.test(needle);
      const rightOk = !/^\d|^[.,]\d/.test(after) && !/^[A-Za-z]/.test(after);
      if (leftOk && rightOk) {
        hit = needle;
        break;
      }
    }
    if (!hit) {
      start += 1;
      continue;
    }
    if (start > plain)
      segments.push({ text: text.slice(plain, start), unsupported: false });
    segments.push({ text: hit, unsupported: true });
    start += hit.length;
    plain = start;
  }
  if (plain < text.length)
    segments.push({ text: text.slice(plain), unsupported: false });
  return segments;
}

/** Splits a source quote around the first occurrence of an operand span. */
export function highlightSpan(text: string, span?: string): Segment[] {
  const at = span ? text.indexOf(span) : -1;
  if (!span || at < 0) return [{ text, unsupported: false }];
  return [
    { text: text.slice(0, at), unsupported: false },
    { text: span, unsupported: true },
    { text: text.slice(at + span.length), unsupported: false },
  ].filter((segment) => segment.text);
}

export const VERIFICATION_LABELS: Record<VerificationStatus, string> = {
  verified: "Numbers verified",
  unverified: "Unverified numbers",
  failed: "Verification failed",
  skipped: "Verification skipped",
};

/** Client text for each server reason code; contract.test.ts checks every code is covered. */
export const REASONS: Record<string, string> = {
  out_of_scope:
    "Forecasts, predictions and investment advice are outside what the disclosures can support.",
  unsupported_numbers:
    "Some numbers are not found in the cited evidence or in a calculation bound to it.",
  verifier_unavailable:
    "The optional groundedness model was unavailable, so only deterministic checks ran.",
  verifier_below_threshold:
    "The optional groundedness model scored this answer below its threshold.",
  no_validated_answer:
    "No answer passed citation validation for the authorized evidence.",
  verification_failed:
    "Numeric verification failed, so the answer was withheld.",
  retrieval_score_below_threshold:
    "The best retrieved evidence scored below the configured relevance threshold.",
  citation_coverage_below_threshold:
    "Too little of the draft answer was supported by citations.",
  tool_errors_above_threshold: "Too many calculation or search steps failed.",
};

export function describeReason(reason: string): string {
  return (
    REASONS[reason] ??
    `${reason.replaceAll("_", " ").replace(/^./, (c) => c.toUpperCase())}.`
  );
}

const STAGE_KINDS: Record<Activity["stage"], ActivityKind> = {
  planning: "planning",
  retrieval: "retrieval",
  generation: "generation",
  tool: "tool_check",
  validation: "citation_validation",
};

/** Machine-readable operation. Labels are display text, read only for older servers. */
export function activityKind(activity: Activity): ActivityKind {
  if (activity.kind) return activity.kind;
  if (activity.stage === "tool")
    return activity.tool === "calculator"
      ? "calculator"
      : activity.tool === "search_evidence"
        ? "evidence_search"
        : "tool_check";
  // Before `kind`, the two validation operations differed only by label.
  if (
    activity.stage === "validation" &&
    activity.label === "Verifying grounding"
  )
    return "grounding_verification";
  return STAGE_KINDS[activity.stage];
}

export type TimelineEntry = Activity & {
  key: string;
  done: boolean;
  kind: ActivityKind;
};

/** Folds started/completed pairs into one row per operation, preserving order. */
export function timeline(activities: Activity[]): TimelineEntry[] {
  const rows: TimelineEntry[] = [];
  activities.forEach((activity, index) => {
    const kind = activityKind(activity);
    if (activity.status === "completed") {
      for (let i = rows.length - 1; i >= 0; i -= 1) {
        const row = rows[i];
        if (
          !row.done &&
          row.kind === kind &&
          (row.tool ?? null) === (activity.tool ?? null)
        ) {
          // The plan summary only arrives with the completed planning event.
          rows[i] = {
            ...row,
            status: "completed",
            done: true,
            plan: activity.plan ?? row.plan,
          };
          return;
        }
      }
    }
    rows.push({
      ...activity,
      kind,
      key: `${index}-${kind}`,
      done: activity.status === "completed",
    });
  });
  return rows;
}

const INTENTS: Record<PlanSummary["intent"], string> = {
  lookup: "Look up a reported figure",
  compare: "Compare figures",
  ratio: "Compute a ratio",
  change: "Measure a change",
  percent_change: "Measure a percentage change",
  aggregate: "Aggregate figures",
};

const ROUTES: Record<PlanSummary["route"], string> = {
  retrieve: "Search the disclosures",
  direct_answer: "Answer without new evidence",
  clarify: "Ask for clarification",
  refuse_out_of_scope: "Decline: outside the disclosures' scope",
};

/** Plain-language plan facts for the planning timeline row. */
export function describePlan(
  plan: PlanSummary,
): { term: string; values: string[] }[] {
  const facts = [
    { term: "Intent", values: [INTENTS[plan.intent]] },
    { term: "Route", values: [ROUTES[plan.route]] },
  ];
  if (plan.companies.length)
    facts.push({ term: "Companies", values: plan.companies });
  if (plan.periods.length)
    facts.push({ term: "Periods", values: plan.periods });
  if (plan.subquestion_count) {
    // The server sends at most the first four sub-questions, each clipped.
    const hidden = plan.subquestion_count - plan.subquestions.length;
    facts.push({
      term: `Searches (${plan.subquestion_count})`,
      values: [
        ...plan.subquestions,
        ...(hidden > 0 ? [`and ${hidden} more`] : []),
      ],
    });
  }
  return facts;
}
