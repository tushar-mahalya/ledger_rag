import type {
  Activity,
  Calculation,
  CalculationKind,
  CalculationOperand,
  CalculationStep,
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

/** Scale word and currency shared by every evidence operand a step depends on. */
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

export function stepRef(index: number): string {
  return `#${index}`;
}

export function formatOperand(
  operand: CalculationOperand,
  steps: CalculationStep[],
  shown: Presentation[],
): string {
  if (operand.kind === "evidence")
    return `“${operand.quote ?? operand.source_value ?? operand.value}” [${operand.citation_id ?? "?"}]`;
  if (operand.kind === "step") {
    const index = operand.step ?? -1;
    const step = steps[index];
    return step
      ? `${stepRef(index)} (${formatValue(step.result, step.kind, shown[index])})`
      : stepRef(index);
  }
  return operand.value;
}

/** e.g. `subtract(“USD 150 million” [C1], “USD 120 million” [C2]) = USD 30 million` */
export function formatDerivation(
  calculation: Calculation,
  index: number,
): string {
  const shown = presentations(calculation);
  const step = calculation.steps[index];
  const args = step.operands
    .map((operand) => formatOperand(operand, calculation.steps, shown))
    .join(", ");
  return `${step.op}(${args}) = ${formatValue(step.result, step.kind, shown[index])}`;
}

export function finalResult(calculation: Calculation): {
  text: string;
  kind: CalculationKind | null | undefined;
} {
  const shown = presentations(calculation);
  const last = calculation.steps.length - 1;
  const kind = calculation.steps[last]?.kind;
  return {
    text: formatValue(calculation.result, kind, shown[last]),
    kind,
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

const REASONS: Record<string, string> = {
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

export type TimelineEntry = Activity & { key: string; done: boolean };

/** Folds started/completed pairs into one row per operation, preserving order. */
export function timeline(activities: Activity[]): TimelineEntry[] {
  const rows: TimelineEntry[] = [];
  activities.forEach((activity, index) => {
    if (activity.status === "completed") {
      for (let i = rows.length - 1; i >= 0; i -= 1) {
        const row = rows[i];
        if (
          !row.done &&
          row.stage === activity.stage &&
          row.label === activity.label &&
          (row.tool ?? null) === (activity.tool ?? null)
        ) {
          rows[i] = { ...row, status: "completed", done: true };
          return;
        }
      }
    }
    rows.push({
      ...activity,
      key: `${index}-${activity.stage}`,
      done: activity.status === "completed",
    });
  });
  return rows;
}
