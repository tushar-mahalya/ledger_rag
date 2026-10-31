"""Deterministic document-level context for chunk profiles v2/v3.

Headers carry only facts derivable from the document itself: the FinQA identifier
(``TICKER/YEAR/page_N.pdf``) yields ticker, fiscal year and page, and the corpus is
10-K pages. Company names are not mapped from tickers: FinQA ships no mapping and
tickers were reused across renamed filers, so the header never invents a name.
Section headings come from a fixed list of canonical 10-K headings and captions from
the sentence that introduces a table; both are omitted when not detected.

Profile v3 adds a per-document LLM summary. Summaries are generated once on Modal
(``compute/summaries.py``), cached by document content hash, and read back by
ingestion so re-indexing never re-bills or drifts.
"""

import asyncio
import hashlib
import json
import random
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import httpx

from ledger_rag.core.models import Document

_FINQA_ID = re.compile(
    r"^(?P<ticker>[A-Z][A-Z0-9.\-]{0,9})/(?P<year>(?:19|20)\d{2})/page_(?P<page>\d+)\.pdf$"
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_SPACE = re.compile(r"\s+")
_APOSTROPHE = r"(?:'|’|\s?2019)?"  # FinQA renders U+2019 as " 2019"
_HEADINGS: tuple[tuple[re.Pattern[str], str], ...] = tuple(
    (re.compile(pattern), label)
    for pattern, label in [
        (r"\bitem\s*1a\s*\.?\s*risk factors\b", "Item 1A. Risk Factors"),
        (r"\bitem\s*1b\s*\.?\s*unresolved staff comments\b", "Item 1B. Unresolved Staff Comments"),
        (r"\bitem\s*1\s*\.?\s*business\b", "Item 1. Business"),
        (r"\bitem\s*2\s*\.?\s*properties\b", "Item 2. Properties"),
        (r"\bitem\s*3\s*\.?\s*legal proceedings\b", "Item 3. Legal Proceedings"),
        (
            r"\bitem\s*5\s*\.?\s*market for (?:the )?registrant",
            "Item 5. Market for Registrant's Common Equity",
        ),
        (
            r"\bitem\s*6\s*\.?\s*selected (?:consolidated )?financial data\b",
            "Item 6. Selected Financial Data",
        ),
        (
            r"\bitem\s*7a\s*\.?\s*quantitative and qualitative disclosures?\b",
            "Item 7A. Quantitative and Qualitative Disclosures About Market Risk",
        ),
        (
            rf"\bitem\s*7\s*\.?\s*management{_APOSTROPHE}s? discussion\b",
            "Item 7. Management's Discussion and Analysis",
        ),
        (
            r"\bitem\s*8\s*\.?\s*financial statements\b",
            "Item 8. Financial Statements and Supplementary Data",
        ),
        (r"\bitem\s*9a\s*\.?\s*controls and procedures\b", "Item 9A. Controls and Procedures"),
        (
            r"\bnotes to (?:the )?consolidated financial statements\b",
            "Notes to Consolidated Financial Statements",
        ),
        (
            rf"\bmanagement{_APOSTROPHE}s? discussion and analysis\b",
            "Management's Discussion and Analysis",
        ),
        (
            r"\bquantitative and qualitative disclosures? about market risk\b",
            "Quantitative and Qualitative Disclosures About Market Risk",
        ),
        (r"\bselected (?:consolidated )?financial data\b", "Selected Financial Data"),
        (
            r"\breport of independent registered public accounting firm\b",
            "Report of Independent Registered Public Accounting Firm",
        ),
    ]
)
_CAPTION = re.compile(
    r"following table|table below|tables? (?:sets?|set) forth|as follows|summari[sz]"
    r"|\bin (?:millions|thousands|billions)\b|:\s*\.?$"
)
CAPTION_CHARS = 240
SUMMARY_CHARS = 600


@dataclass(frozen=True)
class DocumentContext:
    title: str
    ticker: str | None = None
    fiscal_year: int | None = None
    page: int | None = None
    form_type: str | None = None


def document_context(document_id: str, title: str) -> DocumentContext:
    """FinQA identifiers yield filing fields; other documents keep only their title."""
    match = _FINQA_ID.fullmatch(document_id)
    if not match:
        return DocumentContext(title=_clip(title, 300))
    return DocumentContext(
        title=_clip(title, 300),
        ticker=match["ticker"],
        fiscal_year=int(match["year"]),
        page=int(match["page"]),
        form_type="10-K",
    )


def section_heading(text: str) -> str | None:
    """Canonical label of the last recognized 10-K heading in ``text``."""
    lowered = text.lower()
    matches = sorted(
        (match.start(), -match.end(), label)
        for pattern, label in _HEADINGS
        for match in pattern.finditer(lowered)
    )
    label: str | None = None
    end = -1
    for start, negative_end, candidate in matches:
        # A generic heading inside a numbered item ("item 7 . management's discussion")
        # belongs to that item.
        if start < end:
            continue
        label, end = candidate, -negative_end
    return label


def table_caption(preceding_prose: str) -> str | None:
    """The sentence introducing a table, when it reads like a caption or unit line."""
    sentences = [s for s in _SENTENCE.split(preceding_prose.strip()) if s.strip()]
    if not sentences:
        return None
    last = _SPACE.sub(" ", sentences[-1]).strip()
    if not _CAPTION.search(last.lower()):
        return None
    return _clip(last, CAPTION_CHARS)


def render_header(
    context: DocumentContext,
    *,
    section: str | None = None,
    caption: str | None = None,
    summary: str | None = None,
) -> str:
    if context.ticker and context.form_type:
        lines = [
            f"Filing: {context.ticker} Form {context.form_type}, fiscal year "
            f"{context.fiscal_year}, page {context.page}"
        ]
    else:
        lines = [f"Document: {context.title}"]
    if section:
        lines.append(f"Section: {section}")
    if caption:
        lines.append(f"Table: {caption}")
    if summary:
        lines.append(f"Summary: {_clip(summary, SUMMARY_CHARS)}")
    return "\n".join(lines)


def document_hash(title: str, text: str) -> str:
    """Content hash used to cache per-document summaries."""
    return hashlib.sha256(f"{title}\0{text}".encode()).hexdigest()


def _clip(text: str, limit: int) -> str:
    text = _SPACE.sub(" ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rsplit(" ", 1)[0] or text[: limit - 1]
    return cut + "…"


SUMMARY_MODEL = "gemini-3.5-flash-lite"
SUMMARY_PROMPT_VERSION = "s1"
SUMMARY_SYSTEM = (
    "You summarize one page of an SEC 10-K filing for a search index. Write at most "
    "three sentences (under 80 words) naming the topics, line items, segments and "
    "periods the page covers. Use only facts stated on the page; never add company "
    "names, numbers or claims that are not on the page. No preamble."
)
_RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}


class SummaryError(RuntimeError):
    pass


def summary_cache_dir(root: Path, model: str = SUMMARY_MODEL) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", model):
        raise ValueError("Invalid summary model name")
    return root / SUMMARY_PROMPT_VERSION / model


def _summary_file(cache_dir: Path, key: str) -> Path:
    return cache_dir / key[:2] / f"{key}.json"


def load_summaries(cache_dir: Path) -> dict[str, str]:
    """Map document content hash to cached summary."""
    summaries: dict[str, str] = {}
    for path in sorted(cache_dir.glob("*/*.json")):
        record = json.loads(path.read_text())
        key, summary = record.get("document_hash"), record.get("summary")
        if path.stem != key or not isinstance(summary, str) or not summary.strip():
            raise SummaryError(f"Corrupt summary cache entry {path.name}")
        summaries[key] = summary
    return summaries


def _clean_summary(content: object) -> str:
    if not isinstance(content, str) or not content.strip():
        raise SummaryError("Empty summary")
    return _clip(content, SUMMARY_CHARS)


async def summarize_documents(
    documents: Iterable[Document],
    cache_dir: Path,
    client: httpx.AsyncClient,
    *,
    model: str = SUMMARY_MODEL,
    concurrency: int = 8,
    attempts: int = 5,
    base_delay: float = 1.0,
    on_written: Callable[[int], None] | None = None,
) -> dict[str, int]:
    """Fill the summary cache; ``client`` carries the LLM base URL and bearer key.

    At most ``concurrency`` (<= 8) requests are in flight. Rate limits, 5xx and
    transport errors are retried with jittered exponential backoff; other 4xx fail fast.
    """
    if not 1 <= concurrency <= 8:
        raise ValueError("Summary concurrency must be between 1 and 8")
    if attempts < 1:
        raise ValueError("At least one attempt is required")
    semaphore = asyncio.Semaphore(concurrency)
    stats = {"cached": 0, "generated": 0, "failed": 0}
    pending: dict[str, Document] = {}
    for document in documents:
        key = document_hash(document.title, document.text)
        if _summary_file(cache_dir, key).exists():
            stats["cached"] += 1
        else:
            pending.setdefault(key, document)

    async def request(document: Document) -> str:
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": f"{document.title}\n\n{document.text[:24000]}"},
            ],
        }
        for attempt in range(attempts):
            retry = attempt + 1 < attempts
            try:
                response = await client.post("chat/completions", json=body)
            except httpx.TransportError:
                if not retry:
                    raise SummaryError("LLM transport failure") from None
            else:
                if response.status_code == 200:
                    return _clean_summary(response.json()["choices"][0]["message"].get("content"))
                if response.status_code not in _RETRYABLE or not retry:
                    raise SummaryError(f"LLM returned HTTP {response.status_code}")
            await asyncio.sleep(base_delay * 2**attempt * random.uniform(0.5, 1.5))
        raise SummaryError("Summary attempts exhausted")

    async def one(key: str, document: Document) -> None:
        async with semaphore:
            try:
                summary = await request(document)
            except (SummaryError, KeyError, IndexError, TypeError, ValueError) as error:
                stats["failed"] += 1
                reason = str(error) if isinstance(error, SummaryError) else type(error).__name__
                stats[f"failed: {reason}"] = stats.get(f"failed: {reason}", 0) + 1
                return
        target = _summary_file(cache_dir, key)
        record = {
            "document_hash": key,
            "document_id": document.id,
            "model": model,
            "prompt_version": SUMMARY_PROMPT_VERSION,
            "summary": summary,
        }
        await asyncio.to_thread(_write_atomic, target, json.dumps(record))
        stats["generated"] += 1
        if on_written:
            on_written(stats["generated"])

    await asyncio.gather(*(one(key, document) for key, document in pending.items()))
    return stats


def _write_atomic(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(content)
    temporary.replace(target)
