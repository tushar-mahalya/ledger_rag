# External data fixtures

Offline inputs for `tests/unit/test_external_*.py`. No test touches the network.

## SEC EDGAR (`sec/`)

These are trimmed recordings of the official `data.sec.gov` and `www.sec.gov` responses,
captured in 2026 with a declared User-Agent. They keep the original structure, values,
accession numbers, form types and filing dates. Only the fact lists were cut down.

| File | Source | Exercises |
|---|---|---|
| `company_tickers.json` | `www.sec.gov/files/company_tickers.json` | Ticker map. Share classes such as GOOGL/GOOG and BRK-A/BRK-B share one CIK. |
| `concept_0000320193_us-gaap_InterestExpense.json` | companyconcept | Apple FY2021-FY2023 interest expense from 10-K and 10-Q filings |
| `concept_0000320193_us-gaap_SalesRevenueNet.json` | companyconcept | Apple FY2009 revenue: the 10-K filed 2009-10-27 reported 36,537,000,000. The 10-K/A filed 2010-01-25 restated it to 42,905,000,000. |
| `concept_0000320193_us-gaap_Assets.json` | companyconcept | Instant facts. Includes the same-day 10-Q and 10-K/A restatement on 2010-01-25. |
| `concept_0000320193_us-gaap_EarningsPerShareDiluted.json` | companyconcept | `USD/shares` unit |
| `concept_0000320193_us-gaap_EffectiveIncomeTaxRateContinuingOperations.json` | companyconcept | `pure` unit |
| `concept_0000320193_dei_EntityCommonStockSharesOutstanding.json` | companyconcept | `shares` unit and the `dei` taxonomy |
| `concept_0001306965_ifrs-full_Revenue.json` | companyconcept | IFRS filer (Shell): 20-F, 20-F/A and 6-K forms, with a null label |
| `companyfacts_0000320193.json` | companyfacts | Concept names and labels used for suggestions |
| `submissions_0000320193.json` | submissions | Registrant name and tickers |

Two files are synthetic rather than recorded:

- `concept_0001067983_us-gaap_EarningsPerShareBasic.json`: one filing reports two values for the same period, one per share class, with no dimensions. Ledger must refuse to pick one.
- `not_found.xml`: the body that `data.sec.gov` returns for a missing concept (HTTP 404, S3 `NoSuchKey`).

## FRED / ALFRED (`fred/`)

No FRED API key was available, so these files are hand-built to the documented response
shapes:

- [series](https://fred.stlouisfed.org/docs/api/fred/series.html)
- [series/observations](https://fred.stlouisfed.org/docs/api/fred/series_observations.html), requested with `realtime_start=1776-07-04` and `realtime_end=9999-12-31`
- [errors](https://fred.stlouisfed.org/docs/api/fred/errors.html)

FEDFUNDS values match the published monthly averages. Vintage dates and the GDP revision
path are illustrative.

Do not cite these files as data. Use the opt-in live evaluation (`tests/live`) for real
values.
