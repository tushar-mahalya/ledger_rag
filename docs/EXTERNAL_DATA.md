# SEC and FRED evidence

External evidence is an opt-in **chat-only** extension, not a replacement for the
authorized corpus or a financial-advice service. It uses the existing SEC EDGAR
and FRED/ALFRED clients. No external tools are offered or executed on voice.

## Enablement and authorization

`LEDGER_EXTERNAL_DATA_ENABLED=false` is the default. With it enabled,
`LEDGER_SEC_ENABLED` and `LEDGER_FRED_ENABLED` control providers independently.
A principal needs the `reader` role and, when configured,
`LEDGER_EXTERNAL_REQUIRED_GROUP` and membership in
`LEDGER_EXTERNAL_ALLOWED_TENANTS`. These come from authentication, never tool
arguments. A missing or rejected FRED credential disables FRED rather than
falling back to current observations.

SEC requires `LEDGER_SEC_USER_AGENT` with a contact email. The FRED key is
`LEDGER_FRED_API_KEY` in the encrypted store ([secrets](SECRETS.md)); never put
credentials in command arguments, record request URLs, or log upstream bodies.

## Read-only tools

| Tool | Selection | Bounds |
| --- | --- | --- |
| `lookup_company_facts` | Ticker, CIK or exact registrant; XBRL concepts and fiscal/calendar periods | Four concepts, four periods, eight pairs by default |
| `lookup_macro_series` | Series plus up to six dates, a date range, or latest observation | Six requested dates; bounded client output and response bytes |

The chat session fixes the authenticated principal and `as_of`, capped at today,
for the entire turn. **`as_of` means publication availability**, not fiscal year:
SEC filing dates and ALFRED vintage dates must precede the cutoff. Missing
values remain explicit misses; a current FRED response cannot substitute for
unavailable vintage history. Amendment, restatement, staleness and later
revision indicators stay attached to the original value.

Explicit monthly, quarterly, semiannual and annual dates require a published
observation whose calendar-period bounds contain the requested date. An
unpublished current period is not replaced with the previous period. Daily
series alone allow a bounded prior observation for weekends/holidays (less
than seven days); other submonthly series require an exact observation date.

Facts join the existing twelve-citation pool as `C<n>`. Calculator bindings,
numeric verification and final citation validation apply to them exactly as to
corpus evidence. `source_kind` is `corpus`, `sec_xbrl` or `fred`; external
citations require matching typed provenance. Source links use only HTTPS SEC
and FRED human-facing hosts. Tool results and documents remain untrusted data.
SEC evidence IDs include the reported unit as their final component, so
different currencies or per-share units cannot collapse into one citation.
Readback pins the unit as well as accession and period; legacy unitless IDs
are accepted only when the filing's pinned value/unit is unambiguous.
External evidence identities permit at most 261 characters to accommodate the
existing 132-character concept and 60-character unit limits. Corpus document
identities retain their 200-character limit; this does not change index identity.

## Failure and resource limits

- Two external calls per chat turn by default, inside the existing overall tool
  budget; exhausted budgets return explicit tool errors.
- Eight seconds per lookup by default, also bounded by the query deadline.
  Cancellation propagates and releases half-open breaker probes.
- Coalesced HTTP fetches have their own total deadline. Cancelling one waiter
  preserves a fetch needed by another; the last departing waiter cancels and
  drains it. Shutdown cancels and awaits all fetches, including streams already
  draining outside the coalescing map, before closing provider clients.
- Independent provider breakers: five health failures, thirty-second reset by
  default. Input errors do not count as provider failures.
- SEC defaults to five requests/second; FRED to sixty/minute. These limits,
  caches and breakers are **per process**. Coordinate gateway/replica quotas
  before scaling; SEC's fair-access ceiling is not a per-replica allowance.
- Fifteen-minute bounded response cache; sixteen-megabyte response limit by
  default. No arbitrary URL fetches, redirects, ingestion or mutations.
- Activity completion reports `ok`, `error`, `timeout` or `unavailable`.
  Failures contain sanitized categories, never credential-bearing URLs or
  raw upstream exception text. No silent corpus/provider fallback.

## Benchmarks and verification

Closed-corpus evaluation refuses external enablement before reading corpus data
or constructing remote clients. FinQA settings explicitly disable ambient
external enablement and reject attempts to override it. External activity and
provenance are tested separately from benchmark correctness.

```sh
uv run --locked pytest -q services/engine/tests/unit/test_external_*.py services/engine/tests/unit/test_mcp_adapter.py
uv run --locked python scripts/dump_contract_fixtures.py --check
npm --prefix services/web test -- src/external.test.ts src/contract.test.ts
```

Recorded provider fixtures are verified locally; a regression rejects any
`api_key` occurrence or fixture credential value in their contents. The FRED
credential-error fixture has its field spelling and documentation URL sanitized.
The web demo labels its SEC/FRED replay as an example, not a live request.
### Scoped live verification

Verification record `fred-live-20261101T124703Z` used the existing authorized
Vault credential, with no secret provisioning, fixture recording or raw-value
logging. Both FRED requests returned **HTTP 200**:

- `FEDFUNDS` metadata: `realtime_start=realtime_end=2023-10-15`.
- Observations: `observation_start=observation_end=2023-09-01`. The existing
  client requested vintage history and selected only data known at the cutoff.
- Selected publication/vintage date: **2023-10-02**, preceding the **2023-10-15**
  cutoff. The resulting citation passed typed provenance, identity and
  publication-cutoff validation.

This verifies one tightly bounded historical observation, not every series,
ongoing availability or real voice latency. Live SEC verification and broader
provider coverage remain release checks. No paid jobs or deployments ran.

Implementation: [default limits](../services/engine/src/ledger_rag/core/config.py#L175-L192),
[tool schemas](../services/engine/src/ledger_rag/external/tools.py#L29-L47),
[evidence identity limit](../services/engine/src/ledger_rag/core/provenance.py#L21-L25).

Sources: [SEC APIs and fair access](https://www.sec.gov/search-filings/edgar-application-programming-interfaces),
[FRED observations and real-time periods](https://fred.stlouisfed.org/docs/api/fred/series_observations.html).
