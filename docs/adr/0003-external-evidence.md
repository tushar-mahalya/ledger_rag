# ADR 0003: Point-in-time external evidence belongs to chat

- Status: accepted for the opt-in reference implementation
- Date: 2026-11-01

## Context

The corpus may lack a company figure or a macro observation. Fetching the
latest value can leak later filings or revisions into historical answers and
closed-corpus benchmarks. External responses are also untrusted tool data.

## Decision

Reuse official SEC EDGAR XBRL and FRED/ALFRED clients, behind authenticated
entitlements, fixed publication cutoffs, per-turn budgets, deadlines, provider
breakers and bounded read-only requests. Normalize facts into ordinary
citations with typed provenance. Preserve calculator and citation-verification
gates. Never offer these tools to voice or closed-corpus benchmarks.

Current-only FRED fallback, arbitrary browsing, silent provider fallback and
external corpus ingestion are rejected. A missing or unavailable historical
value is an explicit miss or sanitized failure.

## Consequences and release gates

Chat gains traceable structured data without changing index identity. Public
external responses may be cached, but tenant corpus data stays isolated.
Replica-wide SEC/FRED quotas and authorized live provider verification remain
release gates; local fixture parity is not a live availability measurement.
See [external evidence](../EXTERNAL_DATA.md) for bounds and sources.
