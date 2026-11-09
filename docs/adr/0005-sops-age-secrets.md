# ADR 0005: SOPS + age as the secret source of truth

- Status: accepted; supersedes the self-hosted Vault design
- Date: 2026-11-09

## Context

The runtime is Modal (GPU services and jobs) with deployment planned on
Cloudflare; development happens on one macOS workstation. The previous design ran
a single-node Vault with manual unseal plus Vault Agent, AppRole and a Modal JWT
role. That Vault was never reachable from Modal, so jobs depended on a second,
manual projection script, and the operator could not explain or operate the
moving parts. Requirements: no self-hosted service, generous free tier, no extra
cloud account just for secrets, non-interactive token-based operation, least
privilege per runtime, versioned changes, and no secrets in images or the public
repository.

## Decision

Keep encrypted values in a **private git repository** using
[SOPS](https://getsops.io/docs/usage/identities/age/) with an
[age](https://github.com/FiloSottile/age) identity stored in the macOS Keychain.
A public [manifest](../../config/secrets.toml) declares every secret and its
delivery routes. `ledgerctl` decrypts in memory and either runs a local command
with the routed variables or replaces the runtime-native store of each target:
Modal Secrets now, Cloudflare Worker secrets once the Worker exists. Platform
tokens are operator-only by manifest rule. See [secrets](../SECRETS.md).

## Alternatives considered (2026-11-09)

| Option | Why not chosen |
|---|---|
| Self-hosted Vault / OpenBao | Needs an always-on host with persistent storage, TLS, unseal and backups; ephemeral Cloudflare/Modal containers cannot host it |
| Vault image on GHCR / Docker Hub | A registry stores images, it does not run a service; baking secrets into images exposes them to anyone who can pull and persists them in layers |
| Doppler (free) | Good UX, but keyless OIDC identities need the Team plan, there is no Modal integration and the activity log keeps 3 days |
| Infisical Cloud (free) | 5 identities in total, versioning and audit retention are paid; a free SaaS would also become a container start-up dependency |
| GCP Secret Manager + workload identity | Strong properties, but the operator's GCP account could not be set up, and billing on the Gemini project changes its tier |
| AWS Parameter Store + Modal OIDC | Free and the best-documented keyless path, but an AWS account only for secrets is unjustified once deployment targets Cloudflare |
| Modal Secrets only | No history, rollback or single place covering Cloudflare and local runs |

## Consequences

- $0, no new vendor, no service to operate, and no runtime dependency on a
  secrets provider: containers start from platform-native stores.
- Changes are reviewable git commits in the private store; `sops set
  --value-stdin` keeps unrelated ciphertexts unchanged.
- Runtimes hold delivered copies (no keyless access), and reads are not audited.
- The age identity is a crown jewel: Keychain plus one offline backup. Its
  compromise exposes every stored revision and requires rotating all providers.
- The previous Vault artifacts (`deploy/vault`, the production Compose overlay,
  `wait-for-secrets.py`, the Modal projection and JWT scripts and the
  `workload_identity` helper) are removed. Historical validation records that
  mention them remain as history.
- Release gates: a first live `ledgerctl push dev` against the Modal workspace,
  a Cloudflare target once the Worker exists, and a CI identity as a separate
  recipient before automated deploys.
