# Coding-agent workflows

This repository uses the open [Agent Skills specification](https://agentskills.io/specification):
plain `SKILL.md` files, portable YAML frontmatter and task-specific instructions.
No agent framework, plugin server, cloud account or paid model is needed to read
or validate them.

## Single source of truth

```text
AGENTS.md                  shared always-on working agreement
CLAUDE.md                  imports @AGENTS.md
.agents/skills/*/SKILL.md   canonical on-demand skills (ledger-* = project, others vendored)
.agents/skills/*/SOURCE.md  provenance and file hashes of a vendored skill
.claude/skills/<name>      per-skill symlink to ../../.agents/skills/<name>
.mcp.json                  reviewed read-only documentation MCP servers (Claude Code project scope)
```

| Client | Instructions | Skill discovery |
|---|---|---|
| Codex | Root `AGENTS.md` | `.agents/skills` |
| Claude Code | `CLAUDE.md` imports `AGENTS.md` | Per-skill aliases in `.claude/skills` |
| OpenCode | Root `AGENTS.md` | `.agents/skills` / Claude-compatible skill locations |
| VS Code Copilot | Root `AGENTS.md` when instruction discovery is enabled | `.agents/skills` and `.claude/skills` are documented locations |

The **format** is portable; automatic discovery, permissions, slash commands,
configuration and symlink behavior are client/version-dependent. This is not a
claim that every agentic tool supports skills. For another client, configure its
skill path or explicitly load a canonical file. Do not duplicate skill bodies.

Claude's official docs explicitly support symlinked `<skill-name>` entries;
they do not explicitly guarantee a symlinked parent skills directory. This repo
therefore uses individual skill links. Codex, OpenCode and Copilot can use the
canonical directory without depending on symlink support.

On Windows, use a checkout with symlinks enabled. If skill links become text
files, Claude discovery will not work; recreate the directory symlinks or
configure that client to load the canonical files. CI checks every alias target.
Restart/reload a client after adding skills. Clients that scan both locations may
deduplicate by skill name; confirm the selected source rather than assuming it.

## Project workflows and provider skills, not a component instruction stack

Keep `.agents/skills` canonical and `.claude/skills` as discovery aliases, not
copies. Neither directory improves model reasoning by itself; the useful
difference is which clients discover it.

The original eight skills were reduced to project workflows plus unmodified
provider skills. Retrieval, generation, operations and research skills were
retired: they mostly repeated component documentation and general engineering
advice. `ledger-voice` returns only with what docs and training data lack: the
pinned Pipecat 1.14 API names, frame-ordering and Media Streams rules the code
relies on. Shared safety invariants remain in [AGENTS.md](../AGENTS.md).
This follows the [Agent Skills authoring guidance](https://agentskills.io/skill-creation/best-practices):
add non-obvious project knowledge, keep coherent task boundaries, and omit what
the agent already knows. It is not a measured claim of better model performance.

| Skill | Use for |
|---|---|
| [ledger-evaluation](../.agents/skills/ledger-evaluation/SKILL.md) | Executing/comparing benchmarks; corpus, split, cloning and metric pitfalls |
| [ledger-compute](../.agents/skills/ledger-compute/SKILL.md) | Executing/releasing project compute jobs; auth, placement and promotion gates |
| [ledger-voice](../.agents/skills/ledger-voice/SKILL.md) | Twilio Media Streams, Pipecat 1.14 pipeline and ElevenLabs service rules in `services/voice` |
| [modal](../.agents/skills/modal/SKILL.md) | Upstream Modal SDK/CLI guidance |
| [wrangler](../.agents/skills/wrangler/SKILL.md) | Upstream Cloudflare Wrangler guidance; triggers only once a Wrangler config exists |
| [qdrant-multitenancy](../.agents/skills/qdrant-multitenancy/SKILL.md) | Upstream Qdrant tenant-isolation guidance (the repo already uses `is_tenant` payload partitioning) |

Ordinary component edits need no project skill. Navigate through
[architecture](ARCHITECTURE.md), [operations](OPERATIONS.md),
[evaluation](EVALUATION.md) and [research decisions](research/2026-10-31-decisions.md)
as needed. Do not load all skills, force delegation or prescribe a universal
reasoning sequence. Choose relevant evidence and checks for the task.

Descriptions are the discovery surface. "Compare a reranker on frozen dev"
should select `ledger-evaluation`; "fix a voice cancellation test" should select
`ledger-voice` and nothing else. Ask the client which skill it loaded without
executing cloud commands to test discovery after upgrades. This manual client
check is separate from static CI. Upstream descriptions are kept verbatim, so a
provider skill is vendored only if its own trigger is already narrow.

## Provider skills, docs MCP servers and plugins

Researched on 2026-11-09 against provider-owned GitHub organizations and docs.
Vendored skills are byte-identical to the recorded commit; each `SOURCE.md` holds
the URL, commit or version, license, fetch date and file hashes.

| Platform | Adopted | Source, version, license | Rejected, and why |
|---|---|---|---|
| Modal | `modal` skill | [PyPI modal 1.6.3](https://pypi.org/project/modal/1.6.3/) (= `uv.lock`, latest release); body equals modal-client `main` 4791c538; Apache-2.0 | No newer skill; no official docs MCP server or plugin found |
| Cloudflare | `wrangler` skill; `cloudflare-docs` MCP | [cloudflare/skills](https://github.com/cloudflare/skills) 6eb6c781, Apache-2.0; [docs server](https://github.com/cloudflare/mcp-server-cloudflare/tree/a21972e219289337731a066befb5c66886b9ce7b/apps/docs-ai-search) a21972e2, Apache-2.0, no auth | `cloudflare` plugin bundles the OAuth API server `mcp.cloudflare.com` that can change accounts; the umbrella `cloudflare` skill is ~3,900 words and triggers on any architecture question |
| Twilio | `twilio-docs` MCP | [twilio/ai](https://github.com/twilio/ai) 803790a2, MIT; [docs](https://www.twilio.com/docs/ai/mcp): public beta, read-only, no auth | `twilio-developer-kit` plugin adds 57 skills, mostly messaging and SendGrid; none covers Media Streams; the webhook skill validates `request.url`, which may not match the signed URL behind this repo's tunnel/proxy |
| Pipecat | none (`ledger-voice`) | [pipecat-ai/skills](https://github.com/pipecat-ai/skills) cca35d3b, BSD-2-Clause, reviewed | Its skills scaffold projects, deploy to Pipecat Cloud or drive a voice MCP; `pipecat-dev` targets Pipecat contributors; [Context Hub](https://docs.pipecat.ai/api-reference/context-hub/overview.md) needs a tool install and a local ~900 MB embedded index |
| ElevenLabs | none (links in `ledger-voice`) | [elevenlabs/skills](https://github.com/elevenlabs/skills) 9763c39a, MIT, reviewed | Generic `speech-to-text`/`text-to-speech` triggers and direct-SDK, `ELEVENLABS_API_KEY` setup instead of the Pipecat services; the [hosted MCP](https://elevenlabs.io/docs/eleven-api/resources/mcp-server.md) uses OAuth, spends credits and manages agents; no docs MCP exists |
| Qdrant | `qdrant-multitenancy` skill; `qdrant-docs` MCP | [qdrant/skills](https://github.com/qdrant/skills) 9f55f6a7, Apache-2.0; [docs server](https://qdrant.tech/documentation/qdrant-mcp/docs/), no auth | Plugin adds 33 skills (14 top-level); `qdrant-search-quality` overlaps `ledger-evaluation` and sets `allowed-tools`; `qdrant-model-migration` suggests in-place vector migration, against isolated rebuilds; `qdrant-advisor` loads unpinned instructions; Cloud MCP (OAuth) and `mcp-server-qdrant` (key, writes) can act |

Claude Code asks each user to approve `.mcp.json` servers; do not commit
`enableAllProjectMcpServers`. All three servers answered `tools/list` without
credentials on 2026-11-09 with only read-only-annotated tools. Codex users can
add the same URLs with `codex mcp add <name> --url <url>`. `scripts/check_agent_skills.py` accepts only
these reviewed URLs with `{type: http, url}` and rejects headers, commands and
secret-looking values. Account-scoped servers and provider plugins stay opt-in
at user scope, never in the repository; re-review a docs server if it adds
authenticated tools. No `.claude/settings.json` plugin configuration is committed.

## Official Modal skill

Installed using the locked Modal 1.6.3 CLI:

```bash
uv run --locked --group compute modal skills install --yes --no-docs
```

`--no-docs` intentionally avoids vendoring a large, rapidly stale documentation
snapshot. The skill links to official docs; use relevant pages/CLI `--help`
on demand. It is upstream content, not a project-authored workflow.
Its broad Modal trigger is upstream policy; the project skills do not repeat
SDK guidance or expand that trigger to unrelated tasks.
The installed text matches `modal skills show` from the locked SDK (apart from
terminal whitespace). Its Apache-2.0
[license](../.agents/skills/modal/LICENSE) is retained from that distribution.
Source: [Modal Python SDK](https://github.com/modal-labs/modal-client).

To update after a deliberate Modal dependency/lockfile upgrade (the CLI replaces
the whole directory, so restore the license and provenance record):

```bash
uv run --locked --group compute modal skills update --yes --no-docs
git restore .agents/skills/modal/LICENSE .agents/skills/modal/SOURCE.md
# update Version and the SKILL.md hash in SOURCE.md
uv run --locked python scripts/check_agent_skills.py
git diff -- .agents/skills/modal pyproject.toml uv.lock
```

Review upstream instruction changes before committing; do not automatically
execute newly suggested commands. Keep installation workspace-local (no
`--global`). The official reference does not authorize spending or deployment.
No token is needed for this bundled skill installation.

## Headless Modal authentication

**Browser login is not required for token-authenticated CLI, SDK or CI use.**
`modal token new` is optional browser-based token creation, not a prerequisite
when credentials already exist. Revoke exposed credentials before production use.

The Modal token for the project workspace lives in the encrypted secret store
([secrets](SECRETS.md)), not in `~/.modal.toml`. `ledgerctl modal` injects
`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` and the mapped `MODAL_ENVIRONMENT` into the
CLI process only; the SDK reads them directly and they override any local profile.
No `token set`, profile file or browser is needed. This read-only check verifies
access without starting compute:

```bash
ledgerctl modal -- app list
```

These are separate credential boundaries:

| Credential | Consumer | Purpose |
|---|---|---|
| Modal token ID + secret (`operator`) | `ledgerctl` and the Modal CLI/SDK | Deploy/manage apps, submit jobs, deliver Modal Secrets |
| `TEI_API_KEY` in `ledger-tei` | Modal TEI servers and authorized HTTP clients | Protect embedding/reranking endpoints |
| Qdrant URL + key in `ledger-ingestion` | Modal ingestion/evaluation jobs | Direct vector database reads/writes |
| LLM base URL + key in `ledger-llm` | Modal evaluation jobs | OpenAI-compatible answer generation |
| `LEDGER_LLM_SERVE_API_KEY` in `ledger-llm-serve` | Self-hosted vLLM endpoint and its clients | Protect the open-weight LLM endpoint |

`ledgerctl push <env>` provisions those Modal Secrets from the
[manifest](../config/secrets.toml); platform tokens are never delivered to a
runtime. HTTP clients of the deployed TEI endpoints use the TEI key, not platform
credentials or browser login.

## Compute placement

- Run bulk embedding, benchmark model inference and training on Modal.
  A local CLI can orchestrate these remote calls; that is not local model compute.
- [Colocated ingestion](INGESTION.md) loads documents, embeds over loopback and
  uploads directly from one GPU job to a new persistent Qdrant collection.
- Local unit tests and synthetic smoke remain useful; CPU/GPU parity checks are
  explicitly bounded exceptions. Warm query serving is a separate IVR latency
  decision, not a reason to run batch workloads locally.
- Missing credentials, deployment or endpoint connectivity are explicit blockers.
  Do not silently fall back to local CPU model servers. Scope/budget authorization
  is still required; an account credit balance is not a per-run cap.
- Before switching embedding backends, validate rendering/model identity and
  parity. Never append different model/precision outputs to a partial index
  without an explicit compatibility decision or isolated rebuild.

## Maintenance and validation

- Name project skills `ledger-*` and keep them around 40-60 lines; 120 lines and
  800 words are the validator's ceiling, not a target. Add a skill only for a
  demonstrated, repeatable workflow that docs alone do not cover.
  Use common `name`/`description` fields; avoid client-specific tool/model bindings.
- Vendor a provider skill only from the provider's own repository, unmodified, with
  just the files its `SKILL.md` links to, its upstream `LICENSE` and a `SOURCE.md`.
  Vendored ceilings follow the Agent Skills spec: 500 lines and 3,500 words.
- Link to existing source/tests/docs instead of copying implementations.
- When adding a skill, add its relative directory symlink in `.claude/skills`;
  remove that alias when retiring the skill. CI rejects missing or stale aliases.
- Update commands when modules or CLI flags change. Keep proposals separate from
  implemented features; carry durable decisions into the research ledger.
- CI validates frontmatter, naming, descriptions, budgets, local links in every
  skill file, vendored provenance and hashes, relative Claude aliases, the Claude
  import, `.mcp.json` and the official Modal version against `uv.lock`:

  ```bash
  uv run --locked python scripts/check_agent_skills.py
  uv run --locked pytest -q scripts/tests/test_agent_skills.py
  ```

These are static repository checks, not proof a client loaded the skill or obeyed
it. Skills guide behavior; they do not enforce permissions or replace tests.

## Code checks by scope

Run focused tests for the changed component first. The broader service gate is:

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy
uv run --locked --group compute pytest -q
```

For compute changes, also run
`uv run --locked --group compute mypy --strict --ignore-missing-imports compute`.
These checks do not prove cloud execution or serving parity. Documentation-only
changes need the applicable link/skill checks, not model servers or paid jobs.

## Primary compatibility references

- [Agent Skills format](https://agentskills.io/specification)
- [Codex skills](https://developers.openai.com/codex/skills/)
- [Claude Code skills](https://code.claude.com/docs/en/skills)
- [Claude Code memory imports](https://code.claude.com/docs/en/memory)
- [OpenCode skills](https://opencode.ai/docs/skills/)
- [VS Code Agent Skills](https://code.visualstudio.com/docs/copilot/customization/agent-skills)
- [Claude Code MCP project scope](https://code.claude.com/docs/en/mcp)
- [Modal skills CLI](https://modal.com/docs/reference/cli/skills)
- [Modal authentication/configuration](https://modal.com/docs/sdk/py/latest/config)
