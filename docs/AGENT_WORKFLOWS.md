# Coding-agent workflows

This repository uses the open [Agent Skills specification](https://agentskills.io/specification):
plain `SKILL.md` files, portable YAML frontmatter and task-specific instructions.
No agent framework, plugin server, cloud account or paid model is needed to read
or validate them.

## Single source of truth

```text
AGENTS.md                  shared always-on working agreement
CLAUDE.md                  imports @AGENTS.md
.agents/skills/*/SKILL.md   canonical on-demand skills
.claude/skills/<name>      per-skill symlink to ../../.agents/skills/<name>
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

## Three skills, not a component-by-component instruction stack

Keep `.agents/skills` canonical and `.claude/skills` as discovery aliases, not
copies. Neither directory improves model reasoning by itself; the useful
difference is which clients discover it.

The original eight skills are reduced to two project workflows plus the upstream
Modal reference. Retrieval, generation, voice, operations and research skills
were retired: they mostly repeated component documentation and general engineering
advice. Their shared safety invariants remain in [AGENTS.md](../AGENTS.md).
This follows the [Agent Skills authoring guidance](https://agentskills.io/skill-creation/best-practices):
add non-obvious project knowledge, keep coherent task boundaries, and omit what
the agent already knows. It is not a measured claim of better model performance.

| Skill | Use for |
|---|---|
| [ledger-evaluation](../.agents/skills/ledger-evaluation/SKILL.md) | Executing/comparing benchmarks; corpus, split, cloning and metric pitfalls |
| [ledger-compute](../.agents/skills/ledger-compute/SKILL.md) | Executing/releasing project compute jobs; auth, placement and promotion gates |
| [modal](../.agents/skills/modal/SKILL.md) | Unmodified upstream SDK/CLI guidance |

Ordinary component edits need no project skill. Navigate through
[architecture](ARCHITECTURE.md), [operations](OPERATIONS.md),
[evaluation](EVALUATION.md) and [research decisions](research/2026-10-31-decisions.md)
as needed. Do not load all skills, force delegation or prescribe a universal
reasoning sequence. Choose relevant evidence and checks for the task.

Descriptions are the discovery surface. "Compare a reranker on frozen dev"
should select `ledger-evaluation`; "fix a voice cancellation test" should not.
Ask the client which skill it loaded without executing cloud commands to test
discovery after upgrades. This manual client check is separate from static CI.

## Official Modal skill

Installed using the locked Modal 1.6.3 CLI:

```bash
uv run --locked --group compute modal skills install --yes --no-docs
```

`--no-docs` intentionally avoids vendoring a large, rapidly stale documentation
snapshot. The skill links to official docs; use relevant pages/CLI `--help`
on demand. It is upstream content, not a project-authored workflow.
Its broad Modal trigger is upstream policy; the two project skills do not repeat
SDK guidance or expand that trigger to unrelated tasks.
The installed text matches `modal skills show` from the locked SDK (apart from
terminal whitespace). Its Apache-2.0
[license](../.agents/skills/modal/LICENSE) is retained from that distribution.
Source: [Modal Python SDK](https://github.com/modal-labs/modal-client).

To update after a deliberate Modal dependency/lockfile upgrade:

```bash
uv run --locked --group compute modal skills update --yes --no-docs
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

For one-time workstation configuration, enter credentials in your own terminal,
not in command arguments, source files or chat:

```bash
uv run --locked --group compute modal token set --profile tushki10200 --activate --verify
uv run --locked --group compute modal app list --limit 5
```

With token arguments omitted, `token set` prompts for them and writes the local
Modal configuration outside the repository. This requires terminal input, but
not a browser. Use `modal profile activate tushki10200` to select an existing
profile. Profile selection alone does not verify access.

For fully noninteractive CI/deployment, have the secret manager inject
`MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` into the CLI/SDK process environment.
The installed SDK reads them directly; no `token set`, profile file or browser
is needed. Environment values override profile values. Disable shell tracing
and never dump the environment or print secrets. The repository does not yet
provision Vault delivery of these two platform credentials.

After injection, this read-only check verifies access without starting compute:

```bash
uv run --locked --group compute modal app list --limit 5
```

These are separate credential boundaries:

| Credential | Consumer | Purpose |
|---|---|---|
| Modal token ID + secret | CLI/SDK/job submitter | Deploy/manage apps and submit remote jobs |
| `TEI_API_KEY` in `ledger-tei` | Modal TEI servers and authorized HTTP clients | Protect embedding/reranking endpoints |
| Vault identity/token | Secret projection process | Read approved Vault paths |

[Vault projection](../scripts/sync_modal_secrets.sh) provisions `ledger-tei`;
it does **not** log the CLI into Modal. HTTP clients of the deployed TEI endpoints
use the TEI key, not platform credentials or browser login. A backend that also
submits SDK jobs still needs platform authentication.

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

- Keep project skills around 40-60 lines; 120 is the validator's ceiling, not a target.
  Add a skill only for a demonstrated, repeatable workflow that docs alone do not cover.
  Use common `name`/`description` fields; avoid client-specific tool/model bindings.
- Link to existing source/tests/docs instead of copying implementations.
- When adding a skill, add its relative directory symlink in `.claude/skills`;
  remove that alias when retiring the skill. CI rejects missing or stale aliases.
- Update commands when modules or CLI flags change. Keep proposals separate from
  implemented features; carry durable decisions into the research ledger.
- CI validates frontmatter, naming, descriptions, line budgets, local links,
  the Claude import/alias and official Modal version against `uv.lock`:

  ```bash
  uv run --locked python scripts/check_agent_skills.py
  uv run --locked pytest -q tests/unit/test_agent_skills.py
  ```

These are static repository checks, not proof a client loaded the skill or obeyed
it. Skills guide behavior; they do not enforce permissions or replace tests.

## Code checks by scope

Run focused tests for the changed component first. The broader service gate is:

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked mypy src/ledger_rag
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
- [Modal skills CLI](https://modal.com/docs/reference/cli/skills)
- [Modal authentication/configuration](https://modal.com/docs/sdk/py/latest/config)
