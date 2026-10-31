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

## Choose a workflow

| Skill | Use for |
|---|---|
| [ledger-retrieval](../.agents/skills/ledger-retrieval/SKILL.md) | Chunking, model/index identity, hybrid search, quantization and reranking |
| [ledger-evaluation](../.agents/skills/ledger-evaluation/SKILL.md) | FinQA, leakage, ablations, answer correctness and latency/cost evidence |
| [ledger-generation](../.agents/skills/ledger-generation/SKILL.md) | Protocol adapters, grounded tools, citations, abstention and authorization |
| [ledger-voice](../.agents/skills/ledger-voice/SKILL.md) | Audio framing, VAD, barge-in, cancellation and speech latency |
| [ledger-compute](../.agents/skills/ledger-compute/SKILL.md) | Modal training/inference, mining, exports and promotion limitations |
| [ledger-operations](../.agents/skills/ledger-operations/SKILL.md) | Vault/OIDC, deployment, observability and recovery |
| [ledger-research](../.agents/skills/ledger-research/SKILL.md) | Source verification, architecture tradeoffs, experiments and decision retention |
| [modal](../.agents/skills/modal/SKILL.md) | Official SDK/CLI reference; pair with `ledger-compute` for project safeguards |

Descriptions are the discovery surface. For example, "change Qdrant quantization
without changing tenant filters" should select `ledger-retrieval`; "evaluate a new
reranker on frozen dev" should select `ledger-evaluation`. Ask the client to name
the skill it loaded, without executing cloud commands, to smoke-test discovery.
Validate this manually in each target client after upgrades.

## Official Modal skill

Installed using the locked Modal 1.6.3 CLI:

```bash
uv run --locked --group compute modal skills install --yes --no-docs
```

`--no-docs` intentionally avoids vendoring a large, rapidly stale documentation
snapshot. The skill links to official docs; use relevant pages/CLI `--help`
on demand. It is upstream content, not a project-authored workflow.
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

### Authentication is separate

Revoke any token exposed in chat. Then authenticate in your own terminal:

```bash
uv run --locked --group compute modal token new --profile tushki10200 --activate
uv run --locked --group compute modal profile current
```

Do not paste tokens back into chat. Activation only selects a profile; it does not
prove API access. After authentication, a read-only
`uv run --locked --group compute modal app list --limit 5` verifies access.
Training/deployment still requires budget and scope approval.

## Maintenance and validation

- Keep the working agreement brief and project-owned skills under 120 lines.
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

## Primary compatibility references

- [Agent Skills format](https://agentskills.io/specification)
- [Codex skills](https://developers.openai.com/codex/skills/)
- [Claude Code skills](https://code.claude.com/docs/en/skills)
- [Claude Code memory imports](https://code.claude.com/docs/en/memory)
- [OpenCode skills](https://opencode.ai/docs/skills/)
- [VS Code Agent Skills](https://code.visualstudio.com/docs/copilot/customization/agent-skills)
- [Modal skills CLI](https://modal.com/docs/reference/cli/skills)
