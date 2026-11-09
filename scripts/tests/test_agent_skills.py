import json
from pathlib import Path

import pytest

from scripts.check_agent_skills import file_digest, link_errors, validate

ROOT = Path(__file__).resolve().parents[2]
SKILL = ".agents/skills/ledger-example/SKILL.md"
DOCS_URL = "https://search.qdrant.tech/mcp"


def add_alias(repo: Path, name: str) -> None:
    (repo / ".claude/skills" / name).symlink_to(
        f"../../.agents/skills/{name}", target_is_directory=True
    )


def write_source(skill: Path, **overrides: str) -> None:
    fields = {
        "Source": "https://github.com/provider/skills/tree/abc/skills/upstream",
        "Commit": "abc",
        "License": "Apache-2.0",
        "Fetched": "2026-11-09",
        **overrides,
    }
    rows = [
        f"| `{file_digest(path)}` | `{path.relative_to(skill).as_posix()}` |"
        for path in sorted(skill.rglob("*"))
        if path.is_file() and path.name != "SOURCE.md"
    ]
    lines = [f"- {key}: {value}" for key, value in fields.items() if value]
    (skill / "SOURCE.md").write_text("\n".join(["# Source", *lines, "", *rows, ""]))


@pytest.fixture
def skill_repo(tmp_path: Path) -> Path:
    skill = tmp_path / ".agents/skills/ledger-example"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        '---\nname: ledger-example\ndescription: "Use when testing."\n---\n# Example\nDo it.\n'
    )
    for relative in (
        "AGENTS.md",
        "docs/AGENT_WORKFLOWS.md",
        "docs/research/2026-10-31-decisions.md",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Guidance\n")
    (tmp_path / "CLAUDE.md").write_text("@AGENTS.md\n")
    (tmp_path / ".claude/skills").mkdir(parents=True)
    add_alias(tmp_path, "ledger-example")
    return tmp_path


@pytest.fixture
def vendored(skill_repo: Path) -> Path:
    skill = skill_repo / ".agents/skills/upstream"
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: upstream\ndescription: Provider guidance.\n---\n"
        "Read [notes](references/notes.md).\n"
    )
    (skill / "references/notes.md").write_text("Notes [back](../SKILL.md)\n")
    (skill / "LICENSE").write_text("Apache License\n")
    write_source(skill)
    add_alias(skill_repo, "upstream")
    return skill


def test_repository_skills_are_valid() -> None:
    assert validate(ROOT) == []


def test_valid_skill_and_single_source_alias(skill_repo: Path) -> None:
    assert validate(skill_repo) == []


@pytest.mark.parametrize(
    "content,error",
    [
        ("# No YAML\n", "frontmatter"),
        ("---\nname: [\n---\nBody\n", "invalid YAML"),
        ("---\n- list\n---\nBody\n", "mapping"),
        ("---\nname: Other\ndescription: test\n---\nBody\n", "mismatch"),
        ("---\nname: ledger-example\n---\nBody\n", "description"),
        ("---\nname: ledger-example\ndescription: t\nmodel: x\n---\nBody\n", "non-portable"),
        ("---\nname: ledger-example\ndescription: test\n---\n", "empty body"),
    ],
)
def test_invalid_frontmatter_is_rejected(skill_repo: Path, content: str, error: str) -> None:
    (skill_repo / SKILL).write_text(content)
    assert any(error in message for message in validate(skill_repo))


def test_missing_skill_and_broken_alias_fail(skill_repo: Path) -> None:
    (skill_repo / ".agents/skills/empty").mkdir()
    (skill_repo / ".claude/skills/ledger-example").unlink()
    errors = validate(skill_repo)
    assert any("missing SKILL.md" in message for message in errors)
    assert any("symlink" in message for message in errors)


def test_local_links_cannot_escape_repository(skill_repo: Path) -> None:
    path = skill_repo / "AGENTS.md"
    assert not link_errors(path, "[ok](./CLAUDE.md) [web](https://example.com) [h](#h)", skill_repo)
    errors = link_errors(
        path, "[abs](/etc/passwd) [missing](./absent.md) [escape](../outside)", skill_repo
    )
    assert len(errors) == 3


def test_modal_version_must_match_lock(skill_repo: Path) -> None:
    modal = skill_repo / ".agents/skills/modal"
    modal.mkdir()
    add_alias(skill_repo, "modal")
    (modal / "SKILL.md").write_text(
        "---\nname: modal\ndescription: Modal reference\n"
        'metadata:\n  version: "1.0"\n---\nReference\n'
    )
    (skill_repo / "uv.lock").write_text('[[package]]\nname = "modal"\nversion = "2.0"\n')
    assert any("Modal skill version differs" in message for message in validate(skill_repo))


@pytest.mark.parametrize("is_vendored,lines", [(False, 120), (True, 500)])
def test_skill_line_budget_boundary(vendored: Path, is_vendored: bool, lines: int) -> None:
    repo = vendored.parents[2]
    path = vendored / "SKILL.md" if is_vendored else repo / SKILL
    text = path.read_text()
    text += "Step\n" * (lines - len(text.splitlines()))
    for content, valid in ((text, True), (text + "Too much\n", False)):
        path.write_text(content)
        write_source(vendored)
        errors = validate(repo)
        assert errors == [] if valid else any(f"{lines}-line" in m for m in errors)


def test_project_word_budget(skill_repo: Path) -> None:
    path = skill_repo / SKILL
    path.write_text(path.read_text() + "word " * 800)
    assert any("800-word project" in message for message in validate(skill_repo))


def test_description_length_and_yaml_delimiter_in_value(skill_repo: Path) -> None:
    path = skill_repo / SKILL
    for length in (1024, 1025):
        description = "Use --- " + "x" * (length - 8)
        path.write_text(f'---\nname: ledger-example\ndescription: "{description}"\n---\nBody\n')
        errors = validate(skill_repo)
        assert bool(errors) == (length > 1024)
        if errors:
            assert all("description" in message for message in errors)


def test_stale_wrong_target_and_absolute_aliases_fail(skill_repo: Path) -> None:
    aliases = skill_repo / ".claude/skills"
    (aliases / "retired").symlink_to("../../.agents/skills/retired", target_is_directory=True)
    (aliases / "ledger-example").unlink()
    (aliases / "ledger-example").symlink_to("../../docs", target_is_directory=True)
    errors = validate(skill_repo)
    assert any("stale skill alias" in message for message in errors)
    assert any("incorrect per-skill symlink" in message for message in errors)
    (aliases / "ledger-example").unlink()
    (aliases / "ledger-example").symlink_to(
        skill_repo / ".agents/skills/ledger-example", target_is_directory=True
    )
    assert any("absolute" in message for message in validate(skill_repo))


def test_vendored_skill_with_provenance_is_valid(vendored: Path) -> None:
    assert validate(vendored.parents[2]) == []


@pytest.mark.parametrize(
    "change,error",
    [
        (lambda s: (s / "SOURCE.md").unlink(), "needs SOURCE.md"),
        (lambda s: write_source(s, Commit=""), "Commit or Version"),
        (lambda s: write_source(s, Fetched="recently"), "Fetched"),
        (lambda s: (s / "LICENSE").unlink(), "needs its upstream LICENSE"),
        (lambda s: (s / "SKILL.md").write_text((s / "SKILL.md").read_text() + "x\n"), "hash"),
        (lambda s: (s / "references/extra.md").write_text("x\n"), "not recorded"),
        (lambda s: (s / "references/notes.md").unlink(), "missing"),
    ],
)
def test_vendored_skill_needs_provenance_and_unchanged_files(
    vendored: Path, change, error: str
) -> None:
    change(vendored)
    assert any(error in message for message in validate(vendored.parents[2]))


def test_reference_file_links_are_checked(vendored: Path) -> None:
    (vendored / "references/notes.md").write_text("[gone](./absent.md)\n")
    write_source(vendored)
    assert any("notes.md: broken" in message for message in validate(vendored.parents[2]))


def test_crlf_checkout_keeps_vendored_hashes(vendored: Path) -> None:
    path = vendored / "references/notes.md"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    assert validate(vendored.parents[2]) == []


def test_reviewed_docs_mcp_config_is_valid(skill_repo: Path) -> None:
    server = {"type": "http", "url": DOCS_URL}
    (skill_repo / ".mcp.json").write_text(json.dumps({"mcpServers": {"qdrant-docs": server}}))
    assert validate(skill_repo) == []


@pytest.mark.parametrize(
    "config,error",
    [
        ("{not json", "invalid JSON"),
        ('{"servers": {}}', "only an mcpServers"),
        ({"docs": {"type": "http", "url": "https://example.com/mcp"}}, "not a reviewed"),
        ({"docs": {"type": "http", "url": DOCS_URL, "oauth": False}}, "exactly"),
        ({"docs": {"type": "http", "url": [DOCS_URL]}}, "exactly"),
        ({"run": {"command": "npx", "args": ["-y", "server"]}}, "exactly"),
        (
            {"docs": {"type": "http", "url": DOCS_URL, "headers": {"X-Api-Key": "abc"}}},
            "secret-looking",
        ),
        ({"docs": {"type": "http", "url": DOCS_URL + "?k=${QDRANT_KEY}"}}, "secret-looking"),
        ({"docs": {"type": "http", "url": DOCS_URL + "/" + "a1B2" * 10}}, "secret-looking"),
    ],
)
def test_mcp_config_rejects_unreviewed_or_credentialed_servers(
    skill_repo: Path, config: str | dict[str, object], error: str
) -> None:
    text = config if isinstance(config, str) else json.dumps({"mcpServers": config})
    (skill_repo / ".mcp.json").write_text(text)
    assert any(error in message for message in validate(skill_repo))
