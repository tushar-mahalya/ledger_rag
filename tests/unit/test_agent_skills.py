from pathlib import Path

import pytest

from scripts.check_agent_skills import link_errors, validate

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def skill_repo(tmp_path: Path) -> Path:
    skill = tmp_path / ".agents/skills/example"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        '---\nname: example\ndescription: "Use when testing."\n---\n# Example\nDo the task.\n'
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
    (tmp_path / ".claude/skills/example").symlink_to(
        "../../.agents/skills/example", target_is_directory=True
    )
    return tmp_path


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
        ("---\nname: example\n---\nBody\n", "description"),
        ("---\nname: example\ndescription: test\nmodel: x\n---\nBody\n", "non-portable"),
        ("---\nname: example\ndescription: test\n---\n", "empty body"),
    ],
)
def test_invalid_frontmatter_is_rejected(skill_repo: Path, content: str, error: str) -> None:
    (skill_repo / ".agents/skills/example/SKILL.md").write_text(content)
    assert any(error in message for message in validate(skill_repo))


def test_missing_skill_and_broken_alias_fail(skill_repo: Path) -> None:
    (skill_repo / ".agents/skills/empty").mkdir()
    (skill_repo / ".claude/skills/example").unlink()
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
    (skill_repo / ".claude/skills/modal").symlink_to(
        "../../.agents/skills/modal", target_is_directory=True
    )
    (modal / "SKILL.md").write_text(
        "---\nname: modal\ndescription: Modal reference\n"
        'metadata:\n  version: "1.0"\n---\nReference\n'
    )
    (skill_repo / "uv.lock").write_text('[[package]]\nname = "modal"\nversion = "2.0"\n')
    assert any("Modal skill version differs" in message for message in validate(skill_repo))


def test_skill_line_budget_boundary(skill_repo: Path) -> None:
    path = skill_repo / ".agents/skills/example/SKILL.md"
    text = path.read_text()
    text += "Step\n" * (120 - len(text.splitlines()))
    path.write_text(text)
    assert validate(skill_repo) == []
    path.write_text(text + "Too much\n")
    assert any("120-line" in message for message in validate(skill_repo))


def test_description_length_and_yaml_delimiter_in_value(skill_repo: Path) -> None:
    path = skill_repo / ".agents/skills/example/SKILL.md"
    for length in (1024, 1025):
        description = "Use --- " + "x" * (length - 8)
        path.write_text(f'---\nname: example\ndescription: "{description}"\n---\nBody\n')
        errors = validate(skill_repo)
        assert bool(errors) == (length > 1024)
        if errors:
            assert all("description" in message for message in errors)


def test_stale_and_wrong_target_aliases_fail(skill_repo: Path) -> None:
    aliases = skill_repo / ".claude/skills"
    (aliases / "retired").symlink_to("../../.agents/skills/retired", target_is_directory=True)
    (aliases / "example").unlink()
    (aliases / "example").symlink_to("../../docs", target_is_directory=True)
    errors = validate(skill_repo)
    assert any("stale skill alias" in message for message in errors)
    assert any("incorrect per-skill symlink" in message for message in errors)
