"""Validate the repository's portable Agent Skills without invoking an agent or cloud API."""

import re
import sys
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
FIELDS = {"name", "description", "license", "compatibility", "metadata"}


def link_errors(path: Path, text: str, root: Path) -> list[str]:
    errors = []
    for target in LINK.findall(text):
        parsed = urlsplit(target)
        if parsed.scheme in {"http", "https", "mailto"} or target.startswith("#"):
            continue
        if parsed.scheme or target.startswith("/"):
            errors.append(f"{path}: non-portable link: {target}")
            continue
        resolved = (path.parent / unquote(parsed.path)).resolve()
        if not resolved.is_relative_to(root) or not resolved.exists():
            errors.append(f"{path}: broken or external local link: {target}")
    return errors


def validate(root: Path) -> list[str]:
    root = root.resolve()
    errors: list[str] = []
    skills = root / ".agents/skills"
    paths = sorted(skills.glob("*/SKILL.md"))
    if not paths:
        errors.append("No skills found in .agents/skills")
    for directory in sorted(skills.iterdir()) if skills.is_dir() else []:
        if directory.is_dir() and not (directory / "SKILL.md").is_file():
            errors.append(f"{directory}: missing SKILL.md")
    for path in paths:
        text = path.read_text()
        parts = text.removeprefix("---\n").split("\n---\n", 1)
        if not text.startswith("---\n") or len(parts) != 2:
            errors.append(f"{path}: missing YAML frontmatter")
            continue
        try:
            data = yaml.safe_load(parts[0])
        except yaml.YAMLError as exc:
            errors.append(f"{path}: invalid YAML: {exc}")
            continue
        if not isinstance(data, dict):
            errors.append(f"{path}: frontmatter must be a mapping")
            continue
        name, description = data.get("name"), data.get("description")
        if (
            not isinstance(name, str)
            or not NAME.fullmatch(name)
            or len(name) > 64
            or name != path.parent.name
        ):
            errors.append(f"{path}: invalid name or directory mismatch")
        if not isinstance(description, str) or not 1 <= len(description.strip()) <= 1024:
            errors.append(f"{path}: description must contain 1-1024 characters")
        if set(data) - FIELDS:
            errors.append(f"{path}: non-portable frontmatter fields")
        if not parts[1].strip() or len(text.splitlines()) > 120:
            errors.append(f"{path}: empty body or exceeds repository's 120-line skill budget")
        errors.extend(link_errors(path, parts[1], root))
        if name == "modal":
            lock = tomllib.loads((root / "uv.lock").read_text())
            version = next(p["version"] for p in lock["package"] if p["name"] == "modal")
            metadata = data.get("metadata")
            if not isinstance(metadata, dict) or metadata.get("version") != version:
                errors.append(f"{path}: Modal skill version differs from uv.lock")
    for relative in (
        "AGENTS.md",
        "docs/AGENT_WORKFLOWS.md",
        "docs/research/2026-10-31-decisions.md",
    ):
        path = root / relative
        if not path.is_file():
            errors.append(f"{relative}: missing agent guidance")
        else:
            errors.extend(link_errors(path, path.read_text(), root))
    claude = root / "CLAUDE.md"
    if not claude.is_file() or claude.read_text().strip() != "@AGENTS.md":
        errors.append("CLAUDE.md must import @AGENTS.md without duplicating instructions")
    aliases = root / ".claude/skills"
    if not aliases.is_dir() or aliases.is_symlink():
        errors.append(".claude/skills must be a real directory of per-skill symlinks")
    for path in paths:
        alias = aliases / path.parent.name
        if not alias.is_symlink() or alias.resolve() != path.parent.resolve():
            errors.append(f"{alias}: missing or incorrect per-skill symlink")
    if aliases.is_dir():
        for alias in aliases.iterdir():
            if alias.name not in {path.parent.name for path in paths}:
                errors.append(f"{alias}: stale skill alias")
    return errors


def main() -> int:
    errors = validate(Path(__file__).resolve().parents[1])
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Agent Skills: frontmatter, budgets, local links, Claude alias and Modal version OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
