"""Validate the repository's portable Agent Skills without invoking an agent or cloud API."""

import hashlib
import json
import os
import re
import sys
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
FIELDS = {"name", "description", "license", "compatibility", "metadata"}
# Project skills are named ledger-*; every other skill is vendored from its provider.
PROJECT_PREFIX = "ledger-"
# SKILL.md (lines, words). Vendored: the Agent Skills spec's <500 lines / ~5k tokens.
BUDGETS = {"project": (120, 800), "vendored": (500, 3500)}
SOURCE_FIELDS = {
    "Source": re.compile(r"^- Source: https://\S+", re.M),
    "Commit or Version": re.compile(r"^- (?:Commit|Version): \S+", re.M),
    "License": re.compile(r"^- License: \S+", re.M),
    "Fetched": re.compile(r"^- Fetched: \d{4}-\d{2}-\d{2}", re.M),
}
FILE_ROW = re.compile(r"^\| `([0-9a-f]{64})` \| `([^`]+)` \|$", re.M)
IGNORED_FILES = {".DS_Store"}
# Reviewed provider documentation servers: public, read-only tools, no credentials.
DOCS_MCP_URLS = {
    "https://docs.mcp.cloudflare.com/mcp",
    "https://mcp.twilio.com/docs",
    "https://search.qdrant.tech/mcp",
}
SECRETISH = re.compile(
    r"(?i)api[_-]?key|token|secret|passw(?:or)?d|bearer|authorization|\$\{|[\w+/=-]{32,}"
)


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


def file_digest(path: Path) -> str:
    """SHA-256 of LF-normalized content, so a CRLF checkout does not look modified."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def source_errors(directory: Path) -> list[str]:
    """A vendored skill records its provenance and the hash of every file it ships."""
    source = directory / "SOURCE.md"
    if not source.is_file():
        return [f"{directory}: vendored skill needs SOURCE.md"]
    text = source.read_text()
    errors = [
        f"{source}: missing '- {field}:' line"
        for field, pattern in SOURCE_FIELDS.items()
        if not pattern.search(text)
    ]
    if not any(directory.glob("LICENSE*")):
        errors.append(f"{directory}: vendored skill needs its upstream LICENSE")
    recorded = {relative: digest for digest, relative in FILE_ROW.findall(text)}
    shipped = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path != source and path.name not in IGNORED_FILES
    }
    unrecorded = sorted(shipped - set(recorded))
    errors += [f"{directory / relative}: not recorded in SOURCE.md" for relative in unrecorded]
    for relative, digest in sorted(recorded.items()):
        path = directory / relative
        if not path.is_file():
            errors.append(f"{path}: recorded in SOURCE.md but missing")
        elif file_digest(path) != digest:
            errors.append(f"{path}: differs from its SOURCE.md hash")
    return errors


def mcp_errors(root: Path) -> list[str]:
    """A committed .mcp.json may only list reviewed, credential-free docs servers."""
    path = root / ".mcp.json"
    if not path.exists():
        return []
    text = path.read_text()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return [f".mcp.json: invalid JSON: {exc}"]
    errors = []
    if SECRETISH.search(text):
        errors.append(".mcp.json: secret-looking value or variable expansion")
    servers = data.get("mcpServers") if isinstance(data, dict) else None
    if not isinstance(servers, dict) or set(data) != {"mcpServers"}:
        return [*errors, ".mcp.json: must contain only an mcpServers mapping"]
    for name, server in servers.items():
        if (
            not NAME.fullmatch(name)
            or not isinstance(server, dict)
            or set(server) != {"type", "url"}
            or server["type"] != "http"
            or not isinstance(server["url"], str)
        ):
            errors.append(f".mcp.json: {name} must be exactly {{type: http, url}}")
        elif server["url"] not in DOCS_MCP_URLS:
            errors.append(f".mcp.json: {name} is not a reviewed documentation server")
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
        kind = "project" if path.parent.name.startswith(PROJECT_PREFIX) else "vendored"
        if kind == "vendored":
            errors.extend(source_errors(path.parent))
        for document in sorted(path.parent.rglob("*.md")):
            if document != path:
                errors.extend(link_errors(document, document.read_text(), root))
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
        lines, words = BUDGETS[kind]
        if not parts[1].strip():
            errors.append(f"{path}: empty body")
        if len(text.splitlines()) > lines or len(text.split()) > words:
            errors.append(f"{path}: exceeds the {lines}-line/{words}-word {kind} skill budget")
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
        if (
            not alias.is_symlink()
            or Path(os.readlink(alias)) != Path("../../.agents/skills", path.parent.name)
            or alias.resolve() != path.parent.resolve()
        ):
            errors.append(f"{alias}: missing, absolute or incorrect per-skill symlink")
    if aliases.is_dir():
        for alias in aliases.iterdir():
            if alias.name not in {path.parent.name for path in paths}:
                errors.append(f"{alias}: stale skill alias")
    errors.extend(mcp_errors(root))
    return errors


def main() -> int:
    errors = validate(Path(__file__).resolve().parents[1])
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(
        "Agent Skills: frontmatter, budgets, links, provenance, Claude aliases, "
        "docs MCP config and Modal version OK"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
