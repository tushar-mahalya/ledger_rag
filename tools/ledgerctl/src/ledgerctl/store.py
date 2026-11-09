"""The private SOPS + age secret repository.

Layout (a separate, private git repository, by default ``~/.config/ledger/secrets``)::

    .sops.yaml     creation rules: which age recipients encrypt new files
    dev.yaml       SOPS-encrypted values for the dev environment
    prod.yaml      SOPS-encrypted values for the prod environment
    sync-state.json  non-secret record of which revision each target received

Values are decrypted only into this process's memory. Edits use ``sops set
--value-stdin`` so a change re-encrypts one value under the file's existing data key,
keeping git diffs meaningful, and values never appear in argv or temporary files.
"""

import hashlib
import json
import os
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ledgerctl.keys import PUBLIC_KEY, KeySource
from ledgerctl.process import OpsError, Runner, check, run

DEFAULT_HOME = Path("~/.config/ledger/secrets")
FILE_PATTERN = r"(^|/)[a-z][a-z0-9-]*\.yaml$"
MAX_VALUE_CHARS = 16_384
README = """# Ledger secrets (private)

SOPS + age encrypted values for Ledger. Key names are readable; values are not.
Manage this repository only through `ledgerctl` in the ledger_rag checkout:

    ledgerctl status
    ledgerctl set NAME --env dev
    ledgerctl push dev

Never commit decrypted output. Anyone holding the age identity can decrypt every
revision in this history: if it leaks, rotate every provider credential.
"""
GITIGNORE = "*.dec\n*.dec.*\n*.plain\n*.plain.*\n.DS_Store\n"


def secrets_home(environ: Mapping[str, str] | None = None) -> Path:
    environ = os.environ if environ is None else environ
    return Path(environ.get("LEDGER_SECRETS_HOME") or DEFAULT_HOME).expanduser()


def sops_config(recipients: Iterable[str]) -> str:
    keys = sorted(set(recipients))
    if not keys or not all(PUBLIC_KEY.fullmatch(key) for key in keys):
        raise OpsError("At least one valid age public key is required")
    return (
        "# Managed by ledgerctl. Recipients may decrypt every file in this repository.\n"
        "creation_rules:\n"
        f"  - path_regex: '{FILE_PATTERN}'\n"
        f"    age: '{','.join(keys)}'\n"
    )


def revision_of(content: bytes) -> str:
    """A short, non-secret identifier of one encrypted file revision."""
    return hashlib.sha256(content).hexdigest()[:12]


@dataclass
class SecretStore:
    home: Path
    keys: KeySource
    runner: Runner = run

    def path(self, env: str) -> Path:
        return self.home / f"{env}.yaml"

    def _child_env(self) -> dict[str, str]:
        child = {k: v for k, v in os.environ.items() if not k.startswith("SOPS_AGE_")}
        if self.keys.external:
            child.update({k: v for k, v in os.environ.items() if k.startswith("SOPS_AGE_")})
        child.update(self.keys.sops_env())
        return child

    def _require(self, env: str) -> Path:
        path = self.path(env)
        if not path.is_file():
            raise OpsError(f"{path} does not exist; run `ledgerctl init` first")
        return path

    def recipients(self) -> list[str]:
        config = self.home / ".sops.yaml"
        if not config.is_file():
            return []
        return sorted(set(PUBLIC_KEY.findall(config.read_text())))

    def initialize(self, envs: Iterable[str], recipients: Iterable[str]) -> list[str]:
        """Create the repository and empty encrypted files; return the files created."""
        self.home.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.home, 0o700)
        config = self.home / ".sops.yaml"
        if not config.exists():
            config.write_text(sops_config(recipients))
        for name, content in ((".gitignore", GITIGNORE), ("README.md", README)):
            if not (self.home / name).exists():
                (self.home / name).write_text(content)
        created = []
        for env in envs:
            path = self.path(env)
            if path.exists():
                continue
            result = self.runner(
                [
                    "sops",
                    "encrypt",
                    "--filename-override",
                    path.name,
                    "--input-type",
                    "json",
                    "--output-type",
                    "yaml",
                    "/dev/stdin",
                ],
                b"{}",
                self._child_env(),
                self.home,
            )
            check(result, f"Creating {path.name}")
            path.write_bytes(result.stdout)
            created.append(path.name)
        if not (self.home / ".git").exists():
            check(
                self.runner(["git", "init", "-q", "-b", "main"], None, None, self.home), "git init"
            )
        self.commit("chore: initialize encrypted secret store")
        return created

    def read(self, env: str) -> dict[str, str]:
        path = self._require(env)
        result = self.runner(
            ["sops", "decrypt", "--output-type", "json", str(path)],
            None,
            self._child_env(),
            self.home,
        )
        check(result, f"Decrypting {path.name}")
        try:
            data: Any = json.loads(result.stdout)
        except ValueError:
            raise OpsError(f"{path.name} did not decrypt to a JSON object") from None
        if not isinstance(data, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in data.items()
        ):
            raise OpsError(f"{path.name} must contain only string values at the top level")
        return data

    def set(self, env: str, name: str, value: str) -> None:
        if not value or len(value) > MAX_VALUE_CHARS or "\x00" in value:
            raise OpsError(f"{name}: value must be 1-{MAX_VALUE_CHARS} characters without NUL")
        path = self._require(env)
        result = self.runner(
            ["sops", "set", "--value-stdin", str(path), json.dumps([name])],
            json.dumps(value).encode(),
            self._child_env(),
            self.home,
        )
        check(result, f"Setting {name} in {path.name}")

    def unset(self, env: str, name: str) -> None:
        path = self._require(env)
        result = self.runner(
            ["sops", "unset", str(path), json.dumps([name])], None, self._child_env(), self.home
        )
        check(result, f"Removing {name} from {path.name}")

    def revision(self, env: str) -> str:
        return revision_of(self._require(env).read_bytes())

    def commit(self, message: str) -> bool:
        """Commit any repository change; return whether a commit was made."""
        if not re.fullmatch(r"[\x20-\x7e]{1,200}", message):
            raise OpsError("Commit messages must be short printable ASCII")
        check(self.runner(["git", "add", "-A"], None, None, self.home), "git add")
        staged = self.runner(["git", "diff", "--cached", "--quiet"], None, None, self.home)
        if staged.returncode == 0:
            return False
        check(
            self.runner(["git", "commit", "-q", "-m", message], None, None, self.home), "git commit"
        )
        self.push()
        return True

    def has_remote(self) -> bool:
        result = self.runner(["git", "remote"], None, None, self.home)
        return result.returncode == 0 and "origin" in result.stdout.decode().split()

    def push(self) -> bool:
        """Back up the encrypted history to the private ``origin`` remote, if one exists.

        A failed push never undoes the local change; it is reported and retried with the
        next change (``git -C <store> push`` also works).
        """
        if not self.has_remote():
            return False
        result = self.runner(["git", "push", "-q", "origin", "HEAD"], None, None, self.home)
        if result.returncode != 0:
            print("ledgerctl: warning: could not push the secret store to origin", file=sys.stderr)
            return False
        return True

    def attach_remote(self, url: str) -> None:
        """Point the store at its private backup repository and push the history."""
        if not re.fullmatch(r"(git@github\.com:|https://github\.com/)[\w.-]+/[\w.-]+(\.git)?", url):
            raise OpsError("Use a GitHub SSH or HTTPS URL for the private store repository")
        verb = "set-url" if self.has_remote() else "add"
        check(
            self.runner(["git", "remote", verb, "origin", url], None, None, self.home), "git remote"
        )
        if not self.push():
            raise OpsError("Push failed: create the private repository first and check access")

    # Non-secret record of what each remote target last received.
    def _state_path(self) -> Path:
        return self.home / "sync-state.json"

    def sync_state(self) -> dict[str, dict[str, dict[str, Any]]]:
        path = self._state_path()
        if not path.is_file():
            return {}
        try:
            data: Any = json.loads(path.read_text())
        except ValueError:
            raise OpsError("sync-state.json is not valid JSON") from None
        return data if isinstance(data, dict) else {}

    def record_sync(self, env: str, target: str, revision: str, names: list[str]) -> None:
        state = self.sync_state()
        state.setdefault(env, {})[target] = {
            "revision": revision,
            "variables": sorted(names),
            "synced_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        self._state_path().write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
