"""The age identity that unlocks the SOPS files.

On a workstation the private key lives only in the macOS login Keychain: it is
generated in memory, written through ``security -i`` on stdin (never argv or disk) and
read back by SOPS through ``SOPS_AGE_KEY_CMD``. CI or another OS can instead provide
``SOPS_AGE_KEY``, ``SOPS_AGE_KEY_FILE`` or ``SOPS_AGE_KEY_CMD``, which take precedence.
"""

import getpass
import os
import re
import shlex
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from ledgerctl.process import OpsError, Runner, check, run

SERVICE = "ledger-sops-age"
LABEL = "Ledger SOPS age key"
SECRET_KEY = re.compile(r"AGE-SECRET-KEY-1[0-9A-Z]{58}")
PUBLIC_KEY = re.compile(r"age1[0-9a-z]{58}")
ACCOUNT = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
EXTERNAL_VARIABLES = ("SOPS_AGE_KEY", "SOPS_AGE_KEY_FILE", "SOPS_AGE_KEY_CMD")


def _account() -> str:
    account = os.environ.get("USER") or getpass.getuser()
    if not ACCOUNT.fullmatch(account):
        raise OpsError("The login name cannot be used as a Keychain account")
    return account


@dataclass(frozen=True)
class KeySource:
    """Where SOPS obtains the age identity for decryption and editing."""

    runner: Runner = run
    environ: Mapping[str, str] | None = None
    account: str | None = None

    @property
    def _environ(self) -> Mapping[str, str]:
        return os.environ if self.environ is None else self.environ

    @property
    def external(self) -> bool:
        return any(self._environ.get(name) for name in EXTERNAL_VARIABLES)

    @property
    def _account(self) -> str:
        return self.account or _account()

    def keychain_command(self) -> list[str]:
        return ["security", "find-generic-password", "-a", self._account, "-s", SERVICE, "-w"]

    def sops_env(self) -> dict[str, str]:
        """Child-process variables that let SOPS find the identity."""
        if self.external:
            return {}
        if sys.platform != "darwin":
            raise OpsError(
                "No age identity: set SOPS_AGE_KEY_CMD (or SOPS_AGE_KEY_FILE) on this platform"
            )
        return {"SOPS_AGE_KEY_CMD": shlex.join(self.keychain_command())}

    def in_keychain(self) -> bool:
        if sys.platform != "darwin":
            return False
        args = ["security", "find-generic-password", "-a", self._account, "-s", SERVICE]
        return self.runner(args, None, None, None).returncode == 0

    def _private_key(self) -> bytes:
        environ = self._environ
        if environ.get("SOPS_AGE_KEY"):
            return environ["SOPS_AGE_KEY"].encode()
        if environ.get("SOPS_AGE_KEY_FILE"):
            try:
                return Path(environ["SOPS_AGE_KEY_FILE"]).read_bytes()
            except OSError:
                raise OpsError("SOPS_AGE_KEY_FILE cannot be read") from None
        command = (
            shlex.split(environ["SOPS_AGE_KEY_CMD"])
            if environ.get("SOPS_AGE_KEY_CMD")
            else self.keychain_command()
        )
        result = self.runner(command, None, None, None)
        check(result, "Reading the age identity", diagnostic=False)
        return result.stdout

    def public_key(self) -> str:
        key = self._private_key()
        match = SECRET_KEY.search(key.decode(errors="ignore"))
        if not match:
            raise OpsError("The age identity source returned no AGE-SECRET-KEY")
        result = check(
            self.runner(["age-keygen", "-y"], match.group(0).encode() + b"\n", None, None),
            "Deriving the age public key",
        )
        public = result.stdout.decode().strip()
        if not PUBLIC_KEY.fullmatch(public):
            raise OpsError("age-keygen returned an unexpected public key")
        return public

    def create(self, *, replace: bool = False) -> str:
        """Generate a new identity directly into the Keychain and return its public key."""
        if sys.platform != "darwin":
            raise OpsError("Keychain storage is macOS-only; provide SOPS_AGE_KEY_CMD instead")
        if self.in_keychain() and not replace:
            raise OpsError(
                f"A Keychain item {SERVICE!r} already exists; replacing it would orphan the "
                "encrypted files. Re-run with --replace only after re-encrypting to a new key."
            )
        generated = check(self.runner(["age-keygen"], None, None, None), "age-keygen")
        match = SECRET_KEY.search(generated.stdout.decode(errors="ignore"))
        if not match:
            raise OpsError("age-keygen produced no secret key")
        command = (
            f'add-generic-password -U -a {self._account} -s {SERVICE} -l "{LABEL}" '
            f"-w {match.group(0)}\n"
        )
        check(
            self.runner(["security", "-i"], command.encode(), None, None),
            "Writing the age identity to the Keychain",
            diagnostic=False,
        )
        return self.public_key()

    def backup(self, destination: Path) -> None:
        """Write the identity to a new 0600 file, e.g. on removable or encrypted storage."""
        key = SECRET_KEY.search(self._private_key().decode(errors="ignore"))
        if not key:
            raise OpsError("The age identity source returned no AGE-SECRET-KEY")
        content = (
            f"# Ledger SOPS age identity. Keep offline; anyone holding it can decrypt.\n"
            f"# public key: {self.public_key()}\n{key.group(0)}\n"
        )
        try:
            descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            raise OpsError(f"{destination} already exists; choose a new path") from None
        with os.fdopen(descriptor, "w") as stream:
            stream.write(content)
