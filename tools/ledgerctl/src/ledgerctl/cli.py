"""``ledgerctl``: one command to keep, inject and deliver Ledger's secrets.

    ledgerctl init                       first-time setup (idempotent)
    ledgerctl status                     what is set, missing and delivered
    ledgerctl set NAME | unset NAME      change a value (hidden prompt or --stdin)
    ledgerctl rotate [NAME ...]          regenerate internal credentials
    ledgerctl run [--env dev] -- CMD     inject into one process
    ledgerctl push ENV                   deliver to Modal Secrets / Cloudflare
    ledgerctl modal -- ARGS              Modal CLI on the project workspace
    ledgerctl key public | backup PATH   the age identity in the macOS Keychain
    ledgerctl install                    put ledgerctl on PATH (~/.local/bin)

Values are never printed. See docs/SECRETS.md.
"""

import argparse
import getpass
import os
import secrets
import shlex
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NoReturn

from ledgerctl.keys import SERVICE, KeySource
from ledgerctl.manifest import REVISION_KEY, Manifest, ManifestError, load_manifest
from ledgerctl.process import OpsError
from ledgerctl.profiles import load_profile
from ledgerctl.store import SecretStore, secrets_home
from ledgerctl.sync import CloudflarePublisher, ModalPublisher, Publisher, plan

PROG = "ledgerctl"
MANIFEST = Path("config/secrets.toml")
SHIM_MARKER = "# ledgerctl shim"


def repo_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / MANIFEST).is_file():
            return candidate
    fallback = Path(__file__).resolve().parents[4]
    if (fallback / MANIFEST).is_file():
        return fallback
    raise OpsError(f"Cannot find {MANIFEST}; is the ledger_rag checkout installed editable?")


def _manifest() -> Manifest:
    return load_manifest(repo_root() / MANIFEST)


def _store() -> SecretStore:
    return SecretStore(secrets_home(), KeySource())


def _envs(
    manifest: Manifest, requested: Sequence[str] | None, declared: frozenset[str]
) -> list[str]:
    if requested:
        for env in requested:
            manifest.environment(env)
            if env not in declared:
                raise OpsError(f"Not declared for environment {env}")
        return list(requested)
    return sorted(declared)


def _read_value(name: str, env_label: str, from_stdin: bool) -> str:
    if from_stdin:
        value = sys.stdin.read()
        value = value[:-1] if value.endswith("\n") else value
    else:
        if not sys.stdin.isatty():
            raise OpsError("No terminal for a hidden prompt; pipe the value with --stdin")
        value = getpass.getpass(f"{name} ({env_label}), input hidden: ")
    if not value.strip():
        raise OpsError("Empty value; nothing changed")
    return value


def _generate(
    manifest: Manifest,
    store: SecretStore,
    envs: Sequence[str],
    names: set[str] | None,
    *,
    replace: bool,
) -> list[str]:
    made = []
    for env in envs:
        values = store.read(env)
        for name in manifest.stored_names(env):
            spec = manifest.secrets[name]
            if spec.generate != "hex32" or (names is not None and name not in names):
                continue
            if replace or not values.get(name):
                store.set(env, name, secrets.token_hex(32))
                made.append(f"{name} ({env})")
    return made


# --- commands ---------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> int:
    """Identity, encrypted store and internal credentials; safe to re-run."""
    manifest = _manifest()
    store = _store()
    keys = store.keys
    if keys.external:
        print("age identity: provided by SOPS_AGE_KEY* environment (Keychain not used)")
    elif keys.in_keychain():
        print(f"age identity: existing Keychain item {SERVICE!r}")
    else:
        keys.create()
        print(f"age identity: created in the login Keychain as {SERVICE!r}")
        print(f"  back it up once: {PROG} key backup /Volumes/<USB>/ledger-age.txt")
    recipients = store.recipients() or [keys.public_key()]
    created = store.initialize(sorted(manifest.environments), recipients)
    print(f"encrypted store: {store.home} ({', '.join(created) or 'already initialized'})")
    if args.remote:
        store.attach_remote(args.remote)
    print(f"backup remote: {'origin (pushed on every change)' if store.has_remote() else 'none'}")
    made = _generate(manifest, store, sorted(manifest.environments), None, replace=False)
    if made:
        store.commit(f"generate {len(made)} internal credential(s)")
    print(f"internal credentials: {', '.join(made) or 'all present'}")
    missing = sorted(
        {n for env in manifest.environments for n in manifest.missing(env, store.read(env))}
    )
    if missing:
        print("still needed:\n" + "\n".join(f"  {PROG} set {name}" for name in missing))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    manifest = _manifest()
    store = _store()
    envs = list(args.env) if args.env else sorted(manifest.environments)
    values = {env: store.read(env) for env in envs}
    width = max(len(name) for name in manifest.secrets)
    print(f"{'secret'.ljust(width)}  " + "  ".join(e.ljust(8) for e in envs) + "  delivered to")
    incomplete = False
    for name, spec in sorted(manifest.secrets.items()):
        cells = []
        for env in envs:
            if env not in spec.envs:
                cells.append("-")
            elif not spec.stored:
                cells.append("const")
            elif values[env].get(name):
                cells.append("set")
            else:
                cells.append("MISSING" if spec.required else "optional")
                incomplete = incomplete or spec.required
        row = "  ".join(cell.ljust(8) for cell in cells)
        print(f"{name.ljust(width)}  {row}  {', '.join(spec.deliver)}")
    for env in envs:
        unknown = manifest.unknown(env, values[env])
        if unknown:
            print(f"warning: {env} holds undeclared keys: {', '.join(unknown)}")
    state = store.sync_state()
    print()
    for env in envs:
        current = store.revision(env)
        for target in sorted(t for t, spec in manifest.targets.items() if spec.remote):
            record = state.get(env, {}).get(target)
            if record is None:
                status = "never pushed"
            elif record.get("revision") == current:
                status = f"current ({record.get('synced_at')})"
            else:
                status = f"STALE: has {record.get('revision')}, store is {current}"
            print(f"push {env}/{target}: {status}")
    return 1 if incomplete else 0


def cmd_set(args: argparse.Namespace) -> int:
    manifest = _manifest()
    store = _store()
    spec = manifest.secret(args.name)
    if not spec.stored:
        raise OpsError(f"{spec.name} is a public constant in the manifest, not a stored secret")
    if spec.generate and not args.force:
        raise OpsError(f"{spec.name} is generated; use `{PROG} rotate {spec.name}` (or --force)")
    envs = _envs(manifest, args.env, spec.envs)
    value = _read_value(spec.name, "/".join(envs), args.stdin)
    for env in envs:
        store.set(env, spec.name, value)
    store.commit(f"set {spec.name} ({', '.join(envs)})")
    print(f"set {spec.name} in {', '.join(envs)}; deliver with `{PROG} push <env>`")
    return 0


def cmd_unset(args: argparse.Namespace) -> int:
    manifest = _manifest()
    store = _store()
    spec = manifest.secret(args.name)
    envs = _envs(manifest, args.env, spec.envs)
    for env in envs:
        store.unset(env, spec.name)
    store.commit(f"unset {spec.name} ({', '.join(envs)})")
    print(f"unset {spec.name} in {', '.join(envs)}")
    return 0


def cmd_rotate(args: argparse.Namespace) -> int:
    manifest = _manifest()
    store = _store()
    names = set(args.names) if args.names else None
    for name in names or ():
        spec = manifest.secret(name)
        if spec.generate is None:
            raise OpsError(
                f"{name} is a provider credential: rotate it at the provider, "
                f"then `{PROG} set {name}`"
            )
    envs = args.env or sorted(manifest.environments)
    made = _generate(manifest, store, envs, names, replace=True)
    if made:
        store.commit(f"rotate {len(made)} internal credential(s)")
    print(f"rotated: {', '.join(made) or 'nothing'}")
    if made:
        print(f"next: `{PROG} push <env>` and redeploy the apps that use them")
    return 0


def _publisher(platform: str, credentials: Mapping[str, str]) -> Publisher:
    if platform == "modal":
        return ModalPublisher(credentials)
    if platform == "cloudflare":
        return CloudflarePublisher(credentials)
    raise OpsError(f"Cannot push to platform {platform}")


def cmd_push(args: argparse.Namespace) -> int:
    manifest = _manifest()
    store = _store()
    env, dry_run = args.target_env, args.dry_run
    environment = manifest.environment(env)
    values = store.read(env)
    names = args.target or sorted(name for name, t in manifest.targets.items() if t.remote)
    blocked = {name: manifest.missing(env, values, name) for name in names}
    blocked = {name: missing for name, missing in blocked.items() if missing}
    for name, missing in blocked.items():
        print(f"{env}/{name}: missing {', '.join(missing)}")
    if blocked and not dry_run:
        raise OpsError(f"Set the missing secrets (`{PROG} set NAME`) before pushing")
    revision = store.revision(env)
    deliveries = plan(manifest, env, values, revision, names, strict=not dry_run)
    operator = manifest.payload(env, "operator", values, strict=False)
    publishers: dict[str, Publisher] = {}
    for delivery in deliveries:
        target = delivery.target
        variables = [n for n in delivery.variables if n != REVISION_KEY]
        print(
            f"{env}/{target.name} -> {target.platform}:{target.resource} [{', '.join(variables)}]"
        )
        if dry_run:
            continue
        if target.platform not in publishers:
            publishers[target.platform] = _publisher(target.platform, operator)
        publishers[target.platform].publish(environment, delivery)
        store.record_sync(env, target.name, revision, variables)
    if not dry_run and deliveries:
        store.commit(f"push {env} revision {revision}")
    print("dry run: nothing sent" if dry_run else f"pushed revision {revision}")
    return 0


def run_environment(env: str, operator: bool) -> dict[str, str]:
    """Process environment for one local command: shell, profile, then routed secrets."""
    manifest = _manifest()
    environment = manifest.environment(env)
    values = _store().read(env)
    merged = dict(os.environ)
    merged.update(load_profile(repo_root() / "config" / "env" / f"{env}.env"))
    for target in ["local", "operator"] if operator else ["local"]:
        missing = manifest.missing(env, values, target)
        if missing:
            print(f"{PROG}: warning: {env} has no value for {', '.join(missing)}", file=sys.stderr)
        merged.update(manifest.payload(env, target, values, strict=False))
    if operator:
        merged["MODAL_ENVIRONMENT"] = environment.modal_environment
    return merged


def _strip(command: Sequence[str]) -> list[str]:
    command = list(command)
    return command[1:] if command and command[0] == "--" else command


def cmd_run(args: argparse.Namespace) -> NoReturn:
    command = _strip(args.command)
    if not command:
        raise OpsError(f"Nothing to run: {PROG} run --env dev -- <command>")
    os.execvpe(command[0], command, run_environment(args.env, args.operator))


def cmd_modal(args: argparse.Namespace) -> NoReturn:
    environment = run_environment(args.env, operator=True)
    os.execve(sys.executable, [sys.executable, "-m", "modal", *_strip(args.args)], environment)


def cmd_key(args: argparse.Namespace) -> int:
    keys = KeySource()
    if args.action == "public":
        print(keys.public_key())
    else:
        keys.backup(Path(args.path).expanduser())
        print(f"wrote {args.path} (mode 0600); keep it offline and delete other copies")
    return 0


def shim(root: Path) -> str:
    return (
        "#!/usr/bin/env bash\n"
        f"{SHIM_MARKER} for {root}; reinstall with `uv run ledgerctl install`.\n"
        "# Runs the locked project environment, so CI and this machine use the same tool.\n"
        f"exec uv run --project {shlex.quote(str(root))} --frozen --quiet --group compute "
        'ledgerctl "$@"\n'
    )


def cmd_install(args: argparse.Namespace) -> int:
    root = repo_root()
    directory = Path(args.bin_dir).expanduser()
    target = directory / PROG
    if target.exists() and SHIM_MARKER not in target.read_text(errors="ignore"):
        raise OpsError(f"{target} exists and is not a ledgerctl shim; remove it first")
    directory.mkdir(parents=True, exist_ok=True)
    target.write_text(shim(root))
    target.chmod(0o755)
    print(f"installed {target} -> {root}")
    if str(directory) not in os.environ.get("PATH", "").split(os.pathsep):
        print(f"note: {directory} is not on PATH; add it to your shell profile")
    return 0


# --- parser -----------------------------------------------------------------------


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog=PROG,
        description="Keep, inject and deliver Ledger's secrets (SOPS + age).",
        epilog="docs/SECRETS.md explains the model. Values are never printed.",
    )
    sub = root.add_subparsers(dest="command_name", required=True, metavar="COMMAND")

    def command(name: str, help_text: str, func: object) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.set_defaults(func=func)
        return p

    init = command(
        "init", "first-time setup: Keychain identity, encrypted store, internal keys", cmd_init
    )
    init.add_argument("--remote", help="private GitHub repository that backs up the store")
    status = command("status", "which secrets are set, missing and delivered", cmd_status)
    status.add_argument("--env", action="append")

    set_p = command("set", "store a value (hidden prompt; every declared env by default)", cmd_set)
    set_p.add_argument("name")
    set_p.add_argument("--env", action="append")
    set_p.add_argument("--stdin", action="store_true", help="read the value from stdin")
    set_p.add_argument("--force", action="store_true", help="overwrite a generated value")
    unset = command("unset", "remove a value", cmd_unset)
    unset.add_argument("name")
    unset.add_argument("--env", action="append")

    rotate = command(
        "rotate", "regenerate internal credentials (all generated by default)", cmd_rotate
    )
    rotate.add_argument("names", nargs="*", metavar="NAME")
    rotate.add_argument("--env", action="append")

    run_p = command("run", "inject an environment's settings and secrets into one command", cmd_run)
    run_p.add_argument("--env", default="dev")
    run_p.add_argument("--operator", action="store_true", help="also inject platform tokens")
    run_p.add_argument("command", nargs=argparse.REMAINDER)

    push = command("push", "deliver to Modal Secrets / Cloudflare (replaces contents)", cmd_push)
    push.add_argument("target_env", metavar="ENV")
    push.add_argument("--target", action="append")
    push.add_argument("--dry-run", action="store_true")

    modal = command("modal", "Modal CLI authenticated for the project workspace", cmd_modal)
    modal.add_argument("--env", default="dev")
    modal.add_argument("args", nargs=argparse.REMAINDER)

    key = command("key", "the age identity in the macOS Keychain", cmd_key)
    key_sub = key.add_subparsers(dest="action", required=True)
    key_sub.add_parser("public", help="print the public key")
    backup = key_sub.add_parser("backup", help="write the identity to a new 0600 file")
    backup.add_argument("path")

    install = command("install", f"put {PROG} on PATH, bound to this checkout", cmd_install)
    install.add_argument("--bin-dir", default="~/.local/bin")
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result: int = args.func(args)
        return result
    except (OpsError, ManifestError) as exc:
        print(f"{PROG}: error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
