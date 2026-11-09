import json
import shutil
import sys
import types
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from ledgerctl import cli
from ledgerctl.keys import SERVICE, KeySource
from ledgerctl.manifest import REVISION_KEY, ManifestError, load_manifest, parse_manifest
from ledgerctl.process import OpsError, Result
from ledgerctl.profiles import load_profile
from ledgerctl.store import SecretStore, sops_config
from ledgerctl.sync import CloudflarePublisher, Delivery, ModalPublisher, plan

ROOT = Path(__file__).resolve().parents[3]
SECRET_KEY = "AGE-SECRET-KEY-1" + "Q" * 58
PUBLIC_KEY = "age1" + "q" * 58


def manifest_data(**secrets: Any) -> dict[str, Any]:
    return {
        "version": 1,
        "environments": {"dev": {}, "prod": {"modal_environment": "main"}},
        "targets": {
            "operator": {"platform": "operator"},
            "local": {"platform": "local"},
            "modal-a": {"platform": "modal", "secret": "ledger-a"},
            "cf": {"platform": "cloudflare", "worker": "ledger-api"},
        },
        "secrets": secrets
        or {
            "TOKEN_ID": {"description": "operator", "deliver": {"operator": True}},
            "API_KEY": {
                "description": "provider",
                "deliver": {"local": "LEDGER_LLM_API_KEY", "modal-a": True, "cf": ["A", "B"]},
            },
            "OPTIONAL": {"description": "opt", "required": False, "deliver": {"local": True}},
            "BASE_URL": {
                "description": "const",
                "value": "https://x.test",
                "deliver": {"modal-a": True},
            },
            "DEMO": {
                "description": "gen",
                "envs": ["dev"],
                "generate": "hex32",
                "deliver": {"local": True},
            },
        },
    }


class FakeRunner:
    def __init__(self, responses: Mapping[str, Result] | None = None) -> None:
        self.calls: list[tuple[list[str], bytes | None, Mapping[str, str] | None]] = []
        self.responses = dict(responses or {})

    def __call__(
        self,
        args: Sequence[str],
        stdin: bytes | None,
        env: Mapping[str, str] | None,
        cwd: Path | None,
    ) -> Result:
        self.calls.append((list(args), stdin, env))
        for prefix, result in self.responses.items():
            if " ".join(args).startswith(prefix):
                return result
        return Result(0, b"", b"")


def test_project_manifest_never_routes_operator_credentials_to_runtimes() -> None:
    manifest = load_manifest(ROOT / "config/secrets.toml")
    values = {name: "v" for name in manifest.secrets}
    operator = {n for n, s in manifest.secrets.items() if "operator" in s.deliver}
    assert {"MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"} <= operator
    for env in manifest.environments:
        for target, spec in manifest.targets.items():
            if spec.platform == "operator":
                continue
            payload = manifest.payload(env, target, values, strict=False)
            assert not set(payload) & operator, (env, target)
    assert manifest.payload("dev", "modal-tei", values) == {"TEI_API_KEY": "v"}


def test_project_profiles_hold_no_secrets() -> None:
    for env in ("dev", "prod"):
        profile = load_profile(ROOT / f"config/env/{env}.env")
        assert profile["LEDGER_ENVIRONMENT"]
        assert not any(key.endswith(("_API_KEY", "_TOKEN")) for key in profile)


def test_payload_routes_aliases_constants_and_optional_values() -> None:
    manifest = parse_manifest(manifest_data())
    values = {"API_KEY": "k", "DEMO": "d", "TOKEN_ID": "t"}
    assert manifest.payload("dev", "local", values) == {"LEDGER_LLM_API_KEY": "k", "DEMO": "d"}
    assert manifest.payload("prod", "local", values) == {"LEDGER_LLM_API_KEY": "k"}
    assert manifest.payload("dev", "modal-a", values) == {
        "API_KEY": "k",
        "BASE_URL": "https://x.test",
    }
    assert manifest.payload("dev", "cf", values) == {"A": "k", "B": "k"}
    assert manifest.missing("dev", {}) == ["API_KEY", "DEMO", "TOKEN_ID"]
    assert manifest.missing("dev", {}, "modal-a") == ["API_KEY"]
    with pytest.raises(ManifestError, match="API_KEY has no value"):
        manifest.payload("dev", "local", {})
    assert manifest.payload("dev", "local", {}, strict=False) == {}
    assert manifest.unknown("dev", {"API_KEY": "k", "STRAY": "s"}) == ["STRAY"]
    assert "BASE_URL" not in manifest.stored_names("dev")


@pytest.mark.parametrize(
    ("secret", "message"),
    [
        ({"description": "x", "deliver": {"operator": True, "local": True}}, "operator"),
        ({"description": "x", "deliver": {"nowhere": True}}, "unknown target"),
        ({"description": "x", "deliver": {"local": REVISION_KEY}}, "reserved"),
        ({"description": "x", "deliver": {"local": "lower"}}, "invalid variable"),
        (
            {"description": "x", "generate": "hex32", "value": "c", "deliver": {"local": True}},
            "constant",
        ),
        ({"description": "x", "envs": ["staging"], "deliver": {"local": True}}, "envs"),
        ({"description": "", "deliver": {"local": True}}, "description"),
        ({"description": "x", "deliver": {"local": True}, "extra": 1}, "unknown fields"),
    ],
)
def test_manifest_rejects_unsafe_or_ambiguous_entries(secret: dict[str, Any], message: str) -> None:
    with pytest.raises(ManifestError, match=message):
        parse_manifest(manifest_data(BAD=secret))


def test_manifest_rejects_variable_collisions_and_duplicate_resources() -> None:
    clash = manifest_data(
        ONE={"description": "x", "deliver": {"local": "SAME"}},
        TWO={"description": "y", "deliver": {"local": "SAME"}},
    )
    with pytest.raises(ManifestError, match="receives SAME"):
        parse_manifest(clash)
    data = manifest_data()
    data["targets"]["modal-b"] = {"platform": "modal", "secret": "ledger-a"}
    with pytest.raises(ManifestError, match="same remote resource"):
        parse_manifest(data)


def test_profile_rejects_secret_looking_and_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "dev.env"
    path.write_text("# comment\nLEDGER_A=1\n\nLEDGER_B=x=y\n")
    assert load_profile(path) == {"LEDGER_A": "1", "LEDGER_B": "x=y"}
    for body, message in [
        ("LEDGER_LLM_API_KEY=abc\n", "looks secret"),
        ("lower=1\n", "KEY=VALUE"),
        ("LEDGER_A=1\nLEDGER_A=2\n", "duplicate"),
    ]:
        path.write_text(body)
        with pytest.raises(OpsError, match=message):
            load_profile(path)


def test_keychain_identity_is_written_through_stdin_never_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    lookup = f"security find-generic-password -a analyst -s {SERVICE}"
    # Prefixes match in order: the -w read (after the write) before the existence probe.
    runner = FakeRunner(
        {
            f"{lookup} -w": Result(0, SECRET_KEY.encode(), b""),
            lookup: Result(44, b"", b"not found"),
            "age-keygen -y": Result(0, PUBLIC_KEY.encode() + b"\n", b""),
            "age-keygen": Result(0, f"# created\n{SECRET_KEY}\n".encode(), b""),
        }
    )
    keys = KeySource(runner=runner, environ={}, account="analyst")
    assert keys.create() == PUBLIC_KEY
    writes = [call for call in runner.calls if call[0] == ["security", "-i"]]
    assert len(writes) == 1 and SECRET_KEY.encode() in (writes[0][1] or b"")
    assert all(SECRET_KEY not in " ".join(args) for args, _, _ in runner.calls)
    assert keys.sops_env()["SOPS_AGE_KEY_CMD"].endswith(f"-s {SERVICE} -w")


def test_existing_identity_is_never_silently_replaced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    keys = KeySource(runner=FakeRunner(), environ={}, account="analyst")
    with pytest.raises(OpsError, match="already exists"):
        keys.create()


def test_external_identity_takes_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert KeySource(environ={"SOPS_AGE_KEY_FILE": "/k"}).sops_env() == {}
    with pytest.raises(OpsError, match="SOPS_AGE_KEY_CMD"):
        KeySource(environ={}).sops_env()


def test_store_passes_values_on_stdin_and_validates_decrypted_shape(tmp_path: Path) -> None:
    (tmp_path / "dev.yaml").write_text("encrypted")
    runner = FakeRunner({"sops decrypt": Result(0, b'{"A": "1", "B": "2"}', b"")})
    store = SecretStore(tmp_path, KeySource(environ={"SOPS_AGE_KEY": SECRET_KEY}), runner)
    assert store.read("dev") == {"A": "1", "B": "2"}
    store.set("dev", "API_KEY", 'sk-"quoted"')
    args, stdin, _ = runner.calls[-1]
    assert args[:3] == ["sops", "set", "--value-stdin"] and args[-1] == '["API_KEY"]'
    assert json.loads(stdin or b"") == 'sk-"quoted"' and "sk-" not in " ".join(args)
    with pytest.raises(OpsError, match="characters"):
        store.set("dev", "API_KEY", "")
    runner.responses = {"sops decrypt": Result(0, b'{"A": {"nested": 1}}', b"")}
    with pytest.raises(OpsError, match="string values"):
        store.read("dev")
    with pytest.raises(OpsError, match="ledgerctl init"):
        store.read("prod")


def test_failed_decryption_never_echoes_output(tmp_path: Path) -> None:
    (tmp_path / "dev.yaml").write_text("encrypted")
    runner = FakeRunner({"sops decrypt": Result(128, b"partial-secret", b"Error: no key")})
    store = SecretStore(tmp_path, KeySource(environ={"SOPS_AGE_KEY": SECRET_KEY}), runner)
    with pytest.raises(OpsError) as error:
        store.read("dev")
    assert "partial-secret" not in str(error.value)


def test_sops_config_requires_valid_recipients() -> None:
    assert PUBLIC_KEY in sops_config([PUBLIC_KEY])
    with pytest.raises(OpsError):
        sops_config(["not-a-key"])


def test_plan_adds_revision_and_refuses_non_remote_targets() -> None:
    manifest = parse_manifest(manifest_data())
    values = {"API_KEY": "k"}
    deliveries = plan(manifest, "dev", values, "abc123")
    by_target = {d.target.name: dict(d.payload) for d in deliveries}
    assert by_target["modal-a"] == {
        "API_KEY": "k",
        "BASE_URL": "https://x.test",
        REVISION_KEY: "abc123",
    }
    assert set(by_target) == {"modal-a", "cf"}
    with pytest.raises(OpsError, match="never synced"):
        plan(manifest, "dev", values, "abc123", ["local"])


def test_cloudflare_publisher_replaces_secrets_via_stdin() -> None:
    manifest = parse_manifest(manifest_data())
    runner = FakeRunner(
        {"npx --yes wrangler": Result(0, b'[{"name": "A"}, {"name": "STALE"}]', b"")}
    )
    publisher = CloudflarePublisher(
        {"CLOUDFLARE_API_TOKEN": "cf-token", "CLOUDFLARE_ACCOUNT_ID": "acct"}, runner
    )
    delivery = Delivery(manifest.target("cf"), {"A": "secret-a", REVISION_KEY: "r"})
    publisher.publish(manifest.environment("dev"), delivery)
    bulk_args, stdin, env = runner.calls[-1]
    assert bulk_args[-4:] == ["secret", "bulk", "--name", "ledger-api-dev"]
    assert json.loads(stdin or b"") == {"STALE": None, "A": "secret-a", REVISION_KEY: "r"}
    assert all("secret-a" not in arg and "cf-token" not in arg for arg in bulk_args)
    assert env is not None and env["CLOUDFLARE_API_TOKEN"] == "cf-token"


def test_modal_publisher_replaces_named_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Any]] = []

    class Objects:
        @staticmethod
        def delete(name: str, **kwargs: Any) -> None:
            calls.append(("delete", (name, kwargs)))

        @staticmethod
        def create(name: str, env_dict: dict[str, str], **kwargs: Any) -> None:
            calls.append(("create", (name, env_dict, kwargs)))

    fake = types.SimpleNamespace(Secret=types.SimpleNamespace(objects=Objects))
    monkeypatch.setitem(sys.modules, "modal", fake)
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    manifest = parse_manifest(manifest_data())
    publisher = ModalPublisher({"MODAL_TOKEN_ID": "ak", "MODAL_TOKEN_SECRET": "as"})
    delivery = Delivery(manifest.target("modal-a"), {"API_KEY": "k"})
    publisher.publish(manifest.environment("prod"), delivery)
    assert calls == [
        ("delete", ("ledger-a", {"allow_missing": True, "environment_name": "main"})),
        ("create", ("ledger-a", {"API_KEY": "k"}, {"environment_name": "main"})),
    ]
    with pytest.raises(OpsError, match="MODAL_TOKEN_ID"):
        ModalPublisher({})


def test_cli_reports_manifest_errors_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken() -> Any:
        raise ManifestError("bad manifest")

    monkeypatch.setattr(cli, "_manifest", broken)
    assert cli.main(["status"]) == 1
    assert "ledgerctl: error: bad manifest" in capsys.readouterr().err


@pytest.mark.skipif(
    not (shutil.which("sops") and shutil.which("age-keygen")), reason="sops/age not installed"
)
def test_real_sops_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    key = tmp_path / "age.txt"
    subprocess.run(["age-keygen", "-o", str(key)], check=True, capture_output=True)
    monkeypatch.setenv("SOPS_AGE_KEY_FILE", str(key))
    monkeypatch.setenv("GIT_AUTHOR_NAME", "test")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "test@example.invalid")
    monkeypatch.setenv("GIT_COMMITTER_NAME", "test")
    monkeypatch.setenv("GIT_COMMITTER_EMAIL", "test@example.invalid")
    keys = KeySource()
    store = SecretStore(tmp_path / "store", keys)
    assert store.initialize(["dev"], [keys.public_key()]) == ["dev.yaml"]
    store.set("dev", "API_KEY", 'value with spaces and "quotes"')
    store.set("dev", "OTHER", "second")
    before = (tmp_path / "store/dev.yaml").read_text()
    assert "value with spaces" not in before and "API_KEY:" in before
    store.set("dev", "OTHER", "changed")
    after = (tmp_path / "store/dev.yaml").read_text()
    api_line = next(line for line in before.splitlines() if line.startswith("API_KEY:"))
    assert api_line in after  # one edit leaves other ciphertexts untouched
    assert store.read("dev") == {"API_KEY": 'value with spaces and "quotes"', "OTHER": "changed"}
    store.unset("dev", "OTHER")
    assert store.read("dev") == {"API_KEY": 'value with spaces and "quotes"'}
    assert store.commit("test change")


def test_install_writes_a_locked_shim_and_refuses_foreign_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(ROOT)
    assert cli.main(["install", "--bin-dir", str(tmp_path)]) == 0
    shim = (tmp_path / "ledgerctl").read_text()
    assert shim.startswith("#!/usr/bin/env bash\n") and cli.SHIM_MARKER in shim
    assert f"--project {ROOT}" in shim and "--frozen" in shim and 'ledgerctl "$@"' in shim
    assert (tmp_path / "ledgerctl").stat().st_mode & 0o111
    assert cli.main(["install", "--bin-dir", str(tmp_path)]) == 0  # idempotent
    (tmp_path / "ledgerctl").write_text("#!/bin/sh\necho someone else\n")
    assert cli.main(["install", "--bin-dir", str(tmp_path)]) == 1
    assert "not a ledgerctl shim" in capsys.readouterr().err


def test_rotate_refuses_provider_credentials(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = parse_manifest(manifest_data())
    monkeypatch.setattr(cli, "_manifest", lambda: manifest)
    monkeypatch.setattr(cli, "_store", lambda: None)
    assert cli.main(["rotate", "API_KEY"]) == 1
    assert "rotate it at the provider" in capsys.readouterr().err


def test_every_documented_command_parses() -> None:
    parser = cli.parser()
    for argv in (
        ["init"],
        ["status", "--env", "dev"],
        ["set", "NAME", "--stdin"],
        ["unset", "NAME", "--env", "prod"],
        ["rotate"],
        ["run", "--", "true"],
        ["push", "dev", "--dry-run", "--target", "modal-tei"],
        ["modal", "--", "app", "list"],
        ["key", "public"],
        ["install"],
    ):
        assert callable(parser.parse_args(argv).func), argv


def test_commits_are_pushed_only_when_a_private_remote_exists(tmp_path: Path) -> None:
    runner = FakeRunner({"git diff --cached --quiet": Result(1, b"", b"")})
    store = SecretStore(tmp_path, KeySource(environ={"SOPS_AGE_KEY": SECRET_KEY}), runner)
    assert store.commit("set NAME (dev)")
    assert not any(args[:2] == ["git", "push"] for args, _, _ in runner.calls)
    runner.responses["git remote"] = Result(0, b"origin\n", b"")
    runner.calls.clear()
    assert store.commit("set NAME (dev)")
    assert ["git", "push", "-q", "origin", "HEAD"] in [args for args, _, _ in runner.calls]
    with pytest.raises(OpsError, match="GitHub"):
        store.attach_remote("https://example.com/not-github.git")
