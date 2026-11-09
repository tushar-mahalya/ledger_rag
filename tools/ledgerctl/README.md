# ledgerctl

Operator CLI that keeps Ledger's secrets in one SOPS + age store, injects them into
local commands and pushes them to runtime secret stores. Standard library only; the
Modal push uses the workspace's `compute` group. Usage, security model and rotation:
[docs/SECRETS.md](../../docs/SECRETS.md).

```bash
uv run ledgerctl install     # ~/.local/bin/ledgerctl, bound to this checkout's lockfile
ledgerctl status
uv run pytest tools/ledgerctl/tests
```
