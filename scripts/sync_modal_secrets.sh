#!/usr/bin/env bash
# Project Vault secrets into Modal Secrets (Vault stays the single source of truth).
#
#   VAULT_ADDR=https://vault.example:8200 VAULT_TOKEN=... scripts/sync_modal_secrets.sh
#
# ledger-tei.TEI_API_KEY <- secret/ledger/api#embedding_api_key  (the key the service sends)
# Values never appear in argv or shell history: they are written to a 0600 temp file that
# is removed on exit, then loaded with `modal secret create --from-json`.
set -euo pipefail
command -v vault >/dev/null || { echo "vault CLI not found" >&2; exit 1; }
: "${VAULT_ADDR:?VAULT_ADDR must be set}"

umask 077
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

key=$(vault kv get -field=embedding_api_key secret/ledger/api)
[ "${#key}" -ge 32 ] || { echo "embedding_api_key must be at least 32 characters" >&2; exit 1; }
python3 -c 'import json,sys; json.dump({"TEI_API_KEY": sys.stdin.read().strip()}, open(sys.argv[1], "w"))' \
  "$tmp" <<<"$key"
uv run --group compute modal secret create ledger-tei --from-json "$tmp" --force
echo "synced ledger-tei"
