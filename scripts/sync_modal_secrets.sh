#!/usr/bin/env bash
# Project Vault secrets into Modal Secrets (Vault stays the single source of truth).
#
#   VAULT_ADDR=https://vault.example:8200 VAULT_TOKEN=... scripts/sync_modal_secrets.sh
#
# ledger-tei.TEI_API_KEY <- secret/ledger/embedding#api_key  (the key the service sends)
# With argument "ingestion": LEDGER_QDRANT_URL plus secret/ledger/qdrant#api_key
# are projected into ledger-ingestion for direct database writes.
# With argument "llm": LEDGER_LLM_BASE_URL (default: Gemini's OpenAI-compatible endpoint)
# plus secret/ledger/llm#api_key are projected into ledger-llm for evaluation jobs.
# Prefer workload identity (scripts/vault_configure_modal_jwt.sh): jobs then read Vault
# directly with short-lived tokens and no provider credential is copied into Modal.
# Values never appear in argv or shell history: they are written to a 0600 temp file that
# is removed on exit, then loaded with `modal secret create --from-json`.
set -euo pipefail
set +x
case "${1:-tei}" in
  tei) path=embedding; secret=ledger-tei ;;
  ingestion)
    : "${LEDGER_QDRANT_URL:?LEDGER_QDRANT_URL must be an HTTPS URL reachable from Modal}"
    path=qdrant; secret=ledger-ingestion ;;
  llm)
    export LEDGER_LLM_BASE_URL="${LEDGER_LLM_BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai}"
    path=llm; secret=ledger-llm ;;
  *) echo "usage: $0 [tei|ingestion|llm]" >&2; exit 1 ;;
esac
command -v vault >/dev/null || { echo "vault CLI not found" >&2; exit 1; }
: "${VAULT_ADDR:?VAULT_ADDR must be set}"

umask 077
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

key=$(vault kv get -field=api_key "secret/ledger/$path")
python3 - "$tmp" "$secret" 3<<<"$key" <<'PY'
import json
import os
import sys
from urllib.parse import urlsplit

with os.fdopen(3) as stream:
    key = stream.read().strip()
if sys.argv[2] == "ledger-tei":
    if len(key) < 32:
        raise SystemExit("embedding_api_key must be at least 32 characters")
    payload = {"TEI_API_KEY": key}
else:
    llm = sys.argv[2] == "ledger-llm"
    url = os.environ["LEDGER_LLM_BASE_URL" if llm else "LEDGER_QDRANT_URL"]
    parsed = urlsplit(url)
    if (
        not key
        or parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise SystemExit(f"{sys.argv[2]} requires a credential-free HTTPS URL and nonempty key")
    payload = (
        {"LEDGER_LLM_BASE_URL": url, "LEDGER_LLM_API_KEY": key}
        if llm
        else {"LEDGER_QDRANT_URL": url, "LEDGER_QDRANT_API_KEY": key}
    )
with open(sys.argv[1], "w") as stream:
    json.dump(payload, stream)
PY
unset key
uv run --locked --group compute modal secret create "$secret" --from-json "$tmp" --force
echo "synced $secret"
