#!/usr/bin/env bash
# Trust Modal workload identity (OIDC) in Vault so Ledger compute jobs fetch their own
# short-lived secrets instead of reading provider keys copied into Modal Secrets.
#
#   VAULT_ADDR=https://vault.example:8200 VAULT_TOKEN=<admin> \
#   MODAL_WORKSPACE_ID=ac-... [MODAL_ENVIRONMENT=main] scripts/vault_configure_modal_jwt.sh
#
# Idempotent. Each role is bound to the workspace, environment and app (and, where the
# entrypoint is fixed, the function) claims of Modal's RS256 identity token. Vault tokens
# live 5 minutes: jobs read secrets once at start-up and revoke the token immediately.
set -euo pipefail
set +x
: "${VAULT_ADDR:?VAULT_ADDR must be set}"
: "${MODAL_WORKSPACE_ID:?MODAL_WORKSPACE_ID must be set (the workspace_id claim of a Modal identity token)}"
environment="${MODAL_ENVIRONMENT:-main}"
mount="${VAULT_MODAL_JWT_MOUNT:-jwt-modal}"
[[ $MODAL_WORKSPACE_ID =~ ^ac-[A-Za-z0-9]+$ ]] || { echo "invalid workspace id" >&2; exit 1; }
[[ $environment =~ ^[A-Za-z0-9_-]+$ && $mount =~ ^[A-Za-z0-9_-]+$ ]] || {
  echo "invalid environment or mount name" >&2; exit 1; }
command -v vault >/dev/null || { echo "vault CLI not found" >&2; exit 1; }
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! vault auth list -format=json | python3 -c \
  'import json, sys; sys.exit(0 if sys.argv[1] + "/" in json.load(sys.stdin) else 1)' "$mount"; then
  vault auth enable -path="$mount" -description="Modal workload identity" jwt
fi
vault write "auth/$mount/config" \
  oidc_discovery_url=https://oidc.modal.com bound_issuer=https://oidc.modal.com

for name in ingestion evaluation serving; do
  vault policy write "ledger-modal-$name" "$root/deploy/vault/modal/$name.hcl"
done
vault policy write ledger-modal-sync "$root/deploy/vault/modal/sync.hcl"

role_json='
import json
import sys

workspace, environment, policy, function, *apps = sys.argv[1:]
claims = {"workspace_id": workspace, "environment_name": environment, "app_name": apps}
if function:
    claims["function_name"] = function
print(json.dumps({
    "role_type": "jwt",
    "bound_audiences": ["oidc.modal.com"],
    "bound_claims_type": "string",
    "bound_claims": claims,
    "user_claim": "app_name",
    "claim_mappings": {"function_name": "function_name", "app_id": "app_id",
                       "container_id": "container_id"},
    "token_policies": [policy],
    "token_no_default_policy": True,
    "token_type": "service",
    "token_ttl": "5m",
    "token_max_ttl": "10m",
    "token_explicit_max_ttl": "10m",
}))
'

# role <name> <policy> <function-or-empty> <app>...
role() {
  local name=$1 policy=$2 function=$3
  shift 3
  python3 -c "$role_json" "$MODAL_WORKSPACE_ID" "$environment" "$policy" "$function" "$@" |
    vault write "auth/$mount/role/$name" -
}
role ledger-ingestion ledger-modal-ingestion ingest ledger-ingestion
role ledger-evaluation ledger-modal-evaluation "" ledger-evaluation
role ledger-serving ledger-modal-serving "" \
  ledger-inference ledger-reranker ledger-verifier ledger-decisions
echo "configured auth/$mount for workspace $MODAL_WORKSPACE_ID environment $environment"
