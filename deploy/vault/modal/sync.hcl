# Operator projection into Modal Secrets (scripts/sync_modal_secrets.sh).
# Prefer workload identity (modal/*.hcl roles) so jobs read Vault directly.
path "secret/data/ledger/llm" {
  capabilities = ["read"]
}
path "secret/data/ledger/embedding" {
  capabilities = ["read"]
}
path "secret/data/ledger/qdrant" {
  capabilities = ["read"]
}
