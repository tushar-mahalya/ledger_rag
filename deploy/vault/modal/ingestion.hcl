# GPU bulk ingestion: vector database writes and optional contextual summaries.
# Bound to Modal workload identity by scripts/vault_configure_modal_jwt.sh.
path "secret/data/ledger/qdrant" {
  capabilities = ["read"]
}
path "secret/data/ledger/llm" {
  capabilities = ["read"]
}
path "auth/token/lookup-self" {
  capabilities = ["read"]
}
path "auth/token/revoke-self" {
  capabilities = ["update"]
}
