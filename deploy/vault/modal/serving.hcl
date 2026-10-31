# Model-serving endpoints: inbound bearer key only.
# Bound to Modal workload identity by scripts/vault_configure_modal_jwt.sh.
path "secret/data/ledger/embedding" {
  capabilities = ["read"]
}
path "auth/token/lookup-self" {
  capabilities = ["read"]
}
path "auth/token/revoke-self" {
  capabilities = ["update"]
}
