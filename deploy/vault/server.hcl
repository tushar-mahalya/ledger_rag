ui = false
disable_mlock = false
api_addr = "https://vault:8200"
cluster_addr = "https://vault:8201"

storage "raft" {
  path = "/vault/file"
  node_id = "ledger-vault-1"
}

listener "tcp" {
  address = "0.0.0.0:8200"
  tls_cert_file = "/vault/tls/server.crt"
  tls_key_file = "/vault/tls/server.key"
  tls_min_version = "tls12"
}
