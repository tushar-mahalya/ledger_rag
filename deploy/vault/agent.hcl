vault {
  address = "https://vault:8200"
  ca_cert = "/run/vault-ca.crt"
}
auto_auth {
  method "approle" {
    mount_path = "auth/approle"
    config = {
      role_id_file_path = "/run/approle/role-id"
      secret_id_file_path = "/run/approle/secret-id"
      remove_secret_id_file_after_reading = false
    }
  }
}
template_config {
  exit_on_retry_failure = true
  static_secret_render_interval = "1m"
}
template {
  contents = "{{- with secret \"secret/data/ledger/llm\" -}}{{ .Data.data.api_key }}{{- end -}}"
  destination = "/run/ledger-secrets/LEDGER_LLM_API_KEY"
  perms = "0600"
  error_on_missing_key = true
}
template {
  contents = "{{- with secret \"secret/data/ledger/embedding\" -}}{{ .Data.data.api_key }}{{- end -}}"
  destination = "/run/ledger-secrets/LEDGER_EMBEDDING_API_KEY"
  perms = "0600"
  error_on_missing_key = true
}
template {
  contents = "{{- with secret \"secret/data/ledger/qdrant\" -}}{{ .Data.data.api_key }}{{- end -}}"
  destination = "/run/ledger-secrets/LEDGER_QDRANT_API_KEY"
  perms = "0600"
  error_on_missing_key = true
}
template {
  contents = "{{- with secret \"secret/data/ledger/telemetry\" -}}{{ .Data.data.otlp_headers_json }}{{- end -}}"
  destination = "/run/ledger-secrets/LEDGER_OTLP_HEADERS"
  perms = "0600"
  error_on_missing_key = true
}
template {
  contents = "{{- with secret \"secret/data/ledger/telemetry\" -}}{{ .Data.data.metrics_token }}{{- end -}}"
  destination = "/run/telemetry-secrets/metrics-token"
  perms = "0600"
  error_on_missing_key = true
}
