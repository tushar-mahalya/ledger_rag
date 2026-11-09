# deploy

Infrastructure configuration consumed by `compose.yaml` and future deployment targets.

| Path | Contents |
|---|---|
| `observability/` | OpenTelemetry Collector (redaction allowlist), Prometheus, Grafana data sources, Loki and Tempo for `--profile observability` |
| `observability/private/` | Ignored local credentials for that profile (metrics token, Grafana password) |

Service images are defined next to their code (`services/*/Dockerfile`), Modal apps in
`compute/` and `services/voice/modal_app.py`. See [operations](../docs/OPERATIONS.md).
