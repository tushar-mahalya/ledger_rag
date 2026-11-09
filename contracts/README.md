# contracts

The engine's public answer contract, generated from its Pydantic models by
`scripts/dump_contract_fixtures.py` (CI runs `--check`; never edit by hand).

| File | Contents |
|---|---|
| `answers.json` | Real offline chat-pipeline answers, activity streams and evidence events, serialized like the API |
| `contract.schema.json` | JSON Schema of the response models |

Every client tests against these files instead of importing engine code: the web
workspace (`services/web`, zod schemas and bounds) and the voice service
(`services/voice`, its client models). See [the API contract](../docs/CHAT_API.md).
