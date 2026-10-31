"""Wait for Vault Agent files; never interpret secret contents as shell code."""

import os
import time
from pathlib import Path

directory = Path(os.environ["LEDGER_SECRETS_DIR"])
names = ("LEDGER_LLM_API_KEY", "LEDGER_EMBEDDING_API_KEY", "LEDGER_QDRANT_API_KEY")
deadline = time.monotonic() + 120
while not all((directory / name).is_file() and (directory / name).stat().st_size for name in names):
    if time.monotonic() >= deadline:
        raise SystemExit("Vault secret files not available within 120 seconds")
    time.sleep(1)
os.execvp(
    "uvicorn",
    ["uvicorn", "ledger_rag.api.app:app", "--host", "0.0.0.0", "--port", "8080", "--no-access-log"],
)
