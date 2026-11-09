"""Deploy ``ledger-voice`` on Modal (ADR 0006). Not deployed or measured yet.

    ledgerctl push dev --target modal-voice
    ledgerctl modal -- deploy services/voice/modal_app.py

One warm CPU container (``LEDGER_VOICE_MIN_CONTAINERS``, default 0 to avoid idle cost)
serves the Twilio webhook and media WebSockets; each call is one WebSocket on a
container that accepts up to ``LEDGER_VOICE_CALLS_PER_CONTAINER`` concurrent calls.
The function timeout exceeds the longest call. ADR 0006 gate 4 must confirm that a
15-minute Modal WebSocket survives before real traffic. Dependencies come from the
hash-pinned ``requirements.txt`` exported from ``uv.lock`` by
``scripts/export_requirements.py`` (``Image.uv_sync`` does not support uv workspaces).
"""

import os
from pathlib import Path

import modal

APP_NAME = "ledger-voice"
ROOT = Path(__file__).resolve().parents[2]
MIN_CONTAINERS = int(os.environ.get("LEDGER_VOICE_MIN_CONTAINERS", "0"))
CALLS_PER_CONTAINER = int(os.environ.get("LEDGER_VOICE_CALLS_PER_CONTAINER", "8"))
MAX_CALL_SECONDS = 900
# Unpinned until the engine and Qdrant regions are fixed; then set e.g. "us-east" to sit
# beside them and Twilio's us1 processing region (narrow regions cost up to 1.75x).
REGION = os.environ.get("LEDGER_VOICE_REGION") or None


def _profile() -> dict[str, str]:
    """Non-secret ``LEDGER_VOICE_*`` settings of the target environment's profile."""
    name = os.environ.get("LEDGER_PROFILE") or os.environ.get("MODAL_ENVIRONMENT", "dev")
    path = ROOT / "config" / "env" / f"{name}.env"
    values = {}
    for line in path.read_text().splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key.startswith("LEDGER_VOICE_") and not key.startswith("#"):
            values[key] = value
    return values


image = modal.Image.debian_slim(python_version="3.12")
if modal.is_local():
    # Secrets arrive separately through the `ledger-voice` Modal Secret (ledgerctl push).
    image = image.uv_pip_install(requirements=[str(Path(__file__).with_name("requirements.txt"))])
    image = image.env(_profile())
image = image.add_local_python_source("ledger_voice")

app = modal.App(APP_NAME, image=image)


@app.function(
    secrets=[modal.Secret.from_name("ledger-voice")],
    min_containers=MIN_CONTAINERS,
    max_containers=4,
    scaledown_window=300,
    timeout=MAX_CALL_SECONDS + 120,
    cpu=2,
    memory=2048,
    region=REGION,
)
@modal.concurrent(max_inputs=CALLS_PER_CONTAINER)
@modal.asgi_app()
def serve() -> object:
    from ledger_voice.app import create_app

    return create_app()
