"""Non-secret, committed configuration profiles (``config/env/<env>.env``).

Plain ``KEY=VALUE`` lines without interpolation or quoting rules; secrets never go here.
"""

import re
from pathlib import Path

from ledgerctl.process import OpsError

LINE = re.compile(r"^([A-Z][A-Z0-9_]{0,127})=(.*)$")
SECRET_HINT = re.compile(r"(_API_KEY|_TOKEN|_SECRET|_PASSWORD)$")


def load_profile(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise OpsError(f"Profile {path} does not exist")
    values: dict[str, str] = {}
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = LINE.fullmatch(line)
        if not match:
            raise OpsError(f"{path.name}:{number}: expected KEY=VALUE")
        key, value = match.groups()
        if SECRET_HINT.search(key):
            raise OpsError(f"{path.name}:{number}: {key} looks secret; store it with SOPS")
        if key in values:
            raise OpsError(f"{path.name}:{number}: duplicate {key}")
        values[key] = value
    return values
