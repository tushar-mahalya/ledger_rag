"""Download pinned dataset artifacts with SHA-256 verification (stdlib only)."""

import hashlib
import shutil
import urllib.request
from pathlib import Path

from ledger_rag.evaluation.datasets.sources import SOURCES, Artifact

_CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def verify(directory: Path, dataset: str) -> None:
    """Raise unless every pinned artifact of ``dataset`` exists with its pinned digest."""
    for artifact in SOURCES[dataset].artifacts:
        path = directory / artifact.filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run scripts/fetch_datasets.py {dataset}")
        if sha256_file(path) != artifact.sha256:
            raise ValueError(f"Checksum mismatch for {path}")


def fetch_artifact(artifact: Artifact, directory: Path, *, timeout: float = 120) -> bool:
    """Download unless already verified; returns True when a download happened."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / artifact.filename
    if target.is_file() and sha256_file(target) == artifact.sha256:
        return False
    partial = target.with_name(target.name + ".partial")
    try:
        request = urllib.request.Request(artifact.url, headers={"User-Agent": "ledger-rag"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with partial.open("wb") as stream:
                shutil.copyfileobj(response, stream, _CHUNK)
        actual = sha256_file(partial)
        if actual != artifact.sha256:
            raise ValueError(
                f"Checksum mismatch for {artifact.filename}: expected {artifact.sha256}, "
                f"got {actual}"
            )
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return True


def fetch(dataset: str, directory: Path) -> list[str]:
    return [
        artifact.filename
        for artifact in SOURCES[dataset].artifacts
        if fetch_artifact(artifact, directory)
    ]
