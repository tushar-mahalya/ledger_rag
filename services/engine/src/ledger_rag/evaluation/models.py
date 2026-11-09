"""Reader model registry: every LLM is declared once in ``config/models/<id>.toml``.

A spec says how to serve the model (``vllm``: pinned weights served on Modal by
``compute/reasoning.py``; ``hosted``: an OpenAI-compatible API such as Gemini) and
its generation *modes* (for example thinking ``off`` or ``low``), so every
benchmark, pipeline and job calls any model the same way. Adding a model is a new
TOML file, not code. Self-hosted weights must be pinned to a 40-hex Hub commit.

Example::

    id = "qwen3.5-4b"
    publisher = "Alibaba Qwen"
    license = "apache-2.0"
    parameters_b = 4.7

    [serving]
    kind = "vllm"
    repo = "Qwen/Qwen3.5-4B"
    revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    gpu = "L40S"
    vllm_args = ["--language-model-only", "--reasoning-parser", "qwen3"]

    [modes.off]
    extra_body = { chat_template_kwargs = { enable_thinking = false } }
"""

import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from ledger_rag.evaluation.llm import GenerationConfig


def _default_dir() -> Path:
    """``config/models`` of this checkout; jobs pass specs explicitly instead."""
    here = Path(__file__).resolve()
    return here.parents[5] / "config" / "models" if len(here.parents) > 5 else Path("config/models")


MODELS_DIR = _default_dir()
GPUS = frozenset({"T4", "L4", "A10", "L40S", "A100-40GB", "A100-80GB", "H100", "H100!", "H200"})
_ID = re.compile(r"[a-z0-9][a-z0-9.\-]{1,62}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class Mode:
    """One generation setting; raw and Ledger arms of a comparison share it."""

    max_tokens: int = 2048
    temperature: float = 0.0
    reasoning_effort: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelSpec:
    id: str
    kind: Literal["vllm", "hosted"]
    served_name: str
    """The ``model`` field sent to the endpoint (the Hub repo for vLLM)."""
    modes: dict[str, Mode]
    default_mode: str
    publisher: str = ""
    license: str = ""
    parameters_b: float | None = None
    revision: str = ""
    gpu: str = ""
    max_model_len: int = 16384
    vllm_args: tuple[str, ...] = ()
    base_url_env: str = ""
    key_env: str = ""
    requests_per_minute: float | None = None

    def generation(self, mode: str | None = None, *, seed: int = 0) -> GenerationConfig:
        name = mode or self.default_mode
        if name not in self.modes:
            raise ValueError(f"{self.id} has no mode {name!r}; choose from {sorted(self.modes)}")
        chosen = self.modes[name]
        return GenerationConfig(
            model=self.served_name,
            temperature=chosen.temperature,
            max_tokens=chosen.max_tokens,
            reasoning_effort=chosen.reasoning_effort,
            seed=seed,
            extra_body=dict(chosen.extra_body),
            name=f"{self.id}:{name}",
        )

    def manifest(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_manifest(cls, data: dict[str, Any]) -> "ModelSpec":
        fields = dict(data)
        fields["modes"] = {name: Mode(**mode) for name, mode in data["modes"].items()}
        fields["vllm_args"] = tuple(data.get("vllm_args", ()))
        return cls(**fields)


def parse(data: dict[str, Any], source: str = "<memory>") -> ModelSpec:
    model_id = str(data.get("id", ""))
    if not _ID.fullmatch(model_id):
        raise ValueError(f"{source}: id must be lowercase letters, digits, '.' or '-'")
    serving = data.get("serving") or {}
    kind = serving.get("kind")
    modes_raw = data.get("modes") or {}
    if not modes_raw:
        raise ValueError(f"{source}: declare at least one [modes.<name>]")
    modes = {str(name): Mode(**mode) for name, mode in modes_raw.items()}
    default_mode = str(data.get("default_mode") or next(iter(modes)))
    if default_mode not in modes:
        raise ValueError(f"{source}: default_mode {default_mode!r} is not a declared mode")
    common: dict[str, Any] = {
        "id": model_id,
        "modes": modes,
        "default_mode": default_mode,
        "publisher": str(data.get("publisher", "")),
        "license": str(data.get("license", "")),
        "parameters_b": data.get("parameters_b"),
    }
    if kind == "vllm":
        repo, revision, gpu = (
            serving.get("repo", ""),
            serving.get("revision", ""),
            serving.get("gpu"),
        )
        if not repo or not _COMMIT.fullmatch(str(revision)):
            raise ValueError(f"{source}: vllm models need a repo and a 40-hex revision")
        if gpu not in GPUS:
            raise ValueError(f"{source}: gpu must be one of {sorted(GPUS)}")
        return ModelSpec(
            kind="vllm",
            served_name=str(repo),
            revision=str(revision),
            gpu=str(gpu),
            max_model_len=int(serving.get("max_model_len", 16384)),
            vllm_args=tuple(str(arg) for arg in serving.get("vllm_args", ())),
            **common,
        )
    if kind == "hosted":
        if (
            not serving.get("model")
            or not serving.get("base_url_env")
            or not serving.get("key_env")
        ):
            raise ValueError(f"{source}: hosted models need model, base_url_env and key_env")
        rpm = serving.get("requests_per_minute")
        return ModelSpec(
            kind="hosted",
            served_name=str(serving["model"]),
            base_url_env=str(serving["base_url_env"]),
            key_env=str(serving["key_env"]),
            requests_per_minute=float(rpm) if rpm is not None else None,
            **common,
        )
    raise ValueError(f"{source}: serving.kind must be 'vllm' or 'hosted'")


def load(model_id: str, directory: Path = MODELS_DIR) -> ModelSpec:
    path = directory / f"{model_id}.toml"
    if not path.is_file():
        known = ", ".join(sorted(p.stem for p in directory.glob("*.toml")))
        raise ValueError(f"unknown model {model_id!r}; known: {known}")
    spec = parse(tomllib.loads(path.read_text()), str(path))
    if spec.id != model_id:
        raise ValueError(f"{path}: id {spec.id!r} must match the file name")
    return spec


def load_all(directory: Path = MODELS_DIR) -> list[ModelSpec]:
    return [load(path.stem, directory) for path in sorted(directory.glob("*.toml"))]
