"""Pinned, checksummed upstream artifacts for each evaluation dataset."""

from dataclasses import dataclass

from ledger_rag.evaluation.datasets.schema import Usage


@dataclass(frozen=True)
class Artifact:
    filename: str
    url: str
    sha256: str


@dataclass(frozen=True)
class Source:
    name: str
    homepage: str
    revision: str
    license: str
    license_evidence: str
    usage: Usage
    artifacts: tuple[Artifact, ...]


_FINQA_COMMIT = "0f16e2867befa6840783e58be38c9efb9229d742"
_FINQA_SHA256 = {
    "train": "49f237eb9779b569473b26b08048867d04635a7cc39ad6a7a5664c55bb428db6",
    "dev": "a847fb7e0d61a3125a1e2909852df6b89f1ee64d2c5ff1bf689e332214deee51",
    "test": "831dbfb2e785dbc227f895ce3f24046433467aec67b09db2bd6ac7692a8a30dc",
}
_CONVFINQA_COMMIT = "cf3eed2d5984960bf06bb8145bcea5e80b0222a6"
_TATQA_COMMIT = "870accc41953dcde885aabeb963d94aabdc0fbc3"
_FINANCEBENCH_REVISION = "e04404e3a97f69f79c14d42f24981a1c9c3bcd18"

SOURCES: dict[str, Source] = {
    # Consumed raw by ledger_rag.evaluation.benchmarks.finqa (no normalization step).
    # The repository LICENSE is MIT; a Hugging Face re-host labels it CC-BY-4.0
    # (docs/RESEARCH.md): reconcile before compliance-sensitive use.
    "finqa": Source(
        name="FinQA",
        homepage="https://github.com/czyssrs/FinQA",
        revision=_FINQA_COMMIT,
        license="MIT",
        license_evidence=f"https://github.com/czyssrs/FinQA/blob/{_FINQA_COMMIT}/LICENSE",
        usage="train_ok",
        artifacts=tuple(
            Artifact(
                f"{split}.json",
                f"https://raw.githubusercontent.com/czyssrs/FinQA/{_FINQA_COMMIT}/dataset/{split}.json",
                digest,
            )
            for split, digest in _FINQA_SHA256.items()
        ),
    ),
    "convfinqa": Source(
        name="ConvFinQA",
        homepage="https://github.com/czyssrs/ConvFinQA",
        revision=_CONVFINQA_COMMIT,
        license="MIT",
        license_evidence=f"https://github.com/czyssrs/ConvFinQA/blob/{_CONVFINQA_COMMIT}/LICENSE",
        usage="train_ok",
        artifacts=(
            Artifact(
                "data.zip",
                f"https://raw.githubusercontent.com/czyssrs/ConvFinQA/{_CONVFINQA_COMMIT}/data.zip",
                "d764271fae60d81b62e6d58dfc481807ebc8cfbcd633811241723c4a2101072a",
            ),
        ),
    ),
    "tatqa": Source(
        name="TAT-QA",
        homepage="https://github.com/NExTplusplus/TAT-QA",
        revision=_TATQA_COMMIT,
        license="CC-BY-4.0",
        license_evidence=(
            f"https://github.com/NExTplusplus/TAT-QA/blob/{_TATQA_COMMIT}/README.md#license"
        ),
        usage="train_ok",
        artifacts=tuple(
            Artifact(
                f"tatqa_dataset_{split}.json",
                f"https://raw.githubusercontent.com/NExTplusplus/TAT-QA/{_TATQA_COMMIT}"
                f"/dataset_raw/tatqa_dataset_{split}.json",
                digest,
            )
            for split, digest in [
                ("train", "2df6e722cdbaaa37efcbfb280f5c9a15be29a6ec18f618ef936fe63cc6d07c69"),
                ("dev", "8da095a819af6db3c14877c6df2d4d29960e41d1a63dd1fa853507bd2a616af5"),
                ("test_gold", "c4d08418359c1d76468dec420ee748a37f48c06b63cb8ec2766f19d5d314b597"),
            ]
        ),
    ),
    "financebench": Source(
        name="FinanceBench (open-source 150)",
        homepage="https://huggingface.co/datasets/PatronusAI/financebench",
        revision=_FINANCEBENCH_REVISION,
        license="CC-BY-NC-4.0",
        license_evidence=(
            "https://huggingface.co/datasets/PatronusAI/financebench/blob/"
            f"{_FINANCEBENCH_REVISION}/README.md"
        ),
        usage="eval_only",
        artifacts=(
            Artifact(
                "financebench_merged.jsonl",
                "https://huggingface.co/datasets/PatronusAI/financebench/resolve/"
                f"{_FINANCEBENCH_REVISION}/financebench_merged.jsonl",
                "7a1c81789e0fd2f1c37057a7ec0097756d726b05e7228e68e57db8e18c54fd0b",
            ),
        ),
    ),
}
