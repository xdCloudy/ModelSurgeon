"""Pinned, permissively licensed multi-family model evaluation ladder."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from modelsurgeon.adapters import ModelFamily

MODEL_LADDER_SCHEMA_VERSION = 1
PERMISSIVE_LICENSES = frozenset({"apache-2.0", "mit", "bsd-3-clause"})
_EXPECTED_RUNGS = ("100M", "300M", "500M", "1B", "1.5B", "3B", "7B")


class ModelLadderError(ValueError):
    """Raised when a ladder target is not immutable, permissive, or well-scoped."""


class ModelHardwareMode(StrEnum):
    CPU_SMOKE = "cpu_smoke"
    CPU_STREAMING = "cpu_streaming"
    GPU_12_GB = "gpu_12_gb"
    GPU_12_GB_QUANTIZED = "gpu_12_gb_quantized"
    GPU_24_GB = "gpu_24_gb"


class ModelDatasetProtocol(StrEnum):
    LOCAL_SMOKE = "modelsurgeon-smoke-v1"
    WIKITEXT_PERPLEXITY = "wikitext-2-raw-v1:test"


@dataclass(frozen=True, slots=True)
class ModelLadderTarget:
    rung: str
    target_parameters: int
    identifier: str
    revision: str
    family: ModelFamily
    actual_parameters: int
    license: str
    last_modified: date
    purpose: str
    hardware_modes: tuple[ModelHardwareMode, ...]
    datasets: tuple[ModelDatasetProtocol, ...]
    public: bool = True
    gated: bool = False

    def __post_init__(self) -> None:
        if self.rung not in _EXPECTED_RUNGS or self.target_parameters <= 0:
            raise ModelLadderError("model ladder rung and target size must be canonical")
        if not self.identifier or "/" not in self.identifier:
            raise ModelLadderError("model target requires a namespaced Hub identifier")
        if re.fullmatch(r"[0-9a-f]{40}", self.revision) is None:
            raise ModelLadderError("model target revision must be a full lowercase commit")
        if self.license.lower() not in PERMISSIVE_LICENSES:
            raise ModelLadderError(f"model license {self.license!r} is not permissive")
        if not self.public or self.gated:
            raise ModelLadderError("evaluation ladder models must be public and ungated")
        if not self.purpose.strip() or not self.hardware_modes or not self.datasets:
            raise ModelLadderError(
                "every target requires a purpose, hardware mode, and dataset protocol"
            )
        if len(set(self.hardware_modes)) != len(self.hardware_modes) or len(
            set(self.datasets)
        ) != len(self.datasets):
            raise ModelLadderError("target hardware and dataset protocols must be unique")
        ratio = self.actual_parameters / self.target_parameters
        if not 0.65 <= ratio <= 1.40:
            raise ModelLadderError(
                f"{self.identifier} has {self.actual_parameters} parameters, outside the "
                f"declared {self.rung} rung tolerance"
            )

    @property
    def source_url(self) -> str:
        return f"https://huggingface.co/{self.identifier}/tree/{self.revision}"

    def to_record(self) -> dict[str, object]:
        return {
            "rung": self.rung,
            "target_parameters": self.target_parameters,
            "identifier": self.identifier,
            "revision": self.revision,
            "source_url": self.source_url,
            "family": self.family.value,
            "actual_parameters": self.actual_parameters,
            "license": self.license.lower(),
            "last_modified": self.last_modified.isoformat(),
            "purpose": self.purpose,
            "hardware_modes": [mode.value for mode in self.hardware_modes],
            "datasets": [dataset.value for dataset in self.datasets],
            "public": self.public,
            "gated": self.gated,
        }


@dataclass(frozen=True, slots=True)
class ModelEvaluationLadder:
    targets: tuple[ModelLadderTarget, ...]
    metadata_verified_on: date
    schema_version: int = MODEL_LADDER_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MODEL_LADDER_SCHEMA_VERSION:
            raise ModelLadderError("unsupported model ladder schema version")
        rungs = tuple(target.rung for target in self.targets)
        if rungs != _EXPECTED_RUNGS:
            raise ModelLadderError(
                f"model ladder must contain ordered rungs {_EXPECTED_RUNGS}, found {rungs}"
            )
        identities = tuple((target.identifier, target.revision) for target in self.targets)
        if len(set(identities)) != len(identities):
            raise ModelLadderError("model ladder targets must be unique")
        if len({target.family for target in self.targets}) < 3:
            raise ModelLadderError("model ladder must span at least three architecture families")

    @property
    def ladder_id(self) -> str:
        payload = json.dumps(self.to_record(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "metadata_verified_on": self.metadata_verified_on.isoformat(),
            "weights_committed_to_git": False,
            "targets": [target.to_record() for target in self.targets],
        }


_DATASETS = (
    ModelDatasetProtocol.LOCAL_SMOKE,
    ModelDatasetProtocol.WIKITEXT_PERPLEXITY,
)

PERMISSIVE_MODEL_LADDER = ModelEvaluationLadder(
    (
        ModelLadderTarget(
            "100M",
            100_000_000,
            "HuggingFaceTB/SmolLM2-135M",
            "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
            ModelFamily.LLAMA,
            134_515_008,
            "apache-2.0",
            date(2025, 2, 6),
            "fast CPU smoke, codec, and end-to-end correctness baseline",
            (ModelHardwareMode.CPU_SMOKE, ModelHardwareMode.GPU_12_GB),
            _DATASETS,
        ),
        ModelLadderTarget(
            "300M",
            300_000_000,
            "HuggingFaceTB/SmolLM2-360M",
            "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
            ModelFamily.LLAMA,
            361_821_120,
            "apache-2.0",
            date(2025, 2, 6),
            "second Llama-family scale point for within-family scaling",
            (ModelHardwareMode.CPU_SMOKE, ModelHardwareMode.GPU_12_GB),
            _DATASETS,
        ),
        ModelLadderTarget(
            "500M",
            500_000_000,
            "Qwen/Qwen2.5-0.5B",
            "060db6499f32faf8b98477b0a26969ef7d8b9987",
            ModelFamily.QWEN,
            494_032_768,
            "apache-2.0",
            date(2024, 9, 25),
            "first cross-family transfer and Qwen adapter target",
            (ModelHardwareMode.CPU_SMOKE, ModelHardwareMode.GPU_12_GB),
            _DATASETS,
        ),
        ModelLadderTarget(
            "1B",
            1_000_000_000,
            "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
            "59f6f375b26bde864a6ca194a9a3044570490064",
            ModelFamily.LLAMA,
            1_100_048_384,
            "apache-2.0",
            date(2024, 9, 27),
            "consumer-scale Llama base checkpoint and CPU-streaming boundary",
            (ModelHardwareMode.CPU_STREAMING, ModelHardwareMode.GPU_12_GB),
            _DATASETS,
        ),
        ModelLadderTarget(
            "1.5B",
            1_500_000_000,
            "Qwen/Qwen2.5-1.5B",
            "8faed761d45a263340a0528343f099c05c9a4323",
            ModelFamily.QWEN,
            1_543_714_304,
            "apache-2.0",
            date(2024, 10, 8),
            "within-Qwen scaling and cross-model held-out evaluation",
            (ModelHardwareMode.CPU_STREAMING, ModelHardwareMode.GPU_12_GB),
            _DATASETS,
        ),
        ModelLadderTarget(
            "3B",
            3_000_000_000,
            "Qwen/Qwen3-4B-Base",
            "906bfd4b4dc7f14ee4320094d8b41684abff8539",
            ModelFamily.QWEN,
            4_022_468_096,
            "apache-2.0",
            date(2025, 7, 26),
            "large Qwen generation and quantized consumer-GPU boundary",
            (ModelHardwareMode.GPU_12_GB_QUANTIZED, ModelHardwareMode.GPU_24_GB),
            _DATASETS,
        ),
        ModelLadderTarget(
            "7B",
            7_000_000_000,
            "mistralai/Mistral-7B-v0.3",
            "caa1feb0e54d415e2df31207e5f4e273e33509b1",
            ModelFamily.MISTRAL,
            7_248_023_552,
            "apache-2.0",
            date(2025, 7, 24),
            "held-out Mistral family and 12 GB quantized/24 GB reference endpoint",
            (ModelHardwareMode.GPU_12_GB_QUANTIZED, ModelHardwareMode.GPU_24_GB),
            _DATASETS,
        ),
    ),
    metadata_verified_on=date(2026, 8, 24),
)
