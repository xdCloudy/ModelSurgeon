"""Safe, lazy Hugging Face loading boundary with reproducible provenance."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from pathlib import Path
from typing import Any


class HuggingFaceDType(StrEnum):
    """Supported compute dtypes at the Transformers boundary."""

    AUTO = "auto"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


_DEVICE_MAPS = frozenset({"auto", "balanced", "balanced_low_0", "cpu", "sequential"})


@dataclass(frozen=True, slots=True)
class HuggingFaceLoadRequest:
    """Immutable inputs controlling a causal language-model load."""

    model: str
    revision: str | None = None
    trust_remote_code: bool = False
    device_map: str | None = "cpu"
    dtype: HuggingFaceDType = HuggingFaceDType.AUTO
    local_files_only: bool = False
    low_cpu_mem_usage: bool = True

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model cannot be blank")
        if self.revision is not None and not self.revision.strip():
            raise ValueError("revision cannot be blank")
        if self.device_map is not None and self.device_map not in _DEVICE_MAPS:
            allowed = ", ".join(sorted(_DEVICE_MAPS))
            raise ValueError(f"device_map must be one of: {allowed}")


@dataclass(frozen=True, slots=True)
class HuggingFaceLoadProvenance:
    """Persistence-safe record of the resolved source and loader controls."""

    source: str
    requested_revision: str | None
    resolved_revision: str
    trust_remote_code: bool
    device_map: str | None
    dtype: HuggingFaceDType
    local_files_only: bool
    low_cpu_mem_usage: bool

    def to_record(self) -> dict[str, object]:
        return {
            "source": self.source,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "loader_options": {
                "trust_remote_code": self.trust_remote_code,
                "device_map": self.device_map,
                "dtype": self.dtype.value,
                "local_files_only": self.local_files_only,
                "low_cpu_mem_usage": self.low_cpu_mem_usage,
            },
        }


@dataclass(frozen=True, slots=True)
class HuggingFaceLoadResult:
    """Loaded model and the provenance necessary to reproduce the load."""

    model: Any
    provenance: HuggingFaceLoadProvenance


def _torch_dtype(dtype: HuggingFaceDType) -> Any:
    if dtype is HuggingFaceDType.AUTO:
        return "auto"
    try:
        torch = import_module("torch")
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face support is optional; install with `uv sync --extra hf`"
        ) from exc
    return getattr(torch, dtype.value)


def _transformers_device_map(device_map: str | None) -> str | dict[str, str] | None:
    if device_map == "cpu":
        return {"": "cpu"}
    return device_map


def _resolved_revision(model: object, request: HuggingFaceLoadRequest) -> str:
    config = getattr(model, "config", None)
    commit_hash = getattr(config, "_commit_hash", None)
    if isinstance(commit_hash, str) and commit_hash.strip():
        return commit_hash
    if request.revision is not None:
        return request.revision
    local_path = Path(request.model)
    if local_path.exists():
        return str(local_path.resolve())
    raise RuntimeError(
        "Transformers did not expose the resolved Hub commit; pin a revision explicitly"
    )


def load_causal_lm(request: HuggingFaceLoadRequest) -> HuggingFaceLoadResult:
    """Load a causal LM with safe defaults and return reproducible provenance."""
    try:
        auto_model = import_module("transformers").AutoModelForCausalLM
    except (AttributeError, ImportError) as exc:
        raise RuntimeError(
            "Hugging Face support is optional; install with `uv sync --extra hf`"
        ) from exc

    model = auto_model.from_pretrained(
        request.model,
        revision=request.revision,
        trust_remote_code=request.trust_remote_code,
        device_map=_transformers_device_map(request.device_map),
        torch_dtype=_torch_dtype(request.dtype),
        local_files_only=request.local_files_only,
        low_cpu_mem_usage=request.low_cpu_mem_usage,
    )
    provenance = HuggingFaceLoadProvenance(
        source=request.model,
        requested_revision=request.revision,
        resolved_revision=_resolved_revision(model, request),
        trust_remote_code=request.trust_remote_code,
        device_map=request.device_map,
        dtype=request.dtype,
        local_files_only=request.local_files_only,
        low_cpu_mem_usage=request.low_cpu_mem_usage,
    )
    return HuggingFaceLoadResult(model=model, provenance=provenance)
