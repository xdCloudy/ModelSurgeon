"""Safe, lazy Hugging Face loading boundary with reproducible provenance."""

from __future__ import annotations

import json
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from pathlib import Path
from types import MappingProxyType
from typing import Any

from modelsurgeon.adapters.checkpoint_paths import (
    MAX_CHECKPOINT_INDEX_BYTES,
    CheckpointPathError,
    resolve_checkpoint_shard_path,
    validate_checkpoint_index_path,
)


class HuggingFaceDType(StrEnum):
    """Supported compute dtypes at the Transformers boundary."""

    AUTO = "auto"
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


class HuggingFaceLoadError(RuntimeError):
    """Base class for categorized loader failures."""


class HuggingFaceDependencyError(HuggingFaceLoadError):
    """Raised when the optional loading runtime is unavailable."""


class HuggingFaceModelError(HuggingFaceLoadError):
    """Raised when model files or configuration cannot be loaded."""


class HuggingFaceRevisionError(HuggingFaceLoadError):
    """Raised when a requested or resolved revision is invalid."""


_DEVICE_MAPS = frozenset({"auto", "balanced", "balanced_low_0", "cpu", "sequential"})
_AUTO_DEVICE_MAPS = frozenset({"auto", "balanced", "balanced_low_0", "sequential"})
_SHARDED_INDEX_NAMES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")


def _validated_offload_folder(model: str, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise ValueError("offload_folder must be a local path")
    if isinstance(value, str) and not value.strip():
        raise ValueError("offload_folder cannot be blank")
    try:
        folder = Path(value).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError("offload_folder must resolve to a local directory") from exc
    if folder.exists() and not folder.is_dir():
        raise ValueError("offload_folder must be a directory")

    source = Path(model).expanduser()
    if source.exists():
        source_root = source.resolve() if source.is_dir() else source.resolve().parent
        try:
            folder.relative_to(source_root)
        except ValueError:
            pass
        else:
            raise ValueError("offload_folder cannot be inside the source model directory")
    return folder


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
    max_memory: Mapping[str | int, int] | None = None
    offload_folder: str | Path | None = None
    offload_buffers: bool = False

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model cannot be blank")
        if self.revision is not None and not self.revision.strip():
            raise ValueError("revision cannot be blank")
        if self.device_map is not None and self.device_map not in _DEVICE_MAPS:
            allowed = ", ".join(sorted(_DEVICE_MAPS))
            raise ValueError(f"device_map must be one of: {allowed}")
        if self.max_memory is not None:
            if not isinstance(self.max_memory, Mapping) or not self.max_memory:
                raise ValueError("max_memory must be a non-empty device-to-bytes mapping")
            copied: dict[str | int, int] = {}
            serialized_devices: set[str] = set()
            for device, limit in self.max_memory.items():
                if isinstance(device, bool) or not isinstance(device, (str, int)):
                    raise ValueError("max_memory device keys must be strings or integer indexes")
                if isinstance(device, str) and not device.strip():
                    raise ValueError("max_memory device keys cannot be blank")
                if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
                    raise ValueError("max_memory limits must be positive integer byte counts")
                serialized_device = str(device)
                if serialized_device in serialized_devices:
                    raise ValueError("max_memory device keys must be unique when serialized")
                serialized_devices.add(serialized_device)
                copied[device] = limit
            object.__setattr__(self, "max_memory", MappingProxyType(copied))
        if not isinstance(self.offload_buffers, bool):
            raise ValueError("offload_buffers must be a boolean")
        offload_folder = _validated_offload_folder(self.model, self.offload_folder)
        object.__setattr__(self, "offload_folder", offload_folder)
        if (
            self.max_memory is not None or offload_folder is not None or self.offload_buffers
        ) and self.device_map not in _AUTO_DEVICE_MAPS:
            raise ValueError(
                "max_memory and offload options require an automatic or sequential device_map"
            )


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
    max_memory: tuple[tuple[str, int], ...] | None = None
    offload_folder: str | None = None
    offload_buffers: bool = False

    def to_record(self) -> dict[str, object]:
        loader_options: dict[str, object] = {
            "trust_remote_code": self.trust_remote_code,
            "device_map": self.device_map,
            "dtype": self.dtype.value,
            "local_files_only": self.local_files_only,
            "low_cpu_mem_usage": self.low_cpu_mem_usage,
        }
        if self.max_memory is not None:
            loader_options["max_memory"] = dict(self.max_memory)
        if self.offload_folder is not None:
            loader_options["offload_folder"] = self.offload_folder
        if self.offload_buffers:
            loader_options["offload_buffers"] = True
        return {
            "source": self.source,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "loader_options": loader_options,
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
        raise HuggingFaceDependencyError(
            "Hugging Face support is optional; install with `uv sync --extra hf`"
        ) from exc
    return getattr(torch, dtype.value)


def _validate_loaded_model_placement(model: Any) -> None:
    """Reject meta-only models that Accelerate did not dispatch for execution."""
    parameters = getattr(model, "parameters", None)
    modules = getattr(model, "modules", None)
    if not callable(parameters) or not callable(modules):
        return

    try:
        has_meta_parameters = any(
            getattr(getattr(parameter, "device", None), "type", None) == "meta"
            for parameter in parameters()
        )
        has_accelerate_hooks = any(
            getattr(module, "_hf_hook", None) is not None for module in modules()
        )
    except Exception as exc:
        raise HuggingFaceModelError("loaded model placement could not be inspected") from exc

    if has_meta_parameters and not has_accelerate_hooks:
        raise HuggingFaceModelError(
            "Transformers left model parameters on the meta device without execution hooks; "
            "increase max_memory or use a supported device_map"
        )


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
    raise HuggingFaceRevisionError(
        "Transformers did not expose the resolved Hub commit; pin a revision explicitly"
    )


def _validate_shard_index(
    index_path: Path,
    checkpoint_root: Path,
    *,
    hub_snapshot: bool,
) -> None:
    """Reject shard paths that escape the checkpoint or target special files."""
    try:
        validate_checkpoint_index_path(
            index_path,
            checkpoint_root,
            hub_snapshot=hub_snapshot,
        )
    except CheckpointPathError as exc:
        raise HuggingFaceModelError(str(exc)) from exc
    try:
        if not stat.S_ISREG(index_path.stat().st_mode):
            raise HuggingFaceModelError("checkpoint index is not a regular file")
        with index_path.open("rb") as stream:
            raw_index = stream.read(MAX_CHECKPOINT_INDEX_BYTES + 1)
        if len(raw_index) > MAX_CHECKPOINT_INDEX_BYTES:
            raise HuggingFaceModelError(
                f"checkpoint index exceeds {MAX_CHECKPOINT_INDEX_BYTES} bytes"
            )
        payload = json.loads(raw_index.decode("utf-8"))
    except HuggingFaceModelError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HuggingFaceModelError("checkpoint index is unreadable or invalid JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("weight_map"), dict):
        raise HuggingFaceModelError("checkpoint index must contain a weight_map object")

    shard_names = tuple(payload["weight_map"].values())
    string_shard_names = tuple(
        shard_name for shard_name in shard_names if isinstance(shard_name, str)
    )
    if not string_shard_names or len(string_shard_names) != len(shard_names):
        raise HuggingFaceModelError("checkpoint index contains an invalid shard filename")

    for shard_name in set(string_shard_names):
        try:
            shard_path = resolve_checkpoint_shard_path(
                checkpoint_root,
                shard_name,
                hub_snapshot=hub_snapshot,
            )
        except CheckpointPathError as exc:
            raise HuggingFaceModelError(str(exc)) from exc
        try:
            shard_stat = shard_path.stat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise HuggingFaceModelError(
                f"checkpoint shard cannot be inspected: {shard_name!r}"
            ) from exc
        if not stat.S_ISREG(shard_stat.st_mode):
            raise HuggingFaceModelError(
                f"checkpoint shard is not a regular file: {shard_name!r}"
            )


def _preflight_model_checkpoint(request: HuggingFaceLoadRequest) -> str | None:
    """Validate checkpoint indexes before Transformers/Accelerate opens shards.

    For Hub models, first resolve ``config.json`` so preflight and the actual load
    use one immutable snapshot even when the requested revision is a moving ref.
    """
    model_path = Path(request.model)
    if model_path.exists():
        checkpoint_root = (
            model_path.resolve() if model_path.is_dir() else model_path.resolve().parent
        )
        indexes = (
            [model_path.resolve()]
            if model_path.is_file() and model_path.name.endswith(".index.json")
            else list(checkpoint_root.glob("*.index.json"))
        )
        for index_path in indexes:
            _validate_shard_index(index_path, checkpoint_root, hub_snapshot=False)
        return None

    try:
        hub = import_module("huggingface_hub")
        hub_errors = import_module("huggingface_hub.errors")
    except ImportError as exc:
        raise HuggingFaceDependencyError(
            "Hugging Face Hub metadata support is required to inspect model checkpoints"
        ) from exc
    download = hub.hf_hub_download
    try:
        config_path = Path(
            download(
                repo_id=request.model,
                filename="config.json",
                revision=request.revision,
                local_files_only=request.local_files_only,
            )
        )
    except Exception as exc:
        raise HuggingFaceModelError(
            f"failed to resolve immutable Hub snapshot for {request.model!r}: {exc}"
        ) from exc
    if config_path.name != "config.json" or config_path.parent.parent.name != "snapshots":
        raise HuggingFaceRevisionError(
            "Hugging Face Hub did not expose an immutable snapshot for model preflight"
        )
    resolved_revision = config_path.parent.name
    snapshot_root = config_path.parent
    for index_name in _SHARDED_INDEX_NAMES:
        try:
            index_path = Path(
                download(
                    repo_id=request.model,
                    filename=index_name,
                    revision=resolved_revision,
                    local_files_only=request.local_files_only,
                )
            )
        except Exception as exc:
            error_types = tuple(
                error_type
                for error_type in (
                    getattr(hub_errors, "EntryNotFoundError", None),
                    getattr(hub_errors, "LocalEntryNotFoundError", None),
                )
                if isinstance(error_type, type)
            )
            if error_types and isinstance(exc, error_types):
                continue
            raise HuggingFaceModelError(
                f"failed to inspect Hub checkpoint index {index_name!r}: {exc}"
            ) from exc
        if index_path.parent != snapshot_root or index_path.name != index_name:
            raise HuggingFaceModelError(
                f"Hub checkpoint index {index_name!r} was not resolved in the pinned snapshot"
            )
        _validate_shard_index(index_path, snapshot_root, hub_snapshot=True)
    return resolved_revision


def load_causal_lm(request: HuggingFaceLoadRequest) -> HuggingFaceLoadResult:
    """Load a causal LM with safe defaults and return reproducible provenance."""
    try:
        auto_model = import_module("transformers").AutoModelForCausalLM
    except (AttributeError, ImportError) as exc:
        raise HuggingFaceDependencyError(
            "Hugging Face support is optional; install with `uv sync --extra hf`"
        ) from exc

    try:
        resolved_hub_revision = _preflight_model_checkpoint(request)
        loader_options: dict[str, Any] = {
            "revision": resolved_hub_revision or request.revision,
            "trust_remote_code": request.trust_remote_code,
            "device_map": _transformers_device_map(request.device_map),
            "torch_dtype": _torch_dtype(request.dtype),
            "local_files_only": request.local_files_only,
            "low_cpu_mem_usage": request.low_cpu_mem_usage,
        }
        if request.max_memory is not None:
            loader_options["max_memory"] = dict(request.max_memory)
        if request.offload_folder is not None:
            loader_options["offload_folder"] = str(request.offload_folder)
        if request.offload_buffers:
            loader_options["offload_buffers"] = True
        model = auto_model.from_pretrained(
            request.model,
            **loader_options,
        )
        _validate_loaded_model_placement(model)
    except Exception as exc:
        if isinstance(exc, HuggingFaceLoadError):
            raise
        exception_names = {exception_type.__name__ for exception_type in type(exc).mro()}
        if any("Revision" in name for name in exception_names):
            raise HuggingFaceRevisionError(
                f"failed to load revision {request.revision!r} for {request.model!r}: {exc}"
            ) from exc
        raise HuggingFaceModelError(f"failed to load model {request.model!r}: {exc}") from exc
    provenance = HuggingFaceLoadProvenance(
        source=request.model,
        requested_revision=request.revision,
        resolved_revision=_resolved_revision(
            model,
            HuggingFaceLoadRequest(
                model=request.model,
                revision=resolved_hub_revision or request.revision,
                trust_remote_code=request.trust_remote_code,
                device_map=request.device_map,
                dtype=request.dtype,
                local_files_only=request.local_files_only,
                low_cpu_mem_usage=request.low_cpu_mem_usage,
                max_memory=request.max_memory,
                offload_folder=request.offload_folder,
                offload_buffers=request.offload_buffers,
            ),
        ),
        trust_remote_code=request.trust_remote_code,
        device_map=request.device_map,
        dtype=request.dtype,
        local_files_only=request.local_files_only,
        low_cpu_mem_usage=request.low_cpu_mem_usage,
        max_memory=(
            tuple(sorted((str(device), limit) for device, limit in request.max_memory.items()))
            if request.max_memory is not None
            else None
        ),
        offload_folder=str(request.offload_folder) if request.offload_folder is not None else None,
        offload_buffers=request.offload_buffers,
    )
    return HuggingFaceLoadResult(model=model, provenance=provenance)
