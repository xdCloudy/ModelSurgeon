"""Safe, lazy Hugging Face loading boundary with reproducible provenance."""

from __future__ import annotations

import json
import stat
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any


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
_SHARDED_INDEX_NAMES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")


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
        raise HuggingFaceDependencyError(
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
    raise HuggingFaceRevisionError(
        "Transformers did not expose the resolved Hub commit; pin a revision explicitly"
    )


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_shard_index(
    index_path: Path,
    checkpoint_root: Path,
    *,
    hub_snapshot: bool,
) -> None:
    """Reject shard paths that escape the checkpoint or target special files."""
    resolved_root = checkpoint_root.resolve()
    if hub_snapshot:
        # Hub snapshots symlink files into the repository's blobs directory.
        # Allow files in this snapshot or its blobs, not sibling snapshots.
        cache_root = checkpoint_root.parent.parent.resolve()
        blobs_root = (cache_root / "blobs").resolve()
        resolved_index = index_path.resolve()
        if not (
            _inside(resolved_index, resolved_root) or _inside(resolved_index, blobs_root)
        ):
            raise HuggingFaceModelError(
                "Hub checkpoint index escapes its snapshot or model blob cache"
            )
    else:
        if not _inside(index_path.resolve(), resolved_root):
            raise HuggingFaceModelError("checkpoint index escapes its model directory")
    try:
        if not stat.S_ISREG(index_path.stat().st_mode):
            raise HuggingFaceModelError("checkpoint index is not a regular file")
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except HuggingFaceModelError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HuggingFaceModelError("checkpoint index is unreadable or invalid JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("weight_map"), dict):
        raise HuggingFaceModelError("checkpoint index must contain a weight_map object")

    shard_names = tuple(payload["weight_map"].values())
    if not shard_names or any(
        not isinstance(shard_name, str)
        or not shard_name
        or "\x00" in shard_name
        or "\\" in shard_name
        or ":" in shard_name
        for shard_name in shard_names
    ):
        raise HuggingFaceModelError("checkpoint index contains an invalid shard filename")

    for shard_name in set(shard_names):
        if not isinstance(shard_name, str):
            raise HuggingFaceModelError("checkpoint index contains an invalid shard filename")
        posix_name = PurePosixPath(shard_name)
        windows_name = PureWindowsPath(shard_name)
        if (
            posix_name.is_absolute()
            or windows_name.drive
            or windows_name.root
            or ".." in posix_name.parts
            or ".." in windows_name.parts
        ):
            raise HuggingFaceModelError(
                f"checkpoint shard path escapes its model directory: {shard_name!r}"
            )
        shard_path = checkpoint_root.joinpath(*posix_name.parts)
        if hub_snapshot:
            # Hub snapshots intentionally symlink shard files into the blob cache;
            # parent directories must still stay inside the snapshot.
            parent = shard_path.parent
            while parent != checkpoint_root:
                if parent.is_symlink():
                    raise HuggingFaceModelError(
                        f"checkpoint shard path crosses a symlinked directory: {shard_name!r}"
                    )
                if not _inside(parent, checkpoint_root):
                    raise HuggingFaceModelError(
                        f"checkpoint shard path escapes its snapshot: {shard_name!r}"
                    )
                parent = parent.parent
            resolved_shard = shard_path.resolve()
            if not (
                _inside(resolved_shard, resolved_root) or _inside(resolved_shard, blobs_root)
            ):
                raise HuggingFaceModelError(
                    "checkpoint shard target escapes its snapshot or model blob cache: "
                    f"{shard_name!r}"
                )
        else:
            resolved_shard = shard_path.resolve()
            if not _inside(resolved_shard, checkpoint_root.resolve()):
                raise HuggingFaceModelError(
                    f"checkpoint shard path escapes its model directory: {shard_name!r}"
                )
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
        model = auto_model.from_pretrained(
            request.model,
            revision=resolved_hub_revision or request.revision,
            trust_remote_code=request.trust_remote_code,
            device_map=_transformers_device_map(request.device_map),
            torch_dtype=_torch_dtype(request.dtype),
            local_files_only=request.local_files_only,
            low_cpu_mem_usage=request.low_cpu_mem_usage,
        )
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
            ),
        ),
        trust_remote_code=request.trust_remote_code,
        device_map=request.device_map,
        dtype=request.dtype,
        local_files_only=request.local_files_only,
        low_cpu_mem_usage=request.low_cpu_mem_usage,
    )
    return HuggingFaceLoadResult(model=model, provenance=provenance)
