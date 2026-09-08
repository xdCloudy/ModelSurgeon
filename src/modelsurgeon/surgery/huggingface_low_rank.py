"""Reloadable first-party low-rank Hugging Face artifacts."""

from __future__ import annotations

import json
import shutil
from importlib import import_module
from pathlib import Path
from typing import Any

from modelsurgeon.adapters.huggingface.low_rank import (
    LowRankReplacement,
    LowRankReplacementReport,
    replace_huggingface_linears_low_rank,
)

HF_LOW_RANK_SCHEMA_VERSION = 1
HF_LOW_RANK_MANIFEST = "modelsurgeon-low-rank.json"
HF_LOW_RANK_WEIGHTS = "model.safetensors"


class HuggingFaceLowRankError(RuntimeError):
    """Raised when a low-rank artifact cannot be safely published or reloaded."""


def _torch() -> Any:
    try:
        return import_module("torch")
    except ImportError as error:
        raise HuggingFaceLowRankError(
            "Hugging Face low-rank artifacts require the optional runtime"
        ) from error


def _tensor_copy(value: Any, torch: Any) -> Any:
    if not torch.is_tensor(value):
        raise HuggingFaceLowRankError("low-rank artifact contains a non-tensor value")
    return value.detach().cpu().contiguous()


def _replacement_report(model: Any, names: tuple[str, ...]) -> LowRankReplacementReport:
    modules = dict(model.named_modules())
    replacements: list[LowRankReplacement] = []
    for name in names:
        module = modules.get(name)
        rank = getattr(module, "rank", None)
        if module is None or not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise HuggingFaceLowRankError(f"low-rank module {name!r} has no valid rank")
        rows = int(module.out_features)
        columns = int(module.in_features)
        new_parameters = sum(int(parameter.numel()) for parameter in module.parameters())
        old_parameters = rows * columns + (
            rows if getattr(module.up, "bias", None) is not None else 0
        )
        replacements.append(
            LowRankReplacement(
                name,
                rank,
                rank,
                0.0,
                old_parameters,
                new_parameters,
                2 * rows * columns,
                2 * rank * (rows + columns),
            )
        )
    return LowRankReplacementReport(tuple(replacements))


def publish_huggingface_low_rank(
    model: Any,
    destination: Path,
) -> tuple[Path, LowRankReplacementReport]:
    """Publish low-rank modules as safe-tensors plus a deterministic manifest."""

    if destination.exists():
        raise HuggingFaceLowRankError(f"refusing to overwrite {destination}")
    torch = _torch()
    modules = dict(model.named_modules())
    names = tuple(
        sorted(
            name
            for name, module in modules.items()
            if getattr(module, "_modelsurgeon_low_rank", False)
        )
    )
    if not names:
        raise HuggingFaceLowRankError("publish requires at least one low-rank module")
    report = _replacement_report(model, names)
    ranks = {item.module_path: item.effective_rank for item in report.replacements}
    save_config = getattr(getattr(model, "config", None), "save_pretrained", None)
    if not callable(save_config):
        raise HuggingFaceLowRankError("low-rank model has no saveable Hugging Face config")
    tensors = {
        name: _tensor_copy(value, torch)
        for name, value in model.state_dict().items()
        if torch.is_tensor(value)
    }
    try:
        destination.mkdir(parents=True)
        save_config(destination)
        safetensors = import_module("safetensors.torch")
        safetensors.save_file(tensors, str(destination / HF_LOW_RANK_WEIGHTS))
        manifest = {
            "schema_version": HF_LOW_RANK_SCHEMA_VERSION,
            "method": "low_rank",
            "weight_file": HF_LOW_RANK_WEIGHTS,
            "module_names": list(names),
            "ranks": ranks,
        }
        (destination / HF_LOW_RANK_MANIFEST).write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        shutil.rmtree(destination, ignore_errors=True)
        raise HuggingFaceLowRankError(f"could not publish low-rank safetensors: {error}") from error
    return destination / HF_LOW_RANK_WEIGHTS, report


def load_huggingface_low_rank(destination: Path, *, trust_remote_code: bool = False) -> Any:
    """Reload a low-rank artifact by rebuilding its declared module wrappers."""

    manifest_path = destination / HF_LOW_RANK_MANIFEST
    weights_path = destination / HF_LOW_RANK_WEIGHTS
    if not manifest_path.is_file() or not weights_path.is_file():
        raise HuggingFaceLowRankError("low-rank artifact manifest or weights are missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HuggingFaceLowRankError("low-rank artifact manifest is invalid") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != HF_LOW_RANK_SCHEMA_VERSION
        or manifest.get("method") != "low_rank"
        or manifest.get("weight_file") != HF_LOW_RANK_WEIGHTS
    ):
        raise HuggingFaceLowRankError("unsupported low-rank artifact manifest")
    raw_names = manifest.get("module_names")
    raw_ranks = manifest.get("ranks")
    if (
        not isinstance(raw_names, list)
        or not all(isinstance(item, str) for item in raw_names)
        or raw_names != sorted(set(raw_names))
        or not isinstance(raw_ranks, dict)
    ):
        raise HuggingFaceLowRankError("low-rank module metadata is invalid")
    names = tuple(raw_names)
    ranks: tuple[tuple[str, int], ...] = tuple(
        (name, raw_ranks[name])
        for name in names
        if isinstance(raw_ranks.get(name), int) and not isinstance(raw_ranks.get(name), bool)
    )
    if len(ranks) != len(names) or any(rank <= 0 for _, rank in ranks):
        raise HuggingFaceLowRankError("low-rank ranks are invalid")
    try:
        transformers = import_module("transformers")
        config = transformers.AutoConfig.from_pretrained(
            destination, local_files_only=True, trust_remote_code=trust_remote_code
        )
        model = transformers.AutoModelForCausalLM.from_config(
            config, trust_remote_code=trust_remote_code
        ).eval()
        report = replace_huggingface_linears_low_rank(model, ranks)
    except Exception as error:
        raise HuggingFaceLowRankError(
            f"could not construct the low-rank model skeleton: {error}"
        ) from error
    if report.module_names != names:
        raise HuggingFaceLowRankError("artifact names do not match the model architecture")
    torch = _torch()
    try:
        safetensors = import_module("safetensors.torch")
        tensors = safetensors.load_file(str(weights_path), device="cpu")
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise HuggingFaceLowRankError(f"could not load low-rank safetensors: {error}") from error
    expected = {name for name, value in model.state_dict().items() if torch.is_tensor(value)}
    if expected != set(tensors):
        missing = sorted(expected - set(tensors))
        unexpected = sorted(set(tensors) - expected)
        raise HuggingFaceLowRankError(
            f"low-rank artifact state mismatch (missing={missing[:4]}, unexpected={unexpected[:4]})"
        )
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    for key, value in tensors.items():
        target = parameters.get(key)
        if target is None:
            target = buffers.get(key)
        if target is None:
            raise HuggingFaceLowRankError(f"low-rank artifact contains unexpected tensor {key!r}")
        if tuple(target.shape) != tuple(value.shape):
            raise HuggingFaceLowRankError(f"tensor shape mismatch for {key!r}")
        target.data.copy_(value)
    model.eval()
    return model


__all__ = [
    "HF_LOW_RANK_MANIFEST",
    "HF_LOW_RANK_SCHEMA_VERSION",
    "HF_LOW_RANK_WEIGHTS",
    "HuggingFaceLowRankError",
    "load_huggingface_low_rank",
    "publish_huggingface_low_rank",
]
