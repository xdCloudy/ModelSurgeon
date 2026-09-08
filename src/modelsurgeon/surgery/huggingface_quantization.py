"""Reloadable first-party dynamic-int8 artifacts for Hugging Face causal LMs."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

HF_DYNAMIC_INT8_SCHEMA_VERSION = 1
HF_DYNAMIC_INT8_MANIFEST = "modelsurgeon-quantization.json"
HF_DYNAMIC_INT8_WEIGHTS = "model.safetensors"


class HuggingFaceQuantizationError(RuntimeError):
    """Raised when dynamic-int8 quantization cannot be safely materialized."""


@dataclass(frozen=True, slots=True)
class HuggingFaceDynamicInt8Report:
    """Measured model-storage accounting for one dynamic-int8 conversion."""

    method: str
    module_names: tuple[str, ...]
    source_weight_bytes: int
    quantized_weight_bytes: int
    schema_version: int = HF_DYNAMIC_INT8_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.method != "dynamic_int8":
            raise HuggingFaceQuantizationError("unsupported dynamic-int8 report method")
        if not self.module_names or self.module_names != tuple(sorted(self.module_names)):
            raise HuggingFaceQuantizationError(
                "quantized module names must be sorted and non-empty"
            )
        if self.source_weight_bytes <= 0 or self.quantized_weight_bytes <= 0:
            raise HuggingFaceQuantizationError("quantized storage accounting must be positive")

    @property
    def storage_delta_bytes(self) -> int:
        return self.quantized_weight_bytes - self.source_weight_bytes

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "module_names": list(self.module_names),
            "module_count": len(self.module_names),
            "source_weight_bytes": self.source_weight_bytes,
            "quantized_weight_bytes": self.quantized_weight_bytes,
            "storage_delta_bytes": self.storage_delta_bytes,
        }


def _torch() -> Any:
    try:
        return import_module("torch")
    except ImportError as error:
        raise HuggingFaceQuantizationError(
            "dynamic-int8 quantization requires the optional Hugging Face runtime"
        ) from error


def _quantized_linear_type(torch: Any) -> Any:
    try:
        return torch.nn.quantized.dynamic.Linear
    except AttributeError as error:
        raise HuggingFaceQuantizationError(
            "the installed torch runtime has no dynamic quantized Linear"
        ) from error


def _require_cpu(model: Any, torch: Any) -> None:
    devices = {
        str(parameter.device)
        for parameter in model.parameters()
        if getattr(parameter, "device", None) is not None
    }
    if devices and devices != {"cpu"}:
        raise HuggingFaceQuantizationError(
            "dynamic-int8 quantization is supported only for CPU-resident models"
        )


def _quantize_model(model: Any, torch: Any) -> Any:
    try:
        quantize_dynamic = torch.quantization.quantize_dynamic
        return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8).eval()
    except (AttributeError, RuntimeError, TypeError, ValueError) as error:
        raise HuggingFaceQuantizationError(
            f"torch dynamic-int8 quantization failed: {error}"
        ) from error


def quantize_huggingface_dynamic_int8(
    model: Any,
) -> tuple[Any, HuggingFaceDynamicInt8Report]:
    """Quantize every supported Linear in a CPU model and retain real accounting."""

    torch = _torch()
    _require_cpu(model, torch)
    source_bytes = 0
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            source_bytes += int(module.weight.numel() * module.weight.element_size())
            if module.bias is not None:
                source_bytes += int(module.bias.numel() * module.bias.element_size())
    quantized = _quantize_model(model, torch)
    names = tuple(
        sorted(
            name
            for name, module in quantized.named_modules()
            if isinstance(module, _quantized_linear_type(torch))
        )
    )
    if not names:
        raise HuggingFaceQuantizationError(
            "the model contains no quantizable torch.nn.Linear modules"
        )
    quantized_bytes = 0
    for name in names:
        module = dict(quantized.named_modules())[name]
        weight, bias = module._packed_params._weight_bias()
        quantized_bytes += int(weight.numel() * weight.element_size()) + 12
        if bias is not None:
            quantized_bytes += int(bias.numel() * bias.element_size())
    return quantized, HuggingFaceDynamicInt8Report(
        "dynamic_int8", names, source_bytes, quantized_bytes
    )


def _tensor_copy(value: Any, torch: Any) -> Any:
    if not torch.is_tensor(value):
        raise HuggingFaceQuantizationError("artifact state contains a non-tensor value")
    return value.detach().cpu().contiguous()


def publish_huggingface_dynamic_int8(
    model: Any,
    destination: Path,
    *,
    source_weight_bytes: int | None = None,
) -> tuple[Path, HuggingFaceDynamicInt8Report]:
    """Publish a safe-tensors dynamic-int8 artifact with a deterministic loader manifest."""

    if destination.exists():
        raise HuggingFaceQuantizationError(f"refusing to overwrite {destination}")
    torch = _torch()
    quantized_type = _quantized_linear_type(torch)
    modules = dict(model.named_modules())
    quantized_names = tuple(
        sorted(name for name, item in modules.items() if isinstance(item, quantized_type))
    )
    if not quantized_names:
        raise HuggingFaceQuantizationError("publish requires a dynamic-int8 model")
    config = getattr(model, "config", None)
    save_config = getattr(config, "save_pretrained", None)
    if not callable(save_config):
        raise HuggingFaceQuantizationError("quantized model has no saveable Hugging Face config")

    tensors: dict[str, Any] = {}
    source_bytes = 0
    quantized_bytes = 0
    for name in quantized_names:
        module = modules[name]
        weight, bias = module._packed_params._weight_bias()
        prefix = f"__quantized__.{name}."
        tensors[prefix + "weight"] = _tensor_copy(weight.int_repr(), torch)
        tensors[prefix + "weight_scale"] = torch.tensor(weight.q_scale(), dtype=torch.float32)
        tensors[prefix + "weight_zero_point"] = torch.tensor(
            weight.q_zero_point(), dtype=torch.int64
        )
        tensors[name + ".scale"] = torch.tensor(float(module.scale), dtype=torch.float32)
        tensors[name + ".zero_point"] = torch.tensor(int(module.zero_point), dtype=torch.int64)
        quantized_bytes += int(weight.numel() * weight.element_size()) + 12
        if bias is not None:
            tensors[prefix + "bias"] = _tensor_copy(bias, torch)
            quantized_bytes += int(bias.numel() * bias.element_size())

    for name, value in model.state_dict().items():
        if not torch.is_tensor(value) or any(
            name.startswith(item + ".") for item in quantized_names
        ):
            continue
        tensors[name] = _tensor_copy(value, torch)

    if source_weight_bytes is None:
        for name in quantized_names:
            module = modules[name]
            weight, bias = module._packed_params._weight_bias()
            source_bytes += int(weight.numel() * 4)
            if bias is not None:
                source_bytes += int(bias.numel() * bias.element_size())
    else:
        if source_weight_bytes <= 0:
            raise HuggingFaceQuantizationError("source weight accounting must be positive")
        source_bytes = source_weight_bytes

    try:
        destination.mkdir(parents=True)
        save_config(destination)
        safetensors = import_module("safetensors.torch")
        safetensors.save_file(tensors, str(destination / HF_DYNAMIC_INT8_WEIGHTS))
        manifest = {
            "schema_version": HF_DYNAMIC_INT8_SCHEMA_VERSION,
            "method": "dynamic_int8",
            "weight_file": HF_DYNAMIC_INT8_WEIGHTS,
            "module_names": list(quantized_names),
        }
        (destination / HF_DYNAMIC_INT8_MANIFEST).write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        shutil.rmtree(destination, ignore_errors=True)
        raise HuggingFaceQuantizationError(
            f"could not publish quantized safetensors: {error}"
        ) from error
    report = HuggingFaceDynamicInt8Report(
        "dynamic_int8", quantized_names, source_bytes, quantized_bytes
    )
    return destination / HF_DYNAMIC_INT8_WEIGHTS, report


def load_huggingface_dynamic_int8(
    destination: Path,
    *,
    trust_remote_code: bool = False,
) -> Any:
    """Reload a published dynamic-int8 artifact without executing pickle payloads."""

    manifest_path = destination / HF_DYNAMIC_INT8_MANIFEST
    weights_path = destination / HF_DYNAMIC_INT8_WEIGHTS
    if not manifest_path.is_file() or not weights_path.is_file():
        raise HuggingFaceQuantizationError("dynamic-int8 artifact manifest or weights are missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HuggingFaceQuantizationError("dynamic-int8 artifact manifest is invalid") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != HF_DYNAMIC_INT8_SCHEMA_VERSION
        or manifest.get("method") != "dynamic_int8"
        or manifest.get("weight_file") != HF_DYNAMIC_INT8_WEIGHTS
    ):
        raise HuggingFaceQuantizationError("unsupported dynamic-int8 artifact manifest")
    raw_names = manifest.get("module_names")
    if not isinstance(raw_names, list) or not all(isinstance(item, str) for item in raw_names):
        raise HuggingFaceQuantizationError("dynamic-int8 module names are invalid")
    names = tuple(raw_names)
    if names != tuple(sorted(set(names))):
        raise HuggingFaceQuantizationError("dynamic-int8 module names are not canonical")

    torch = _torch()
    try:
        transformers = import_module("transformers")
        config = transformers.AutoConfig.from_pretrained(
            destination,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
        model = transformers.AutoModelForCausalLM.from_config(
            config,
            trust_remote_code=trust_remote_code,
        ).eval()
    except Exception as error:
        raise HuggingFaceQuantizationError(
            f"could not construct the dynamic-int8 model skeleton: {error}"
        ) from error
    quantized_model = _quantize_model(model, torch)
    quantized_type = _quantized_linear_type(torch)
    modules = dict(quantized_model.named_modules())
    if any(name not in modules or not isinstance(modules[name], quantized_type) for name in names):
        raise HuggingFaceQuantizationError("artifact names do not match the model architecture")
    try:
        safetensors = import_module("safetensors.torch")
        tensors = safetensors.load_file(str(weights_path), device="cpu")
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        raise HuggingFaceQuantizationError(
            f"could not load quantized safetensors: {error}"
        ) from error
    names_set = set(names)
    for name in names:
        module = modules[name]
        prefix = f"__quantized__.{name}."
        required = (prefix + "weight", prefix + "weight_scale", prefix + "weight_zero_point")
        if any(key not in tensors for key in required):
            raise HuggingFaceQuantizationError(f"quantized weights for {name!r} are incomplete")
        weight = torch._make_per_tensor_quantized_tensor(
            tensors[prefix + "weight"],
            float(tensors[prefix + "weight_scale"].item()),
            int(tensors[prefix + "weight_zero_point"].item()),
        )
        module._packed_params.set_weight_bias(weight, tensors.get(prefix + "bias"))
        scale_key = name + ".scale"
        zero_point_key = name + ".zero_point"
        if scale_key not in tensors or zero_point_key not in tensors:
            raise HuggingFaceQuantizationError(
                f"output quantization metadata for {name!r} is missing"
            )
        module.scale = float(tensors[scale_key].item())
        module.zero_point = int(tensors[zero_point_key].item())

    parameters = dict(quantized_model.named_parameters())
    buffers = dict(quantized_model.named_buffers())
    ordinary = {
        key: value
        for key, value in tensors.items()
        if not key.startswith("__quantized__.")
        and not any(key.startswith(name + ".") for name in names_set)
    }
    expected = {
        key
        for key, value in quantized_model.state_dict().items()
        if torch.is_tensor(value)
        and not any(key.startswith(name + ".") for name in names_set)
    }
    missing = sorted(expected - set(ordinary))
    if missing:
        raise HuggingFaceQuantizationError(
            "dynamic-int8 artifact is missing ordinary model tensors: " + ", ".join(missing[:8])
        )
    for key, value in ordinary.items():
        target = parameters.get(key)
        if target is None:
            target = buffers.get(key)
        if target is None:
            raise HuggingFaceQuantizationError(
                f"dynamic-int8 artifact contains an unexpected tensor {key!r}"
            )
        if tuple(target.shape) != tuple(value.shape):
            raise HuggingFaceQuantizationError(f"tensor shape mismatch for {key!r}")
        target.data.copy_(value)
    quantized_model.eval()
    return quantized_model


__all__ = [
    "HF_DYNAMIC_INT8_MANIFEST",
    "HF_DYNAMIC_INT8_SCHEMA_VERSION",
    "HF_DYNAMIC_INT8_WEIGHTS",
    "HuggingFaceDynamicInt8Report",
    "HuggingFaceQuantizationError",
    "load_huggingface_dynamic_int8",
    "publish_huggingface_dynamic_int8",
    "quantize_huggingface_dynamic_int8",
]
