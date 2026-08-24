"""Benchmark model-ladder stages on the local reference host."""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from modelsurgeon.adapters.huggingface.loader import (
    HuggingFaceDType,
    HuggingFaceLoadRequest,
    load_causal_lm,
)
from modelsurgeon.evaluation.model_ladder import PERMISSIVE_MODEL_LADDER
from modelsurgeon.evaluation.scale_study import choose_scale_default
from modelsurgeon.experiments.hardware import collect_hardware_inventory
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.instrumentation.memory_telemetry import (
    MemoryTelemetryConfig,
    MemoryTelemetryReport,
    TorchCudaMemoryProvider,
    collect_memory_telemetry,
)


def _snapshot_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _stage(
    name: str,
    operation: Callable[[], object],
    *,
    cuda: TorchCudaMemoryProvider | None,
) -> tuple[object, dict[str, object]]:
    captured: list[MemoryTelemetryReport] = []
    result_box: list[object] = []
    started_wall = time.perf_counter()
    started_cpu = time.process_time()

    def measured_operation() -> object:
        result = operation()
        result_box.append(result)
        return result

    collect_memory_telemetry(
        name,
        measured_operation,
        MemoryTelemetryConfig(True, 0.01, 8192),
        cuda=cuda,
        report_callback=captured.append,
    )
    if len(captured) != 1 or len(result_box) != 1:
        raise RuntimeError("stage operation and telemetry must each run exactly once")
    report = captured[0]
    measurement = {
        "stage": name,
        "status": "complete",
        "wall_seconds": time.perf_counter() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu,
        "peak_rss_bytes": report.peak_rss_bytes,
        "peak_cuda_allocated_bytes": report.peak_cuda_allocated_bytes,
        "peak_cuda_reserved_bytes": report.peak_cuda_reserved_bytes,
        "scratch_disk_peak_bytes": 0,
        "memory_sample_count": len(report.samples),
    }
    return result_box[0], measurement


def _first_matrix(model: object) -> tuple[str, Any]:
    candidates = [
        (name, parameter)
        for name, parameter in model.named_parameters()  # type: ignore[attr-defined]
        if parameter.ndim == 2 and "mlp" in name and "weight" in name
    ]
    if not candidates:
        candidates = [
            (name, parameter)
            for name, parameter in model.named_parameters()  # type: ignore[attr-defined]
            if parameter.ndim == 2
        ]
    if not candidates:
        raise RuntimeError("model exposes no matrix parameter")
    return candidates[0]


def _run_target(target: Any, accelerator_bytes: int | None) -> dict[str, object]:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    base: dict[str, object] = {
        "rung": target.rung,
        "identifier": target.identifier,
        "requested_revision": target.revision,
        "parameters": target.actual_parameters,
        "default": choose_scale_default(
            target.actual_parameters,
            accelerator_memory_bytes=accelerator_bytes,
        ).to_record(),
    }
    try:
        snapshot = Path(
            snapshot_download(
                target.identifier,
                revision=target.revision,
                local_files_only=True,
            )
        )
    except Exception as error:
        return {
            **base,
            "status": "failed",
            "failure_stage": "model_availability",
            "failure_type": "model_unavailable",
            "failure_message": str(error),
            "stages": [],
        }

    cuda = TorchCudaMemoryProvider() if torch.cuda.is_available() else None
    stages: list[dict[str, object]] = []
    try:
        loaded, load_measurement = _stage(
            "load",
            lambda: load_causal_lm(
                HuggingFaceLoadRequest(
                    target.identifier,
                    revision=target.revision,
                    device_map="auto" if cuda is not None else "cpu",
                    dtype=HuggingFaceDType.FLOAT16,
                    local_files_only=True,
                )
            ),
            cuda=cuda,
        )
        stages.append(load_measurement)
        model = loaded.model

        def analyze() -> dict[str, object]:
            named = tuple(model.named_parameters())
            name, tensor = _first_matrix(model)
            return {
                "observed_parameters": sum(parameter.numel() for _, parameter in named),
                "parameter_tensors": len(named),
                "selected_tensor": name,
                "selected_shape": list(tensor.shape),
            }

        analysis, measurement = _stage("analysis", analyze, cuda=cuda)
        stages.append({**measurement, "result": analysis})
        tensor_name, tensor = _first_matrix(model)

        def features() -> dict[str, object]:
            flat = tensor.detach().reshape(-1)
            count = min(65_536, flat.numel())
            indexes = torch.linspace(0, flat.numel() - 1, steps=count, device=flat.device).long()
            sample = flat[indexes].float()
            return {
                "tensor": tensor_name,
                "sample_elements": count,
                "mean": float(sample.mean().item()),
                "rms": float(sample.square().mean().sqrt().item()),
                "maximum_magnitude": float(sample.abs().max().item()),
            }

        feature_result, measurement = _stage("feature", features, cuda=cuda)
        stages.append({**measurement, "result": feature_result})

        def mutate() -> dict[str, object]:
            row = tensor[0]
            with torch.no_grad():
                original = row.detach().clone()
                row.zero_()
                changed = int(torch.count_nonzero(original).item())
                row.copy_(original)
            return {
                "tensor": tensor_name,
                "aligned_row_elements": row.numel(),
                "changed_elements": changed,
                "restored": True,
            }

        mutation_result, measurement = _stage("mutation", mutate, cuda=cuda)
        stages.append({**measurement, "result": mutation_result})
        tokenizer = AutoTokenizer.from_pretrained(
            target.identifier,
            revision=target.revision,
            local_files_only=True,
        )

        def evaluate() -> dict[str, object]:
            inputs = tokenizer(
                "Model surgery should preserve useful behavior after a bounded edit.",
                return_tensors="pt",
                truncation=True,
                max_length=32,
            )
            device = next(model.parameters()).device
            inputs = {name: value.to(device) for name, value in inputs.items()}
            with torch.inference_mode():
                output = model(**inputs)
            logits = output.logits
            return {
                "tokens": int(inputs["input_ids"].numel()),
                "finite_logits": bool(torch.isfinite(logits).all().item()),
                "execution_device": str(device),
            }

        evaluation, measurement = _stage("evaluation", evaluate, cuda=cuda)
        stages.append({**measurement, "result": evaluation})
        return {
            **base,
            "status": "complete",
            "resolved_revision": loaded.provenance.resolved_revision,
            "model_cache_bytes": _snapshot_bytes(snapshot),
            "stages": stages,
        }
    except Exception as error:
        return {
            **base,
            "status": "failed",
            "failure_stage": "execution",
            "failure_type": type(error).__name__,
            "failure_message": str(error),
            "model_cache_bytes": _snapshot_bytes(snapshot),
            "stages": stages,
        }
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rung", action="append")
    args = parser.parse_args()
    selected = set(args.rung or [target.rung for target in PERMISSIVE_MODEL_LADDER.targets])
    unknown = selected - {target.rung for target in PERMISSIVE_MODEL_LADDER.targets}
    if unknown:
        raise ValueError(f"unknown ladder rungs: {sorted(unknown)}")

    hardware = collect_hardware_inventory(args.output.parent)
    accelerator_bytes = max(
        (
            device.total_memory_bytes
            for device in hardware.cuda.devices
            if device.total_memory_bytes is not None
        ),
        default=None,
    )
    results = [
        _run_target(target, accelerator_bytes)
        for target in PERMISSIVE_MODEL_LADDER.targets
        if target.rung in selected
    ]
    record = {
        "record_type": "v0.8_consumer_scale_study",
        "version": "1",
        "ladder_id": PERMISSIVE_MODEL_LADDER.ladder_id,
        "protocol": {
            "local_files_only": True,
            "dtype": "float16",
            "feature_sample_elements": 65_536,
            "mutation": "zero_and_restore_first_aligned_matrix_row",
            "evaluation": "single_local_prompt_forward_max_32_tokens",
            "memory_sample_interval_seconds": 0.01,
            "scratch_writes": False,
        },
        "hardware": hardware.to_record(),
        "results": results,
    }
    args.output.write_text(canonical_identity_json(record) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
