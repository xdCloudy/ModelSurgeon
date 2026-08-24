"""Run the v0.8 matched surgery and requantization loss decomposition."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import struct
from pathlib import Path

from modelsurgeon.adapters.huggingface.loader import (
    HuggingFaceDType,
    HuggingFaceLoadRequest,
    load_causal_lm,
)
from modelsurgeon.evaluation.quantization_loss_study import run_quantization_loss_study
from modelsurgeon.evaluation.quantized_feature_reliability import QuantizedTensorSample
from modelsurgeon.experiments.identity import canonical_identity_json

_MLP = re.compile(r"(?:^|\.)model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight$")


def _parse_model(value: str) -> tuple[str, str]:
    identifier, separator, revision = value.rpartition("@")
    if not separator or not identifier or not revision:
        raise argparse.ArgumentTypeError("models must use IDENTIFIER@REVISION")
    return identifier, revision


def _sample(parameter: object, count: int) -> tuple[float, ...]:
    import torch

    tensor = parameter.detach().float().reshape(-1).cpu()  # type: ignore[attr-defined]
    indexes = torch.linspace(0, tensor.numel() - 1, steps=count, dtype=torch.long)
    return tuple(float(value) for value in tensor[indexes].tolist())


def _digest(values: tuple[float, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack("<d", value))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", type=_parse_model, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--elements", type=int, default=4096)
    parser.add_argument("--surgery-stride", type=int, default=16)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    args = parser.parse_args()
    if args.elements <= 0 or args.elements % 256:
        raise ValueError("elements must be a positive multiple of 256")

    samples: list[QuantizedTensorSample] = []
    sample_manifest: list[dict[str, object]] = []
    models: list[dict[str, str]] = []
    for identifier, revision in args.model:
        loaded = load_causal_lm(
            HuggingFaceLoadRequest(
                identifier,
                revision=revision,
                device_map="cpu",
                dtype=HuggingFaceDType.FLOAT16,
                local_files_only=True,
            )
        )
        matched: dict[tuple[int, str], tuple[str, object]] = {}
        for name, parameter in loaded.model.named_parameters():
            match = _MLP.search(name)
            if match is not None:
                matched[(int(match.group(1)), match.group(2))] = (name, parameter)
        layers = sorted({layer for layer, _ in matched})
        selected_layers = tuple(sorted({layers[0], layers[len(layers) // 2], layers[-1]}))
        for layer in selected_layers:
            for projection in ("gate", "up", "down"):
                name, parameter = matched[(layer, projection)]
                values = _sample(parameter, args.elements)
                samples.append(QuantizedTensorSample(identifier, name, values))
                sample_manifest.append(
                    {
                        "model": identifier,
                        "tensor": name,
                        "elements": len(values),
                        "float64_sample_sha256": _digest(values),
                    }
                )
        models.append(
            {
                "identifier": identifier,
                "requested_revision": revision,
                "resolved_revision": loaded.provenance.resolved_revision,
            }
        )
        del loaded
        gc.collect()

    result = run_quantization_loss_study(
        tuple(samples),
        surgery_stride=args.surgery_stride,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    record: dict[str, object] = {
        "record_type": "v0.8_quantization_loss_study",
        "version": "1",
        "protocol": {
            "codecs": ["Q4_K", "Q5_K", "Q6_K", "Q8_0"],
            "elements_per_tensor": args.elements,
            "surgery": "zero_every_nth_aligned_weight",
            "surgery_stride": args.surgery_stride,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "quality_metric": "mean_squared_weight_error",
            "effect_model": "combined_minus_requantization_minus_surgery",
        },
        "models": models,
        "samples": sample_manifest,
        "result": result.to_record(),
    }
    args.output.write_text(canonical_identity_json(record) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
