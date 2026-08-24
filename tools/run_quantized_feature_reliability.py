"""Run v0.8 quantized-feature reliability on sampled real model tensors."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import struct

from modelsurgeon.adapters.huggingface.loader import (
    HuggingFaceDType,
    HuggingFaceLoadRequest,
    load_causal_lm,
)
from modelsurgeon.evaluation.quantized_feature_reliability import (
    QuantizedTensorSample,
    run_quantized_feature_reliability,
)
from modelsurgeon.experiments.identity import canonical_identity_json

_MLP = re.compile(r"(?:^|\.)model\.layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight$")


def _parse_model(value: str) -> tuple[str, str]:
    identifier, separator, revision = value.rpartition("@")
    if not separator or not identifier or not revision:
        raise argparse.ArgumentTypeError("models must use IDENTIFIER@REVISION")
    return identifier, revision


def _sample_values(parameter: object, count: int) -> tuple[float, ...]:
    import torch

    tensor = parameter.detach().float().reshape(-1).cpu()  # type: ignore[attr-defined]
    if tensor.numel() < count:
        raise ValueError("sampled tensor is smaller than the requested study sample")
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--elements", type=int, default=4096)
    args = parser.parse_args()
    if args.elements <= 0 or args.elements % 256:
        raise ValueError("elements must be a positive multiple of 256")

    samples: list[QuantizedTensorSample] = []
    sample_manifest: list[dict[str, object]] = []
    model_manifest: list[dict[str, str]] = []
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
        if not layers:
            raise ValueError(f"no gated MLP tensors found for {identifier}")
        selected_layers = tuple(sorted({layers[0], layers[len(layers) // 2], layers[-1]}))
        for layer in selected_layers:
            for projection in ("gate", "up", "down"):
                name, parameter = matched[(layer, projection)]
                values = _sample_values(parameter, args.elements)
                samples.append(QuantizedTensorSample(identifier, name, values))
                sample_manifest.append(
                    {
                        "model": identifier,
                        "tensor": name,
                        "elements": len(values),
                        "float64_sample_sha256": _digest(values),
                    }
                )
        model_manifest.append(
            {
                "identifier": identifier,
                "requested_revision": revision,
                "resolved_revision": loaded.provenance.resolved_revision,
            }
        )
        del loaded
        gc.collect()

    result = run_quantized_feature_reliability(tuple(samples))
    record: dict[str, object] = {
        "record_type": "v0.8_quantized_feature_reliability_study",
        "version": "1",
        "protocol": {
            "codecs": ["Q4_K", "Q5_K", "Q6_K", "Q8_0"],
            "elements_per_tensor": args.elements,
            "layers_per_model": 3,
            "projections_per_layer": 3,
            "sampling": "deterministic_evenly_spaced_flattened_weights",
            "direct_feature": "primary_scale_abs_mean",
            "decoded_features": [
                "weight_mean",
                "weight_variance",
                "weight_abs_mean",
                "weight_rms",
                "sparsity",
            ],
        },
        "models": model_manifest,
        "samples": sample_manifest,
        "result": result.to_record(),
    }
    with open(args.output, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_identity_json(record) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
