"""Augment one persisted Q1/Q2 dataset with real MLP-channel gradient features."""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from modelsurgeon.adapters.huggingface.loader import HuggingFaceDType
from modelsurgeon.adapters.huggingface.proof_runtime import (
    HuggingFaceMLPProofConfig,
    HuggingFaceMLPProofRuntime,
)
from modelsurgeon.cli.surgeon import load_surgeon_records
from modelsurgeon.experiments.hardware import collect_hardware_inventory
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.graph import ComponentId


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _coordinate(record: Mapping[str, object]) -> tuple[ComponentId, int, int]:
    components = record.get("components")
    mutation = record.get("mutation")
    if (
        not isinstance(components, list)
        or len(components) != 1
        or not isinstance(mutation, Mapping)
    ):
        raise ValueError("gradient augmentation requires one-component mutations")
    plan = mutation.get("plan")
    if not isinstance(plan, Mapping) or not isinstance(plan.get("request"), Mapping):
        raise ValueError("gradient augmentation requires a mutation request")
    parameters = cast(Mapping[str, object], plan["request"]).get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("gradient augmentation requires mutation coordinates")
    layer = parameters.get("layer_index")
    channel = parameters.get("channel_index")
    if not isinstance(layer, int) or not isinstance(channel, int):
        raise ValueError("gradient augmentation coordinates must be integers")
    return ComponentId.parse(str(components[0])), layer, channel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--calibration-text", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    examples_path = args.dataset / "examples.jsonl"
    split_path = args.dataset / "split.json"
    records = load_surgeon_records(examples_path)
    coordinates = tuple(_coordinate(record) for record in records)
    runtime = HuggingFaceMLPProofRuntime(
        HuggingFaceMLPProofConfig(
            model=args.model,
            revision=args.revision,
            calibration_text=args.calibration_text,
            device_map="auto",
            dtype=HuggingFaceDType.FLOAT16,
            sequence_length=args.max_tokens,
            max_tokens=args.max_tokens,
            seed=42,
            tool_revision="q3-gradient-feature-v1",
        )
    )
    collection = runtime.collect_mlp_channel_gradient_features(coordinates)
    by_component = {
        str(component): [feature.to_record() for feature in features]
        for component, features in collection.records
    }

    args.output.mkdir(parents=True, exist_ok=False)
    output_examples = args.output / "examples.jsonl"
    with output_examples.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            record_components = record.get("components")
            if not isinstance(record_components, list) or not record_components:
                raise ValueError("gradient feature join requires components")
            component = str(record_components[0])
            features = by_component.get(component)
            raw = record.get("pre_mutation_features")
            if features is None or not isinstance(raw, list):
                raise ValueError(f"gradient feature join failed for {component}")
            augmented = dict(record)
            augmented["pre_mutation_features"] = [*raw, *features]
            stream.write(canonical_identity_json(augmented) + "\n")
    output_split = args.output / "split.json"
    output_split.write_bytes(split_path.read_bytes())
    manifest: dict[str, object] = {
        "record_type": "v0.8_q3_gradient_feature_dataset",
        "version": "1",
        "model": {"identifier": args.model, "revision": args.revision},
        "source": {
            "examples": f"{args.dataset.name}/examples.jsonl",
            "examples_sha256": _sha256(examples_path),
            "split_sha256": _sha256(split_path),
        },
        "output": {
            "examples_sha256": _sha256(output_examples),
            "split_sha256": _sha256(output_split),
        },
        "collection": collection.to_record(),
        "hardware": collect_hardware_inventory(args.calibration_text.parent).to_record(),
    }
    manifest_path = args.output / "gradient-manifest.json"
    manifest_path.write_text(canonical_identity_json(manifest) + "\n", encoding="utf-8")
    print(canonical_identity_json(manifest))


if __name__ == "__main__":
    main()
