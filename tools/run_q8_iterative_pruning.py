"""Run the matched v0.8 Q8 iterative-versus-one-shot cumulative-mask study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from modelsurgeon.adapters.huggingface.loader import HuggingFaceDType
from modelsurgeon.adapters.huggingface.proof_runtime import (
    HuggingFaceMLPProofConfig,
    HuggingFaceMLPProofRuntime,
)
from modelsurgeon.cli.proof_evidence import load_grouped_proof_split
from modelsurgeon.cli.surgeon import load_surgeon_records
from modelsurgeon.evaluation.matched_pruning_study import run_matched_pruning_selection
from modelsurgeon.evaluation.static_feature_study import StaticFeatureStudyConfig
from modelsurgeon.experiments.identity import canonical_identity_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--calibration-text", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--safe-perplexity-delta", type=float, default=0.01)
    parser.add_argument("--latency-repetitions", type=int, default=7)
    args = parser.parse_args()

    examples = args.dataset / "examples.jsonl"
    split = args.dataset / "split.json"
    manifest = args.dataset / "gradient-manifest.json"
    selections = run_matched_pruning_selection(
        load_surgeon_records(examples),
        load_grouped_proof_split(split),
        budget=args.budget,
        config=StaticFeatureStudyConfig(
            safe_perplexity_delta=args.safe_perplexity_delta,
            top_n=args.budget,
        ),
    )
    runtime = HuggingFaceMLPProofRuntime(
        HuggingFaceMLPProofConfig(
            model=args.model,
            revision=args.revision,
            calibration_text=args.calibration_text,
            device_map="auto",
            dtype=HuggingFaceDType.FLOAT16,
            local_files_only=True,
            sequence_length=64,
            max_tokens=64,
            seed=42,
            tool_revision="q8-iterative-pruning-v1",
        )
    )
    one_shot_coordinates = tuple(
        (item.layer_index, item.channel_index) for item in selections.one_shot
    )
    iterative_coordinates = tuple(
        (item.layer_index, item.channel_index) for item in selections.iterative
    )
    one_shot = runtime.measure_channel_set(
        one_shot_coordinates,
        warmup=1,
        repetitions=args.latency_repetitions,
    )
    iterative_path = []
    for count in range(1, len(iterative_coordinates) + 1):
        final = count == len(iterative_coordinates)
        measurement = runtime.measure_channel_set(
            iterative_coordinates[:count],
            warmup=1 if final else 0,
            repetitions=args.latency_repetitions if final else 1,
        )
        iterative_path.append(measurement)
    iterative = iterative_path[-1]
    record: dict[str, object] = {
        "record_type": "v0.8_q8_iterative_pruning_study",
        "version": "1",
        "protocol": {
            "compression_kind": "cumulative_structured_mlp_channel_mask",
            "matched_channel_budget": args.budget,
            "safe_perplexity_delta": args.safe_perplexity_delta,
            "feature_profile": "static_activation_gradient",
            "iterative_policy": "reveal_single_channel_label_then_retrain",
            "latency_warmup": 1,
            "latency_repetitions": args.latency_repetitions,
            "quantization": "none_float16_high_precision_mask",
        },
        "source": {
            "examples_sha256": _sha256(examples),
            "split_sha256": _sha256(split),
            "gradient_manifest_sha256": _sha256(manifest),
            "calibration_text_sha256": _sha256(args.calibration_text),
        },
        "selections": selections.to_record(),
        "one_shot": {
            "measurement": one_shot.to_record(),
            "total_study_seconds": (
                selections.one_shot_training_seconds + one_shot.measurement_wall_seconds
            ),
            "gpu_hours": one_shot.measurement_wall_seconds / 3600.0,
        },
        "iterative": {
            "path": [item.to_record() for item in iterative_path],
            "measurement": iterative.to_record(),
            "total_study_seconds": (
                selections.iterative_training_seconds
                + selections.iterative_revealed_evaluation_seconds
                + sum(item.measurement_wall_seconds for item in iterative_path)
            ),
            "gpu_hours": (
                selections.iterative_revealed_evaluation_seconds
                + sum(item.measurement_wall_seconds for item in iterative_path)
            )
            / 3600.0,
        },
    }
    args.output.write_text(canonical_identity_json(record) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
