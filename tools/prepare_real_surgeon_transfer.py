"""Prepare leakage-safe surgeon examples from retained physical observations.

This utility deliberately accepts only candidate observations with measured
baseline and post-mutation perplexity. It does not synthesize labels or turn
planning records into training data. Model identities are kept as split groups
so a held-out model cannot contribute rows to training or validation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if result == result and abs(result) != float("inf") else None


def _model_key(model: dict[str, Any]) -> str:
    return f"{model['identifier']}@{model['revision']}"


def _observation_to_example(observation: dict[str, Any]) -> dict[str, Any] | None:
    measurement = observation.get("measurement")
    model = observation.get("model")
    features = observation.get("features")
    lineage = observation.get("lineage")
    if not isinstance(measurement, dict) or not isinstance(model, dict):
        return None
    if not isinstance(features, list) or not features or not isinstance(lineage, dict):
        return None
    baseline = _number(measurement.get("baseline_perplexity"))
    candidate = _number(measurement.get("candidate_perplexity"))
    if baseline is None or candidate is None:
        return None
    if not all(
        isinstance(model.get(name), str) and model[name]
        for name in ("identifier", "revision", "family", "format")
    ):
        return None
    mutation = lineage.get("mutation")
    if not isinstance(mutation, dict):
        return None
    targets = mutation.get("targets")
    if not isinstance(targets, list) or not targets or not all(
        isinstance(item, str) and item for item in targets
    ):
        return None
    parameters = mutation.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {}
    scope = parameters.get("candidate_scope")
    if not isinstance(scope, str) or not scope:
        scope = str(measurement.get("scope") or observation.get("stage") or "unknown")
    model_record = {
        "identifier": model["identifier"],
        "revision": model["revision"],
        "family": model["family"],
        "format": model["format"],
        "parameter_count": model.get("parameter_count"),
        "quantization": model.get("quantization"),
    }
    return {
        "schema_version": 1,
        "example_id": observation.get("observation_id"),
        "experiment_id": observation.get("run_id") or observation.get("observation_id"),
        "model": model_record,
        "components": sorted(set(targets)),
        "mutation": {
            "plan": {
                "request": {
                    "kind": str(mutation.get("kind") or "mask"),
                    "targets": sorted(set(targets)),
                    "parameters": {"candidate_scope": scope},
                }
            }
        },
        "pre_mutation_features": features,
        "baseline_metrics": [
            {"name": "perplexity", "state": "measured", "value": baseline, "unit": "perplexity"}
        ],
        "post_metrics": [
            {"name": "perplexity", "state": "measured", "value": candidate, "unit": "perplexity"}
        ],
        "versions": {"feature_schema_version": 1},
        "source_observation": {
            "observation_id": observation.get("observation_id"),
            "run_id": observation.get("run_id"),
            "stage": observation.get("stage"),
            "outcome": observation.get("outcome"),
            "measurement_authority": measurement.get("measurement_authority"),
        },
    }


def _iter_observations(roots: list[Path]) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("optimization-evidence/observations/*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and payload.get("stage") == "active_search":
                observations.append(payload)
    return observations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-model", action="append", default=[])
    parser.add_argument("--validation-model", action="append", default=[])
    args = parser.parse_args()

    selected: dict[str, dict[str, Any]] = {}
    for observation in _iter_observations(args.root):
        example = _observation_to_example(observation)
        if example is None or not isinstance(example.get("example_id"), str):
            continue
        selected[example["example_id"]] = example
    if not selected:
        raise SystemExit("no measured active-search observations found")

    test_models = set(args.test_model)
    validation_models = set(args.validation_model)
    overlap = test_models & validation_models
    if overlap:
        raise SystemExit(f"model identities cannot be in both held-out sets: {sorted(overlap)}")

    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in selected.values():
        grouped[_model_key(example["model"])].append(example)
    unknown = (test_models | validation_models) - set(grouped)
    if unknown:
        raise SystemExit(f"requested split model identities were not observed: {sorted(unknown)}")

    assignments: dict[str, str] = {}
    split_counts: Counter[str] = Counter()
    model_partitions: dict[str, str] = {}
    for model_key in sorted(grouped):
        if model_key in test_models:
            partition = "test"
        elif model_key in validation_models:
            partition = "validation"
        else:
            partition = "train"
        model_partitions[model_key] = partition
        for example in grouped[model_key]:
            example_id = str(example["example_id"])
            assignments[example_id] = partition
            split_counts[partition] += 1

    args.output.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output / "examples.jsonl"
    split_path = args.output / "split.json"
    manifest_path = args.output / "manifest.json"
    with dataset_path.open("w", encoding="utf-8", newline="\n") as handle:
        for example_id in sorted(selected):
            handle.write(json.dumps(selected[example_id], sort_keys=True) + "\n")
    split_path.write_text(
        json.dumps(
            {
                "version": "model-identity-transfer-v1",
                "mode": "model_identity",
                "assignments": assignments,
                "model_partitions": model_partitions,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "record_type": "real_surgeon_transfer_dataset",
                "schema_version": 1,
                "source_roots": [str(root) for root in args.root],
                "example_count": len(selected),
                "split_counts": dict(sorted(split_counts.items())),
                "model_counts": {
                    model: len(grouped[model]) for model in sorted(grouped)
                },
                "model_partitions": model_partitions,
                "test_models": sorted(test_models),
                "validation_models": sorted(validation_models),
                "dataset": str(dataset_path),
                "split": str(split_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json.loads(manifest_path.read_text(encoding="utf-8")), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
