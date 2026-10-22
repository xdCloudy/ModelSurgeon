"""Prepare leakage-safe surgeon examples from retained physical observations.

This utility deliberately accepts only candidate observations with measured
baseline and post-mutation perplexity. It does not synthesize labels or turn
planning records into training data. Model identities are kept as split groups
so a held-out model cannot contribute rows to training or validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from modelsurgeon.datasets.grouped_splits import (
    GroupedSplitManifest,
    GroupedSplitMode,
    SplitGroup,
    SplitPartition,
    SplitRatios,
)


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
            raise ValueError(f"transfer evidence root does not exist: {root}")
        if not root.is_dir():
            raise ValueError(f"transfer evidence root is not a directory: {root}")
        for path in sorted(root.rglob("optimization-evidence/observations/*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"cannot read transfer evidence observation {path}: {error}"
                ) from error
            if not isinstance(payload, dict):
                raise ValueError(f"transfer evidence observation must be a JSON object: {path}")
            if payload.get("stage") == "active_search":
                observations.append(payload)
    return observations


def _select_examples(observations: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    source_by_id: dict[str, dict[str, Any]] = {}
    for observation in observations:
        observation_id = observation.get("observation_id")
        if not isinstance(observation_id, str) or not observation_id:
            continue
        previous = source_by_id.get(observation_id)
        if previous is not None:
            if previous != observation:
                raise ValueError(
                    f"conflicting duplicate transfer observation id: {observation_id}"
                )
            continue
        source_by_id[observation_id] = observation
        example = _observation_to_example(observation)
        if example is None:
            continue
        example_id = example["example_id"]
        if not isinstance(example_id, str) or not example_id:
            continue
        selected[example_id] = example
    return selected


def _file_integrity(path: Path) -> dict[str, int | str]:
    hasher = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            hasher.update(chunk)
            size_bytes += len(chunk)
    return {"sha256": f"sha256:{hasher.hexdigest()}", "size_bytes": size_bytes}


def _publish_dataset(
    output: Path,
    selected: dict[str, dict[str, Any]],
    split_record: dict[str, Any],
    manifest_record: dict[str, Any],
) -> dict[str, Any]:
    output = output.absolute()
    if output == output.parent:
        raise ValueError("transfer dataset output must be a dedicated directory")
    if output.exists() or output.is_symlink():
        raise ValueError(f"transfer dataset output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.partial-{uuid.uuid4().hex}")
    staging.mkdir()
    try:
        with (staging / "examples.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for example_id in sorted(selected):
                handle.write(json.dumps(selected[example_id], sort_keys=True) + "\n")
        (staging / "split.json").write_text(
            json.dumps(split_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        published_manifest = dict(manifest_record)
        published_manifest["artifacts"] = {
            "examples.jsonl": _file_integrity(staging / "examples.jsonl"),
            "split.json": _file_integrity(staging / "split.json"),
        }
        (staging / "manifest.json").write_text(
            json.dumps(published_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output.exists() or output.is_symlink():
            raise ValueError(f"transfer dataset output already exists: {output}")
        os.rename(staging, output)
        return published_manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-model", action="append", default=[])
    parser.add_argument("--validation-model", action="append", default=[])
    args = parser.parse_args()

    try:
        selected = _select_examples(_iter_observations(args.root))
    except ValueError as error:
        raise SystemExit(str(error)) from error
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

    split_counts: Counter[str] = Counter()
    model_partitions: dict[str, str] = {}
    split_groups: list[SplitGroup] = []
    for model_key in sorted(grouped):
        if model_key in test_models:
            partition = SplitPartition.TEST
        elif model_key in validation_models:
            partition = SplitPartition.VALIDATION
        else:
            partition = SplitPartition.TRAIN
        model_partitions[model_key] = partition.value
        example_ids = tuple(sorted(str(example["example_id"]) for example in grouped[model_key]))
        group_key = f"model_identity:{model_key}"
        group_digest = hashlib.sha256(model_key.encode("utf-8")).hexdigest()
        split_groups.append(
            SplitGroup(
                f"group_{group_digest}",
                partition,
                (group_key,),
                example_ids,
            )
        )
        split_counts[partition.value] += len(example_ids)

    missing_partitions = [
        partition.value
        for partition in SplitPartition
        if split_counts[partition.value] == 0
    ]
    if missing_partitions:
        raise SystemExit(
            "model-identity transfer split must populate train, validation, and test; "
            f"missing {missing_partitions}"
        )
    total_examples = sum(split_counts.values())
    split_ratios = SplitRatios(
        split_counts[SplitPartition.TRAIN.value] / total_examples,
        split_counts[SplitPartition.VALIDATION.value] / total_examples,
        split_counts[SplitPartition.TEST.value] / total_examples,
    )
    split_manifest = GroupedSplitManifest(
        GroupedSplitMode.MODEL_IDENTITY,
        0,
        split_ratios,
        tuple(sorted(split_groups, key=lambda group: group.group_id)),
    )

    output = args.output
    split_record = split_manifest.to_record()
    manifest_record = {
        "record_type": "real_surgeon_transfer_dataset",
        "schema_version": 2,
        "source_roots": [str(root) for root in args.root],
        "example_count": len(selected),
        "split_counts": dict(sorted(split_counts.items())),
        "model_counts": {
            model: len(grouped[model]) for model in sorted(grouped)
        },
        "model_partitions": model_partitions,
        "test_models": sorted(test_models),
        "validation_models": sorted(validation_models),
        "dataset": str(output / "examples.jsonl"),
        "split": str(output / "split.json"),
    }
    try:
        manifest_record = _publish_dataset(output, selected, split_record, manifest_record)
    except (OSError, ValueError) as error:
        raise SystemExit(f"cannot publish transfer dataset: {error}") from error
    print(json.dumps(manifest_record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
