#!/usr/bin/env python3
"""Measure leave-one-profile-out stability of aggregated REAP expert rankings.

This CPU-only analysis compares score aggregation rules. It does not prune a
model or measure candidate quality, inference performance, or resource use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    # Git may check JSON profile files out with CRLF on Windows. Hash the
    # canonical LF bytes so an identical committed profile has one identity.
    payload = path.read_bytes().replace(b"\r\n", b"\n")
    record = json.loads(payload)
    if not isinstance(record, dict):
        raise ValueError(f"unsupported POE profile: {path}")
    if record.get("poeprofile") != 0 or not isinstance(record.get("layers"), list):
        raise ValueError(f"unsupported POE profile: {path}")
    layers = record["layers"]
    layer_count = record.get("n_layers")
    expert_count = record.get("n_experts")
    if (
        isinstance(layer_count, bool)
        or not isinstance(layer_count, int)
        or layer_count < 1
        or len(layers) != layer_count
        or isinstance(expert_count, bool)
        or not isinstance(expert_count, int)
        or expert_count < 1
    ):
        raise ValueError(f"invalid profile dimensions: {path}")
    for layer_index, layer in enumerate(layers):
        scores = layer.get("reap_mean") if isinstance(layer, dict) else None
        if not isinstance(scores, list) or len(scores) != expert_count:
            raise ValueError(f"invalid REAP scores in layer {layer_index}: {path}")
        for score in scores:
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or float(score) < 0
            ):
                raise ValueError(f"invalid REAP score in layer {layer_index}: {path}")
    return {
        "record": record,
        "profile_sha256": hashlib.sha256(payload).hexdigest(),
        "name": path.name,
    }


def _validate_compatible(profiles: list[dict[str, Any]]) -> None:
    first = profiles[0]["record"]
    for profile in profiles[1:]:
        record = profile["record"]
        for key in ("model_fingerprint", "arch", "n_layers", "n_experts", "top_k"):
            if record.get(key) != first.get(key):
                raise ValueError(f"profile mismatch for {key}")


def _minmax(values: list[float]) -> list[float]:
    low = min(values)
    high = max(values)
    if high == low:
        return [0.5] * len(values)
    return [(value - low) / (high - low) for value in values]


def _average_percentile_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    start = 0
    denominator = max(1, len(values) - 1)
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        percentile = ((start + end) / 2) / denominator
        for offset in range(start, end + 1):
            result[order[offset]] = percentile
        start = end + 1
    return result


def _bottom_set(values: list[float], count: int) -> set[int]:
    # POE sorts high-to-low and breaks score ties by ascending expert ID;
    # pruning uses the final ``count`` entries of that order.
    return set(sorted(range(len(values)), key=lambda index: (-values[index], index))[-count:])


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _aggregate(profile_layers: list[list[float]], method: str) -> list[float]:
    if method in {"mean_minmax", "max_minmax"}:
        normalized = [_minmax(values) for values in profile_layers]
    elif method == "mean_rank":
        normalized = [_average_percentile_ranks(values) for values in profile_layers]
    else:
        raise ValueError(f"unsupported aggregation method: {method}")
    if method == "max_minmax":
        return [max(values) for values in zip(*normalized, strict=True)]
    return [sum(values) / len(values) for values in zip(*normalized, strict=True)]


def compare_aggregations(
    profiles: list[dict[str, Any]], prune_fraction: float = 0.125
) -> dict[str, Any]:
    if len(profiles) < 3:
        raise ValueError("at least three profiles are required for leave-one-out analysis")
    if not math.isfinite(prune_fraction) or not 0 < prune_fraction <= 1:
        raise ValueError("prune_fraction must be finite and in (0, 1]")
    _validate_compatible(profiles)
    first = profiles[0]["record"]
    layer_count = first["n_layers"]
    expert_count = first["n_experts"]
    prune_count = min(expert_count, max(1, math.floor(expert_count * prune_fraction + 0.5)))
    profile_scores = [
        [[float(value) for value in layer["reap_mean"]] for layer in profile["record"]["layers"]]
        for profile in profiles
    ]
    methods = ("mean_minmax", "max_minmax", "mean_rank")
    results: dict[str, Any] = {}
    for method in methods:
        folds: list[dict[str, Any]] = []
        for held_out_index, held_out_scores in enumerate(profile_scores):
            training_profiles = [
                scores for index, scores in enumerate(profile_scores) if index != held_out_index
            ]
            per_layer = [
                _jaccard(
                    _bottom_set(
                        _aggregate([profile[layer] for profile in training_profiles], method),
                        prune_count,
                    ),
                    _bottom_set(held_out_scores[layer], prune_count),
                )
                for layer in range(layer_count)
            ]
            folds.append(
                {
                    "held_out_profile": profiles[held_out_index]["name"],
                    "mean_bottom_set_jaccard": sum(per_layer) / len(per_layer),
                    "min_layer_jaccard": min(per_layer),
                    "layer_jaccards": per_layer,
                }
            )
        fold_means = [fold["mean_bottom_set_jaccard"] for fold in folds]
        results[method] = {
            "leave_one_profile_out_mean_jaccard": sum(fold_means) / len(fold_means),
            "worst_held_out_profile_mean_jaccard": min(fold_means),
            "folds": folds,
        }

    single_profile_folds: list[dict[str, Any]] = []
    pairwise: list[float] = []
    for held_out_index, held_out_scores in enumerate(profile_scores):
        training_profiles = [
            scores for index, scores in enumerate(profile_scores) if index != held_out_index
        ]
        fold_values = [
            _jaccard(
                _bottom_set(training_scores[layer], prune_count),
                _bottom_set(held_out_scores[layer], prune_count),
            )
            for training_scores in training_profiles
            for layer in range(layer_count)
        ]
        fold_mean = sum(fold_values) / len(fold_values)
        single_profile_folds.append(
            {
                "held_out_profile": profiles[held_out_index]["name"],
                "mean_training_profile_jaccard": fold_mean,
            }
        )
        pairwise.extend(fold_values)
    single_profile_fold_means = [
        fold["mean_training_profile_jaccard"] for fold in single_profile_folds
    ]
    return {
        "schema_version": 1,
        "analysis": "leave_one_profile_out_reap_bottom_set_stability",
        "poe_compare_source_revision": "3908873326bcdbe097f4184729db7c33fff1cf60",
        "model_fingerprint": first["model_fingerprint"],
        "architecture": first["arch"],
        "layers": layer_count,
        "experts_per_layer": expert_count,
        "prune_fraction": prune_fraction,
        "experts_pruned_per_layer": prune_count,
        "profiles": [
            {
                "name": profile["name"],
                "tokens": profile["record"].get("tokens"),
                "dataset_bytes": profile["record"].get("dataset_bytes"),
                "dataset_hash_poe_fnv1a": profile["record"].get("dataset_hash"),
                "profile_sha256": profile["profile_sha256"],
            }
            for profile in profiles
        ],
        "single_profile_pairwise_mean_jaccard": sum(pairwise) / len(pairwise),
        "single_profile_worst_held_out_mean_jaccard": min(single_profile_fold_means),
        "single_profile_folds": single_profile_folds,
        "aggregation_results": results,
        "scope": "ranking stability only; no candidate artifact, quality, or runtime comparison",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, action="append", required=True)
    parser.add_argument("--prune-fraction", type=float, default=0.125)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = compare_aggregations([_load(path) for path in args.profile], args.prune_fraction)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.write_text(serialized, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
