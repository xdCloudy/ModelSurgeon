#!/usr/bin/env python3
"""Compare expert-score stability across two POE calibration profiles.

Implements the reported REAP bottom-set Jaccard and average-tie Spearman
statistics for POE ``.poeprofile`` JSON files. Score ranking follows POE's
deterministic expert-ID tie break. This analysis does not compare model quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, dataset: Path) -> dict[str, Any]:
    profile_bytes = path.read_bytes()
    record = json.loads(profile_bytes)
    if record.get("poeprofile") != 0 or not isinstance(record.get("layers"), list):
        raise ValueError(f"unsupported POE profile: {path}")
    if len(record["layers"]) != record.get("n_layers"):
        raise ValueError(f"inconsistent layer count in {path}")
    if dataset.stat().st_size != record.get("dataset_bytes"):
        raise ValueError(f"dataset byte count does not match profile {path}")
    return {
        "record": record,
        "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
        "dataset_sha256": _sha256(dataset),
        "dataset_bytes": dataset.stat().st_size,
    }


def _average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: (-values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start
        while end + 1 < len(ordered) and values[ordered[end + 1]] == values[ordered[start]]:
            end += 1
        rank = (start + end) / 2
        for offset in range(start, end + 1):
            ranks[ordered[offset]] = rank
        start = end + 1
    return ranks


def _spearman(left: list[float], right: list[float]) -> float:
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    covariance = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left_ranks, right_ranks, strict=True)
    )
    left_ss = sum((value - left_mean) ** 2 for value in left_ranks)
    right_ss = sum((value - right_mean) ** 2 for value in right_ranks)
    denominator = math.sqrt(left_ss * right_ss)
    return covariance / denominator if denominator else 0.0


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _ranked_set(values: list[float], count: int, *, bottom: bool) -> set[int]:
    # Match POE's descending score, ascending expert-ID tie break. Its bottom
    # set is the last ``count`` entries in that order.
    ordered = sorted(range(len(values)), key=lambda index: (-values[index], index))
    chosen = ordered[-count:] if bottom else ordered[:count]
    return set(chosen)


def compare(
    first: dict[str, Any], second: dict[str, Any], fraction: float
) -> dict[str, Any]:
    a = first["record"]
    b = second["record"]
    for key in ("model_fingerprint", "arch", "n_layers", "n_experts", "top_k"):
        if a.get(key) != b.get(key):
            raise ValueError(f"profile mismatch for {key}")
    if a["n_layers"] <= 0 or a["n_experts"] <= 0:
        raise ValueError("profile dimensions must be positive")
    layers_a = a["layers"]
    layers_b = b["layers"]
    expert_count = a["n_experts"]
    top_count = min(expert_count, max(1, math.floor(expert_count * fraction + 0.5)))
    reap_jaccards: list[float] = []
    route_jaccards: list[float] = []
    spearmans: list[float] = []
    for index, (layer_a, layer_b) in enumerate(zip(layers_a, layers_b, strict=True)):
        reap_a = layer_a["reap_mean"]
        reap_b = layer_b["reap_mean"]
        route_a = layer_a["sel_count"]
        route_b = layer_b["sel_count"]
        if any(len(values) != expert_count for values in (reap_a, reap_b, route_a, route_b)):
            raise ValueError(f"expert score/count length mismatch in layer {index}")
        reap_jaccards.append(
            _jaccard(
                _ranked_set(reap_a, top_count, bottom=True),
                _ranked_set(reap_b, top_count, bottom=True),
            )
        )
        route_jaccards.append(
            _jaccard(
                _ranked_set(route_a, top_count, bottom=False),
                _ranked_set(route_b, top_count, bottom=False),
            )
        )
        spearmans.append(_spearman(reap_a, reap_b))
    return {
        "method": "per-layer REAP bottom-set Jaccard and average-tie Spearman",
        "poe_compare_source_revision": "3908873326bcdbe097f4184729db7c33fff1cf60",
        "model_fingerprint": a["model_fingerprint"],
        "architecture": a["arch"],
        "layers": a["n_layers"],
        "experts_per_layer": expert_count,
        "prune_fraction": fraction,
        "experts_pruned_per_layer": top_count,
        "calibrations": [
            {
                "tokens": a["tokens"],
                "dataset_hash_poe_fnv1a": a["dataset_hash"],
                "dataset_bytes": first["dataset_bytes"],
                "dataset_sha256": first["dataset_sha256"],
                "profile_sha256": first["profile_sha256"],
            },
            {
                "tokens": b["tokens"],
                "dataset_hash_poe_fnv1a": b["dataset_hash"],
                "dataset_bytes": second["dataset_bytes"],
                "dataset_sha256": second["dataset_sha256"],
                "profile_sha256": second["profile_sha256"],
            },
        ],
        "reap_prune_set_jaccard_mean": sum(reap_jaccards) / len(reap_jaccards),
        "reap_prune_set_jaccard_min": min(reap_jaccards),
        "reap_prune_set_jaccard_max": max(reap_jaccards),
        "reap_rank_spearman_mean": sum(spearmans) / len(spearmans),
        "router_frequency_top_set_jaccard_mean": sum(route_jaccards) / len(route_jaccards),
        "layers_with_zero_reap_prune_set_overlap": sum(value == 0 for value in reap_jaccards),
        "layer_reap_prune_set_jaccards": reap_jaccards,
        "scope": "ranking stability only; no candidate artifact, quality, or runtime comparison",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-a", type=Path, required=True)
    parser.add_argument("--dataset-a", type=Path, required=True)
    parser.add_argument("--profile-b", type=Path, required=True)
    parser.add_argument("--dataset-b", type=Path, required=True)
    parser.add_argument("--prune-fraction", type=float, default=0.125)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 < args.prune_fraction <= 1:
        parser.error("prune-fraction must be in (0, 1]")
    result = compare(
        _load(args.profile_a, args.dataset_a),
        _load(args.profile_b, args.dataset_b),
        args.prune_fraction,
    )
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(serialized, end="")
    else:
        args.output.write_text(serialized, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
