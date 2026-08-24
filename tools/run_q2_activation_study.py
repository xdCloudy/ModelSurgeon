"""Run the paired v0.8 Q2 activation-feature ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from modelsurgeon.cli.proof_evidence import load_grouped_proof_split
from modelsurgeon.cli.surgeon import load_surgeon_records
from modelsurgeon.evaluation.activation_feature_study import (
    run_activation_feature_ablation,
)
from modelsurgeon.evaluation.static_feature_study import StaticFeatureStudyConfig
from modelsurgeon.experiments.identity import canonical_identity_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_cost(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"cost artifact must be an object: {path}")
    return cast(Mapping[str, object], value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", type=Path, required=True)
    parser.add_argument("--cost", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safe-perplexity-delta", type=float, default=0.01)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    args = parser.parse_args()
    if len(args.dataset) != len(args.cost):
        raise ValueError("Q2 requires one feature-cost artifact per dataset")

    costs: dict[str, tuple[Mapping[str, object], Path]] = {}
    for path in args.cost:
        record = _load_cost(path)
        model = record.get("model")
        if not isinstance(model, Mapping) or not isinstance(model.get("identifier"), str):
            raise ValueError(f"cost artifact lacks model identity: {path}")
        costs[str(model["identifier"])] = (record, path)

    results: list[dict[str, object]] = []
    sources: list[dict[str, str]] = []
    identifiers: set[str] = set()
    config = StaticFeatureStudyConfig(
        safe_perplexity_delta=args.safe_perplexity_delta,
        top_n=args.top_n,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    for directory in args.dataset:
        examples = directory / "examples.jsonl"
        split = directory / "split.json"
        result = run_activation_feature_ablation(
            load_surgeon_records(examples),
            load_grouped_proof_split(split),
            config,
        )
        identifier = result.static.model_identifier
        cost_source = costs.get(identifier)
        if cost_source is None:
            raise ValueError(f"no collection-cost artifact for {identifier}")
        cost, cost_path = cost_source
        cost_model = cost.get("model")
        if (
            not isinstance(cost_model, Mapping)
            or cost_model.get("revision") != result.static.model_revision
        ):
            raise ValueError(f"collection-cost revision mismatch for {identifier}")
        item = result.to_record()
        item["feature_collection_cost"] = cost.get("costs")
        results.append(item)
        identifiers.add(identifier)
        sources.append(
            {
                "examples": f"{directory.name}/examples.jsonl",
                "examples_sha256": _sha256(examples),
                "split": f"{directory.name}/split.json",
                "split_sha256": _sha256(split),
                "feature_cost": cost_path.name,
                "feature_cost_sha256": _sha256(cost_path),
            }
        )
    if len(identifiers) < 3:
        raise ValueError("Q2 requires at least three distinct target models")
    record: dict[str, object] = {
        "record_type": "v0.8_q2_activation_feature_study",
        "version": "1",
        "protocol": {
            "comparison": "static_only_vs_static_activation",
            "safe_perplexity_delta": args.safe_perplexity_delta,
            "top_n": args.top_n,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_confidence": 0.95,
            "seed": 42,
            "split_seed": 43,
        },
        "sources": sources,
        "results": results,
    }
    args.output.write_text(canonical_identity_json(record) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
