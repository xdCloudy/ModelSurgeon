"""Run the paired v0.8 Q3 gradient-feature ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from modelsurgeon.cli.proof_evidence import load_grouped_proof_split
from modelsurgeon.cli.surgeon import load_surgeon_records
from modelsurgeon.evaluation.gradient_feature_study import (
    run_gradient_feature_ablation,
)
from modelsurgeon.evaluation.static_feature_study import StaticFeatureStudyConfig
from modelsurgeon.experiments.identity import canonical_identity_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return cast(Mapping[str, object], value)


def _q2_costs(path: Path) -> dict[str, Mapping[str, object]]:
    report = _object(path)
    raw_results = report.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("Q2 report lacks results")
    output: dict[str, Mapping[str, object]] = {}
    for raw in raw_results:
        if not isinstance(raw, Mapping) or not isinstance(raw.get("model"), Mapping):
            raise ValueError("Q2 result lacks model identity")
        model = cast(Mapping[str, object], raw["model"])
        identifier = model.get("identifier")
        costs = raw.get("feature_collection_cost")
        if not isinstance(identifier, str) or not isinstance(costs, Mapping):
            raise ValueError("Q2 result lacks feature collection cost")
        output[identifier] = cast(Mapping[str, object], costs)
    return output


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    return float(value)


def _resource_comparison(
    gradient: Mapping[str, object], q2: Mapping[str, object]
) -> dict[str, object]:
    activation = q2.get("activation")
    if not isinstance(activation, Mapping):
        raise ValueError("Q2 cost lacks activation measurement")
    fields = (
        "wall_seconds",
        "incremental_peak_rss_bytes",
        "incremental_peak_cuda_allocated_bytes",
    )
    overhead = {
        field: _number(gradient.get(field), f"gradient {field}")
        - _number(activation.get(field), f"activation {field}")
        for field in fields
    }
    return {
        "activation": dict(activation),
        "gradient": dict(gradient),
        "gradient_minus_activation": overhead,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", type=Path, required=True)
    parser.add_argument("--q2-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safe-perplexity-delta", type=float, default=0.01)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    args = parser.parse_args()

    q2_costs = _q2_costs(args.q2_report)
    config = StaticFeatureStudyConfig(
        safe_perplexity_delta=args.safe_perplexity_delta,
        top_n=args.top_n,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    results: list[dict[str, object]] = []
    sources: list[dict[str, str]] = []
    identifiers: set[str] = set()
    for directory in args.dataset:
        examples = directory / "examples.jsonl"
        split = directory / "split.json"
        manifest_path = directory / "gradient-manifest.json"
        manifest = _object(manifest_path)
        result = run_gradient_feature_ablation(
            load_surgeon_records(examples),
            load_grouped_proof_split(split),
            config,
        )
        identifier = result.static_activation.model_identifier
        q2_cost = q2_costs.get(identifier)
        collection = manifest.get("collection")
        if q2_cost is None or not isinstance(collection, Mapping):
            raise ValueError(f"resource evidence is missing for {identifier}")
        gradient_cost = collection.get("cost")
        if not isinstance(gradient_cost, Mapping):
            raise ValueError(f"gradient cost is missing for {identifier}")
        item = result.to_record()
        item["feature_collection_cost"] = _resource_comparison(
            cast(Mapping[str, object], gradient_cost), q2_cost
        )
        results.append(item)
        identifiers.add(identifier)
        sources.append(
            {
                "examples": f"{directory.name}/examples.jsonl",
                "examples_sha256": _sha256(examples),
                "split": f"{directory.name}/split.json",
                "split_sha256": _sha256(split),
                "gradient_manifest": f"{directory.name}/gradient-manifest.json",
                "gradient_manifest_sha256": _sha256(manifest_path),
            }
        )
    if len(identifiers) < 3:
        raise ValueError("Q3 requires at least three distinct target models")
    record: dict[str, object] = {
        "record_type": "v0.8_q3_gradient_feature_study",
        "version": "1",
        "protocol": {
            "comparison": "static_activation_vs_static_activation_gradient",
            "safe_perplexity_delta": args.safe_perplexity_delta,
            "top_n": args.top_n,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_confidence": 0.95,
            "seed": 42,
            "split_seed": 43,
            "gradient_batches": 1,
            "gradient_tokens": 63,
        },
        "q2_report_sha256": _sha256(args.q2_report),
        "sources": sources,
        "results": results,
    }
    args.output.write_text(canonical_identity_json(record) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
