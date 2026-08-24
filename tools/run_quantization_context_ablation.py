"""Run the v0.8 source quantization and hardware context ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from modelsurgeon.cli.proof_evidence import load_grouped_proof_split
from modelsurgeon.cli.surgeon import load_surgeon_records
from modelsurgeon.evaluation.quantization_context_study import (
    run_quantization_context_ablation,
)
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
    parser.add_argument("--dataset", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safe-perplexity-delta", type=float, default=0.01)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    args = parser.parse_args()

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
        manifest = directory / "gradient-manifest.json"
        result = run_quantization_context_ablation(
            load_surgeon_records(examples),
            load_grouped_proof_split(split),
            config,
        )
        results.append(result.to_record())
        identifiers.add(result.context_aware.model_identifier)
        sources.append(
            {
                "dataset": directory.name,
                "examples_sha256": _sha256(examples),
                "split_sha256": _sha256(split),
                "gradient_manifest_sha256": _sha256(manifest),
            }
        )
    if len(identifiers) < 3:
        raise ValueError("context ablation requires at least three models")
    record: dict[str, object] = {
        "record_type": "v0.8_quantization_context_ablation",
        "version": "1",
        "protocol": {
            "comparison": "context_blind_vs_context_aware",
            "context": [
                "codec",
                "bits_per_weight",
                "feature_error",
                "feature_source_precision",
                "model_parameter_count",
                "cpu_logical_cores",
                "system_memory_bytes",
                "accelerator",
                "accelerator_memory_bytes",
                "operating_system",
            ],
            "feature_profile": "static_activation_gradient",
            "safe_perplexity_delta": args.safe_perplexity_delta,
            "top_n": args.top_n,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "seed": 42,
        },
        "sources": sources,
        "results": results,
    }
    args.output.write_text(canonical_identity_json(record) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
