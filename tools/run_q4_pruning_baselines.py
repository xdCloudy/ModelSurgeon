"""Run the equal-budget v0.8 Q4 learned/magnitude/random comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from modelsurgeon.cli.proof_evidence import load_grouped_proof_split
from modelsurgeon.cli.surgeon import load_surgeon_records
from modelsurgeon.evaluation.pruning_baseline_study import (
    DEFAULT_Q4_SEEDS,
    PruningBaselineStudyConfig,
    run_pruning_baseline_study,
)
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
    parser.add_argument("--selection-budget", type=int, default=10)
    parser.add_argument("--safe-perplexity-delta", type=float, default=0.01)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    args = parser.parse_args()

    config = PruningBaselineStudyConfig(
        selection_budget=args.selection_budget,
        seeds=DEFAULT_Q4_SEEDS,
        bootstrap_repetitions=args.bootstrap_repetitions,
        safe_perplexity_delta=args.safe_perplexity_delta,
    )
    results: list[dict[str, object]] = []
    sources: list[dict[str, str]] = []
    identifiers: set[str] = set()
    for directory in args.dataset:
        examples = directory / "examples.jsonl"
        split = directory / "split.json"
        manifest = directory / "gradient-manifest.json"
        result = run_pruning_baseline_study(
            load_surgeon_records(examples),
            load_grouped_proof_split(split),
            config,
        )
        results.append(result.to_record())
        identifiers.add(result.model_identifier)
        sources.append(
            {
                "examples": f"{directory.name}/examples.jsonl",
                "examples_sha256": _sha256(examples),
                "split": f"{directory.name}/split.json",
                "split_sha256": _sha256(split),
                "gradient_manifest": f"{directory.name}/gradient-manifest.json",
                "gradient_manifest_sha256": _sha256(manifest),
            }
        )
    if len(identifiers) < 3:
        raise ValueError("Q4 requires at least three distinct target models")
    record: dict[str, object] = {
        "record_type": "v0.8_q4_pruning_baseline_study",
        "version": "1",
        "protocol": {
            "selection_budget": args.selection_budget,
            "safe_perplexity_delta": args.safe_perplexity_delta,
            "seeds": list(DEFAULT_Q4_SEEDS),
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_confidence": 0.95,
            "feature_profile": "static_activation_gradient",
            "ranking_target": "safe_mutation_probability",
        },
        "sources": sources,
        "results": results,
    }
    args.output.write_text(canonical_identity_json(record) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
