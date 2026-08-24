"""Run the v0.8 Q7 active-versus-random study over multiple measured models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from modelsurgeon.cli.proof_evidence import load_grouped_proof_split
from modelsurgeon.cli.surgeon import load_surgeon_records
from modelsurgeon.evaluation.multi_model_active_learning import (
    MultiModelActiveLearningConfig,
    run_model_active_learning_study,
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
    parser.add_argument("--budget", action="append", type=int)
    parser.add_argument("--target-auc", type=float, default=0.8)
    parser.add_argument("--safe-perplexity-delta", type=float, default=0.01)
    args = parser.parse_args()

    budgets = tuple(args.budget or (64, 128, 256, 384))
    config = MultiModelActiveLearningConfig(
        budgets=budgets,
        target_auc=args.target_auc,
        safe_perplexity_delta=args.safe_perplexity_delta,
    )
    results: list[dict[str, object]] = []
    sources: list[dict[str, str]] = []
    identifiers: set[str] = set()
    for directory in args.dataset:
        examples = directory / "examples.jsonl"
        split = directory / "split.json"
        manifest = directory / "gradient-manifest.json"
        result = run_model_active_learning_study(
            load_surgeon_records(examples),
            load_grouped_proof_split(split),
            config,
        )
        results.append(result.to_record())
        identifiers.add(result.model[0])
        sources.append(
            {
                "dataset": directory.name,
                "examples_sha256": _sha256(examples),
                "split_sha256": _sha256(split),
                "gradient_manifest_sha256": _sha256(manifest),
            }
        )
    if len(identifiers) < 3:
        raise ValueError("Q7 requires at least three distinct models")
    record: dict[str, object] = {
        "record_type": "v0.8_q7_multi_model_active_learning_study",
        "version": "1",
        "protocol": {
            "strategies": ["active_uncertainty", "seeded_random"],
            "budgets": list(budgets),
            "seeds": list(config.seeds),
            "target_auc": args.target_auc,
            "safe_perplexity_delta": args.safe_perplexity_delta,
            "feature_profile": "static_activation_gradient",
            "test_metric": "roc_auc",
            "gpu_hour_accounting": "sum_selected_cuda_evaluation_wall_seconds",
        },
        "sources": sources,
        "results": results,
    }
    args.output.write_text(canonical_identity_json(record) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
