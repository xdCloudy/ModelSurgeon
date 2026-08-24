"""Run the zero-target-example v0.8 Q5 cross-model transfer study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from modelsurgeon.cli.proof_evidence import load_grouped_proof_split
from modelsurgeon.cli.surgeon import load_surgeon_records
from modelsurgeon.evaluation.cross_model_transfer import (
    TransferDataset,
    run_cross_model_transfer,
)
from modelsurgeon.evaluation.static_feature_study import StaticFeatureStudyConfig
from modelsurgeon.experiments.identity import canonical_identity_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(directory: Path) -> TransferDataset:
    return TransferDataset(
        load_surgeon_records(directory / "examples.jsonl"),
        load_grouped_proof_split(directory / "split.json"),
    )


def _source_record(directory: Path, role: str) -> dict[str, str]:
    examples = directory / "examples.jsonl"
    split = directory / "split.json"
    manifest = directory / "gradient-manifest.json"
    return {
        "role": role,
        "examples": f"{directory.name}/examples.jsonl",
        "examples_sha256": _sha256(examples),
        "split": f"{directory.name}/split.json",
        "split_sha256": _sha256(split),
        "gradient_manifest": f"{directory.name}/gradient-manifest.json",
        "gradient_manifest_sha256": _sha256(manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safe-perplexity-delta", type=float, default=0.01)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    args = parser.parse_args()

    if len(args.source) < 2:
        raise ValueError("Q5 requires at least two source models")
    config = StaticFeatureStudyConfig(
        safe_perplexity_delta=args.safe_perplexity_delta,
        top_n=args.top_n,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    result = run_cross_model_transfer(
        tuple(_load(directory) for directory in args.source),
        _load(args.target),
        config,
    )
    record: dict[str, object] = {
        "record_type": "v0.8_q5_cross_model_transfer_study",
        "version": "1",
        "protocol": {
            "training_scope": "source_train_and_validation_only",
            "target_scope": "target_test_only_after_source_fit",
            "feature_profile": "static_activation_gradient",
            "safe_perplexity_delta": args.safe_perplexity_delta,
            "top_n": args.top_n,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_confidence": 0.95,
            "seed": 42,
            "split_seed": 43,
        },
        "sources": [
            *(_source_record(directory, "training_source") for directory in args.source),
            _source_record(args.target, "unseen_target"),
        ],
        "result": result.to_record(),
    }
    args.output.write_text(canonical_identity_json(record) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
