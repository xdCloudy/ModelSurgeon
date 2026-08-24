"""Run the pinned v0.8 Q1 static-only study from persisted proof datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from modelsurgeon.cli.proof_evidence import load_grouped_proof_split
from modelsurgeon.cli.surgeon import load_surgeon_records
from modelsurgeon.evaluation.static_feature_study import (
    StaticFeatureStudyConfig,
    run_static_feature_study,
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
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    parser.add_argument("--safe-perplexity-delta", type=float, default=0.01)
    args = parser.parse_args()

    results: list[dict[str, object]] = []
    sources: list[dict[str, str]] = []
    model_identifiers: set[str] = set()
    for directory in args.dataset:
        examples = directory / "examples.jsonl"
        split = directory / "split.json"
        result = run_static_feature_study(
            load_surgeon_records(examples),
            load_grouped_proof_split(split),
            StaticFeatureStudyConfig(
                safe_perplexity_delta=args.safe_perplexity_delta,
                top_n=args.top_n,
                bootstrap_repetitions=args.bootstrap_repetitions,
            ),
        )
        results.append(result.to_record())
        model_identifiers.add(result.model_identifier)
        sources.append(
            {
                "examples": f"{directory.name}/examples.jsonl",
                "examples_sha256": _sha256(examples),
                "split": f"{directory.name}/split.json",
                "split_sha256": _sha256(split),
            }
        )
    if len(results) < 3:
        raise ValueError("Q1 requires at least three target-model datasets")
    if len(model_identifiers) < 3:
        raise ValueError("Q1 requires at least three distinct target models")
    record: dict[str, object] = {
        "record_type": "v0.8_q1_static_feature_study",
        "version": "1",
        "protocol": {
            "feature_profile": "static_only",
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
