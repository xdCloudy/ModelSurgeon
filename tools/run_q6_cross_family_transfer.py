"""Run v0.8 Q6 single-, multi-, and held-out-family transfer protocols."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from modelsurgeon.cli.proof_evidence import load_grouped_proof_split
from modelsurgeon.cli.surgeon import load_surgeon_records
from modelsurgeon.evaluation.cross_family_transfer import (
    TransferExperiment,
    TransferProtocol,
    run_cross_family_transfer_study,
)
from modelsurgeon.evaluation.cross_model_transfer import TransferDataset
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


def _source_record(directory: Path) -> dict[str, str]:
    examples = directory / "examples.jsonl"
    split = directory / "split.json"
    manifest = directory / "gradient-manifest.json"
    return {
        "dataset": directory.name,
        "examples_sha256": _sha256(examples),
        "split_sha256": _sha256(split),
        "gradient_manifest_sha256": _sha256(manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family-source", type=Path, required=True)
    parser.add_argument("--represented-target", type=Path, required=True)
    parser.add_argument("--other-family", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safe-perplexity-delta", type=float, default=0.01)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--bootstrap-repetitions", type=int, default=1000)
    args = parser.parse_args()

    family_source = _load(args.family_source)
    represented_target = _load(args.represented_target)
    other_family = _load(args.other_family)
    config = StaticFeatureStudyConfig(
        safe_perplexity_delta=args.safe_perplexity_delta,
        top_n=args.top_n,
        bootstrap_repetitions=args.bootstrap_repetitions,
    )
    result = run_cross_family_transfer_study(
        (
            TransferExperiment(
                TransferProtocol.SINGLE_FAMILY,
                (family_source,),
                represented_target,
            ),
            TransferExperiment(
                TransferProtocol.MULTI_FAMILY,
                (family_source, other_family),
                represented_target,
            ),
            TransferExperiment(
                TransferProtocol.HELD_OUT_FAMILY,
                (family_source, represented_target),
                other_family,
            ),
        ),
        config,
    )
    record: dict[str, object] = {
        "record_type": "v0.8_q6_cross_family_transfer_study",
        "version": "1",
        "protocol": {
            "feature_profile": "static_activation_gradient",
            "safe_perplexity_delta": args.safe_perplexity_delta,
            "top_n": args.top_n,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_confidence": 0.95,
            "seed": 42,
            "split_seed": 43,
            "failure_policy": "retain_expected_schema_failures",
        },
        "sources": [
            _source_record(args.family_source),
            _source_record(args.represented_target),
            _source_record(args.other_family),
        ],
        "result": result.to_record(),
    }
    args.output.write_text(canonical_identity_json(record) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
