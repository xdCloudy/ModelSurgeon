"""Publish a signed pretrained-surgeon card for an existing immutable bundle."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from modelsurgeon.surgeon.pretrained_registry import (
    PretrainedSurgeonCard,
    PretrainedSurgeonRegistry,
)


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--secret-env", required=True)
    args = parser.parse_args()
    secret_text = os.environ.get(args.secret_env)
    if not secret_text:
        raise SystemExit(f"signing secret environment variable is missing: {args.secret_env}")
    registry = PretrainedSurgeonRegistry(args.registry)
    bundle = registry.bundles.load(args.bundle)
    source_evidence = tuple(sorted({_file_digest(args.dataset), _file_digest(args.split)}))
    card = PretrainedSurgeonCard(
        bundle_digest=str(bundle.artifact.metadata.digest),
        model_id=args.model_id,
        model_revision=args.model_revision,
        training_models=bundle.card.training_models,
        source_evidence_digests=source_evidence,
        feature_schema_version=bundle.preprocessor.source_feature_schema_version,
        target_schema_version=bundle.preprocessor.target_schema_version,
        state_schema_version=1,
        supported_mutations=("mask",),
        supported_codecs=("none",),
        supported_hardware=("cpu",),
        calibration_policy={
            "source": "real_physical_observations",
            "split_policy": "model_identity_grouped_transfer",
            "target": bundle.card.target_name,
        },
        compatibility={
            "feature_names": list(bundle.preprocessor.output_feature_names),
            "model_formats": ["huggingface"],
            "candidate_scope": "mlp_channel",
        },
        license="Apache-2.0",
        metrics=bundle.card.metrics,
        limitations=(
            "held-out transfer evidence is required before treating guidance as generally "
            "predictive",
            "pilot transfer bundle trained from retained CPU Hugging Face observations",
        ),
    )
    published = registry.publish(
        card,
        key_id=args.key_id,
        secret=secret_text.encode("utf-8"),
    )
    print(
        {
            "card_digest": str(published.artifact.metadata.digest),
            "bundle_digest": str(published.card.bundle_digest),
            "key_id": args.key_id,
            "source_evidence_digests": list(source_evidence),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
