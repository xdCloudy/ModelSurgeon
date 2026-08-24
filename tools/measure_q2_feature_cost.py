"""Measure static and activation feature collection cost on one pinned HF model."""

from __future__ import annotations

import argparse
from pathlib import Path

from modelsurgeon.adapters.huggingface.loader import HuggingFaceDType
from modelsurgeon.adapters.huggingface.proof_runtime import (
    HuggingFaceMLPProofConfig,
    HuggingFaceMLPProofRuntime,
)
from modelsurgeon.experiments.hardware import collect_hardware_inventory
from modelsurgeon.experiments.identity import canonical_identity_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--calibration-text", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()

    runtime = HuggingFaceMLPProofRuntime(
        HuggingFaceMLPProofConfig(
            model=args.model,
            revision=args.revision,
            calibration_text=args.calibration_text,
            device_map="auto",
            dtype=HuggingFaceDType.FLOAT16,
            sequence_length=args.max_tokens,
            max_tokens=args.max_tokens,
            seed=42,
            tool_revision="q2-feature-cost-v1",
        )
    )
    record: dict[str, object] = {
        "record_type": "v0.8_q2_feature_collection_cost",
        "version": "1",
        "model": {"identifier": args.model, "revision": args.revision},
        "calibration": {
            "path": args.calibration_text.name,
            "max_tokens": args.max_tokens,
        },
        "costs": runtime.measure_feature_collection_costs().to_record(),
        "hardware": collect_hardware_inventory(args.calibration_text.parent).to_record(),
    }
    args.output.write_text(canonical_identity_json(record) + "\n", encoding="utf-8")
    print(canonical_identity_json(record))


if __name__ == "__main__":
    main()
