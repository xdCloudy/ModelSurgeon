"""Run a revision-pinned real-model short fine-tuning repair smoke study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from modelsurgeon.adapters.huggingface.loader import (
    HuggingFaceDType,
    HuggingFaceLoadRequest,
    load_causal_lm,
)
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.surgery.short_finetune import (
    FineTuneParameterMode,
    ShortFineTuneConfig,
    run_short_finetune_repair,
)


def _tensor_sha256(tensor: object) -> str:
    host = tensor.detach().float().cpu().contiguous()  # type: ignore[attr-defined]
    return hashlib.sha256(host.numpy().tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--parameter", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer

    loaded = load_causal_lm(
        HuggingFaceLoadRequest(
            args.model,
            revision=args.revision,
            device_map="auto",
            dtype=HuggingFaceDType.FLOAT32,
            local_files_only=True,
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=True,
    )
    encoded = tokenizer(
        "A bounded repair should preserve the useful behavior of a surgically edited model.",
        return_tensors="pt",
        truncation=True,
        max_length=32,
    )
    example = {**encoded, "labels": encoded["input_ids"].clone()}
    parameter = dict(loaded.model.named_parameters())[args.parameter]
    before = _tensor_sha256(parameter)
    result = run_short_finetune_repair(
        loaded.model,
        (example,),
        (example,),
        ShortFineTuneConfig(
            parameter_mode=FineTuneParameterMode.SELECTED,
            parameter_names=(args.parameter,),
            learning_rate=1e-3,
            max_steps=args.steps,
            validation_patience=args.steps,
            max_trainable_parameters=int(parameter.numel()),
            seed=42,
        ),
        source_checkpoint_id=f"checkpoint_{args.revision}",
        candidate_parent_checkpoint_id="checkpoint_real_short_finetune_candidate",
    )
    after = _tensor_sha256(parameter)
    record = {
        "record_type": "v0.9_short_finetune_repair_smoke",
        "version": "1",
        "model": {
            "identifier": args.model,
            "requested_revision": args.revision,
            "resolved_revision": loaded.provenance.resolved_revision,
        },
        "parameter": args.parameter,
        "tokens": int(encoded["input_ids"].numel()),
        "parameter_sha256_before": before,
        "parameter_sha256_after": after,
        "parameter_changed": before != after,
        "result": result.to_record(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    args.output.write_text(canonical_identity_json(record) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
