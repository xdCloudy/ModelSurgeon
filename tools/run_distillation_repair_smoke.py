"""Run a revision-pinned real-model immutable-logit distillation smoke study."""

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
from modelsurgeon.surgery.distillation_repair import (
    DistillationRepairConfig,
    TokenizerSignature,
    run_distillation_repair,
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
    parser.add_argument("--candidate-scale", type=float, default=0.9)
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
    signature = TokenizerSignature.from_tokenizer(tokenizer)
    encoded = tokenizer(
        "A distilled repair should recover behavior from an immutable baseline.",
        return_tensors="pt",
        truncation=True,
        max_length=32,
    )
    example = {**encoded, "labels": encoded["input_ids"].clone()}
    device = next(loaded.model.parameters()).device
    loaded.model.eval()
    with torch.inference_mode():
        teacher_logits = (
            loaded.model(**{name: value.to(device) for name, value in encoded.items()})
            .logits.detach()
            .float()
            .cpu()
            .contiguous()
            .clone(),
        )
    parameter = dict(loaded.model.named_parameters())[args.parameter]
    source_hash = _tensor_sha256(parameter)
    with torch.no_grad():
        parameter.mul_(args.candidate_scale)
    candidate_hash = _tensor_sha256(parameter)
    result = run_distillation_repair(
        loaded.model,
        (example,),
        DistillationRepairConfig(
            parameter_names=(args.parameter,),
            temperature=2.0,
            distillation_weight=0.75,
            supervised_weight=0.25,
            learning_rate=1e-3,
            max_steps=args.steps,
            max_trainable_parameters=int(parameter.numel()),
            seed=42,
        ),
        teacher_tokenizer=signature,
        candidate_tokenizer=signature,
        source_checkpoint_id=f"checkpoint_{args.revision}",
        candidate_parent_checkpoint_id="checkpoint_real_distillation_candidate",
        baseline_logits=teacher_logits,
    )
    repaired_hash = _tensor_sha256(parameter)
    record = {
        "record_type": "v0.9_distillation_repair_smoke",
        "version": "1",
        "model": {
            "identifier": args.model,
            "requested_revision": args.revision,
            "resolved_revision": loaded.provenance.resolved_revision,
        },
        "parameter": args.parameter,
        "candidate_scale": args.candidate_scale,
        "tokens": int(encoded["input_ids"].numel()),
        "source_parameter_sha256": source_hash,
        "candidate_parameter_sha256": candidate_hash,
        "repaired_parameter_sha256": repaired_hash,
        "result": result.to_record(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    args.output.write_text(canonical_identity_json(record) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
