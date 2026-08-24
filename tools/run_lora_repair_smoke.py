"""Run a revision-pinned real-model LoRA repair smoke study."""

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
from modelsurgeon.surgery.lora_repair import (
    LoRAOutputMode,
    LoRARepairConfig,
    lora_adapter_state_dict,
    run_bounded_lora_repair,
)


def _tensor_sha256(tensor: object) -> str:
    host = tensor.detach().float().cpu().contiguous()  # type: ignore[attr-defined]
    return hashlib.sha256(host.numpy().tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--rank", type=int, default=2)
    args = parser.parse_args()

    import torch
    from safetensors.torch import save
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
    labels = encoded["input_ids"].clone()
    example = {**encoded, "labels": labels}
    module = loaded.model.get_submodule(args.target)
    before = _tensor_sha256(module.weight)
    result = run_bounded_lora_repair(
        loaded.model,
        (example,),
        (args.target,),
        LoRARepairConfig(
            rank=args.rank,
            alpha=2 * args.rank,
            learning_rate=1e-3,
            max_steps=args.steps,
            seed=42,
            output_mode=LoRAOutputMode.SEPARATE,
        ),
        source_checkpoint_id=f"checkpoint_{args.revision}",
        candidate_checkpoint_id="checkpoint_real_lora_smoke_candidate",
    )
    wrapper = loaded.model.get_submodule(args.target)
    after = _tensor_sha256(wrapper.base.weight)
    adapter_bytes = save(lora_adapter_state_dict(loaded.model, (args.target,)))
    record = {
        "record_type": "v0.9_lora_repair_smoke",
        "version": "1",
        "model": {
            "identifier": args.model,
            "requested_revision": args.revision,
            "resolved_revision": loaded.provenance.resolved_revision,
        },
        "target": args.target,
        "tokens": int(encoded["input_ids"].numel()),
        "base_weight_sha256_before": before,
        "base_weight_sha256_after": after,
        "base_weight_preserved": before == after,
        "adapter_safetensors_bytes": len(adapter_bytes),
        "adapter_safetensors_sha256": hashlib.sha256(adapter_bytes).hexdigest(),
        "result": result.to_record(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    args.output.write_text(canonical_identity_json(record) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
