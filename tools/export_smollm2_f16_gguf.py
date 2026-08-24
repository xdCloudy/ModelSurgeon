"""Export the pinned cached SmolLM2 Llama checkpoint to a native F16 GGUF."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def export_model(model: Any, tokenizer_assets: Path, output: Path) -> None:
    """Write one exact Llama-family model using bounded local tokenizer assets."""

    import gguf
    import numpy as np

    config = model.config
    if getattr(config, "model_type", None) != "llama" or not bool(
        getattr(config, "tie_word_embeddings", False)
    ):
        raise ValueError("this bounded exporter requires a tied Llama-family checkpoint")
    if output.exists():
        raise FileExistsError(f"GGUF output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    vocabulary = json.loads((tokenizer_assets / "vocab.json").read_text(encoding="utf-8"))
    if not isinstance(vocabulary, dict) or len(vocabulary) != int(config.vocab_size):
        raise ValueError("tokenizer vocabulary does not match model config")
    tokens: list[str | None] = [None] * len(vocabulary)
    for token, index in vocabulary.items():
        if not isinstance(token, str) or not isinstance(index, int) or not 0 <= index < len(tokens):
            raise ValueError("tokenizer vocabulary entry is invalid")
        tokens[index] = token
    if any(token is None for token in tokens):
        raise ValueError("tokenizer vocabulary IDs are not contiguous")
    tokenizer_config = json.loads(
        (tokenizer_assets / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    special_ids = {
        int(index)
        for index, record in tokenizer_config.get("added_tokens_decoder", {}).items()
        if isinstance(record, dict) and record.get("special") is True
    }
    merges = (tokenizer_assets / "merges.txt").read_text(encoding="utf-8").splitlines()
    if merges and merges[0].startswith("#version"):
        merges = merges[1:]

    writer = gguf.GGUFWriter(output, "llama", use_temp_file=True)
    try:
        writer.add_name("SmolLM2-135M")
        writer.add_description("Revision-pinned local ModelSurgeon F16 export")
        writer.add_context_length(int(config.max_position_embeddings))
        writer.add_embedding_length(int(config.hidden_size))
        writer.add_feed_forward_length(int(config.intermediate_size))
        writer.add_block_count(int(config.num_hidden_layers))
        writer.add_head_count(int(config.num_attention_heads))
        writer.add_head_count_kv(int(config.num_key_value_heads))
        writer.add_rope_dimension_count(int(config.head_dim))
        rope_parameters = getattr(config, "rope_parameters", None)
        rope_base = (
            rope_parameters.get("rope_theta")
            if isinstance(rope_parameters, dict)
            else getattr(config, "rope_theta", None)
        )
        if not isinstance(rope_base, (int, float)) or not math.isfinite(rope_base):
            raise ValueError("Llama config has no finite RoPE frequency base")
        writer.add_rope_freq_base(float(rope_base))
        writer.add_layer_norm_rms_eps(float(config.rms_norm_eps))
        writer.add_file_type(int(gguf.GGMLQuantizationType.F16))
        writer.add_tokenizer_model("gpt2")
        writer.add_tokenizer_pre("smollm")
        writer.add_token_list([token for token in tokens if token is not None])
        writer.add_token_merges(merges)
        writer.add_token_types(
            [
                gguf.TokenType.CONTROL if index in special_ids else gguf.TokenType.NORMAL
                for index in range(len(tokens))
            ]
        )
        writer.add_bos_token_id(int(config.bos_token_id))
        writer.add_eos_token_id(int(config.eos_token_id))
        writer.add_unk_token_id(int(config.eos_token_id))
        writer.add_add_bos_token(False)
        writer.add_add_eos_token(False)
        tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.LLAMA, int(config.num_hidden_layers))
        for source_name, tensor in model.state_dict().items():
            if source_name == "lm_head.weight":
                continue
            destination = tensor_map.get_name(source_name, try_suffixes=(".weight", ".bias"))
            if destination is None:
                raise ValueError(f"no GGUF tensor mapping for {source_name}")
            dtype = np.float32 if tensor.ndim == 1 else np.float16
            writer.add_tensor(
                destination,
                tensor.detach().float().cpu().numpy().astype(dtype, copy=False),
            )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
    finally:
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM

    snapshot = args.cache / f"models--{args.model.replace('/', '--')}" / "snapshots" / args.revision
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=args.cache,
        local_files_only=True,
        dtype="auto",
    )
    export_model(model, snapshot, args.output)


if __name__ == "__main__":
    main()
