"""Collect input second moments for supported Llama GGUF projection tensors.

This research tool uses the exact quantized source GGUF through Transformers'
GGUF loader, which dequantizes the complete model on CPU. It refuses models
whose dense float32 weight size exceeds the explicit configured limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from modelsurgeon.adapters.gguf.container import open_gguf

_PROJECTIONS = {
    "q_proj": ("self_attn", "attn_q"),
    "k_proj": ("self_attn", "attn_k"),
    "v_proj": ("self_attn", "attn_v"),
    "o_proj": ("self_attn", "attn_output"),
    "gate_proj": ("mlp", "ffn_gate"),
    "up_proj": ("mlp", "ffn_up"),
    "down_proj": ("mlp", "ffn_down"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_text(path: Path, max_bytes: int) -> str:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"calibration text is not a file: {resolved}")
    with resolved.open("rb") as stream:
        payload = stream.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError("calibration text exceeds its configured byte limit")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("calibration text must be valid UTF-8") from error


def _metadata(container: Any) -> dict[str, Any]:
    return {entry.key: entry.value for entry in container.metadata}


def _verify_model_and_tokenizer(config: Any, tokenizer: Any, metadata: dict[str, Any]) -> None:
    if metadata.get("general.architecture") != "llama" or config.model_type != "llama":
        raise ValueError("activation calibration currently supports GGUF architecture llama only")
    checks = (
        ("llama.block_count", "num_hidden_layers"),
        ("llama.embedding_length", "hidden_size"),
        ("llama.feed_forward_length", "intermediate_size"),
        ("llama.attention.head_count", "num_attention_heads"),
        ("llama.attention.head_count_kv", "num_key_value_heads"),
    )
    for gguf_key, config_key in checks:
        if metadata.get(gguf_key) != getattr(config, config_key, None):
            raise ValueError(f"model configuration does not match GGUF metadata: {gguf_key}")
    float_checks = (
        ("llama.rope.freq_base", "rope_theta"),
        ("llama.attention.layer_norm_rms_epsilon", "rms_norm_eps"),
    )
    for gguf_key, config_key in float_checks:
        actual = metadata.get(gguf_key)
        expected = getattr(config, config_key, None)
        if config_key == "rope_theta" and expected is None:
            rope_parameters = getattr(config, "rope_parameters", None)
            if isinstance(rope_parameters, dict):
                expected = rope_parameters.get("rope_theta")
        if not isinstance(actual, (int, float)) or not isinstance(expected, (int, float)):
            raise ValueError(f"model configuration is missing {gguf_key} or {config_key}")
        if not math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-8):
            raise ValueError(f"model configuration does not match GGUF metadata: {gguf_key}")

    tokens = metadata.get("tokenizer.ggml.tokens")
    merges = metadata.get("tokenizer.ggml.merges")
    if not isinstance(tokens, (list, tuple)) or not isinstance(merges, (list, tuple)):
        raise ValueError("source GGUF is missing its tokenizer vocabulary or merges")
    if len(tokens) != tokenizer.vocab_size:
        raise ValueError("model tokenizer vocabulary size differs from source GGUF")
    if list(tokens) != tokenizer.convert_ids_to_tokens(list(range(tokenizer.vocab_size))):
        raise ValueError("model tokenizer vocabulary differs from source GGUF")
    tokenizer_merges = getattr(tokenizer, "_merges", None)
    if not isinstance(tokenizer_merges, list) or [
        f"{left} {right}" for left, right in tokenizer_merges
    ] != list(merges):
        raise ValueError("model tokenizer merges differ from source GGUF")
    for key, attribute in (
        ("tokenizer.ggml.bos_token_id", "bos_token_id"),
        ("tokenizer.ggml.eos_token_id", "eos_token_id"),
        ("tokenizer.ggml.unknown_token_id", "unk_token_id"),
    ):
        if metadata.get(key) != getattr(tokenizer, attribute, None):
            raise ValueError(f"model tokenizer special token differs from source GGUF: {key}")
    if metadata.get("tokenizer.ggml.add_bos_token") != getattr(tokenizer, "add_bos_token", None):
        raise ValueError("model tokenizer BOS behavior differs from source GGUF")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local snapshot")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--calibration-text", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-size", type=int, default=512)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-dense-weight-bytes", type=int, default=1024**3)
    parser.add_argument("--max-text-bytes", type=int, default=64 * 1024 * 1024)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if (
        min(
            args.context_size,
            args.chunks,
            args.threads,
            args.max_dense_weight_bytes,
            args.max_text_bytes,
        )
        <= 0
    ):
        raise ValueError("context, chunk, thread, and resource limits must be positive")
    source = args.source.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists() or output == source:
        raise FileExistsError("moment output must be a new path distinct from source GGUF")
    calibration_text = args.calibration_text.expanduser().resolve(strict=True)
    text = _read_text(calibration_text, args.max_text_bytes)

    with open_gguf(source) as mapped:
        metadata = _metadata(mapped.container)
        dense_weight_bytes = sum(
            math.prod(tensor.dimensions) * 4 for tensor in mapped.container.tensors
        )
        if dense_weight_bytes > args.max_dense_weight_bytes:
            raise ValueError(
                "GGUF float32 weight size exceeds --max-dense-weight-bytes; "
                "the calibration loader dequantizes the complete model"
            )
        if metadata.get("general.architecture") != "llama":
            raise ValueError(
                "activation calibration currently supports GGUF architecture llama only"
            )
        tensor_dimensions = {tensor.name: tensor.dimensions for tensor in mapped.container.tensors}
        source_sha256 = _sha256(source)

    try:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint
    except ImportError as error:
        raise RuntimeError(
            "GGUF activation calibration requires the optional HF dependencies; "
            "run with `uv run --extra hf`"
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    config = AutoConfig.from_pretrained(args.model, revision=args.model_revision)
    _verify_model_and_tokenizer(config, tokenizer, metadata)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(args.threads)
    model = AutoModelForCausalLM.from_config(config)  # type: ignore[no-untyped-call]
    parsed = load_gguf_checkpoint(  # type: ignore[no-untyped-call]
        str(source), return_tensors=True, model_to_load=model
    )
    incompatibility = model.load_state_dict(parsed["tensors"], strict=False)
    expected_missing = {"lm_head.weight"} if config.tie_word_embeddings else set()
    if set(incompatibility.missing_keys) != expected_missing or incompatibility.unexpected_keys:
        raise ValueError("GGUF tensors did not load exactly into the revision-pinned HF model")
    if config.tie_word_embeddings and (
        model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()
    ):
        raise ValueError("GGUF source requires tied embeddings but HF model weights are untied")
    del parsed
    model.eval()

    encoded = tokenizer(
        text, return_tensors="pt", add_special_tokens=False, verbose=False
    ).input_ids[0]
    required_tokens = args.context_size * args.chunks
    if len(encoded) < required_tokens:
        raise ValueError(
            f"calibration text contains {len(encoded)} tokens; {required_tokens} are required"
        )
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None or len(layers) != config.num_hidden_layers:
        raise ValueError("loaded llama model does not expose the expected decoder layers")

    accumulators: dict[str, tuple[Any, int]] = {}
    handles = []
    for layer_index, layer in enumerate(layers):
        for projection_name, (module_group, gguf_suffix) in _PROJECTIONS.items():
            module = getattr(getattr(layer, module_group), projection_name)
            tensor_name = f"blk.{layer_index}.{gguf_suffix}.weight"
            dimensions = tensor_dimensions.get(tensor_name)
            if (
                dimensions is None
                or len(dimensions) != 2
                or dimensions[0] != module.in_features
                or dimensions[1] != module.out_features
            ):
                raise ValueError(f"projection geometry does not match GGUF tensor {tensor_name!r}")
            accumulators[tensor_name] = (None, 0)

            def collect_input(
                _module: Any, inputs: tuple[Any, ...], *, name: str = tensor_name
            ) -> None:
                values = inputs[0].detach().float()
                second_moment_sum = values.square().sum(dim=tuple(range(values.ndim - 1))).cpu()
                previous_sum, previous_count = accumulators[name]
                accumulators[name] = (
                    second_moment_sum if previous_sum is None else previous_sum + second_moment_sum,
                    previous_count + values.numel() // values.shape[-1],
                )

            handles.append(module.register_forward_pre_hook(collect_input))

    add_bos = bool(metadata["tokenizer.ggml.add_bos_token"])
    try:
        with torch.inference_mode():
            for index in range(args.chunks):
                tokens = encoded[
                    index * args.context_size : (index + 1) * args.context_size
                ].clone()
                if add_bos:
                    tokens[0] = tokenizer.bos_token_id
                model(input_ids=tokens.unsqueeze(0), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    tensors: dict[str, dict[str, Any]] = {}
    for name, (sum_values, sample_count) in accumulators.items():
        if sum_values is None or sample_count <= 0:
            raise ValueError(f"no calibration activations were recorded for {name!r}")
        values = (sum_values / sample_count).tolist()
        if len(values) != tensor_dimensions[name][0]:
            raise ValueError(f"input activation axis mismatch for {name!r}")
        tensors[name] = {
            "input_axis": 0,
            "sample_count": sample_count,
            "second_moments": values,
        }

    record = {
        "schema_version": 1,
        "source_sha256": source_sha256,
        "model": {
            "model_id": args.model,
            "revision": args.model_revision,
            "architecture": "llama",
            "dense_weight_bytes": dense_weight_bytes,
            "dtype": "float32",
        },
        "calibration": {
            "dataset": args.dataset,
            "dataset_revision": args.dataset_revision,
            "split": args.split,
            "text_sha256": _sha256(calibration_text),
            "tokenizer": args.model,
            "tokenizer_revision": args.model_revision,
            "token_count": len(encoded),
            "calibration_token_count": required_tokens,
            "context_size": args.context_size,
            "chunks": args.chunks,
            "activation_samples_per_projection": args.context_size * args.chunks,
            "torch_seed": 0,
            "torch_threads": args.threads,
        },
        "tensors": tensors,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {"output": str(output), "source_sha256": source_sha256, "tensor_count": len(tensors)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
