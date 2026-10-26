"""Compare ShortGPT, random, and weight-magnitude layer choices on held-out text."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import random
import statistics
import time
from pathlib import Path
from typing import Any

from modelsurgeon.adapters.huggingface.block_influence import (
    measure_huggingface_block_influence,
)
from modelsurgeon.adapters.huggingface.loader import (
    HuggingFaceDType,
    HuggingFaceLoadRequest,
    load_causal_lm,
)
from modelsurgeon.adapters.huggingface.physical_layers import (
    remove_huggingface_transformer_layers,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _model_identifier(model_argument: str, model_path: Path, explicit: str | None) -> str:
    if explicit is not None:
        if not explicit.strip():
            raise ValueError("model identifier cannot be blank")
        return explicit
    if not Path(model_argument).exists():
        return model_argument
    for parent in (model_path, *model_path.parents):
        if parent.name.startswith("models--"):
            return parent.name.removeprefix("models--").replace("--", "/")
    raise ValueError(
        "--model-identifier is required when --model is a local path outside the Hugging Face cache"
    )


def _limitations(random_seed_count: int) -> list[str]:
    return [
        "One pinned model and one removed layer; cross-model generalization is untested.",
        f"{random_seed_count} predeclared random seeds give descriptive, not powered, "
        "uncertainty estimates.",
        "Prefill throughput on short contexts does not establish autoregressive decode speed.",
        "Timed evaluation excludes model loading and candidate surgery costs.",
        "Scores use WikiText-2 train tokens; quality uses its held-out test split.",
    ]


def _token_chunks(
    tokenizer: Any,
    path: Path,
    *,
    sequence_length: int,
    max_tokens: int,
) -> tuple[tuple[int, ...], ...]:
    text = path.read_text(encoding="utf-8")
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        truncation=True,
        max_length=max_tokens,
    )
    token_ids = encoded.get("input_ids")
    if not isinstance(token_ids, list) or not all(
        isinstance(token, int) and not isinstance(token, bool) and token >= 0 for token in token_ids
    ):
        raise ValueError(f"tokenizer returned invalid input_ids for {path}")
    bounded = token_ids[:max_tokens]
    chunks = tuple(
        tuple(bounded[offset : offset + sequence_length])
        for offset in range(0, len(bounded), sequence_length)
        if len(bounded[offset : offset + sequence_length]) >= 2
    )
    if not chunks:
        raise ValueError(f"text produced fewer than two tokens: {path}")
    return chunks


def _measure_perplexity(
    model: Any,
    torch: Any,
    chunks: tuple[tuple[int, ...], ...],
    *,
    repeats: int = 1,
    warmup_chunks: tuple[tuple[int, ...], ...] = (),
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("measurement repeats must be positive")
    embeddings = model.get_input_embeddings()
    input_device = embeddings.weight.device
    model.eval()

    def synchronize() -> None:
        if input_device.type == "cuda":
            torch.cuda.synchronize(input_device)

    with torch.inference_mode():
        for chunk in warmup_chunks:
            input_ids = torch.tensor((chunk,), dtype=torch.long, device=input_device)
            model(input_ids=input_ids, use_cache=False)
    synchronize()

    timing_samples: list[float] = []
    total_nll = 0.0
    token_count = 0
    for _ in range(repeats):
        repeat_nll = 0.0
        repeat_token_count = 0
        synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            for chunk in chunks:
                input_ids = torch.tensor((chunk,), dtype=torch.long, device=input_device)
                output = model(input_ids=input_ids, use_cache=False)
                logits = output.logits[:, :-1, :].float().contiguous()
                targets = input_ids[:, 1:].to(logits.device).contiguous()
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, int(logits.shape[-1])),
                    targets.view(-1),
                    reduction="sum",
                )
                repeat_nll += float(loss.detach().cpu().item())
                repeat_token_count += int(targets.numel())
        synchronize()
        timing_samples.append(time.perf_counter() - started)
        total_nll += repeat_nll
        token_count = repeat_token_count
    mean_loss = total_nll / token_count
    mean_loss /= repeats
    perplexity = math.exp(mean_loss)
    if not math.isfinite(perplexity):
        raise ValueError("held-out perplexity is not finite")
    tokens_per_second = tuple(token_count / sample for sample in timing_samples)
    return {
        "mean_loss": mean_loss,
        "perplexity": perplexity,
        "token_count": token_count,
        "wall_seconds": statistics.median(timing_samples),
        "wall_seconds_samples": timing_samples,
        "tokens_per_second": statistics.median(tokens_per_second),
        "tokens_per_second_samples": tokens_per_second,
        "measurement_repeats": repeats,
        "device": str(input_device),
    }


def _parameter_count(model: Any) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def _layer_magnitude_scores(model: Any) -> tuple[float, ...]:
    return tuple(
        sum(
            float(parameter.detach().float().abs().sum().cpu().item())
            for parameter in layer.parameters()
        )
        for layer in model.model.layers
    )


def _evaluate_removal(
    model: Any,
    layer_index: int,
    *,
    torch: Any,
    evaluation_chunks: tuple[tuple[int, ...], ...],
    measurement_repeats: int,
    warmup_chunks: tuple[tuple[int, ...], ...],
) -> tuple[Any, dict[str, Any]]:
    candidate = copy.deepcopy(model)
    surgery = remove_huggingface_transformer_layers(candidate, (layer_index,))
    measurement = _measure_perplexity(
        candidate,
        torch,
        evaluation_chunks,
        repeats=measurement_repeats,
        warmup_chunks=warmup_chunks,
    )
    return candidate, {
        "removed_layer": layer_index,
        **measurement,
        "parameter_delta": surgery.parameter_delta,
        "candidate_parameters": surgery.new_parameter_count,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.cpu_threads)
    if args.device_map == "auto" and not torch.cuda.is_available():
        raise ValueError("device-map auto requires an available CUDA device in this benchmark")
    model_path = Path(args.model).expanduser().resolve()
    calibration_path = Path(args.calibration_text).expanduser().resolve()
    evaluation_path = Path(args.evaluation_text).expanduser().resolve()
    output = Path(args.output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)

    loaded = load_causal_lm(
        HuggingFaceLoadRequest(
            model=str(model_path),
            revision=args.revision,
            device_map=args.device_map,
            dtype=HuggingFaceDType.AUTO,
            local_files_only=True,
        )
    )
    model = loaded.model
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), revision=args.revision, local_files_only=True, trust_remote_code=False
    )
    calibration_chunks = _token_chunks(
        tokenizer,
        calibration_path,
        sequence_length=args.sequence_length,
        max_tokens=args.calibration_tokens,
    )
    evaluation_chunks = _token_chunks(
        tokenizer,
        evaluation_path,
        sequence_length=args.sequence_length,
        max_tokens=args.evaluation_tokens,
    )
    influence = measure_huggingface_block_influence(
        model,
        calibration_chunks,
        input_device=model.get_input_embeddings().weight.device,
    )
    influence_order = sorted(influence, key=lambda item: (item.score, item.layer_index))
    magnitude_scores = _layer_magnitude_scores(model)
    magnitude_order = sorted(
        range(len(model.model.layers)), key=lambda index: (magnitude_scores[index], index)
    )
    warmup_chunks = calibration_chunks[:1] if args.measurement_repeats > 1 else ()
    baseline = _measure_perplexity(
        model,
        torch,
        evaluation_chunks,
        repeats=args.measurement_repeats,
        warmup_chunks=warmup_chunks,
    )
    random_results: list[dict[str, object]] = []
    random_layers: list[int] = []
    for seed in args.random_seeds:
        layer = random.Random(seed).randrange(len(model.model.layers))
        random_layers.append(layer)
        candidate, result = _evaluate_removal(
            model,
            layer,
            torch=torch,
            evaluation_chunks=evaluation_chunks,
            measurement_repeats=args.measurement_repeats,
            warmup_chunks=warmup_chunks,
        )
        result["seed"] = seed
        result["perplexity_delta"] = float(result["perplexity"]) - float(baseline["perplexity"])
        random_results.append(result)
        del candidate

    method_results: dict[str, dict[str, object]] = {}
    for method, layer_index in (
        ("shortgpt_block_influence", influence_order[0].layer_index),
        ("weight_magnitude", magnitude_order[0]),
    ):
        candidate, result = _evaluate_removal(
            model,
            layer_index,
            torch=torch,
            evaluation_chunks=evaluation_chunks,
            measurement_repeats=args.measurement_repeats,
            warmup_chunks=warmup_chunks,
        )
        result["perplexity_delta"] = float(result["perplexity"]) - float(baseline["perplexity"])
        method_results[method] = result
        if method == "shortgpt_block_influence":
            artifact = output / "shortgpt-artifact"
            candidate.save_pretrained(artifact, safe_serialization=True)
            tokenizer.save_pretrained(artifact)
            del candidate
            reloaded = AutoModelForCausalLM.from_pretrained(
                str(artifact),
                local_files_only=True,
                trust_remote_code=False,
                device_map=args.device_map,
            )
            reload_measurement = _measure_perplexity(
                reloaded,
                torch,
                evaluation_chunks,
                repeats=args.measurement_repeats,
                warmup_chunks=warmup_chunks,
            )
            method_results[method]["artifact_reload"] = {
                "perplexity": reload_measurement["perplexity"],
                "perplexity_delta": float(reload_measurement["perplexity"])
                - float(baseline["perplexity"]),
                "parameter_count": _parameter_count(reloaded),
                "disk_bytes": _directory_bytes(artifact),
                "finite_logits": True,
            }
            del reloaded
        else:
            del candidate

    source_file = model_path / "model.safetensors"
    return {
        "record_type": "shortgpt_depth_pruning_physical_comparison",
        "schema_version": 1,
        "benchmark": {
            "script": "tools/benchmark_shortgpt_depth.py",
            "script_sha256": _sha256(Path(__file__).resolve()),
        },
        "model": {
            "identifier": _model_identifier(args.model, model_path, args.model_identifier),
            "revision": args.revision,
            "weight_file_sha256": _sha256(source_file),
            "parameters": _parameter_count(model),
            "source_disk_bytes": _directory_bytes(model_path),
        },
        "environment": {
            "os": platform.platform(),
            "cpu": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device_map": args.device_map,
            "device": str(model.get_input_embeddings().weight.device),
            "dtype": str(model.get_input_embeddings().weight.dtype),
            "torch_threads": torch.get_num_threads(),
            "measurement_repeats": args.measurement_repeats,
        },
        "calibration": {
            "name": calibration_path.name,
            "sha256": _sha256(calibration_path),
            "bytes": calibration_path.stat().st_size,
            "tokens": sum(len(chunk) for chunk in calibration_chunks),
            "split": "train",
        },
        "evaluation": {
            "name": evaluation_path.name,
            "sha256": _sha256(evaluation_path),
            "bytes": evaluation_path.stat().st_size,
            "tokens": baseline["token_count"],
            "split": "test",
            "baseline": baseline,
        },
        "selection": {
            "metric": "mean(1 - cosine(block_input, block_output))",
            "shortgpt_layer_order": [
                {"layer_index": item.layer_index, "score": item.score, "tokens": item.token_count}
                for item in influence_order
            ],
            "weight_magnitude_layer_order": magnitude_order,
            "random_seeds": args.random_seeds,
            "random_selected_layers": random_layers,
        },
        "candidates": {
            "shortgpt_block_influence": method_results["shortgpt_block_influence"],
            "weight_magnitude": method_results["weight_magnitude"],
            "random": random_results,
        },
        "limitations": _limitations(len(args.random_seeds)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-identifier")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--calibration-text", required=True)
    parser.add_argument("--evaluation-text", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--calibration-tokens", type=int, default=2048)
    parser.add_argument("--evaluation-tokens", type=int, default=2048)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--device-map", choices=("cpu", "auto"), default="cpu")
    parser.add_argument("--measurement-repeats", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--random-seeds", type=int, nargs="+", default=[11, 23, 47])
    args = parser.parse_args()
    if (
        min(
            args.calibration_tokens,
            args.evaluation_tokens,
            args.sequence_length,
            args.measurement_repeats,
            args.cpu_threads,
        )
        <= 0
    ):
        parser.error("token limits, sequence length, and CPU thread count must be positive")
    report = run(args)
    report_path = Path(args.output_directory).expanduser().resolve() / "shortgpt-depth-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"report": str(report_path), "report_sha256": _sha256(report_path)}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
