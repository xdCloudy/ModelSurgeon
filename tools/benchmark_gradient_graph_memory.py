"""Compare retained-graph and chunk-streamed HF gradient memory on CUDA."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, cast

import torch

from modelsurgeon.adapters.huggingface.loader import HuggingFaceDType
from modelsurgeon.adapters.huggingface.proof_runtime import (
    HuggingFaceMLPProofConfig,
    HuggingFaceMLPProofRuntime,
)


def _run_mode(
    runtime: HuggingFaceMLPProofRuntime,
    selected_parameters: tuple[Any, ...],
    total_tokens: int,
    mode: str,
) -> tuple[dict[str, object], tuple[torch.Tensor, ...]]:
    model = runtime.model
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline_bytes = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    total_loss: Any = None
    for chunk in runtime._chunks:
        input_ids = torch.tensor([chunk], dtype=torch.long, device=runtime._input_device)
        output = model(input_ids=input_ids, use_cache=False)
        logits = output.logits[:, :-1, :].float().contiguous()
        targets = input_ids[:, 1:].to(logits.device).contiguous()
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, int(logits.shape[-1])),
            targets.view(-1),
            reduction="sum",
        )
        if mode == "retained_graphs":
            total_loss = loss if total_loss is None else total_loss + loss
        else:
            (loss / total_tokens).backward()  # type: ignore[no-untyped-call]
            del output, logits, targets, loss
    if mode == "retained_graphs":
        if total_loss is None:
            raise RuntimeError("calibration produced no gradient loss")
        (total_loss / total_tokens).backward()
        del total_loss
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    gradients = tuple(
        parameter.grad.detach().float().cpu().clone() for parameter in selected_parameters
    )
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return (
        {
            "mode": mode,
            "baseline_allocated_bytes": baseline_bytes,
            "peak_allocated_bytes": peak_allocated,
            "peak_incremental_bytes": peak_allocated - baseline_bytes,
            "peak_reserved_bytes": peak_reserved,
            "wall_seconds": elapsed,
        },
        gradients,
    )


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _first_order_removal_scores(
    weights: tuple[torch.Tensor, ...], gradients: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    oriented_weights = (weights[0], weights[1], weights[2].transpose(0, 1))
    oriented_gradients = (gradients[0], gradients[1], gradients[2].transpose(0, 1))
    score = (oriented_weights[0] * oriented_gradients[0]).sum(dim=1)
    for weight, gradient in zip(
        oriented_weights[1:], oriented_gradients[1:], strict=True
    ):
        score = score + (weight * gradient).sum(dim=1)
    return -score


def _spearman_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left_rank = torch.argsort(torch.argsort(left)).float()
    right_rank = torch.argsort(torch.argsort(right)).float()
    correlation = torch.corrcoef(torch.stack((left_rank, right_rank)))[0, 1]
    return float(correlation)


def _top_fraction_overlap(left: torch.Tensor, right: torch.Tensor, fraction: float) -> float:
    count = max(1, int(left.numel() * fraction))
    left_top = set(torch.argsort(left)[:count].tolist())
    right_top = set(torch.argsort(right)[:count].tolist())
    return len(left_top & right_top) / count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model identifier or local directory")
    parser.add_argument("--revision", required=True, help="immutable model revision or commit")
    parser.add_argument("--calibration-text", required=True, type=Path)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument(
        "--dtype",
        choices=tuple(item.value for item in HuggingFaceDType),
        default="float16",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.sequence_length < 2 or args.max_tokens < 2 or args.layer < 0 or args.repeat < 1:
        parser.error("sequence length, token budget, repeat count, and layer must be positive")
    if not args.calibration_text.is_file():
        parser.error("calibration text must be an existing file")
    if not torch.cuda.is_available():
        parser.error("gradient graph memory comparison requires CUDA")

    runtime = HuggingFaceMLPProofRuntime(
        HuggingFaceMLPProofConfig(
            model=args.model,
            revision=args.revision,
            calibration_text=args.calibration_text,
            device_map=args.device_map,
            dtype=HuggingFaceDType(args.dtype),
            local_files_only=args.local_files_only,
            sequence_length=args.sequence_length,
            max_tokens=args.max_tokens,
            safe_perplexity_delta=1_000_000.0,
            seed=args.seed,
            tool_revision="gradient-graph-memory-benchmark-v1",
        )
    )
    selected_parameters = tuple(runtime._projection_weights(args.layer))
    selected_ids = {id(parameter) for parameter in selected_parameters}
    original_requires_grad = tuple(
        (parameter, bool(parameter.requires_grad)) for parameter in runtime.model.parameters()
    )
    weights_cpu = tuple(
        parameter.detach().float().cpu().clone() for parameter in selected_parameters
    )
    try:
        for parameter, _ in original_requires_grad:
            parameter.requires_grad_(id(parameter) in selected_ids)
        total_tokens = sum(len(chunk) - 1 for chunk in runtime._chunks)
        if total_tokens <= 0:
            parser.error("calibration text produced no causal targets")

        retained_records: list[dict[str, object]] = []
        streamed_records: list[dict[str, object]] = []
        maximum_gradient_difference = 0.0
        maximum_relative_gradient_difference = 0.0
        maximum_first_order_difference = 0.0
        maximum_relative_first_order_difference = 0.0
        gradients_allclose = True
        first_order_scores_allclose = True
        minimum_score_spearman = 1.0
        minimum_top_1_percent_overlap = 1.0
        minimum_top_5_percent_overlap = 1.0
        score_maximum_absolute_value = 0.0
        for iteration in range(args.repeat):
            order = (
                ("retained_graphs", "streamed_backward")
                if iteration % 2 == 0
                else ("streamed_backward", "retained_graphs")
            )
            gradients_by_mode: dict[str, tuple[torch.Tensor, ...]] = {}
            for mode in order:
                record, gradients = _run_mode(
                    runtime, selected_parameters, total_tokens, mode
                )
                (retained_records if mode == "retained_graphs" else streamed_records).append(
                    record
                )
                gradients_by_mode[mode] = gradients
            for retained, streamed in zip(
                gradients_by_mode["retained_graphs"],
                gradients_by_mode["streamed_backward"],
                strict=True,
            ):
                difference = float((retained - streamed).abs().max())
                maximum_gradient_difference = max(maximum_gradient_difference, difference)
                denominator = torch.maximum(retained.abs(), streamed.abs()).clamp_min(1e-12)
                relative_difference = float(((retained - streamed).abs() / denominator).max())
                maximum_relative_gradient_difference = max(
                    maximum_relative_gradient_difference, relative_difference
                )
                gradients_allclose = gradients_allclose and torch.allclose(
                    retained, streamed, rtol=1e-4, atol=1e-6
                )
            retained_scores = _first_order_removal_scores(
                weights_cpu, gradients_by_mode["retained_graphs"]
            )
            streamed_scores = _first_order_removal_scores(
                weights_cpu, gradients_by_mode["streamed_backward"]
            )
            score_difference = (retained_scores - streamed_scores).abs()
            maximum_first_order_difference = max(
                maximum_first_order_difference, float(score_difference.max())
            )
            score_maximum_absolute_value = max(
                score_maximum_absolute_value,
                float(torch.maximum(retained_scores.abs(), streamed_scores.abs()).max()),
            )
            score_denominator = torch.maximum(
                retained_scores.abs(), streamed_scores.abs()
            ).clamp_min(1e-12)
            maximum_relative_first_order_difference = max(
                maximum_relative_first_order_difference,
                float((score_difference / score_denominator).max()),
            )
            first_order_scores_allclose = first_order_scores_allclose and torch.allclose(
                retained_scores, streamed_scores, rtol=1e-4, atol=1e-6
            )
            minimum_score_spearman = min(
                minimum_score_spearman,
                _spearman_correlation(retained_scores, streamed_scores),
            )
            minimum_top_1_percent_overlap = min(
                minimum_top_1_percent_overlap,
                _top_fraction_overlap(retained_scores, streamed_scores, 0.01),
            )
            minimum_top_5_percent_overlap = min(
                minimum_top_5_percent_overlap,
                _top_fraction_overlap(retained_scores, streamed_scores, 0.05),
            )

        def summary(records: list[dict[str, object]]) -> dict[str, object]:
            return {
                "runs": records,
                "median_peak_allocated_bytes": int(
                    statistics.median(
                        int(cast(int, item["peak_allocated_bytes"])) for item in records
                    )
                ),
                "median_peak_incremental_bytes": int(
                    statistics.median(
                        int(cast(int, item["peak_incremental_bytes"])) for item in records
                    )
                ),
                "median_peak_reserved_bytes": int(
                    statistics.median(
                        int(cast(int, item["peak_reserved_bytes"])) for item in records
                    )
                ),
                "median_wall_seconds": statistics.median(
                    float(cast(float, item["wall_seconds"])) for item in records
                ),
            }

        record = {
            "record_type": "hf_gradient_graph_memory_comparison",
            "schema_version": 1,
            "git_revision": _git_revision(),
            "model": args.model,
            "revision": args.revision,
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "dtype": args.dtype,
            "layer": args.layer,
            "sequence_length": args.sequence_length,
            "max_tokens": args.max_tokens,
            "chunks": len(runtime._chunks),
            "causal_target_tokens": total_tokens,
            "calibration_text_sha256": hashlib.sha256(
                args.calibration_text.read_bytes()
            ).hexdigest(),
            "repeat_count": args.repeat,
            "gradient_allclose_rtol_1e-4_atol_1e-6": gradients_allclose,
            "max_abs_gradient_difference": maximum_gradient_difference,
            "max_relative_gradient_difference": maximum_relative_gradient_difference,
            "first_order_scores_allclose_rtol_1e-4_atol_1e-6": first_order_scores_allclose,
            "max_abs_first_order_difference": maximum_first_order_difference,
            "max_relative_first_order_difference": maximum_relative_first_order_difference,
            "first_order_score_spearman_min": minimum_score_spearman,
            "first_order_top_1_percent_overlap_min": minimum_top_1_percent_overlap,
            "first_order_top_5_percent_overlap_min": minimum_top_5_percent_overlap,
            "first_order_scores_max_abs_value": score_maximum_absolute_value,
            "retained_graphs": summary(retained_records),
            "streamed_backward": summary(streamed_records),
        }
        serialized = json.dumps(record, indent=2, sort_keys=True) + "\n"
        if args.output is None:
            print(serialized, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
            print(args.output)
    finally:
        for parameter, requires_grad in original_requires_grad:
            parameter.requires_grad_(requires_grad)
        runtime.model.zero_grad(set_to_none=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
