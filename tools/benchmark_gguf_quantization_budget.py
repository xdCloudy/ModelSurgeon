"""Run a reproducible mixed-codec GGUF byte-budget experiment.

This research utility estimates per-tensor reconstruction MSE against the
decoded source artifact, plans an exact file-size budget, writes the selected
candidate transactionally, and optionally measures paired llama.cpp
perplexity. It does not promote or publish the candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from modelsurgeon.adapters.gguf import (
    DENSE_GGUF_CODECS,
    Q2_K_CODEC,
    Q3_K_CODEC,
    Q4_K_CODEC,
    Q5_0_CODEC,
    Q5_K_CODEC,
    Q6_K_CODEC,
    Q8_0_CODEC,
    CodecRegistry,
    GGMLQuantizationType,
    GGUFDiskEstimate,
    GGUFTensorReader,
    estimate_tensor_quantization_losses,
    execute_gguf_quantization_plan,
    open_gguf,
    plan_gguf_quantization_budget,
    preflight_gguf_disk,
)
from modelsurgeon.adapters.gguf.container import gguf_tensor_byte_size
from modelsurgeon.evaluation import (
    LlamaCppPerplexityConfig,
    LlamaCppPerplexityManifest,
    benchmark_gguf_perplexity,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _codecs() -> CodecRegistry:
    registry = CodecRegistry()
    codecs = (
        *DENSE_GGUF_CODECS,
        Q2_K_CODEC,
        Q3_K_CODEC,
        Q4_K_CODEC,
        Q5_0_CODEC,
        Q5_K_CODEC,
        Q6_K_CODEC,
        Q8_0_CODEC,
    )
    registered: set[GGMLQuantizationType] = set()
    for codec in codecs:
        quant_type = codec.identity.quant_type
        if quant_type not in registered:
            registry.register(codec)
            registered.add(quant_type)
    return registry


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--target-file-bytes", type=int, required=True)
    parser.add_argument(
        "--target-type",
        choices=("F16", "F32", "Q2_K", "Q3_K", "Q4_K", "Q5_0", "Q5_K", "Q6_K", "Q8_0"),
        default="Q3_K",
    )
    parser.add_argument("--max-working-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--evaluation-text", type=Path)
    parser.add_argument("--dataset", default="Salesforce/wikitext:wikitext-2-raw-v1")
    parser.add_argument("--dataset-revision", required=False)
    parser.add_argument("--split", default="test")
    parser.add_argument("--tokenizer", required=False)
    parser.add_argument("--tokenizer-revision", required=False)
    parser.add_argument("--llama-perplexity", type=Path)
    parser.add_argument("--llama-revision", required=False)
    parser.add_argument("--context-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    return parser


def _check_evaluation_arguments(args: argparse.Namespace) -> bool:
    supplied = (
        args.evaluation_text,
        args.dataset_revision,
        args.tokenizer,
        args.tokenizer_revision,
        args.llama_perplexity,
        args.llama_revision,
    )
    if all(value is None for value in supplied):
        return False
    if any(value is None for value in supplied):
        raise ValueError(
            "perplexity evaluation requires text, dataset/tokenizer revisions, and llama.cpp"
        )
    return True


def main() -> int:
    args = _parser().parse_args()
    if args.target_file_bytes <= 0 or args.max_working_bytes <= 0:
        raise ValueError("target file size and working-memory limit must be positive")
    source = args.source.expanduser().resolve(strict=True)
    candidate = args.candidate.expanduser().resolve()
    report_path = args.report.expanduser().resolve()
    if candidate == source or candidate.exists():
        raise FileExistsError("candidate must be a new path distinct from the source")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite experiment report: {report_path}")
    run_perplexity = _check_evaluation_arguments(args)
    codecs = _codecs()
    target_type = GGMLQuantizationType(args.target_type)
    source_size = source.stat().st_size
    started = time.perf_counter()
    estimates: dict[str, dict[str, object]] = {}
    source_tensor_bytes: dict[str, int] = {}

    with open_gguf(source) as mapped:
        reader = GGUFTensorReader(mapped)
        alternatives = {}
        for tensor in mapped.container.tensors:
            source_tensor_bytes[tensor.name] = tensor.byte_size
            try:
                target_size = gguf_tensor_byte_size(target_type, tensor.dimensions)
            except ValueError:
                continue
            if target_size >= tensor.byte_size or tensor.quant_type is target_type:
                continue
            tensor_handle = reader.index.tensor(tensor.name)
            try:
                estimate = estimate_tensor_quantization_losses(
                    reader,
                    tensor_handle,
                    (target_type,),
                    codecs,
                    byte_order=mapped.container.byte_order,
                    max_working_bytes=args.max_working_bytes,
                )[0]
            except (KeyError, ValueError) as error:
                estimates[tensor.name] = {"skipped": str(error)}
                continue
            alternatives[tensor.name] = (estimate.to_alternative(),)
            estimates[tensor.name] = {
                "source_type": tensor.quant_type.value,
                "target_type": target_type.value,
                "source_bytes": tensor.byte_size,
                "target_bytes": target_size,
                "weight_mse": estimate.mean_squared_error,
                "sample_count": estimate.sample_count,
                "estimated_peak_working_bytes": estimate.estimated_peak_working_bytes,
            }

        plan = plan_gguf_quantization_budget(
            mapped.container,
            alternatives,
            codecs,
            target_file_bytes=args.target_file_bytes,
        )
        candidate.parent.mkdir(parents=True, exist_ok=True)
        disk_plan = preflight_gguf_disk(
            candidate,
            candidate.parent,
            GGUFDiskEstimate(
                output_bytes=plan.planned_file_bytes,
                scratch_bytes=0,
                alignment_bytes=mapped.container.alignment,
            ),
        )
        write_result = execute_gguf_quantization_plan(
            reader,
            plan,
            codecs,
            disk_plan,
            max_working_bytes=args.max_working_bytes,
        )

    output_counts = Counter(choice.output_quant_type.value for choice in plan.choices)
    changed = [
        choice
        for choice in plan.choices
        if choice.output_quant_type is not choice.source_quant_type
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "objective": "exact_gguf_byte_budget_minimize_sum_of_codec_local_weight_mse",
        "limitations": [
            (
                "codec loss compares against decoded source weights, "
                "not original full-precision weights"
            ),
            "weight MSE does not predict model quality",
            "no deployment promotion is performed",
        ],
        "source": {
            "path": str(source),
            "sha256": _sha256(source),
            "file_bytes": source_size,
            "tensor_count": len(plan.choices),
        },
        "planner": {
            "target_type": target_type.value,
            "target_file_bytes": plan.target_file_bytes,
            "planned_file_bytes": plan.planned_file_bytes,
            "estimated_loss": plan.estimated_loss,
            "explored_states": plan.explored_states,
            "output_type_counts": dict(sorted(output_counts.items())),
            "changed_tensor_count": len(changed),
            "changed_tensors": [
                {
                    "name": choice.tensor_name,
                    "source_type": choice.source_quant_type.value,
                    "output_type": choice.output_quant_type.value,
                    "source_bytes": source_tensor_bytes[choice.tensor_name],
                    "output_bytes": choice.encoded_bytes,
                    "estimated_loss": choice.estimated_loss,
                }
                for choice in changed
            ],
        },
        "codec_estimates": estimates,
        "candidate": {
            "path": str(write_result.path),
            "sha256": write_result.sha256,
            "file_bytes": write_result.file_size,
        },
        "elapsed_seconds_through_write": time.perf_counter() - started,
    }

    if run_perplexity:
        text_path = args.evaluation_text.expanduser().resolve(strict=True)
        manifest = LlamaCppPerplexityManifest(
            text_path=text_path,
            text_sha256=_sha256(text_path),
            sample_count=1,
            dataset=args.dataset,
            dataset_revision=args.dataset_revision,
            split=args.split,
            tokenizer=args.tokenizer,
            tokenizer_revision=args.tokenizer_revision,
        )
        evaluation = benchmark_gguf_perplexity(
            source,
            candidate,
            manifest,
            config=LlamaCppPerplexityConfig(
                executable=args.llama_perplexity,
                expected_revision=args.llama_revision,
                context_size=args.context_size,
                batch_size=min(args.context_size, 512),
                microbatch_size=min(args.context_size, 512),
                threads=args.threads,
                gpu_layers=0,
                chunks=args.chunks,
                timeout_seconds=args.timeout_seconds,
            ),
        )
        report["evaluation"] = {
            "manifest": manifest.to_record(),
            "successful": evaluation.successful,
            "perplexity_delta": evaluation.perplexity_delta,
            "report": evaluation.to_record(),
        }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_name(f".{report_path.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    print(
        json.dumps(
            {"report": str(report_path), "candidate": str(candidate), "sha256": write_result.sha256}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
