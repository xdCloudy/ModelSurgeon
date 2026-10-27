"""Run a bounded, CPU-only physical KFAC versus magnitude/random MLP-channel pilot."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as parquet
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from modelsurgeon.adapters.huggingface.kfac_mlp_channels import (
    score_huggingface_gated_mlp_channels_kfac,
)
from modelsurgeon.adapters.huggingface.physical_mlp import remove_huggingface_mlp_channels


class PeakRSS:
    def __init__(self) -> None:
        import psutil

        self._process = psutil.Process(os.getpid())
        self._peak = self._process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def __enter__(self) -> PeakRSS:
        self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        self._thread.join()

    @property
    def peak_bytes(self) -> int:
        return self._peak

    def _sample(self) -> None:
        while not self._stop.wait(0.2):
            self._peak = max(self._peak, self._process.memory_info().rss)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _directory_record(path: Path) -> dict[str, object]:
    files = []
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        files.append(
            {
                "path": file_path.relative_to(path).as_posix(),
                "bytes": file_path.stat().st_size,
                "sha256": _sha256(file_path),
            }
        )
    return {
        "disk_bytes": sum(int(item["bytes"]) for item in files),
        "files": files,
    }


def _text_from_parquet(parquet_path: Path, text_path: Path) -> dict[str, object]:
    table = parquet.read_table(parquet_path, columns=["text"])
    text = "\n".join(value or "" for value in table.column("text").to_pylist())
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(text, encoding="utf-8")
    return {
        "parquet_path": str(parquet_path.resolve()),
        "parquet_bytes": parquet_path.stat().st_size,
        "parquet_sha256": _sha256(parquet_path),
        "serialized_text_bytes": text_path.stat().st_size,
        "serialized_text_sha256": _sha256(text_path),
        "row_count": table.num_rows,
    }


def _token_ids(tokenizer: Any, text_path: Path) -> list[int]:
    encoded = tokenizer(
        text_path.read_text(encoding="utf-8"),
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return [int(value) for value in encoded["input_ids"]]


def _chunks(token_ids: list[int], target_tokens: int, sequence_length: int) -> list[torch.Tensor]:
    chunks: list[torch.Tensor] = []
    cursor = 0
    collected = 0
    stride = sequence_length - 1
    while collected < target_tokens:
        count = min(stride, target_tokens - collected)
        end = cursor + count + 1
        if end > len(token_ids):
            raise ValueError(f"dataset provides {collected} of {target_tokens} requested targets")
        chunks.append(torch.tensor([token_ids[cursor:end]], dtype=torch.long))
        cursor += count
        collected += count
    return chunks


def _evaluate(model: Any, chunks: list[torch.Tensor]) -> dict[str, float | int]:
    model.eval()
    target_count = 0
    nll_sum = 0.0
    started = time.perf_counter()
    with torch.inference_mode():
        for chunk in chunks:
            outputs = model(input_ids=chunk, use_cache=False)
            logits = outputs.logits[:, :-1, :].contiguous()
            labels = chunk[:, 1:].contiguous()
            nll_sum += float(
                torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1),
                    reduction="sum",
                )
            )
            target_count += int(labels.numel())
    elapsed_seconds = time.perf_counter() - started
    mean_nll = nll_sum / target_count
    return {
        "target_tokens": target_count,
        "nll_sum": nll_sum,
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "evaluation_seconds": elapsed_seconds,
        "tokens_per_second": target_count / elapsed_seconds,
    }


def _magnitude_scores(model: Any) -> np.ndarray:
    width = int(model.config.intermediate_size)
    scores = np.zeros(width, dtype=np.float64)
    layers = model.model.layers
    for layer in layers:
        mlp = layer.mlp
        for values in (
            mlp.gate_proj.weight.detach(),
            mlp.up_proj.weight.detach(),
            mlp.down_proj.weight.detach().T,
        ):
            scores += values.to(dtype=torch.float64).square().sum(dim=1).cpu().numpy()
    return scores


def _load_model(model_path: Path) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device="cpu", dtype=torch.float32)
    return model


def _git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.set_num_threads(args.torch_threads)
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if torch.cuda.is_available():
        raise RuntimeError("CPU-only benchmark refuses an available CUDA device")

    model_path = args.model.resolve()
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    storage_dtype_name = config.get("torch_dtype", "float32")
    storage_dtype = getattr(torch, storage_dtype_name, None)
    if storage_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"unsupported checkpoint storage dtype: {storage_dtype_name!r}")
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_path = output_path.parent / "kfac-mlp-channel-pilot-inputs"
    train_text = work_path / "wikitext-train.txt"
    test_text = work_path / "wikitext-test.txt"
    train_record = _text_from_parquet(args.train_parquet.resolve(), train_text)
    test_record = _text_from_parquet(args.test_parquet.resolve(), test_text)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    calibration_ids_all = _token_ids(tokenizer, train_text)
    evaluation_ids_all = _token_ids(tokenizer, test_text)
    calibration_chunks = _chunks(calibration_ids_all, args.calibration_tokens, args.sequence_length)
    evaluation_chunks = _chunks(evaluation_ids_all, args.evaluation_tokens, args.sequence_length)
    calibration_ids = [int(value) for chunk in calibration_chunks for value in chunk[0].tolist()]
    evaluation_ids = [int(value) for chunk in evaluation_chunks for value in chunk[0].tolist()]
    calibration_digest = hashlib.sha256(
        np.asarray(calibration_ids, dtype="<i8").tobytes()
    ).hexdigest()
    evaluation_digest = hashlib.sha256(
        np.asarray(evaluation_ids, dtype="<i8").tobytes()
    ).hexdigest()

    model_start = time.perf_counter()
    source_model = _load_model(model_path)
    model_load_seconds = time.perf_counter() - model_start
    source_parameter_count = sum(int(parameter.numel()) for parameter in source_model.parameters())
    intermediate_size = int(source_model.config.intermediate_size)
    removal_counts = sorted(set(args.removed_channels))
    if any(count >= intermediate_size for count in removal_counts):
        raise ValueError("removed channel counts must be smaller than the MLP intermediate width")

    with PeakRSS() as peak_rss:
        scoring_start = time.perf_counter()
        kfac_report = score_huggingface_gated_mlp_channels_kfac(
            source_model,
            calibration_chunks,
            max_calibration_target_tokens=args.calibration_tokens,
            max_workspace_bytes=args.max_workspace_gib * 1024**3,
            collector_workspace_bytes=args.collector_workspace_mib * 1024**2,
        )
        scoring_seconds = time.perf_counter() - scoring_start
        kfac_scores = np.asarray(kfac_report.aggregate_costs, dtype=np.float64)
        magnitude_scores = _magnitude_scores(source_model)
        if not np.isfinite(kfac_scores).all() or not np.isfinite(magnitude_scores).all():
            raise ValueError("candidate scores must be finite")

        selected: dict[str, tuple[int, ...]] = {}
        kfac_order = np.argsort(kfac_scores, kind="stable")
        magnitude_order = np.argsort(magnitude_scores, kind="stable")
        for count in removal_counts:
            selected[f"kfac_singleton_additive_{count}"] = tuple(
                sorted(int(value) for value in kfac_order[:count])
            )
            selected[f"magnitude_l2_{count}"] = tuple(
                sorted(int(value) for value in magnitude_order[:count])
            )
        for seed in args.random_seeds:
            random_order = random.Random(seed).sample(range(intermediate_size), intermediate_size)
            for count in removal_counts:
                selected[f"random_seed_{seed}_{count}"] = tuple(sorted(random_order[:count]))

        evaluation_start = time.perf_counter()
        baseline = _evaluate(source_model, evaluation_chunks)
        baseline_seconds = time.perf_counter() - evaluation_start
        baseline["benchmark_call_seconds"] = baseline_seconds

        baseline_files = _directory_record(model_path)
        candidate_results: dict[str, object] = {}
        source_state = source_model.state_dict()
        for method, indices in selected.items():
            method_start = time.perf_counter()
            candidate_model = _load_model(model_path)
            surgery_start = time.perf_counter()
            surgery = remove_huggingface_mlp_channels(candidate_model, indices)
            surgery_seconds = time.perf_counter() - surgery_start
            if any(
                not torch.equal(source_state[name], value)
                for name, value in source_model.state_dict().items()
            ):
                raise AssertionError("source model weights changed during candidate surgery")

            with tempfile.TemporaryDirectory(
                prefix=f"{method}-", dir=output_path.parent
            ) as temporary:
                artifact_dir = Path(temporary)
                with torch.inference_mode():
                    expected_logits = candidate_model(
                        input_ids=evaluation_chunks[0], use_cache=False
                    ).logits
                candidate_model.to(dtype=storage_dtype)
                candidate_model.save_pretrained(artifact_dir, safe_serialization=True)
                tokenizer.save_pretrained(artifact_dir)
                artifact_record = _directory_record(artifact_dir)
                del candidate_model
                gc.collect()
                reloaded_model = AutoModelForCausalLM.from_pretrained(
                    artifact_dir,
                    local_files_only=True,
                    dtype=torch.float32,
                    low_cpu_mem_usage=True,
                )
                reloaded_model.to(device="cpu", dtype=torch.float32)
                with torch.inference_mode():
                    actual_logits = reloaded_model(
                        input_ids=evaluation_chunks[0], use_cache=False
                    ).logits
                reload_max_logit_error = float(
                    torch.max(torch.abs(expected_logits - actual_logits)).item()
                )
                if not torch.allclose(expected_logits, actual_logits, rtol=1e-5, atol=1e-6):
                    raise AssertionError("saved and reloaded logits differ")
                del expected_logits, actual_logits
                reload_evaluation = _evaluate(reloaded_model, evaluation_chunks)
                del reloaded_model
                gc.collect()

            candidate_results[method] = {
                "removed_channels": list(indices),
                "removed_channel_count": len(indices),
                "parameter_count": surgery.new_parameter_count,
                "parameter_delta": surgery.parameter_delta,
                "parameter_reduction_fraction": (-surgery.parameter_delta / source_parameter_count),
                "surgery_seconds": surgery_seconds,
                "end_to_end_method_seconds": time.perf_counter() - method_start,
                "reloaded": reload_evaluation,
                "reload_max_logit_error": reload_max_logit_error,
                "candidate_perplexity_delta": (
                    float(reload_evaluation["perplexity"]) - float(baseline["perplexity"])
                ),
                "quality_gate_passed": abs(
                    float(reload_evaluation["perplexity"]) - float(baseline["perplexity"])
                )
                <= args.max_ppl_delta,
                "artifact": artifact_record,
            }
        peak_rss_bytes = peak_rss.peak_bytes

    return {
        "record_type": "kfac_mlp_channel_physical_pilot",
        "schema_version": 1,
        "source_commit": _git_revision(),
        "benchmark_script_sha256": _sha256(Path(__file__).resolve()),
        "model": {
            "identifier": args.model_id,
            "revision": args.model_revision,
            "path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
            "weight_files": [
                {
                    "path": item.name,
                    "bytes": item.stat().st_size,
                    "sha256": _sha256(item),
                }
                for item in sorted(model_path.glob("*.safetensors"))
            ],
            "parameter_count": source_parameter_count,
            "parameter_dtypes": sorted(
                {str(parameter.dtype) for parameter in source_model.parameters()}
            ),
            "checkpoint_storage_dtype": storage_dtype_name,
            "model_type": source_model.config.model_type,
            "layer_count": int(source_model.config.num_hidden_layers),
            "hidden_size": int(source_model.config.hidden_size),
            "intermediate_size": intermediate_size,
            "source_disk": baseline_files,
        },
        "dataset": {
            "identifier": "Salesforce/wikitext",
            "revision": args.dataset_revision,
            "configuration": "wikitext-2-raw-v1",
            "train": train_record,
            "test": test_record,
            "calibration_target_tokens": args.calibration_tokens,
            "evaluation_target_tokens": args.evaluation_tokens,
            "sequence_length": args.sequence_length,
            "calibration_input_ids_sha256": calibration_digest,
            "evaluation_input_ids_sha256": evaluation_digest,
        },
        "environment": {
            "os": platform.platform(),
            "cpu": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "device": "cpu",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "transformers_version": transformers.__version__,
            "numpy_version": np.__version__,
            "python_version": sys.version,
            "model_load_seconds": model_load_seconds,
            "peak_rss_bytes": peak_rss_bytes,
        },
        "protocol": {
            "removed_channel_counts": removal_counts,
            "quality_gate_absolute_ppl_delta": args.max_ppl_delta,
            "random_seeds": args.random_seeds,
            "measurement_repeats": 1,
            "candidate_comparison": "same held-out targets; globally shared MLP-channel set",
            "kfac_selection": (
                "Sum layer-local singleton down_proj input-column KFAC costs; choose lowest sum."
            ),
            "magnitude_selection": (
                "Sum squared gate/up row and down column weights across layers; choose lowest L2."
            ),
            "limitations": [
                "Single model and one CPU run; no statistical uncertainty or family transfer.",
                "Additive singleton costs omit channel-channel and cross-layer interactions.",
                "No compensation or recovery is applied.",
                "Only WikiText perplexity is measured; no task or decode tests.",
                "KFAC costs are proxies; held-out artifact measurements are authoritative.",
            ],
        },
        "baseline": baseline,
        "scoring": {
            "seconds": scoring_seconds,
            "planned_workspace_bytes": kfac_report.planned_workspace_bytes,
            "max_workspace_bytes": kfac_report.max_workspace_bytes,
            "activation_factor_sha256": list(kfac_report.activation_factor_sha256),
            "gradient_factor_sha256": list(kfac_report.gradient_factor_sha256),
            "selected_score_range": {
                "kfac_min": float(np.min(kfac_scores)),
                "kfac_max": float(np.max(kfac_scores)),
                "magnitude_min": float(np.min(magnitude_scores)),
                "magnitude_max": float(np.max(magnitude_scores)),
            },
        },
        "candidates": candidate_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--model-revision", default="93efa2f097d58c2a74874c7e644dbc9b0cee75a2")
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--test-parquet", type=Path, required=True)
    parser.add_argument("--dataset-revision", default="b08601e04326c79dfdd32d625aee71d232d685c3")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-tokens", type=int, default=256)
    parser.add_argument("--evaluation-tokens", type=int, default=2048)
    parser.add_argument("--sequence-length", type=int, default=129)
    parser.add_argument("--removed-channels", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--random-seeds", type=int, nargs="+", default=[11, 29, 47])
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--max-workspace-gib", type=int, default=4)
    parser.add_argument("--collector-workspace-mib", type=int, default=256)
    parser.add_argument("--max-ppl-delta", type=float, default=0.05)
    args = parser.parse_args()
    if any(
        value <= 0
        for value in (
            args.calibration_tokens,
            args.evaluation_tokens,
            args.sequence_length,
            args.torch_threads,
        )
    ):
        parser.error(
            "token counts, sequence length, removed channels, and thread count must be positive"
        )
    if any(seed < 0 for seed in args.random_seeds):
        parser.error("random seeds must be nonnegative")
    if any(value <= 0 for value in args.removed_channels):
        parser.error("removed channel counts must be positive")
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "baseline_perplexity": report["baseline"]["perplexity"],
                "scoring_seconds": report["scoring"]["seconds"],
                "candidate_perplexities": {
                    name: value["reloaded"]["perplexity"]
                    for name, value in report["candidates"].items()
                },
                "peak_rss_bytes": report["environment"]["peak_rss_bytes"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
