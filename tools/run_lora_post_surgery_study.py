"""Compare no-repair and held-out-selected LoRA after physical MLP pruning."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

from modelsurgeon.adapters.huggingface.loader import (
    HuggingFaceDType,
    HuggingFaceLoadRequest,
    load_causal_lm,
)
from modelsurgeon.adapters.huggingface.physical_mlp import remove_huggingface_mlp_channels
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.surgery.lora_repair import (
    LoRAOutputMode,
    LoRARepairConfig,
    run_bounded_lora_repair,
)

MODEL = "HuggingFaceTB/SmolLM2-135M"
MODEL_REVISION = "93efa2f097d58c2a74874c7e644dbc9b0cee75a2"
DATASET = "Salesforce/wikitext"
DATASET_REVISION = "00aa25585682d4957f9e86edc73f59be7419af99"
DATASET_CONFIG = "wikitext-2-raw-v1"
SPLITS = ("train", "validation", "test")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_manifest(path: Path) -> dict[str, object]:
    files = tuple(sorted(item for item in path.rglob("*") if item.is_file()))
    manifest = [
        {
            "path": item.relative_to(path).as_posix(),
            "size_bytes": item.stat().st_size,
            "sha256": _sha256_file(item),
        }
        for item in files
    ]
    manifest_json = canonical_identity_json(manifest)
    return {
        "file_count": len(files),
        "file_bytes": sum(int(item["size_bytes"]) for item in manifest),
        "manifest_sha256": hashlib.sha256(manifest_json.encode("utf-8")).hexdigest(),
    }


def _load_dataset_rows(split: str) -> tuple[tuple[tuple[int, str], ...], Path]:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    import pyarrow.parquet as parquet
    from huggingface_hub import hf_hub_download

    relative_path = f"{DATASET_CONFIG}/{split}-00000-of-00001.parquet"
    path = Path(
        hf_hub_download(
            repo_id=DATASET,
            repo_type="dataset",
            revision=DATASET_REVISION,
            filename=relative_path,
        )
    )
    table = parquet.read_table(path, columns=["text"])
    rows: list[tuple[int, str]] = []
    for index, raw in enumerate(table.column("text").to_pylist()):
        if not isinstance(raw, str):
            continue
        text = " ".join(raw.split())
        if len(text.split()) < 50 or text.startswith("="):
            continue
        rows.append((index, text))
    return tuple(rows), path


def _ordered_test_rows(
    rows: tuple[tuple[int, str], ...], sample_seed: int | None
) -> tuple[tuple[int, str], ...]:
    if sample_seed is None:
        return rows
    shuffled = list(rows)
    random.Random(sample_seed).shuffle(shuffled)
    return tuple(shuffled)


def _examples(
    texts: tuple[tuple[int, str], ...], tokenizer: Any, count: int, max_length: int
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, object], ...]]:
    examples: list[dict[str, Any]] = []
    provenance: list[dict[str, object]] = []
    for row_index, text in texts:
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        if int(encoded["input_ids"].numel()) < 16:
            continue
        example = dict(encoded)
        example["labels"] = encoded["input_ids"].clone()
        examples.append(example)
        provenance.append(
            {
                "row_index": row_index,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "tokens": int(encoded["input_ids"].numel()),
            }
        )
        if len(examples) == count:
            break
    if len(examples) != count:
        raise RuntimeError(f"dataset split supplied only {len(examples)} usable rows; need {count}")
    return tuple(examples), tuple(provenance)


def _parameter_metrics(model: Any) -> dict[str, int]:
    parameters = tuple(model.parameters())
    return {
        "parameters": sum(int(parameter.numel()) for parameter in parameters),
        "parameter_bytes": sum(
            int(parameter.numel() * parameter.element_size()) for parameter in parameters
        ),
    }


def _low_magnitude_channels(model: Any, count: int) -> tuple[int, ...]:
    import torch

    config = model.config
    width = int(config.intermediate_size)
    layers = int(config.num_hidden_layers)
    if not 0 < count < width:
        raise ValueError(f"channels must be between 1 and {width - 1}")
    scores = torch.zeros(width, dtype=torch.float64)
    modules = dict(model.named_modules())
    for layer in range(layers):
        mlp = modules.get(f"model.layers.{layer}.mlp")
        if mlp is None:
            raise RuntimeError(f"unsupported MLP path at layer {layer}")
        gate_norm = mlp.gate_proj.weight.detach().float().norm(dim=1).cpu()
        up_norm = mlp.up_proj.weight.detach().float().norm(dim=1).cpu()
        down_norm = mlp.down_proj.weight.detach().float().norm(dim=0).cpu()
        scores += (gate_norm * up_norm * down_norm).double()
    selected = sorted(range(width), key=lambda index: (float(scores[index]), index))[:count]
    return tuple(sorted(selected))


def _measure(
    model: Any, examples: tuple[dict[str, Any], ...], device: Any, *, repeats: int = 3
) -> dict[str, object]:
    import torch

    model.to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    input_examples = tuple(
        {
            name: tensor.to(device) if hasattr(tensor, "to") else tensor
            for name, tensor in example.items()
        }
        for example in examples
    )
    durations: list[float] = []
    weighted_loss = 0.0
    token_count = 0
    with torch.inference_mode():
        for example in input_examples:
            for repeat in range(repeats + 1):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                output = model(**example)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                loss = getattr(output, "loss", None)
                if loss is None or not bool(torch.isfinite(loss).item()):
                    raise RuntimeError("evaluation returned no finite loss")
                valid_targets = int((example["labels"][:, 1:] != -100).sum().item())
                if repeat == repeats:
                    weighted_loss += float(loss.float().item()) * valid_targets
                    token_count += valid_targets
                else:
                    durations.append(elapsed)
    if token_count <= 0 or not durations:
        raise RuntimeError("evaluation has no target tokens or timing samples")
    median_seconds = statistics.median(durations)
    loss_value = weighted_loss / token_count
    record: dict[str, object] = {
        "mean_token_loss": loss_value,
        "perplexity": math.exp(loss_value),
        "target_tokens": token_count,
        "median_prefill_latency_ms": 1000.0 * median_seconds,
        "median_prefill_tokens_per_second": token_count / median_seconds,
    }
    if device.type == "cuda":
        record["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        record["peak_cuda_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    else:
        record["peak_cuda_allocated_bytes"] = None
        record["peak_cuda_reserved_bytes"] = None
    return record


def _save_and_reload(
    model: Any,
    output: Path,
    source_revision: str,
    source_snapshot: Path,
    tokenizer: Any,
) -> tuple[Any, dict[str, object]]:
    model.to(device="cpu", dtype=__import__("torch").float16)
    model.save_pretrained(output, safe_serialization=True, max_shard_size="1GB")
    for source_file in source_snapshot.iterdir():
        if not source_file.is_file():
            continue
        if source_file.name in {
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        }:
            continue
        if source_file.suffix in {".safetensors", ".bin"}:
            continue
        shutil.copy2(source_file, output / source_file.name)
    from transformers import AutoTokenizer

    reloaded_tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True)
    if tokenizer.get_vocab() != reloaded_tokenizer.get_vocab():
        raise RuntimeError("candidate tokenizer vocabulary differs from the pinned source")
    manifest = _artifact_manifest(output)
    reloaded = load_causal_lm(
        HuggingFaceLoadRequest(
            str(output),
            revision=source_revision,
            device_map="cpu",
            dtype=HuggingFaceDType.FLOAT16,
            local_files_only=True,
        )
    )
    return reloaded.model, manifest


def run(args: argparse.Namespace) -> dict[str, object]:
    import torch

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    import pyarrow
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("this consumer-hardware study requires CUDA")
    device = torch.device("cuda:0")
    dataset_rows: dict[str, tuple[tuple[int, str], ...]] = {}
    dataset_files: dict[str, dict[str, object]] = {}
    for split in SPLITS:
        rows, path = _load_dataset_rows(split)
        dataset_rows[split] = rows
        dataset_files[split] = {
            "path": f"{DATASET_CONFIG}/{split}-00000-of-00001.parquet",
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=MODEL_REVISION, local_files_only=True
    )
    training_examples, training_rows = _examples(
        dataset_rows["train"], tokenizer, args.examples_per_split, args.max_length
    )
    validation_examples, validation_rows = _examples(
        dataset_rows["validation"], tokenizer, args.examples_per_split, args.max_length
    )
    test_texts = _ordered_test_rows(dataset_rows["test"], args.test_sample_seed)
    test_example_count = (
        args.examples_per_split
        if args.test_examples_per_split is None
        else args.test_examples_per_split
    )
    extended_test_sampling = (
        args.test_examples_per_split is not None or args.test_sample_seed is not None
    )
    test_examples, test_rows = _examples(
        test_texts, tokenizer, test_example_count, args.max_length
    )
    input_hashes = {
        split: {
            hashlib.sha256(
                example["input_ids"].detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest()
            for example in examples
        }
        for split, examples in (
            ("train", training_examples),
            ("validation", validation_examples),
            ("test", test_examples),
        )
    }
    if (
        input_hashes["train"] & input_hashes["validation"]
        or input_hashes["train"] & input_hashes["test"]
        or input_hashes["validation"] & input_hashes["test"]
    ):
        raise RuntimeError("WikiText splits contain an identical selected token sequence")

    snapshot = Path(
        snapshot_download(repo_id=MODEL, revision=MODEL_REVISION, local_files_only=True)
    )
    source_files = _artifact_manifest(snapshot)
    baseline_loaded = load_causal_lm(
        HuggingFaceLoadRequest(
            MODEL,
            revision=MODEL_REVISION,
            device_map="cpu",
            dtype=HuggingFaceDType.FLOAT16,
            local_files_only=True,
        )
    )
    baseline_model = baseline_loaded.model
    baseline_parameters = _parameter_metrics(baseline_model)
    baseline_quality = _measure(baseline_model, test_examples, device)
    del baseline_model, baseline_loaded
    torch.cuda.empty_cache()

    loaded = load_causal_lm(
        HuggingFaceLoadRequest(
            MODEL,
            revision=MODEL_REVISION,
            device_map="cpu",
            dtype=HuggingFaceDType.FLOAT32,
            local_files_only=True,
        )
    )
    model = loaded.model
    model.to("cpu")
    removed = _low_magnitude_channels(model, args.remove_channels)
    surgery = remove_huggingface_mlp_channels(model, removed)
    with tempfile.TemporaryDirectory(prefix="modelsurgeon-lora-study-") as temp_name:
        root = Path(temp_name)
        no_repair_model, no_repair_manifest = _save_and_reload(
            model, root / "no-repair", MODEL_REVISION, snapshot, tokenizer
        )
        pruned_parameters = _parameter_metrics(no_repair_model)
        no_repair_quality = _measure(no_repair_model, test_examples, device)
        no_repair_model.to(device="cpu", dtype=torch.float32)

        target_modules = tuple(
            sorted(
                name
                for name, module in no_repair_model.named_modules()
                if name.endswith(".mlp.down_proj") and isinstance(module, torch.nn.Linear)
            )
        )
        if len(target_modules) != int(no_repair_model.config.num_hidden_layers):
            raise RuntimeError("could not identify exactly one down_proj per transformer layer")
        no_repair_model.to(device)
        repair = run_bounded_lora_repair(
            no_repair_model,
            training_examples,
            target_modules,
            LoRARepairConfig(
                rank=args.rank,
                alpha=2 * args.rank,
                learning_rate=args.learning_rate,
                max_steps=args.steps,
                seed=args.seed,
                output_mode=LoRAOutputMode.MERGED,
            ),
            source_checkpoint_id=f"checkpoint_{MODEL_REVISION}",
            candidate_checkpoint_id="checkpoint_wikitext_pruned_candidate",
            validation_examples=validation_examples,
            validation_dataset_sha256=dataset_files["validation"]["sha256"],
            evaluation_source="wikitext_validation_split",
        )
        repaired_model, repaired_manifest = _save_and_reload(
            no_repair_model, root / "repaired", MODEL_REVISION, snapshot, tokenizer
        )
        repaired_quality = _measure(repaired_model, test_examples, device)
        del no_repair_model, repaired_model
        torch.cuda.empty_cache()

    return {
        "schema_version": 2 if extended_test_sampling else 1,
        "protocol": (
            "lora-post-surgery-wikitext-smoke-v2"
            if extended_test_sampling
            else "lora-post-surgery-wikitext-smoke-v1"
        ),
        "model": {
            "identifier": MODEL,
            "revision": MODEL_REVISION,
            "resolved_revision": loaded.provenance.resolved_revision,
            "license": "Apache-2.0 (upstream model card)",
            "architecture": model.config.model_type,
            "layer_count": int(model.config.num_hidden_layers),
        },
        "run_parameters": {
            "output_dtype": "float16",
            "remove_channels": args.remove_channels,
            "examples_per_split": args.examples_per_split,
            "test_examples_per_split": test_example_count,
            "test_sample_seed": args.test_sample_seed,
            "max_length": args.max_length,
            "rank": args.rank,
            "alpha": 2 * args.rank,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            **(
                {
                    "test_examples_per_split": test_example_count,
                    "test_sample_seed": args.test_sample_seed,
                }
                if extended_test_sampling
                else {}
            ),
        },
        "dataset": {
            "identifier": DATASET,
            "revision": DATASET_REVISION,
            "config": DATASET_CONFIG,
            "license": "CC-BY-SA-3.0 and GFDL (upstream metadata)",
            "split_files": dataset_files,
            "selected_rows": {
                "train": list(training_rows),
                "validation": list(validation_rows),
                "test": list(test_rows),
            },
            "selected_token_sequences_disjoint": True,
            "raw_text_included": False,
        },
        "runtime": {
            "python_version": __import__("sys").version.split()[0],
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "transformers_version": __import__("transformers").__version__,
            "peft_version": __import__("peft").__version__,
            "pyarrow_version": pyarrow.__version__,
            "device": torch.cuda.get_device_name(device),
            "model_snapshot": source_files,
        },
        "surgery": {
            "method": (
                "global lowest gated-MLP channel proxy: sum over layers of gate-row "
                "L2 norm x up-row L2 norm x down-column L2 norm"
            ),
            "removed_channels": list(removed),
            "result": surgery.to_record(),
        },
        "repair": repair.to_record(),
        "arms": {
            "source": {
                **baseline_parameters,
                "test": baseline_quality,
            },
            "pruned_no_repair": {
                **pruned_parameters,
                "artifact": no_repair_manifest,
                "test": no_repair_quality,
            },
            "pruned_lora_selected": {
                **pruned_parameters,
                "artifact": repaired_manifest,
                "test": repaired_quality,
            },
        },
        "limitations": [
            (
                "One 135M model, fixed pruning severity, one seed, four examples per split; "
                "this is a feasibility smoke, not a generalization or superiority claim."
            ),
            (
                "Quality is next-token loss and perplexity on four held-out WikiText-2 test "
                "rows; selection uses the separate validation split."
            ),
            (
                "Latency is short-batch prefill under observed workstation load; it is not "
                "a decode-throughput or deployment benchmark."
            ),
            "No distillation, multiple-precision, or CPU-offload arm is included.",
            (
                "Temporary candidate artifacts were reloaded for verification and are not "
                "retained by this script."
            ),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remove-channels", type=int, default=64)
    parser.add_argument("--examples-per-split", type=int, default=4)
    parser.add_argument("--test-examples-per-split", type=int)
    parser.add_argument("--test-sample-seed", type=int)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (
        args.examples_per_split <= 0
        or args.max_length < 16
        or (args.test_examples_per_split is not None and args.test_examples_per_split <= 0)
        or (args.test_sample_seed is not None and args.test_sample_seed < 0)
    ):
        parser.error(
            "example counts must be positive, test-sample-seed non-negative, "
            "and max-length at least 16"
        )
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = canonical_identity_json(result) + "\n"
    args.output.write_text(serialized, encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
