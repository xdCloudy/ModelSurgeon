"""Run the v0.9 iterative HF/native-GGUF surgery comparison on cached assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

from export_smollm2_f16_gguf import export_model

from modelsurgeon.adapters import ModelFamily
from modelsurgeon.adapters.gguf import (
    CodecRegistry,
    GGUFDiskEstimate,
    discover_gguf_components,
    open_gguf,
    preflight_gguf_disk,
)
from modelsurgeon.adapters.huggingface.loader import (
    HuggingFaceDType,
    HuggingFaceLoadRequest,
    load_causal_lm,
)
from modelsurgeon.adapters.huggingface.physical_mlp import (
    remove_huggingface_mlp_channels,
)
from modelsurgeon.evaluation.iterative_search_study import (
    BackendStudy,
    IterativeSearchStudy,
    SearchGeneration,
    SearchGoals,
    StudyArm,
    StudyBackend,
    StudyMeasurement,
    compare_arm,
)
from modelsurgeon.evaluation.llama_cpp_perplexity import (
    LlamaCppPerplexityConfig,
    LlamaCppPerplexityManifest,
    benchmark_gguf_perplexity,
)
from modelsurgeon.evaluation.llama_cpp_throughput import (
    LlamaCppThroughputConfig,
    benchmark_gguf_throughput,
)
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.surgery.native_mlp_execute import (
    execute_native_gguf_mlp_channel_removal,
)
from modelsurgeon.surgery.native_mlp_plan import (
    plan_native_gguf_model_mlp_channel_removal,
)
from modelsurgeon.surgery.short_finetune import (
    FineTuneParameterMode,
    ShortFineTuneConfig,
    run_short_finetune_repair,
)

_TEXT = (
    "Reliable model surgery needs immutable inputs, measured quality, bounded repair, "
    "and deployment validation. Every candidate is compared under one token contract."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load(model: str, revision: str) -> Any:
    return load_causal_lm(
        HuggingFaceLoadRequest(
            model,
            revision=revision,
            device_map="cpu",
            dtype=HuggingFaceDType.FLOAT32,
            local_files_only=True,
        )
    ).model


def _hf_measure(
    measurement_id: str,
    model: Any,
    encoded: dict[str, Any],
    removed: tuple[int, ...],
    *,
    extra_wall: float = 0.0,
    peak_vram: int | None = None,
) -> StudyMeasurement:
    import torch

    started = time.perf_counter()
    model.to("cuda")
    device_batch = {name: value.to("cuda") for name, value in encoded.items()}
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        loss = float(model(**device_batch, labels=device_batch["input_ids"]).loss.item())
        for _ in range(2):
            model(**device_batch)
        torch.cuda.synchronize()
        samples = []
        for _ in range(7):
            begin = time.perf_counter()
            model(**device_batch)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - begin)
    measured_wall = time.perf_counter() - started
    parameters = tuple(model.parameters())
    size = sum(int(item.numel() * item.element_size()) for item in parameters)
    cuda_peak = max(torch.cuda.max_memory_allocated(), peak_vram or 0)
    return StudyMeasurement(
        measurement_id,
        StudyBackend.HUGGING_FACE,
        removed,
        loss,
        statistics.median(samples),
        size,
        sum(int(item.numel()) for item in parameters),
        measured_wall + extra_wall,
        measured_wall + extra_wall,
        0.0,
        None,
        cuda_peak,
        None,
    )


def _hf_study(
    args: argparse.Namespace, tokenizer: Any
) -> tuple[BackendStudy, Any, tuple[int, ...], dict[str, object]]:
    import torch

    encoded = tokenizer(_TEXT, return_tensors="pt", truncation=True, max_length=64)
    goals = SearchGoals(args.max_quality_increase, args.min_latency_gain, args.min_size_gain)
    measurements: list[StudyMeasurement] = []
    baseline = _load(args.model, args.revision)
    measurements.append(_hf_measure("measurement_hf_baseline", baseline, encoded, ()))
    del baseline
    torch.cuda.empty_cache()

    generation_one: list[StudyMeasurement] = []
    for channel in args.channels:
        started = time.perf_counter()
        candidate = _load(args.model, args.revision)
        remove_huggingface_mlp_channels(candidate, (channel,))
        mutation_wall = time.perf_counter() - started
        item = _hf_measure(
            f"measurement_hf_g1_c{channel}",
            candidate,
            encoded,
            (channel,),
            extra_wall=mutation_wall,
        )
        generation_one.append(item)
        measurements.append(item)
        del candidate
        torch.cuda.empty_cache()
    selected_one = min(generation_one, key=lambda item: (item.quality_value, item.measurement_id))
    selected_channel = selected_one.removed_channels[0]

    generation_two: list[StudyMeasurement] = []
    for channel in args.channels:
        if channel == selected_channel:
            continue
        removed = tuple(sorted((selected_channel, channel)))
        started = time.perf_counter()
        candidate = _load(args.model, args.revision)
        remove_huggingface_mlp_channels(candidate, removed)
        mutation_wall = time.perf_counter() - started
        item = _hf_measure(
            f"measurement_hf_g2_c{removed[0]}_{removed[1]}",
            candidate,
            encoded,
            removed,
            extra_wall=mutation_wall,
        )
        generation_two.append(item)
        measurements.append(item)
        del candidate
        torch.cuda.empty_cache()
    no_repair = min(generation_two, key=lambda item: (item.quality_value, item.measurement_id))

    best_two = tuple(sorted(item.removed_channels[0] for item in sorted(
        generation_one, key=lambda item: (item.quality_value, item.measurement_id)
    )[:2]))
    one_shot = next((item for item in generation_two if item.removed_channels == best_two), None)
    if one_shot is None:
        started = time.perf_counter()
        candidate = _load(args.model, args.revision)
        remove_huggingface_mlp_channels(candidate, best_two)
        one_shot = _hf_measure(
            f"measurement_hf_one_shot_c{best_two[0]}_{best_two[1]}",
            candidate,
            encoded,
            best_two,
            extra_wall=time.perf_counter() - started,
        )
        measurements.append(one_shot)
        del candidate
        torch.cuda.empty_cache()

    repaired_model = _load(args.model, args.revision)
    remove_huggingface_mlp_channels(repaired_model, no_repair.removed_channels)
    repaired_model.to("cuda")
    example = {**encoded, "labels": encoded["input_ids"].clone()}
    repair = run_short_finetune_repair(
        repaired_model,
        (example,),
        (example,),
        ShortFineTuneConfig(
            parameter_mode=FineTuneParameterMode.SELECTED,
            parameter_names=("model.norm.weight",),
            learning_rate=1e-3,
            max_steps=2,
            validation_patience=2,
            max_trainable_parameters=1_000,
            seed=args.seed,
        ),
        source_checkpoint_id=f"checkpoint_{args.revision}",
        candidate_parent_checkpoint_id=f"checkpoint_{no_repair.measurement_id}",
    )
    repaired = _hf_measure(
        "measurement_hf_repair",
        repaired_model,
        encoded,
        no_repair.removed_channels,
        extra_wall=repair.wall_seconds,
        peak_vram=repair.peak_cuda_allocated_bytes,
    )
    measurements.append(repaired)
    generations = (
        SearchGeneration(
            1,
            measurements[0].measurement_id,
            tuple(sorted(item.measurement_id for item in generation_one)),
            selected_one.measurement_id,
        ),
        SearchGeneration(
            2,
            selected_one.measurement_id,
            tuple(sorted(item.measurement_id for item in generation_two)),
            no_repair.measurement_id,
        ),
    )
    outcomes = (
        compare_arm(StudyArm.NO_REPAIR, measurements[0], no_repair, goals),
        compare_arm(StudyArm.REPAIR, measurements[0], repaired, goals),
        compare_arm(StudyArm.ONE_SHOT, measurements[0], one_shot, goals),
    )
    return (
        BackendStudy(
            StudyBackend.HUGGING_FACE,
            "causal_language_model_loss",
            goals,
            measurements[0].measurement_id,
            generations,
            outcomes,
            tuple(measurements),
        ),
        repaired_model,
        no_repair.removed_channels,
        repair.to_record(),
    )


def _native_mutate(
    source_path: Path, destination: Path, channels: tuple[int, ...]
) -> tuple[float, dict[str, object]]:
    started = time.perf_counter()
    disk = preflight_gguf_disk(
        destination,
        destination.parent,
        GGUFDiskEstimate(source_path.stat().st_size * 2, 0),
    )
    with open_gguf(source_path) as source:
        discovery = discover_gguf_components(source.container, family=ModelFamily.LLAMA)
        plan = plan_native_gguf_model_mlp_channel_removal(
            discovery, removed_channels=channels
        )
        result = execute_native_gguf_mlp_channel_removal(
            source, plan, destination, disk, CodecRegistry()
        )
    return time.perf_counter() - started, {
        "plan": {
            "family": plan.family.value,
            "architecture": plan.architecture,
            "layer_count": len(plan.layer_indices),
            "removed_channels": list(plan.removed_channels),
            "coupled_tensor_count": len(plan.coupled_tensor_names),
            "expected_parameter_delta": plan.expected_parameter_delta,
            "expected_storage_delta": plan.expected_storage_delta,
        },
        "result": {
            "path": str(result.write_result.path),
            "file_size": result.write_result.file_size,
            "sha256": result.write_result.sha256,
            "feed_forward_length": result.output_discovery.shape.feed_forward_length,
            "unchanged_tensor_count": len(result.unchanged_tensor_sha256),
            "requantization_errors": len(result.requantization_errors),
            "peak_row_working_bytes": result.peak_row_working_bytes,
        },
    }


def _native_measure(
    measurement_id: str,
    path: Path,
    removed: tuple[int, ...],
    baseline: Path,
    manifest: LlamaCppPerplexityManifest,
    perplexity_config: LlamaCppPerplexityConfig,
    throughput_config: LlamaCppThroughputConfig,
    *,
    mutation_wall: float = 0.0,
) -> tuple[StudyMeasurement, dict[str, object]]:
    started = time.perf_counter()
    quality = benchmark_gguf_perplexity(
        baseline, path, manifest, config=perplexity_config
    )
    throughput = benchmark_gguf_throughput(path, config=throughput_config)
    wall = time.perf_counter() - started + mutation_wall
    if not quality.successful or not throughput.successful:
        raise RuntimeError(
            f"native measurement failed: {quality.candidate.failure_reason}; "
            f"{throughput.failure_reason}"
        )
    assert quality.candidate.perplexity is not None
    assert throughput.generation is not None
    assert throughput.environment is not None
    latency = statistics.median(throughput.generation.latency_samples_ns) / 1e9
    measurement = StudyMeasurement(
        measurement_id,
        StudyBackend.NATIVE_GGUF,
        removed,
        quality.candidate.perplexity,
        latency,
        path.stat().st_size,
        throughput.environment.model_parameters,
        wall,
        0.0,
        wall,
        throughput.peak_rss_bytes,
        throughput.peak_vram_bytes,
        _sha256(path),
    )
    return measurement, {
        "perplexity": quality.to_record(),
        "throughput": throughput.to_record(),
    }


def _native_study(
    args: argparse.Namespace,
    repaired_model: Any,
    tokenizer_assets: Path,
) -> tuple[BackendStudy, dict[str, object]]:
    goals = SearchGoals(args.max_quality_increase, args.min_latency_gain, args.min_size_gain)
    args.artifacts.mkdir(parents=True, exist_ok=True)
    manifest = LlamaCppPerplexityManifest(
        args.perplexity_text,
        _sha256(args.perplexity_text),
        1,
        "ModelSurgeon bounded v1 fixture",
        "fixture-v1",
        "all",
        args.model,
        args.revision,
    )
    perplexity_config = LlamaCppPerplexityConfig(
        executable=args.llama_perplexity,
        expected_revision=args.llama_revision,
        context_size=128,
        batch_size=128,
        microbatch_size=128,
        threads=args.threads,
        chunks=1,
        timeout_seconds=300,
    )
    throughput_config = LlamaCppThroughputConfig(
        executable=args.llama_bench,
        expected_revision=args.llama_revision,
        prompt_tokens=64,
        generation_tokens=16,
        batch_size=64,
        microbatch_size=64,
        threads=args.threads,
        repetitions=3,
        timeout_seconds=300,
    )
    measurements: list[StudyMeasurement] = []
    evidence: dict[str, object] = {"mutations": {}, "evaluations": {}}
    baseline, baseline_eval = _native_measure(
        "measurement_native_baseline",
        args.baseline_gguf,
        (),
        args.baseline_gguf,
        manifest,
        perplexity_config,
        throughput_config,
    )
    measurements.append(baseline)
    evidence["evaluations"][baseline.measurement_id] = baseline_eval  # type: ignore[index]

    generation_one: list[StudyMeasurement] = []
    paths: dict[tuple[int, ...], Path] = {}
    for channel in args.channels:
        removed = (channel,)
        path = args.artifacts / f"native-g1-c{channel}.gguf"
        mutation_wall, mutation = _native_mutate(args.baseline_gguf, path, removed)
        item, evaluation = _native_measure(
            f"measurement_native_g1_c{channel}", path, removed, args.baseline_gguf,
            manifest, perplexity_config, throughput_config,
            mutation_wall=mutation_wall,
        )
        paths[removed] = path
        generation_one.append(item)
        measurements.append(item)
        evidence["mutations"][item.measurement_id] = mutation  # type: ignore[index]
        evidence["evaluations"][item.measurement_id] = evaluation  # type: ignore[index]
    selected_one = min(generation_one, key=lambda item: (item.quality_value, item.measurement_id))
    selected_channel = selected_one.removed_channels[0]

    generation_two: list[StudyMeasurement] = []
    for original_channel in args.channels:
        if original_channel == selected_channel:
            continue
        removed_pair = (
            min(selected_channel, original_channel),
            max(selected_channel, original_channel),
        )
        current_channel = original_channel - int(selected_channel < original_channel)
        path = args.artifacts / f"native-g2-c{removed_pair[0]}-{removed_pair[1]}.gguf"
        mutation_wall, mutation = _native_mutate(
            paths[(selected_channel,)], path, (current_channel,)
        )
        item, evaluation = _native_measure(
            f"measurement_native_g2_c{removed_pair[0]}_{removed_pair[1]}",
            path,
            removed_pair,
            args.baseline_gguf, manifest, perplexity_config, throughput_config,
            mutation_wall=mutation_wall,
        )
        paths[removed_pair] = path
        generation_two.append(item)
        measurements.append(item)
        evidence["mutations"][item.measurement_id] = mutation  # type: ignore[index]
        evidence["evaluations"][item.measurement_id] = evaluation  # type: ignore[index]
    no_repair = min(generation_two, key=lambda item: (item.quality_value, item.measurement_id))

    ranked = sorted(generation_one, key=lambda item: (item.quality_value, item.measurement_id))
    best_two = tuple(sorted((ranked[0].removed_channels[0], ranked[1].removed_channels[0])))
    one_shot = next((item for item in generation_two if item.removed_channels == best_two), None)
    if one_shot is None:
        path = args.artifacts / f"native-one-shot-c{best_two[0]}-{best_two[1]}.gguf"
        mutation_wall, mutation = _native_mutate(args.baseline_gguf, path, best_two)
        one_shot, evaluation = _native_measure(
            "measurement_native_one_shot", path, best_two, args.baseline_gguf,
            manifest, perplexity_config, throughput_config,
            mutation_wall=mutation_wall,
        )
        measurements.append(one_shot)
        evidence["mutations"][one_shot.measurement_id] = mutation  # type: ignore[index]
        evidence["evaluations"][one_shot.measurement_id] = evaluation  # type: ignore[index]

    repaired_path = args.artifacts / "native-repaired-f16.gguf"
    export_started = time.perf_counter()
    export_model(repaired_model.to("cpu"), tokenizer_assets, repaired_path)
    export_wall = time.perf_counter() - export_started
    repaired, evaluation = _native_measure(
        "measurement_native_repair", repaired_path, no_repair.removed_channels,
        args.baseline_gguf, manifest, perplexity_config, throughput_config,
        mutation_wall=export_wall,
    )
    measurements.append(repaired)
    evidence["evaluations"][repaired.measurement_id] = evaluation  # type: ignore[index]
    evidence["repair_deployment"] = {
        "method": "bounded HF short fine-tune followed by F16 GGUF export",
        "export_wall_seconds": export_wall,
    }
    generations = (
        SearchGeneration(
            1,
            baseline.measurement_id,
            tuple(sorted(item.measurement_id for item in generation_one)),
            selected_one.measurement_id,
        ),
        SearchGeneration(
            2,
            selected_one.measurement_id,
            tuple(sorted(item.measurement_id for item in generation_two)),
            no_repair.measurement_id,
        ),
    )
    outcomes = (
        compare_arm(StudyArm.NO_REPAIR, baseline, no_repair, goals),
        compare_arm(StudyArm.REPAIR, baseline, repaired, goals),
        compare_arm(StudyArm.ONE_SHOT, baseline, one_shot, goals),
    )
    return BackendStudy(
        StudyBackend.NATIVE_GGUF, "llama_cpp_perplexity", goals,
        baseline.measurement_id, generations, outcomes, tuple(measurements)
    ), evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    parser.add_argument("--revision", default="93efa2f097d58c2a74874c7e644dbc9b0cee75a2")
    parser.add_argument("--baseline-gguf", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--perplexity-text",
        type=Path,
        default=Path("tests/fixtures/llama_cpp_perplexity_v1.txt"),
    )
    parser.add_argument(
        "--llama-perplexity",
        type=Path,
        default=Path("D:/llamacpp/bin/llama-perplexity.exe"),
    )
    parser.add_argument("--llama-bench", type=Path, default=Path("D:/llamacpp/bin/llama-bench.exe"))
    parser.add_argument("--llama-revision", default="d7fa69b7d")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--channels", type=int, nargs=3, default=(0, 1, 2))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-quality-increase", type=float, default=0.05)
    parser.add_argument("--min-latency-gain", type=float, default=0.0)
    parser.add_argument("--min-size-gain", type=float, default=0.0001)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, local_files_only=True
    )
    tokenizer_assets = Path(tokenizer.name_or_path)
    if not tokenizer_assets.is_dir():
        from huggingface_hub import snapshot_download

        tokenizer_assets = Path(snapshot_download(
            args.model, revision=args.revision, local_files_only=True
        ))
    hf, repaired_model, _, repair_record = _hf_study(args, tokenizer)
    native, native_evidence = _native_study(args, repaired_model, tokenizer_assets)
    study = IterativeSearchStudy(args.seed, (hf, native))
    record = {
        "record_type": "v0.9_iterative_search_study",
        "version": "1",
        "model": {"identifier": args.model, "revision": args.revision},
        "protocol": {
            "candidate_channels": list(args.channels),
            "generation_budget": 2,
            "selection": "lowest quality metric with measurement-id tie break",
            "repair": "selected model.norm.weight, two FP32 steps",
            "native_runtime_revision": args.llama_revision,
        },
        "study": study.to_record(),
        "repair": repair_record,
        "native_evidence": native_evidence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_identity_json(record) + "\n", encoding="utf-8")
    print(json.dumps(record["study"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
