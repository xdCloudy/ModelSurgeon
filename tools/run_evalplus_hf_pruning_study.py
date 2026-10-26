"""Compare a pinned HF coder model with a physically MLP-pruned candidate on HumanEval+."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import statistics
import tempfile
import time
import urllib.request
import zipfile
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from modelsurgeon.adapters.huggingface.loader import (
    HuggingFaceDType,
    HuggingFaceLoadRequest,
    load_causal_lm,
)
from modelsurgeon.adapters.huggingface.physical_mlp import remove_huggingface_mlp_channels
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.utils.subprocesses import run_bounded_process, sanitized_subprocess_environment

MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
MODEL_REVISION = "ea3f2471cf1b1f0db85067f1ef93848e38e88c25"
EVALPLUS_REVISION = "e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2"
EVALPLUS_VERSION = "0.3.1"
EVALPLUS_IMAGE_REVISION_LABEL = "org.modelsurgeon.evalplus.revision"
HUMANEVAL_PLUS_VERSION = "v0.1.10"
HUMANEVAL_PLUS_RELEASE_COMMIT = "200defce9e3429d28ca215b6dd061c0f7f31c18b"
EVALPLUS_IMAGE_DEFAULT = f"modelsurgeon-evalplus:{EVALPLUS_REVISION[:7]}"
REPORT_MARKER = "MODELSURGEON_EVALPLUS_REPORT="
WORKSPACE_MANIFEST = ".modelsurgeon-evalplus-study.json"
WIKITEXT_URL = (
    "https://huggingface.co/datasets/ggml-org/ci/resolve/"
    "927b3642933080f1b0e811e2f916e14c292992f9/wikitext-2-raw-v1.zip"
)
WIKITEXT_SHA256 = "ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11"
WIKITEXT_MEMBER = "wikitext-2-raw/wiki.train.raw"
WIKITEXT_ARCHIVE_MAX_BYTES = 5_000_000
WIKITEXT_TRAIN_MAX_BYTES = 12_000_000


def _workspace_identity(args: argparse.Namespace) -> dict[str, object]:
    channel_scorer = getattr(args, "channel_scorer", "magnitude")
    calibration_tokens = getattr(args, "calibration_tokens", 2048)
    return {
        "schema_version": 4,
        "model": args.model,
        "model_revision": args.model_revision,
        "remove_channels": args.remove_channels,
        "max_new_tokens": args.max_new_tokens,
        "evalplus_image": args.evalplus_image,
        "evalplus_revision": EVALPLUS_REVISION,
        "dataset_version": HUMANEVAL_PLUS_VERSION,
        "stop_criteria_policy": "restore-after-each-task-v1",
        "runtime_measurement_policy": "cuda-synchronized-generate-wall-time-v1",
        "channel_scorer": channel_scorer,
        "calibration_tokens": calibration_tokens,
        "calibration_dataset_sha256": (
            WIKITEXT_SHA256 if channel_scorer == "activation_weighted" else None
        ),
    }


def _prepare_workspace(path: Path, args: argparse.Namespace) -> Path:
    """Create or validate a persistent workspace before resuming any generated samples."""
    path.mkdir(parents=True, exist_ok=True)
    manifest_path = path / WORKSPACE_MANIFEST
    expected = _workspace_identity(args)
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("persistent study workspace has an unreadable manifest") from exc
        if existing != expected:
            raise RuntimeError("persistent study workspace settings do not match this run")
    else:
        if any(path.iterdir()):
            raise RuntimeError(
                "refusing to resume a non-empty workspace without a study manifest"
            )
        manifest_path.write_text(
            canonical_identity_json(expected) + "\n", encoding="utf-8"
        )
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_manifest(path: Path) -> dict[str, object]:
    records = [
        {
            "path": item.relative_to(path).as_posix(),
            "size_bytes": item.stat().st_size,
            "sha256": _sha256_file(item),
        }
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]
    payload = canonical_identity_json(records).encode("utf-8")
    return {
        "file_count": len(records),
        "file_bytes": sum(int(item["size_bytes"]) for item in records),
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
    }


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


def _activation_weighted_channels(
    model: Any,
    count: int,
    activation_rms_by_layer: tuple[Any, ...],
) -> tuple[int, ...]:
    """Rank coupled MLP channels using layer-mean weight and activation scales."""
    import torch

    config = model.config
    width = int(config.intermediate_size)
    layers = int(config.num_hidden_layers)
    if not 0 < count < width:
        raise ValueError(f"channels must be between 1 and {width - 1}")
    if len(activation_rms_by_layer) != layers:
        raise ValueError("activation calibration must provide one RMS vector per layer")
    modules = dict(model.named_modules())
    layer_weight_norms: list[Any] = []
    for layer in range(layers):
        mlp = modules.get(f"model.layers.{layer}.mlp")
        if mlp is None:
            raise RuntimeError(f"unsupported MLP path at layer {layer}")
        gate = mlp.gate_proj.weight.detach().float().cpu()
        up = mlp.up_proj.weight.detach().float().cpu()
        down = mlp.down_proj.weight.detach().float().cpu()
        if gate.shape[0] != width or up.shape[0] != width or down.shape[1] != width:
            raise RuntimeError(f"unsupported MLP projection shape at layer {layer}")
        layer_weight_norms.append(
            (gate.square().sum(dim=1) + up.square().sum(dim=1) + down.square().sum(dim=0)).sqrt()
        )
    activation = torch.stack(
        [torch.as_tensor(value, dtype=torch.float64).cpu() for value in activation_rms_by_layer]
    )
    weights = torch.stack([value.to(torch.float64) for value in layer_weight_norms])
    if activation.shape != (layers, width) or not torch.isfinite(activation).all():
        raise ValueError("activation RMS calibration has an invalid shape or non-finite values")
    scores = weights.mean(dim=0) * activation.mean(dim=0)
    if not torch.isfinite(scores).all():
        raise ValueError("activation-weighted channel scores are non-finite")
    selected = sorted(range(width), key=lambda index: (float(scores[index]), index))[:count]
    return tuple(sorted(selected))


def _ensure_wikitext_train(root: Path) -> tuple[Path, dict[str, object]]:
    """Fetch the pinned WikiText-2 train split with bounded, hash-checked extraction."""
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / "wikitext-2-raw-v1.zip"
    train_path = root / "wiki.train.raw"
    if not archive_path.exists():
        request = urllib.request.Request(WIKITEXT_URL, headers={"User-Agent": "ModelSurgeon/1"})
        descriptor, temporary_name = tempfile.mkstemp(prefix=".wikitext-", suffix=".zip", dir=root)
        try:
            with (
                os.fdopen(descriptor, "wb") as target,
                urllib.request.urlopen(request, timeout=60) as response,
            ):
                total = 0
                while chunk := response.read(64 * 1024):
                    total += len(chunk)
                    if total > WIKITEXT_ARCHIVE_MAX_BYTES:
                        raise RuntimeError("WikiText calibration archive exceeds its size limit")
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
            if _sha256_file(Path(temporary_name)) != WIKITEXT_SHA256:
                raise RuntimeError("WikiText calibration archive digest does not match its pin")
            os.replace(temporary_name, archive_path)
        except BaseException:
            Path(temporary_name).unlink(missing_ok=True)
            raise
    if _sha256_file(archive_path) != WIKITEXT_SHA256:
        raise RuntimeError("cached WikiText calibration archive digest does not match its pin")
    with zipfile.ZipFile(archive_path) as archive:
        matches = [entry for entry in archive.infolist() if entry.filename == WIKITEXT_MEMBER]
        if len(matches) != 1 or matches[0].file_size > WIKITEXT_TRAIN_MAX_BYTES:
            raise RuntimeError("WikiText archive has an invalid train member")
        if not train_path.exists():
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".wiki-train-", suffix=".tmp", dir=root
            )
            try:
                with os.fdopen(descriptor, "wb") as target, archive.open(matches[0]) as source:
                    total = 0
                    while chunk := source.read(64 * 1024):
                        total += len(chunk)
                        if total > WIKITEXT_TRAIN_MAX_BYTES:
                            raise RuntimeError("WikiText train member exceeds its size limit")
                        target.write(chunk)
                    target.flush()
                    os.fsync(target.fileno())
                os.replace(temporary_name, train_path)
            except BaseException:
                Path(temporary_name).unlink(missing_ok=True)
                raise
    return train_path, {
        "identifier": "WikiText-2 raw v1 train split",
        "url": WIKITEXT_URL,
        "archive_sha256": WIKITEXT_SHA256,
        "member": WIKITEXT_MEMBER,
        "member_sha256": _sha256_file(train_path),
        "license": "CC-BY-SA-4.0",
        "archive_revision": "ggml-org/ci@927b3642933080f1b0e811e2f916e14c292992f9",
    }


def _calibrate_down_projection_rms(
    snapshot: Path,
    train_path: Path,
    *,
    revision: str,
    token_budget: int,
) -> tuple[tuple[Any, ...], dict[str, object]]:
    """Measure gated-MLP intermediate RMS on a pinned, non-evaluation text split."""
    import torch
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    text = train_path.read_text(encoding="utf-8")
    token_ids = tokenizer.encode(text, add_special_tokens=False)[:token_budget]
    if len(token_ids) < token_budget:
        raise RuntimeError("WikiText calibration split contains fewer tokens than requested")
    model = load_causal_lm(
        HuggingFaceLoadRequest(
            str(snapshot),
            revision=revision,
            device_map="cpu",
            dtype=HuggingFaceDType.FLOAT16,
            local_files_only=True,
        )
    ).model
    model.to(device="cuda")
    model.eval()
    layers = int(model.config.num_hidden_layers)
    width = int(model.config.intermediate_size)
    sums = [torch.zeros(width, dtype=torch.float64, device="cpu") for _ in range(layers)]
    counts = [0 for _ in range(layers)]
    handles = []
    modules = dict(model.named_modules())
    for layer in range(layers):
        module = modules.get(f"model.layers.{layer}.mlp.down_proj")
        if module is None:
            raise RuntimeError(f"unsupported MLP down projection at layer {layer}")

        def collect(_module: Any, inputs: tuple[Any, ...], *, layer_index: int = layer) -> None:
            if not inputs:
                raise RuntimeError("down projection hook received no activation tensor")
            activation = inputs[0].detach().float().reshape(-1, width).cpu().double()
            sums[layer_index].add_(activation.square().sum(dim=0))
            counts[layer_index] += int(activation.shape[0])

        handles.append(module.register_forward_pre_hook(collect))
    try:
        with torch.inference_mode():
            for start in range(0, token_budget, 128):
                chunk = token_ids[start : start + 128]
                input_ids = torch.tensor([chunk], dtype=torch.long, device="cuda")
                model(input_ids=input_ids)
        if any(count == 0 for count in counts):
            raise RuntimeError("WikiText calibration did not reach every MLP layer")
        rms = tuple((total / count).sqrt() for total, count in zip(sums, counts, strict=True))
    finally:
        for handle in handles:
            handle.remove()
        model.to(device="cpu")
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return rms, {
        "tokens": len(token_ids),
        "token_ids_sha256": hashlib.sha256(
            b"".join(int(token).to_bytes(8, "little", signed=False) for token in token_ids)
        ).hexdigest(),
        "sequence_length": 128,
        "activation_measure": "post-gated-MLP down-projection input RMS per channel",
        "activation_samples_per_layer": counts,
        "model_revision": revision,
    }


def _docker_wrapper(sample_filename: str) -> bytes:
    if Path(sample_filename).name != sample_filename or not sample_filename.endswith(".jsonl"):
        raise ValueError("EvalPlus sample name must be a plain JSONL filename")
    sample_name = json.dumps(sample_filename)
    script = f'''import hashlib, json, shutil
from pathlib import Path
from evalplus.data.humaneval import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.evaluate import evaluate

sample_path = Path("/input") / {sample_name}
dataset_path = Path("/input/HumanEvalPlus.jsonl")
evaluation_sample_path = Path("/tmp") / sample_path.name
shutil.copyfile(sample_path, evaluation_sample_path)
problems = get_human_eval_plus(version="default")
evaluate(
    dataset="humaneval",
    samples=str(evaluation_sample_path),
    parallel=2,
    test_details=False,
)
result_path = evaluation_sample_path.with_name(
    evaluation_sample_path.stem + "_eval_results.json"
)
raw = json.loads(result_path.read_text(encoding="utf-8"))
per_task = {{}}
for task_id, rows in raw.get("eval", {{}}).items():
    if len(rows) != 1:
        raise RuntimeError("HumanEval+ study expects one greedy sample per task")
    per_task[task_id] = {{
        "base_status": rows[0].get("base_status"),
        "plus_status": rows[0].get("plus_status"),
    }}
for task_id in problems:
    if task_id not in per_task:
        raise RuntimeError("EvalPlus result is missing a task")
base_passes = sum(row["base_status"] == "pass" for row in per_task.values())
plus_passes = sum(row["plus_status"] == "pass" for row in per_task.values())
payload = {{
    "schema_version": 1,
    "evalplus_version": "0.3.1",
    "evalplus_dataset_hash": get_human_eval_plus_hash(version="default"),
    "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
    "task_count": len(problems),
    "base_passes": base_passes,
    "plus_passes": plus_passes,
    "base_pass_at_1": base_passes / len(problems),
    "plus_pass_at_1": plus_passes / len(problems),
    "per_task": per_task,
}}
print("{REPORT_MARKER}" + json.dumps(payload, sort_keys=True, separators=(",", ":")))
'''
    return script.encode("utf-8")


def _inspect_evalplus_image(docker: str, image: str) -> dict[str, str]:
    command = [
        docker,
        "image",
        "inspect",
        "--format",
        f'{{{{ index .Config.Labels "{EVALPLUS_IMAGE_REVISION_LABEL}" }}}}',
        image,
    ]
    completed = run_bounded_process(
        command,
        b"",
        timeout_seconds=30,
        stdout_limit=4096,
        stderr_limit=4096,
        env=sanitized_subprocess_environment(),
    )
    if completed.returncode != 0 or completed.timed_out or completed.output_exceeded:
        raise RuntimeError("could not inspect the pinned EvalPlus Docker image")
    revision = completed.stdout.decode("utf-8", errors="strict").strip()
    if revision != EVALPLUS_REVISION:
        raise RuntimeError(
            f"EvalPlus image has revision {revision!r}; expected {EVALPLUS_REVISION}"
        )
    id_result = run_bounded_process(
        [docker, "image", "inspect", "--format", "{{.Id}}", image],
        b"",
        timeout_seconds=30,
        stdout_limit=4096,
        stderr_limit=4096,
        env=sanitized_subprocess_environment(),
    )
    if id_result.returncode != 0 or id_result.timed_out or id_result.output_exceeded:
        raise RuntimeError("could not read the EvalPlus image digest")
    return {"revision": revision, "image_id": id_result.stdout.decode().strip()}


def _evaluate_in_docker(
    samples: Path,
    dataset_path: Path,
    *,
    docker: str,
    image: str,
    image_identity: dict[str, str],
    timeout_seconds: float,
) -> dict[str, object]:
    input_dir = samples.parent
    mounted_dataset = input_dir / dataset_path.name
    if not mounted_dataset.is_file() or _sha256_file(mounted_dataset) != _sha256_file(dataset_path):
        raise RuntimeError("the per-arm sandbox input does not contain the pinned HumanEval+ data")
    command = [
        docker,
        "run",
        "--rm",
        "-i",
        "--platform",
        "linux/amd64",
        "--network",
        "none",
        "--read-only",
        "--user",
        "65534:65534",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--pids-limit",
        "32",
        "--memory",
        "3g",
        "--cpus",
        "2",
        "--ulimit",
        "core=0",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=512m,mode=1777",
        "--mount",
        f"type=bind,source={input_dir.resolve()},target=/input,readonly",
        "--env",
        "HUMANEVAL_OVERRIDE_PATH=/input/HumanEvalPlus.jsonl",
        "--env",
        "EVALPLUS_MAX_MEMORY_BYTES=1073741824",
        "--env",
        "HOME=/tmp",
        "--env",
        "XDG_CACHE_HOME=/tmp/.cache",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        image,
        "python",
        "-",
    ]
    completed = run_bounded_process(
        command,
        _docker_wrapper(samples.name),
        timeout_seconds=timeout_seconds,
        stdout_limit=16 * 1024 * 1024,
        stderr_limit=1024 * 1024,
        total_output_limit=17 * 1024 * 1024,
        env=sanitized_subprocess_environment(),
    )
    if completed.timed_out:
        raise RuntimeError("sandboxed EvalPlus evaluation exceeded its wall-time budget")
    if completed.output_exceeded:
        raise RuntimeError("sandboxed EvalPlus evaluation exceeded its output budget")
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace")[-1000:]
        raise RuntimeError(f"sandboxed EvalPlus evaluation failed: {detail}")
    lines = completed.stdout.decode("utf-8", errors="strict").splitlines()
    records = [line[len(REPORT_MARKER) :] for line in lines if line.startswith(REPORT_MARKER)]
    if len(records) != 1:
        raise RuntimeError("sandboxed EvalPlus evaluation returned no unique report")
    result = json.loads(records[0])
    if result.get("schema_version") != 1 or result.get("evalplus_version") != EVALPLUS_VERSION:
        raise RuntimeError("sandboxed EvalPlus report has an unsupported schema or version")
    task_count = result.get("task_count")
    per_task = result.get("per_task")
    if not isinstance(task_count, int) or task_count != 164 or not isinstance(per_task, dict):
        raise RuntimeError("sandboxed EvalPlus report has an incomplete task set")
    if len(per_task) != task_count:
        raise RuntimeError("sandboxed EvalPlus report has duplicate or missing task results")
    for metric in ("base_pass_at_1", "plus_pass_at_1"):
        value = result.get(metric)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise RuntimeError(f"sandboxed EvalPlus report has an invalid {metric}")
    status_measures = (
        ("base_status_counts", "base_status"),
        ("plus_status_counts", "plus_status"),
    )
    counts = {
        measure: dict(Counter(str(row.get(key)) for row in per_task.values()))
        for measure, key in status_measures
    }
    return {
        "measurement_authority": "evalplus_0.3.1_sandboxed_physical_execution",
        "docker_image": image,
        "docker_image_id": image_identity["image_id"],
        "evalplus_source_revision": image_identity["revision"],
        "evalplus_dataset_hash": result["evalplus_dataset_hash"],
        "dataset_sha256": result["dataset_sha256"],
        "task_count": task_count,
        "base_passes": result["base_passes"],
        "plus_passes": result["plus_passes"],
        "base_pass_at_1": result["base_pass_at_1"],
        "plus_pass_at_1": result["plus_pass_at_1"],
        **counts,
        "per_task": per_task,
    }


def _build_pruned_candidate(
    snapshot: Path,
    destination: Path,
    *,
    revision: str,
    remove_channels: int,
    channel_scorer: str = "magnitude",
    activation_rms_by_layer: tuple[Any, ...] | None = None,
) -> dict[str, object]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    loaded = load_causal_lm(
        HuggingFaceLoadRequest(
            str(snapshot),
            revision=revision,
            device_map="cpu",
            dtype=HuggingFaceDType.FLOAT16,
            local_files_only=True,
        )
    )
    model = loaded.model
    old_width = int(model.config.intermediate_size)
    original_parameters = _parameter_metrics(model)
    if channel_scorer == "magnitude":
        removed = _low_magnitude_channels(model, remove_channels)
    elif channel_scorer == "activation_weighted" and activation_rms_by_layer is not None:
        removed = _activation_weighted_channels(model, remove_channels, activation_rms_by_layer)
    else:
        raise ValueError("activation-weighted scoring requires calibrated layer RMS vectors")
    surgery = remove_huggingface_mlp_channels(model, removed)
    model.to(device="cpu", dtype=torch.float16)
    model.save_pretrained(destination, safe_serialization=True, max_shard_size="1GB")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    tokenizer.save_pretrained(destination)
    pruned_parameters = _parameter_metrics(model)
    if pruned_parameters["parameters"] >= original_parameters["parameters"]:
        raise RuntimeError("physical MLP surgery did not reduce the parameter count")
    del loaded, model
    gc.collect()

    reloaded = AutoModelForCausalLM.from_pretrained(
        destination,
        local_files_only=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    expected_width = old_width - remove_channels
    if int(reloaded.config.intermediate_size) != expected_width:
        raise RuntimeError(
            f"reloaded model width {reloaded.config.intermediate_size} != {expected_width}"
        )
    if _parameter_metrics(reloaded) != pruned_parameters:
        raise RuntimeError("reloaded candidate parameter metrics do not match the saved model")
    del reloaded
    gc.collect()
    return {
        "original_parameters": original_parameters,
        "parameters": pruned_parameters,
        "removed_channels": list(removed),
        "surgery": surgery.to_record(),
        "artifact": _artifact_manifest(destination),
        "reloadable": True,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    # Avoid the optional hf_xet extension: standard Hub downloads work on this Windows setup.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    import torch
    from evalplus.data import get_human_eval_plus
    from evalplus.data.humaneval import HUMANEVAL_PLUS_VERSION as installed_dataset_version
    from evalplus.data.utils import get_dataset_metadata, make_cache
    from huggingface_hub import snapshot_download

    if installed_dataset_version != HUMANEVAL_PLUS_VERSION:
        raise RuntimeError("installed EvalPlus dataset version differs from this study protocol")
    if importlib.metadata.version("evalplus") != EVALPLUS_VERSION:
        raise RuntimeError(f"this study requires EvalPlus {EVALPLUS_VERSION}")
    if not torch.cuda.is_available():
        raise RuntimeError("this physical study requires CUDA")
    docker = shutil.which("docker")
    if docker is None:
        raise RuntimeError("Docker is required for sandboxed execution of generated code")
    image_identity = _inspect_evalplus_image(docker, args.evalplus_image)

    snapshot = Path(
        snapshot_download(
            repo_id=args.model,
            revision=args.model_revision,
        )
    ).resolve(strict=True)
    dataset_url, dataset_cache_path = get_dataset_metadata(
        "HumanEvalPlus", HUMANEVAL_PLUS_VERSION, False
    )
    make_cache(dataset_url, dataset_cache_path)
    dataset_path = Path(dataset_cache_path).resolve(strict=True)
    tasks = get_human_eval_plus(version="default")
    if len(tasks) != 164:
        raise RuntimeError(f"expected all 164 HumanEval+ tasks, found {len(tasks)}")
    dataset_record = {
        "identifier": "HumanEval+",
        "version": HUMANEVAL_PLUS_VERSION,
        "release_commit": HUMANEVAL_PLUS_RELEASE_COMMIT,
        "license": "Apache-2.0",
        "url": dataset_url,
        "sha256": _sha256_file(dataset_path),
        "example_count": len(tasks),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite study report: {args.output}")
    if args.work_dir is None:
        workspace = tempfile.TemporaryDirectory(
            prefix="modelsurgeon-evalplus-pruning-", dir=args.output.parent
        )
    else:
        workspace = nullcontext(_prepare_workspace(args.work_dir, args))
    with workspace as workspace_path:
        root = Path(workspace_path)
        calibration_record: dict[str, object] | None = None
        activation_rms_by_layer = None
        if args.channel_scorer == "activation_weighted":
            train_path, dataset_identity = _ensure_wikitext_train(root / "calibration")
            activation_rms_by_layer, calibration_record = _calibrate_down_projection_rms(
                snapshot,
                train_path,
                revision=args.model_revision,
                token_budget=args.calibration_tokens,
            )
            calibration_record = {**dataset_identity, **calibration_record}
        candidate_path = root / "pruned-candidate"
        candidate_path.mkdir(exist_ok=True)
        candidate_record = _build_pruned_candidate(
            snapshot,
            candidate_path,
            revision=args.model_revision,
            remove_channels=args.remove_channels,
            channel_scorer=args.channel_scorer,
            activation_rms_by_layer=activation_rms_by_layer,
        )
        dataset_input = root / "HumanEvalPlus.jsonl"
        shutil.copyfile(dataset_path, dataset_input)
        # Evaluation inputs share a read-only mount; each generated arm sees only its own samples.
        source_input = root / "source-input"
        candidate_input = root / "candidate-input"
        source_input.mkdir(exist_ok=True)
        candidate_input.mkdir(exist_ok=True)
        shutil.copyfile(dataset_input, source_input / dataset_input.name)
        shutil.copyfile(dataset_input, candidate_input / dataset_input.name)

        source_samples = source_input / "source_predictions.jsonl"
        candidate_samples = candidate_input / "candidate_predictions.jsonl"
        source_generation = _generate_predictions(
            snapshot, source_samples, tasks, max_new_tokens=args.max_new_tokens
        )
        candidate_generation = _generate_predictions(
            candidate_path, candidate_samples, tasks, max_new_tokens=args.max_new_tokens
        )
        # Restrict each read-only bind mount to its own outputs and the pinned test data.
        source_eval = _evaluate_in_docker(
            source_samples,
            dataset_input,
            docker=docker,
            image=args.evalplus_image,
            image_identity=image_identity,
            timeout_seconds=args.evalplus_timeout_seconds,
        )
        candidate_eval = _evaluate_in_docker(
            candidate_samples,
            dataset_input,
            docker=docker,
            image=args.evalplus_image,
            image_identity=image_identity,
            timeout_seconds=args.evalplus_timeout_seconds,
        )

    source_score = float(source_eval["plus_pass_at_1"])
    candidate_score = float(candidate_eval["plus_pass_at_1"])
    return {
        "schema_version": 3,
        "protocol": "physical-mlp-pruning-humanevalplus-pass-at-1-v3",
        "model": {
            "identifier": args.model,
            "revision": args.model_revision,
            "license": "Apache-2.0 (model card)",
            "source_snapshot": _artifact_manifest(snapshot),
        },
        "dataset": dataset_record,
        "method": {
            "name": (
                "activation-weighted-global-gated-MLP-channel-removal"
                if args.channel_scorer == "activation_weighted"
                else "global-lowest-magnitude-gated-MLP-channel-removal"
            ),
            "score": (
                "mean across layers of combined gate/up/down channel L2 norm times "
                "mean across layers of post-gated MLP activation RMS"
                if args.channel_scorer == "activation_weighted"
                else "sum across layers of gate-row L2 x up-row L2 x down-column L2"
            ),
            "scorer": args.channel_scorer,
            "calibration": calibration_record,
            **candidate_record,
        },
        "environment": {
            "python_version": __import__("sys").version.split()[0],
            "torch_version": str(torch.__version__),
            "cuda_runtime_version": torch.version.cuda,
            "transformers_version": importlib.metadata.version("transformers"),
            "evalplus_version": EVALPLUS_VERSION,
            "evalplus_source_revision": EVALPLUS_REVISION,
            "evalplus_license": "Apache-2.0",
            "evalplus_docker_image": args.evalplus_image,
            "evalplus_docker_image_id": image_identity["image_id"],
            "device": torch.cuda.get_device_name(0),
        },
        "run_parameters": {
            "channel_scorer": args.channel_scorer,
            "calibration_tokens": (
                args.calibration_tokens if args.channel_scorer == "activation_weighted" else None
            ),
            "calibration_corpus": "WikiText-2 raw v1 train split"
            if args.channel_scorer == "activation_weighted"
            else None,
            "remove_channels": args.remove_channels,
            "max_new_tokens": args.max_new_tokens,
            "prompt_mode": "EvalPlus base-prompt function completion",
            "greedy": True,
            "samples_per_task": 1,
            "execution_parallelism": 2,
            "runtime_measurement": (
                "per-task model.generate wall time, CUDA-synchronized before and after; "
                "includes prompt prefill and token generation, excludes model loading, "
                "tokenization, and EvalPlus sandbox execution"
            ),
            "sandbox_network": "disabled",
            "sandbox_memory_limit_bytes": 3 * 1024**3,
            "sandbox_cpu_limit": 2,
            "sandbox_wall_time_limit_seconds_per_arm": args.evalplus_timeout_seconds,
        },
        "arms": {
            "source": {
                "artifact": {
                    "file_bytes": sum(
                        int(record["size_bytes"])
                        for record in _artifact_file_records(snapshot)
                    ),
                    **_artifact_manifest(snapshot),
                },
                "generation": source_generation,
                "task_quality": source_eval,
            },
            "pruned": {
                "artifact": candidate_record["artifact"],
                "generation": candidate_generation,
                "task_quality": candidate_eval,
            },
        },
        "comparison": {
            "plus_pass_at_1_delta": candidate_score - source_score,
            "plus_pass_at_1_retention_ratio": (
                None if source_score <= 0 else candidate_score / source_score
            ),
            "retention_inconclusive": source_score <= 0,
            "generated_tokens_per_second_ratio": (
                float(candidate_generation["runtime"]["generated_tokens_per_second"])
                / float(source_generation["runtime"]["generated_tokens_per_second"])
                if float(source_generation["runtime"]["generated_tokens_per_second"]) > 0
                else None
            ),
            "median_generation_latency_delta_seconds": (
                float(candidate_generation["runtime"]["median_generation_latency_seconds"])
                - float(source_generation["runtime"]["median_generation_latency_seconds"])
            ),
            "p95_generation_latency_delta_seconds": (
                float(candidate_generation["runtime"]["p95_generation_latency_seconds"])
                - float(source_generation["runtime"]["p95_generation_latency_seconds"])
            ),
        },
        "limitations": [
            (
                "One code-specialized model revision, one magnitude-based MLP pruning severity, "
                "and one greedy sample per task."
            ),
            (
                "Runtime timing is one generation pass per arm; token throughput includes "
                "prompt prefill and is not a repeated decode-only latency benchmark."
            ),
            (
                "HumanEval+ is public and may be present in model training data; contamination "
                "is not ruled out."
            ),
            (
                "Pass@1 measures executable correctness, not coding quality on repository-scale "
                "tasks or inference speed."
            ),
            (
                "EvalPlus generated code runs only in local Docker with networking disabled, "
                "no capabilities, a read-only root, resource limits, and a read-only input mount."
            ),
        ],
    }


def _artifact_file_records(path: Path) -> list[dict[str, object]]:
    return [
        {
            "path": item.relative_to(path).as_posix(),
            "size_bytes": item.stat().st_size,
            "sha256": _sha256_file(item),
        }
        for item in sorted(path.rglob("*"))
        if item.is_file()
    ]


def _sample_digests(path: Path, tasks: dict[str, Any]) -> dict[str, object]:
    seen: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        task_id = row.get("task_id")
        solution = row.get("solution")
        if not isinstance(task_id, str) or task_id not in tasks or task_id in seen:
            raise RuntimeError(f"invalid or duplicate EvalPlus sample on line {line_number}")
        if not isinstance(solution, str) or not solution.strip():
            raise RuntimeError(f"empty EvalPlus solution on line {line_number}")
        seen[task_id] = hashlib.sha256(solution.encode("utf-8")).hexdigest()
    if set(seen) != set(tasks):
        raise RuntimeError("EvalPlus generated an incomplete task sample")
    return {
        "sample_count": len(seen),
        "sample_file_sha256": _sha256_file(path),
        "solution_sha256_by_task": dict(sorted(seen.items())),
    }


def _repair_prediction_checkpoint(path: Path) -> None:
    """Recover a valid or interrupted final JSONL record before EvalPlus resumes it."""
    if not path.is_file():
        return
    content = path.read_bytes()
    if not content or content.endswith(b"\n"):
        return
    last_newline = content.rfind(b"\n")
    final_record = content[last_newline + 1 :]
    try:
        json.loads(final_record.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        repaired = content[: last_newline + 1] if last_newline >= 0 else b""
    else:
        repaired = content + b"\n"

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(repaired)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _decoder_with_task_local_stopping_criteria(decoder_type: type) -> type:
    """Keep EvalPlus's stop-sequence wrapper local to each generated task.

    The pinned EvalPlus HF decoder installs a StopSequencer wrapper on the
    model's private Transformers hook and does not restore it. EvalPlus reuses
    the decoder across tasks, so without this adapter the wrappers accumulate.
    """

    class TaskLocalStoppingCriteriaDecoder(decoder_type):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self._modelsurgeon_base_get_stopping_criteria = (
                self.model._get_stopping_criteria
            )
            self._modelsurgeon_generation_metrics: list[dict[str, int | float]] = []

        def codegen(self, *args: object, **kwargs: object) -> object:
            self.model._get_stopping_criteria = (
                self._modelsurgeon_base_get_stopping_criteria
            )
            original_generate = getattr(self.model, "generate", None)
            had_instance_generate = "generate" in self.model.__dict__
            if callable(original_generate):
                import torch

                def measured_generate(*generate_args: object, **generate_kwargs: object) -> object:
                    input_ids = (
                        generate_args[0]
                        if generate_args
                        else generate_kwargs.get("inputs")
                    )
                    if input_ids is None or not hasattr(input_ids, "shape"):
                        raise RuntimeError("generation runtime measurement lacks input token IDs")
                    input_tokens = int(input_ids.shape[-1])
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    started_at = time.perf_counter()
                    output_ids = original_generate(*generate_args, **generate_kwargs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started_at
                    generated_per_sequence = max(
                        0, int(output_ids.shape[-1]) - input_tokens
                    )
                    sequence_count = (
                        int(output_ids.shape[0]) if len(output_ids.shape) > 1 else 1
                    )
                    self._modelsurgeon_generation_metrics.append(
                        {
                            "seconds": elapsed,
                            "prompt_tokens": input_tokens,
                            "generated_tokens": generated_per_sequence * sequence_count,
                        }
                    )
                    return output_ids

                self.model.generate = measured_generate
            try:
                return super().codegen(*args, **kwargs)
            finally:
                if callable(original_generate):
                    if had_instance_generate:
                        self.model.generate = original_generate
                    else:
                        del self.model.generate
                self.model._get_stopping_criteria = (
                    self._modelsurgeon_base_get_stopping_criteria
                )

    return TaskLocalStoppingCriteriaDecoder


def _summarize_generation_metrics(
    records: list[dict[str, int | float]],
) -> dict[str, object]:
    if not records:
        raise RuntimeError("EvalPlus generated no measurable model outputs")
    latencies = sorted(float(record["seconds"]) for record in records)
    total_seconds = sum(latencies)
    generated_tokens = sum(int(record["generated_tokens"]) for record in records)
    return {
        "measurement_scope": (
            "model.generate wall time, CUDA-synchronized before and after each call; "
            "includes prompt prefill and token generation, excludes model loading, "
            "tokenization, and EvalPlus sandbox execution"
        ),
        "generation_calls": len(records),
        "prompt_tokens": sum(int(record["prompt_tokens"]) for record in records),
        "generated_tokens": generated_tokens,
        "generate_wall_seconds": round(total_seconds, 6),
        "generated_tokens_per_second": round(
            generated_tokens / total_seconds if total_seconds > 0 else 0.0, 6
        ),
        "median_generation_latency_seconds": round(
            statistics.median(latencies), 6
        ),
        "p95_generation_latency_seconds": round(
            latencies[math.ceil(0.95 * len(latencies)) - 1], 6
        ),
    }


def _generate_predictions(
    model_path: Path,
    target_path: Path,
    tasks: dict[str, Any],
    *,
    max_new_tokens: int,
) -> dict[str, object]:
    import torch
    from evalplus.codegen import codegen
    from evalplus.provider.hf import HuggingFaceDecoder

    TaskLocalDecoder = _decoder_with_task_local_stopping_criteria(HuggingFaceDecoder)

    decoder = TaskLocalDecoder(
        name=str(model_path),
        dataset="humaneval",
        force_base_prompt=True,
        temperature=0.0,
        max_new_tokens=max_new_tokens,
        dtype="float16",
    )
    _repair_prediction_checkpoint(target_path)
    raw_path = target_path.with_name(f"{target_path.stem}.raw.jsonl")
    raw_path.unlink(missing_ok=True)
    codegen(
        target_path=str(target_path),
        model=decoder,
        dataset="humaneval",
        greedy=True,
        n_samples=1,
        resume=True,
    )
    runtime_records = decoder._modelsurgeon_generation_metrics
    if len(runtime_records) != len(tasks):
        raise RuntimeError(
            "generation timing did not cover every task; refusing to report partial runtime data"
        )
    generation_metrics = _summarize_generation_metrics(runtime_records)
    del decoder
    if raw_path.exists():
        raw_path.unlink()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not target_path.is_file():
        raise RuntimeError("EvalPlus code generation did not write its JSONL sample file")
    return {**_sample_digests(target_path, tasks), "runtime": generation_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="retain model artifacts and resumable EvalPlus predictions in this directory",
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--remove-channels", type=int, default=256)
    parser.add_argument(
        "--channel-scorer",
        choices=("magnitude", "activation_weighted"),
        default="magnitude",
        help="candidate-only scorer; activation calibration uses pinned WikiText-2 train text",
    )
    parser.add_argument("--calibration-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--evalplus-image", default=EVALPLUS_IMAGE_DEFAULT)
    parser.add_argument("--evalplus-timeout-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    if args.remove_channels <= 0 or args.max_new_tokens <= 0 or args.calibration_tokens <= 0:
        parser.error("remove-channels, max-new-tokens, and calibration-tokens must be positive")
    if not math.isfinite(args.evalplus_timeout_seconds) or args.evalplus_timeout_seconds <= 0:
        parser.error("evalplus-timeout-seconds must be finite and positive")
    if args.output.exists():
        parser.error(f"refusing to overwrite output: {args.output}")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_identity_json(result) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "source_plus_pass_at_1": result["arms"]["source"]["task_quality"]["plus_pass_at_1"],
                "pruned_plus_pass_at_1": result["arms"]["pruned"]["task_quality"]["plus_pass_at_1"],
                "artifact_size_bytes": result["arms"]["pruned"]["artifact"]["file_bytes"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
