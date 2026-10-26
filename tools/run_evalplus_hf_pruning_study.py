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
import tempfile
from collections import Counter
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


def _docker_wrapper(sample_filename: str) -> bytes:
    if Path(sample_filename).name != sample_filename or not sample_filename.endswith(".jsonl"):
        raise ValueError("EvalPlus sample name must be a plain JSONL filename")
    sample_name = json.dumps(sample_filename)
    script = f'''import hashlib, json
from pathlib import Path
from evalplus.data.humaneval import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.evaluate import evaluate

sample_path = Path("/input") / {sample_name}
dataset_path = Path("/input/HumanEvalPlus.jsonl")
problems = get_human_eval_plus(version="default")
evaluate(
    dataset="humaneval",
    samples=str(sample_path),
    parallel=2,
    test_details=False,
    output_file="/tmp/evalplus-results.json",
)
raw = json.loads(Path("/tmp/evalplus-results.json").read_text(encoding="utf-8"))
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
        "EVALPLUS_TIMEOUT_PER_TASK=30",
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
    removed = _low_magnitude_channels(model, remove_channels)
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
    with tempfile.TemporaryDirectory(
        prefix="modelsurgeon-evalplus-pruning-", dir=args.output.parent
    ) as temp_name:
        root = Path(temp_name)
        candidate_path = root / "pruned-candidate"
        candidate_path.mkdir()
        candidate_record = _build_pruned_candidate(
            snapshot,
            candidate_path,
            revision=args.model_revision,
            remove_channels=args.remove_channels,
        )
        dataset_input = root / "HumanEvalPlus.jsonl"
        shutil.copyfile(dataset_path, dataset_input)
        # Evaluation inputs share a read-only mount; each generated arm sees only its own samples.
        source_input = root / "source-input"
        candidate_input = root / "candidate-input"
        source_input.mkdir()
        candidate_input.mkdir()
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
        "schema_version": 1,
        "protocol": "physical-mlp-pruning-humanevalplus-pass-at-1-v1",
        "model": {
            "identifier": args.model,
            "revision": args.model_revision,
            "license": "Apache-2.0 (model card)",
            "source_snapshot": _artifact_manifest(snapshot),
        },
        "dataset": dataset_record,
        "method": {
            "name": "global-lowest-magnitude-gated-MLP-channel-removal",
            "score": "sum across layers of gate-row L2 x up-row L2 x down-column L2",
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
            "remove_channels": args.remove_channels,
            "max_new_tokens": args.max_new_tokens,
            "prompt_mode": "EvalPlus base-prompt function completion",
            "greedy": True,
            "samples_per_task": 1,
            "execution_parallelism": 2,
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
        },
        "limitations": [
            (
                "One code-specialized model revision, one magnitude-based MLP pruning severity, "
                "and one greedy sample per task."
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

    decoder = HuggingFaceDecoder(
        name=str(model_path),
        dataset="humaneval",
        force_base_prompt=True,
        temperature=0.0,
        max_new_tokens=max_new_tokens,
        dtype="float16",
    )
    codegen(
        target_path=str(target_path),
        model=decoder,
        dataset="humaneval",
        greedy=True,
        n_samples=1,
        resume=False,
    )
    del decoder
    raw_path = target_path.with_name(f"{target_path.stem}.raw.jsonl")
    if raw_path.exists():
        raw_path.unlink()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not target_path.is_file():
        raise RuntimeError("EvalPlus code generation did not write its JSONL sample file")
    return _sample_digests(target_path, tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--remove-channels", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--evalplus-image", default=EVALPLUS_IMAGE_DEFAULT)
    parser.add_argument("--evalplus-timeout-seconds", type=float, default=1800.0)
    args = parser.parse_args()
    if args.remove_channels <= 0 or args.max_new_tokens <= 0:
        parser.error("remove-channels and max-new-tokens must be positive")
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
