"""Summarize paired per-task EvalPlus outcomes from physical pruning runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

BOOTSTRAP_SEED = 20261026


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _paired_outcomes(
    left: dict[str, str], right: dict[str, str]
) -> tuple[int, int, int, int, list[int]]:
    """Return both-pass, left-only, right-only, neither, and paired score deltas."""
    if left.keys() != right.keys():
        raise ValueError("paired runs must contain the same task IDs")
    both_pass = left_only = right_only = neither = 0
    deltas: list[int] = []
    for task_id in sorted(left):
        left_pass = left[task_id] == "pass"
        right_pass = right[task_id] == "pass"
        deltas.append(int(left_pass) - int(right_pass))
        if left_pass and right_pass:
            both_pass += 1
        elif left_pass:
            left_only += 1
        elif right_pass:
            right_only += 1
        else:
            neither += 1
    return both_pass, left_only, right_only, neither, deltas


def _exact_mcnemar_p_value(left_only: int, right_only: int) -> float:
    """Two-sided exact McNemar p-value, using the binomial distribution."""
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    smaller = min(left_only, right_only)
    lower_tail = sum(math.comb(discordant, index) for index in range(smaller + 1))
    return min(1.0, 2.0 * lower_tail / (2**discordant))


def _percentile(sorted_values: list[float], quantile: float) -> float:
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _paired_bootstrap_interval(
    deltas: list[int], *, samples: int, seed: int = BOOTSTRAP_SEED
) -> list[float]:
    """Return a deterministic percentile interval in percentage points."""
    if not deltas:
        raise ValueError("paired task outcomes cannot be empty")
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    generator = random.Random(seed)
    count = len(deltas)
    estimates = sorted(
        100.0 * sum(generator.choices(deltas, k=count)) / count for _ in range(samples)
    )
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def _task_statuses(report: dict[str, Any], arm: str, name: str) -> dict[str, str]:
    try:
        tasks = report["arms"][arm]["task_quality"]["per_task"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"run {name!r} is missing {arm} per-task outcomes") from error
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError(f"run {name!r} has no {arm} per-task outcomes")
    statuses: dict[str, str] = {}
    for task_id, result in tasks.items():
        if not isinstance(task_id, str) or not isinstance(result, dict):
            raise ValueError(f"run {name!r} has a malformed {arm} task result")
        status = result.get("plus_status")
        if not isinstance(status, str) or not status:
            raise ValueError(f"run {name!r} has a missing EvalPlus+ status")
        statuses[task_id] = status
    return statuses


def _source_hash(path: Path) -> str:
    normalized = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def _compare(
    left_name: str,
    left: dict[str, str],
    right_name: str,
    right: dict[str, str],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    both_pass, left_only, right_only, neither, deltas = _paired_outcomes(left, right)
    task_count = len(deltas)
    left_passes = both_pass + left_only
    right_passes = both_pass + right_only
    return {
        "left": left_name,
        "right": right_name,
        "task_count": task_count,
        "left_plus_passes": left_passes,
        "right_plus_passes": right_passes,
        "left_plus_status_counts": dict(sorted(Counter(left.values()).items())),
        "right_plus_status_counts": dict(sorted(Counter(right.values()).items())),
        "left_minus_right_percentage_points": 100.0 * (left_passes - right_passes) / task_count,
        "paired_bootstrap_95_percent_interval_percentage_points": _paired_bootstrap_interval(
            deltas, samples=bootstrap_samples
        ),
        "paired_bootstrap_samples": bootstrap_samples,
        "paired_bootstrap_seed": BOOTSTRAP_SEED,
        "discordant_tasks": {"left_only_pass": left_only, "right_only_pass": right_only},
        "both_pass": both_pass,
        "neither_pass": neither,
        "mcnemar_exact_two_sided_p_value": _exact_mcnemar_p_value(left_only, right_only),
    }


def build_report(run_paths: dict[str, Path], *, bootstrap_samples: int = 20_000) -> dict[str, Any]:
    if len(run_paths) < 2:
        raise ValueError("at least two candidate run reports are required")
    if "source" in run_paths:
        raise ValueError("'source' is reserved for each run's unpruned baseline arm")
    if "magnitude" not in run_paths:
        raise ValueError("named reports must include 'magnitude'")

    reports = {name: _load_json(path) for name, path in run_paths.items()}
    reference = reports["magnitude"]
    reference_model = reference.get("model", {}).get("revision")
    reference_dataset = reference.get("dataset", {}).get("sha256")
    reference_evalplus = reference.get("arms", {}).get("source", {}).get("task_quality", {})
    reference_evalplus_revision = reference_evalplus.get("evalplus_source_revision")
    reference_image = reference_evalplus.get("docker_image_id")
    reference_source_hash = (
        reference.get("arms", {}).get("source", {}).get("generation", {}).get("sample_file_sha256")
    )
    controls = (reference_model, reference_dataset, reference_evalplus_revision, reference_image)
    if any(value is None for value in controls) or reference_source_hash is None:
        raise ValueError(
            "magnitude report is missing model, dataset, evaluator, or source controls"
        )

    task_maps: dict[str, dict[str, str]] = {
        "source": _task_statuses(reports["magnitude"], "source", "magnitude")
    }
    for name, report in reports.items():
        run_controls = (
            report.get("model", {}).get("revision"),
            report.get("dataset", {}).get("sha256"),
            report.get("arms", {})
            .get("source", {})
            .get("task_quality", {})
            .get("evalplus_source_revision"),
            report.get("arms", {}).get("source", {}).get("task_quality", {}).get("docker_image_id"),
        )
        if run_controls != controls:
            raise ValueError(f"run {name!r} does not share the reference model/dataset/evaluator")
        source_hash = (
            report.get("arms", {}).get("source", {}).get("generation", {}).get("sample_file_sha256")
        )
        if source_hash != reference_source_hash:
            raise ValueError(f"run {name!r} has a different source-arm sample set")
        source_tasks = _task_statuses(report, "source", name)
        candidate_tasks = _task_statuses(report, "pruned", name)
        if source_tasks != task_maps["source"]:
            raise ValueError(f"run {name!r} has different source-arm task outcomes")
        task_maps[name] = candidate_tasks

    if task_maps["source"].keys() != task_maps["magnitude"].keys():
        raise ValueError("source and candidate runs must contain the same task IDs")

    comparisons = [
        _compare(
            "source",
            task_maps["source"],
            "magnitude",
            task_maps["magnitude"],
            bootstrap_samples=bootstrap_samples,
        )
    ]
    for name in run_paths:
        if name != "magnitude":
            comparisons.append(
                _compare(
                    "source",
                    task_maps["source"],
                    name,
                    task_maps[name],
                    bootstrap_samples=bootstrap_samples,
                )
            )
            comparisons.append(
                _compare(
                    "magnitude",
                    task_maps["magnitude"],
                    name,
                    task_maps[name],
                    bootstrap_samples=bootstrap_samples,
                )
            )

    return {
        "schema_version": 1,
        "protocol": "paired-evalplus-task-outcome-inference-v1",
        "controls": {
            "model_revision": reference_model,
            "dataset_sha256": reference_dataset,
            "evalplus_source_revision": reference_evalplus_revision,
            "evalplus_docker_image_id": reference_image,
            "source_generated_samples_sha256": reference_source_hash,
            "task_count": len(task_maps["source"]),
        },
        "method": {
            "bootstrap": "paired percentile bootstrap over benchmark tasks",
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "paired_test": "two-sided exact McNemar test on discordant pass/fail outcomes",
        },
        "interpretation_limits": [
            "Intervals describe task-sample variation on this fixed benchmark; they do not capture "
            "generation-seed, model, or benchmark contamination uncertainty.",
            "Exact McNemar p-values compare observed paired task outcomes and do not establish "
            "generalization to other models or task distributions.",
            "Each arm uses one greedy completion per task, so the analysis does not estimate "
            "pass@k or decode variability.",
        ],
        "source_reports": [
            {"name": name, "path": path.as_posix(), "sha256_lf_normalized": _source_hash(path)}
            for name, path in run_paths.items()
        ],
        "comparisons": comparisons,
    }


def _parse_run(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("runs must use NAME=PATH")
    return name.strip(), Path(path.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", type=_parse_run, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    args = parser.parse_args()
    run_paths = dict(args.run)
    if len(run_paths) != len(args.run):
        parser.error("run names must be unique")
    report = build_report(run_paths, bootstrap_samples=args.bootstrap_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(args.output), "comparisons": len(report["comparisons"])}))


if __name__ == "__main__":
    main()
