"""Run the v0.8 Q7 active-versus-random study over multiple measured models."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

from modelsurgeon.cli.proof_evidence import load_grouped_proof_split
from modelsurgeon.cli.surgeon import load_surgeon_records
from modelsurgeon.evaluation.multi_model_active_learning import (
    MultiModelActiveLearningConfig,
    run_model_active_learning_study,
)
from modelsurgeon.experiments.identity import canonical_identity_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _tool_provenance(root: Path) -> dict[str, object]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
        dirty = True
    source_files = (
        Path(__file__).resolve(),
        root / "src/modelsurgeon/evaluation/multi_model_active_learning.py",
        root / "src/modelsurgeon/active_learning/acquisition.py",
        root / "src/modelsurgeon/active_learning/diversity.py",
    )
    return {
        "git_revision": revision,
        "worktree_dirty": dirty,
        "runtime": {
            "python": platform.python_version(),
            "lightgbm": _package_version("lightgbm"),
            "numpy": _package_version("numpy"),
            "scikit_learn": _package_version("scikit-learn"),
            "platform": platform.platform(),
        },
        "source_sha256": {
            path.relative_to(root).as_posix(): _sha256(path) for path in source_files
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", action="append", type=int)
    parser.add_argument("--target-auc", type=float, default=0.8)
    parser.add_argument("--safe-perplexity-delta", type=float, default=0.01)
    parser.add_argument("--feature-diversity-fraction", type=float, default=0.25)
    args = parser.parse_args()

    budgets = tuple(args.budget or (64, 128, 256, 384))
    config = MultiModelActiveLearningConfig(
        budgets=budgets,
        target_auc=args.target_auc,
        safe_perplexity_delta=args.safe_perplexity_delta,
        feature_diversity_fraction=args.feature_diversity_fraction,
    )
    repo_root = Path(__file__).resolve().parents[1]
    results: list[dict[str, object]] = []
    sources: list[dict[str, str]] = []
    identifiers: set[str] = set()
    for directory in args.dataset:
        examples = directory / "examples.jsonl"
        split = directory / "split.json"
        manifest = directory / "gradient-manifest.json"
        result = run_model_active_learning_study(
            load_surgeon_records(examples),
            load_grouped_proof_split(split),
            config,
        )
        results.append(result.to_record())
        identifiers.add(result.model[0])
        sources.append(
            {
                "dataset": directory.name,
                "examples_sha256": _sha256(examples),
                "split_sha256": _sha256(split),
                "gradient_manifest_sha256": _sha256(manifest),
            }
        )
    if len(identifiers) < 3:
        raise ValueError("Q7 requires at least three distinct models")
    record: dict[str, object] = {
        "record_type": "v0.8_q7_multi_model_active_learning_study",
        "version": "2",
        "protocol": {
            "strategies": [
                "active_uncertainty",
                "uncertainty_diversity",
                "seeded_random",
            ],
            "budgets": list(budgets),
            "seeds": list(config.seeds),
            "target_auc": args.target_auc,
            "safe_perplexity_delta": args.safe_perplexity_delta,
            "acquisition_policy": {
                "high_value_fraction": 0.0,
                "uncertainty_fraction": 1.0 - config.feature_diversity_fraction,
                "diversity_fraction": config.feature_diversity_fraction,
                "initial_centers": "currently selected labeled examples",
                "numeric_features": "source-only trained-preprocessor output",
                "categorical_and_topology": "pre-mutation model and mutation metadata",
            },
            "feature_profile": "static_activation_gradient",
            "test_metric": "roc_auc",
            "gpu_hour_accounting": "sum_selected_cuda_evaluation_wall_seconds",
        },
        "tool_provenance": _tool_provenance(repo_root),
        "sources": sources,
        "results": results,
    }
    args.output.write_text(canonical_identity_json(record) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
