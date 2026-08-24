"""Focused CLI contracts for train-surgeon and predict-surgeon."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from modelsurgeon.cli.app import app


def _metric(value: float) -> dict[str, object]:
    return {
        "name": "perplexity",
        "state": "measured",
        "value": value,
        "unit": "perplexity",
        "reason": None,
    }


def _example(example_id: str, feature: float, delta: float) -> dict[str, object]:
    component = f"model.layers.0.mlp.channel.{example_id[-1]}"
    return {
        "example_id": example_id,
        "model": {
            "identifier": "tiny/model",
            "revision": "model-revision",
            "family": "llama",
            "format": "safetensors",
            "quantization": None,
        },
        "components": [component],
        "mutation": {
            "plan": {
                "request": {
                    "kind": "mask",
                    "targets": [component],
                    "parameters": {"candidate_scope": "channel"},
                }
            }
        },
        "pre_mutation_features": [
            {
                "schema_version": 1,
                "component_id": component,
                "name": "activation_rms",
                "kind": "scalar",
                "value": feature,
            }
        ],
        "baseline_metrics": [_metric(10.0)],
        "post_metrics": [_metric(10.0 + delta)],
        "versions": {"feature_schema_version": 1},
    }


def _write_inputs(root: Path) -> tuple[Path, Path, tuple[dict[str, object], ...]]:
    examples = (
        _example("example-0", 0.1, 0.1),
        _example("example-1", 0.2, 0.2),
        _example("example-2", 0.3, 0.3),
        _example("example-3", 0.4, 0.4),
    )
    dataset = root / "examples.jsonl"
    dataset.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in examples),
        encoding="utf-8",
    )
    split = root / "split.json"
    split.write_text(
        json.dumps(
            {
                "version": "1",
                "algorithm": "connected-groups-greedy-v1",
                "mode": "component",
                "seed": 17,
                "ratios": {"train": 0.5, "validation": 0.25, "test": 0.25},
                "groups": [
                    {
                        "group_id": "group-train",
                        "partition": "train",
                        "keys": ["component:train"],
                        "example_ids": ["example-0", "example-1"],
                    },
                    {
                        "group_id": "group-validation",
                        "partition": "validation",
                        "keys": ["component:validation"],
                        "example_ids": ["example-2"],
                    },
                    {
                        "group_id": "group-test",
                        "partition": "test",
                        "keys": ["component:test"],
                        "example_ids": ["example-3"],
                    },
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return dataset, split, examples


def test_train_and_predict_cli_emit_resolved_versions_and_split_metrics(
    tmp_path: Path,
) -> None:
    dataset, split, examples = _write_inputs(tmp_path)
    registry = tmp_path / "registry"
    runner = CliRunner()

    trained = runner.invoke(
        app,
        [
            "train-surgeon",
            str(dataset),
            "--split",
            str(split),
            "--registry",
            str(registry),
            "--target",
            "perplexity",
            "--baseline",
            "linear",
            "--json",
        ],
    )
    assert trained.exit_code == 0, trained.output
    training = json.loads(trained.output.strip().splitlines()[-1])
    assert training["training_models"] == [
        {
            "identifier": "tiny/model",
            "quantization": None,
            "revision": "model-revision",
        }
    ]
    assert training["split_manifest_version"] == "1"
    assert training["split_algorithm"] == "connected-groups-greedy-v1"
    assert training["split_group_counts"] == {
        "test": 1,
        "train": 1,
        "validation": 1,
    }

    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(examples[-1]), encoding="utf-8")
    predicted = runner.invoke(
        app,
        [
            "predict-surgeon",
            str(candidate),
            "--registry",
            str(registry),
            "--bundle",
            str(training["artifact_digest"]),
            "--json",
        ],
    )
    assert predicted.exit_code == 0, predicted.output
    prediction = json.loads(predicted.output.strip().splitlines()[-1])
    assert prediction["training_models"] == training["training_models"]
    assert prediction["source_feature_schema_version"] == 1
    assert prediction["target_schema_version"] == 1
    assert prediction["artifact_digest"] == training["artifact_digest"]
