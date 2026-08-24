"""Optional LightGBM integration smoke for the complete First Surgeon evidence path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelsurgeon.cli.proof_evidence import (
    FirstSurgeonEvidenceConfig,
    run_first_surgeon_evidence,
)

pytest.importorskip("lightgbm")


def _metric(name: str, value: float) -> dict[str, object]:
    return {
        "name": name,
        "state": "measured",
        "value": value,
        "unit": "perplexity",
        "reason": None,
    }


def _example(index: int, delta: float) -> dict[str, object]:
    layer = index // 8
    channel = index % 8
    component = f"model.layers.{layer}.mlp.channel.{channel}"
    magnitude = 0.1 + delta
    return {
        "example_id": f"example-{index:02d}",
        "model": {
            "identifier": "synthetic/tiny-llama",
            "revision": "model-rev",
            "family": "llama",
            "format": "safetensors",
            "quantization": None,
        },
        "dataset": {
            "identifier": "synthetic-calibration",
            "revision": "dataset-rev",
            "split": "calibration",
            "manifest_id": "manifest-rev",
            "tokenizer": "synthetic-tokenizer",
            "tokenizer_revision": "tokenizer-rev",
        },
        "components": [component],
        "mutation": {
            "plan": {
                "request": {
                    "kind": "mask",
                    "targets": [component],
                    "parameters": {
                        "candidate_scope": "channel",
                        "layer_index": layer,
                        "channel_index": channel,
                    },
                }
            }
        },
        "pre_mutation_features": [
            {
                "schema_version": 1,
                "component_id": component,
                "name": "activation_rms",
                "kind": "scalar",
                "value": magnitude * 0.5,
            },
            {
                "schema_version": 1,
                "component_id": component,
                "name": "weight_count",
                "kind": "scalar",
                "value": 32.0,
            },
            {
                "schema_version": 1,
                "component_id": component,
                "name": "weight_l1_norm",
                "kind": "scalar",
                "value": magnitude * 32.0,
            },
            {
                "schema_version": 1,
                "component_id": component,
                "name": "weight_l2_norm",
                "kind": "scalar",
                "value": magnitude * (32.0**0.5),
            },
            {
                "schema_version": 1,
                "component_id": component,
                "name": "weight_max_magnitude",
                "kind": "scalar",
                "value": magnitude,
            },
        ],
        "baseline_metrics": [_metric("perplexity", 10.0)],
        "post_metrics": [_metric("perplexity", 10.0 + delta)],
        "versions": {
            "tool_revision": "test-tool-revision",
            "config_digest": "test-config-digest",
            "evaluator_version": "test-evaluator-v1",
            "feature_schema_version": 1,
            "mutation_record_schema_version": 1,
        },
    }


def _group(example_id: str, partition: str, component: str) -> dict[str, object]:
    return {
        "group_id": f"group-{example_id}",
        "partition": partition,
        "keys": [f"component:{component}"],
        "example_ids": [example_id],
    }


def _write_fixture(root: Path) -> tuple[Path, Path]:
    # Every partition contains both safe (delta <= .25) and unsafe mutations.
    deltas = (
        0.05,
        0.75,
        0.10,
        0.90,
        0.15,
        0.80,
        0.20,
        1.00,
        0.08,
        0.70,
        0.12,
        0.95,
        0.06,
        0.85,
        0.18,
        1.10,
    )
    examples = tuple(_example(index, delta) for index, delta in enumerate(deltas))
    dataset = root / "examples.jsonl"
    dataset.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in reversed(examples)),
        encoding="utf-8",
    )

    groups: list[dict[str, object]] = []
    for index, record in enumerate(examples):
        partition = "train" if index < 8 else "validation" if index < 12 else "test"
        component = str(record["components"][0])
        groups.append(_group(str(record["example_id"]), partition, component))
    split = root / "split.json"
    split.write_text(
        json.dumps(
            {
                "version": "1",
                "algorithm": "connected-groups-greedy-v1",
                "mode": "component",
                "seed": 43,
                "ratios": {"train": 0.5, "validation": 0.25, "test": 0.25},
                "groups": groups,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return dataset, split


def test_complete_lightgbm_evidence_path(tmp_path: Path) -> None:
    dataset, split = _write_fixture(tmp_path)
    result = run_first_surgeon_evidence(
        dataset,
        split,
        tmp_path / "registry",
        FirstSurgeonEvidenceConfig(
            safe_perplexity_delta=0.25,
            threads=1,
            seed=7,
            top_n=2,
            bootstrap_repetitions=20,
        ),
    )

    assert result.classifier.report.metric("auc").value is not None
    assert result.classifier.report.metric("precision_at_2").value is not None
    assert result.regressor.report.metric("mae").value is not None
    assert result.regressor.report.metric("rmse").value is not None
    assert len(result.random_ranking.entries) == 4
    assert len(result.magnitude_ranking.entries) == 4
    assert result.classifier.digest.startswith("sha256:")
    assert result.regressor.digest.startswith("sha256:")
    assert 0.0 <= result.classifier_smoke_prediction <= 1.0
