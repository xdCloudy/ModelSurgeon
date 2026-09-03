from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import modelsurgeon.cli.features as features_module
from modelsurgeon.cli.app import app
from modelsurgeon.features.schema import (
    FeatureKind,
    FeatureRecord,
    PrecisionProvenance,
    PrecisionSource,
)
from modelsurgeon.graph import ComponentId


def _record(component: ComponentId) -> FeatureRecord:
    return FeatureRecord(
        component,
        "weight_mean",
        FeatureKind.SCALAR,
        1.0,
        "float64",
        "weight_statistics",
        "1",
        PrecisionProvenance(PrecisionSource.HIGH_PRECISION, "float32", "float64"),
    )


class _Runtime:
    def __init__(self) -> None:
        self.requests: list[features_module.FeatureRequest] = []

    def extract(
        self, request: features_module.FeatureRequest
    ) -> dict[ComponentId, tuple[FeatureRecord, ...]]:
        self.requests.append(request)
        if request.group.name == "unsupported":
            raise RuntimeError("extractor dependency is unavailable")
        return {component: (_record(component),) for component in request.components}


def _config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_revision": "model-rev",
                "input_revision": "data-rev",
                "groups": [
                    {
                        "name": "statistics",
                        "extractor": "weight_statistics",
                        "extractor_version": "1",
                    },
                {
                    "name": "unsupported",
                    "extractor": "unavailable_extractor",
                        "extractor_version": "1",
                    },
                ],
                "components": ["model.layers.0", "model.layers.1"],
                "max_records": 4,
            }
        ),
        encoding="utf-8",
    )


def test_features_extracts_reuses_and_reports_skipped_groups(tmp_path: Path, monkeypatch) -> None:
    config, cache = tmp_path / "features.json", tmp_path / "cache"
    _config(config)
    runtime = _Runtime()
    monkeypatch.setattr(features_module, "load_feature_runtime", lambda _: runtime)

    first = CliRunner().invoke(
        app,
        [
            "features",
            str(config),
            "--runtime",
            "fixture:runtime",
            "--cache",
            str(cache),
            "--component",
            "model.layers.0",
        ],
    )
    assert first.exit_code == 0, first.output
    payload = json.loads(first.stdout)
    assert payload["components"] == ["model.layers.0"]
    assert payload["groups"][0]["state"] == "extracted"
    assert payload["groups"][1] == {
        "group": "unsupported",
        "record_count": 0,
        "reason": "RuntimeError: extractor dependency is unavailable",
        "state": "skipped",
    }
    assert runtime.requests[0].cpu_only is True

    second = CliRunner().invoke(
        app,
        [
            "features",
            str(config),
            "--runtime",
            "fixture:runtime",
            "--cache",
            str(cache),
            "--component",
            "model.layers.0",
        ],
    )
    assert second.exit_code == 0, second.output
    assert json.loads(second.stdout)["groups"][0]["state"] == "reused"


def test_features_rejects_empty_component_filters(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "features.json"
    _config(config)
    monkeypatch.setattr(features_module, "load_feature_runtime", lambda _: _Runtime())

    result = CliRunner().invoke(
        app,
        [
            "features",
            str(config),
            "--runtime",
            "fixture:runtime",
            "--cache",
            str(tmp_path / "cache"),
            "--component",
            "model.layers.9",
        ],
    )

    assert result.exit_code == 2
    assert "select no declared components" in result.output
