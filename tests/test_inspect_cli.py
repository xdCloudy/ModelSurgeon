"""Tests for stable model inspection CLI output and categorized errors."""

from __future__ import annotations

import json
from collections.abc import Iterable
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from modelsurgeon.adapters.huggingface import (
    HuggingFaceDependencyError,
    HuggingFaceDType,
    HuggingFaceLoadProvenance,
    HuggingFaceLoadResult,
    HuggingFaceModelError,
    HuggingFaceRevisionError,
)
from modelsurgeon.cli import inspection
from modelsurgeon.cli.app import app


class Parameter:
    def __init__(self, count: int) -> None:
        self.count = count

    def numel(self) -> int:
        return self.count


class TinyModel:
    config = SimpleNamespace(
        model_type="llama",
        architectures=["LlamaForCausalLM"],
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=2,
    )

    def __init__(self) -> None:
        self.weight = Parameter(8)

    def named_modules(self) -> Iterable[tuple[str, object]]:
        return [
            ("", self),
            ("model", SimpleNamespace()),
            ("model.layers", SimpleNamespace()),
            ("model.layers.0", SimpleNamespace()),
            ("model.layers.0.self_attn", SimpleNamespace()),
            ("model.layers.0.mlp", SimpleNamespace()),
        ]

    def named_parameters(self) -> Iterable[tuple[str, Parameter]]:
        return [("model.layers.0.mlp.up_proj.weight", self.weight)]


def _fake_load() -> HuggingFaceLoadResult:
    return HuggingFaceLoadResult(
        TinyModel(),
        HuggingFaceLoadProvenance(
            source="tiny/model",
            requested_revision="tag",
            resolved_revision="a" * 40,
            trust_remote_code=False,
            device_map="cpu",
            dtype=HuggingFaceDType.AUTO,
            local_files_only=False,
            low_cpu_mem_usage=True,
        ),
    )


def test_json_inspection_order_and_totals_are_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inspection, "load_causal_lm", lambda request: _fake_load())
    runner = CliRunner()

    first = runner.invoke(app, ["inspect", "tiny/model", "--revision", "tag", "--json"])
    second = runner.invoke(app, ["inspect", "tiny/model", "--revision", "tag", "--json"])

    assert first.exit_code == 0
    assert first.stdout == second.stdout
    records = [json.loads(line) for line in first.stdout.splitlines()]
    assert records[0]["record_type"] == "model"
    assert records[0]["resolved_revision"] == "a" * 40
    assert records[0]["family"] == "llama"
    assert records[0]["parameter_count"] == 8
    assert records[1]["component_id"] == "model"
    assert all(record["record_type"] == "component" for record in records[1:])


@pytest.mark.parametrize(
    ("error", "category", "exit_code"),
    [
        (HuggingFaceDependencyError("install hf"), "dependency", 3),
        (HuggingFaceModelError("missing model"), "model", 4),
        (HuggingFaceRevisionError("missing revision"), "revision", 5),
        (ValueError("unsupported adapter"), "adapter", 6),
    ],
)
def test_error_categories_have_distinct_json_records_and_exit_codes(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    category: str,
    exit_code: int,
) -> None:
    def fail(request: object) -> object:
        raise error

    monkeypatch.setattr("modelsurgeon.cli.app.inspect_huggingface_model", fail)

    result = CliRunner().invoke(app, ["inspect", "bad/model", "--json"])

    assert result.exit_code == exit_code
    payload = json.loads(result.stderr.strip())
    assert payload["record_type"] == "error"
    assert payload["category"] == category
