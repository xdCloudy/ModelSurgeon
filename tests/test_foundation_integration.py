"""Golden offline integration from HF loader boundary through CLI inspection."""

from __future__ import annotations

import json
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from typer.testing import CliRunner

from modelsurgeon.adapters import ModelFamily
from modelsurgeon.adapters.huggingface import HuggingFaceLoadRequest, loader
from modelsurgeon.cli.app import app
from modelsurgeon.cli.inspection import inspect_huggingface_model
from modelsurgeon.graph import build_component_graph, validate_component_graph
from modelsurgeon.testing import tiny_transformer

MODEL_ID = "HuggingFaceM4/tiny-random-LlamaForCausalLM"
REVISION = "d3040b7c81a0a810fa13c6f392f3e304a0e121d5"


class OfflineAutoModel:
    calls: ClassVar[list[tuple[str, dict[str, Any]]]] = []

    @classmethod
    def from_pretrained(cls, model_id: str, **options: Any) -> object:
        cls.calls.append((model_id, options))
        model = tiny_transformer(ModelFamily.LLAMA)
        model.config = SimpleNamespace(**asdict(model.config), _commit_hash=REVISION)
        return model


@pytest.fixture(autouse=True)
def offline_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    OfflineAutoModel.calls.clear()

    def fake_import(name: str) -> object:
        if name == "transformers":
            return SimpleNamespace(AutoModelForCausalLM=OfflineAutoModel)
        raise AssertionError(f"integration test attempted unexpected dependency import {name!r}")

    monkeypatch.setattr(loader, "import_module", fake_import)


def test_load_discover_graph_validate_and_cli_json_golden_path() -> None:
    inspection = inspect_huggingface_model(
        HuggingFaceLoadRequest(
            model=MODEL_ID,
            revision=REVISION,
            local_files_only=True,
        )
    )
    graph = build_component_graph(inspection.discovery.components())
    validation = validate_component_graph(graph)

    assert inspection.family.family is ModelFamily.LLAMA
    assert inspection.provenance.resolved_revision == REVISION
    assert inspection.discovery.parameter_count == 1512
    assert inspection.discovery.to_record() == {
        "family": "llama",
        "module_count": 29,
        "parameter_tensor_count": 21,
        "parameter_count": 1512,
        "logical_component_count": 42,
    }
    assert len(graph.nodes) == 96
    assert len(graph.constraints) == 4
    assert validation.valid
    graph_ids = {str(node.component_id) for node in graph.nodes}
    assert {
        "model",
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.residual_attn",
        "model.layers.1.mlp.channel.11",
        "model.lm_head.weight",
    } <= graph_ids

    source, options = OfflineAutoModel.calls[-1]
    assert source == MODEL_ID
    assert options["revision"] == REVISION
    assert options["device_map"] == {"": "cpu"}
    assert options["local_files_only"] is True
    assert options["low_cpu_mem_usage"] is True

    runner = CliRunner()
    command = ["inspect", MODEL_ID, "--revision", REVISION, "--json"]
    first = runner.invoke(app, command)
    second = runner.invoke(app, command)

    assert first.exit_code == 0
    assert first.stdout == second.stdout
    records = [json.loads(line) for line in first.stdout.splitlines()]
    assert len(records) == 93
    assert records[0] == {
        "family": "llama",
        "family_evidence": ["architecture:llamaforcausallm", "model_type:llama"],
        "loader_options": {
            "device_map": "cpu",
            "dtype": "auto",
            "local_files_only": False,
            "low_cpu_mem_usage": True,
            "trust_remote_code": False,
        },
        "logical_component_count": 42,
        "module_count": 29,
        "parameter_count": 1512,
        "parameter_tensor_count": 21,
        "record_type": "model",
        "requested_revision": REVISION,
        "resolved_revision": REVISION,
        "source": MODEL_ID,
    }
    component_ids = [record["component_id"] for record in records[1:]]
    assert component_ids[0] == "model"
    assert component_ids[-1] == "model.layers.1.mlp.channel.11"
    assert "model.layers.0.self_attn.q_proj.weight" in component_ids
