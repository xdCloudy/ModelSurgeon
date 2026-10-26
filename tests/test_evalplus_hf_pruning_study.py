from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
from tools.run_evalplus_hf_pruning_study import (
    _docker_wrapper,
    _generate_predictions,
    _low_magnitude_channels,
)


def _toy_model(
    gate: list[list[float]],
    up: list[list[float]],
    down: list[list[float]],
) -> object:
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class ToyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = len(gate)
            self.config = SimpleNamespace(intermediate_size=width, num_hidden_layers=1)
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([nn.Module()])
            self.model.layers[0].mlp = nn.Module()
            self.model.layers[0].mlp.gate_proj = nn.Linear(width, width, bias=False)
            self.model.layers[0].mlp.up_proj = nn.Linear(width, width, bias=False)
            self.model.layers[0].mlp.down_proj = nn.Linear(width, width, bias=False)
            with torch.no_grad():
                self.model.layers[0].mlp.gate_proj.weight.copy_(torch.tensor(gate))
                self.model.layers[0].mlp.up_proj.weight.copy_(torch.tensor(up))
                self.model.layers[0].mlp.down_proj.weight.copy_(torch.tensor(down))

    return ToyModel()


def test_channel_selection_uses_coupled_global_magnitude_score() -> None:
    model = _toy_model(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        [[1.0, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 2.0]],
    )

    assert _low_magnitude_channels(model, 1) == (1,)


def test_channel_selection_breaks_equal_scores_by_index() -> None:
    torch = pytest.importorskip("torch")
    identity = torch.eye(3).tolist()
    model = _toy_model(identity, identity, identity)

    assert _low_magnitude_channels(model, 2) == (0, 1)


@pytest.mark.parametrize("count", [0, 3])
def test_channel_selection_rejects_invalid_removal_count(count: int) -> None:
    torch = pytest.importorskip("torch")
    identity = torch.eye(3).tolist()
    model = _toy_model(identity, identity, identity)

    with pytest.raises(ValueError, match="channels must be"):
        _low_magnitude_channels(model, count)


def test_docker_wrapper_accepts_only_plain_jsonl_filename() -> None:
    wrapper = _docker_wrapper("source_predictions.jsonl")

    assert b'Path("/input") / "source_predictions.jsonl"' in wrapper
    with pytest.raises(ValueError, match="plain JSONL filename"):
        _docker_wrapper("../outside.jsonl")


def test_generation_uses_evalplus_dataset_name_and_removes_raw_output(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evalplus_module = ModuleType("evalplus")
    evalplus_module.__path__ = []
    provider_module = ModuleType("evalplus.provider")
    provider_module.__path__ = []
    codegen_module = ModuleType("evalplus.codegen")
    hf_module = ModuleType("evalplus.provider.hf")
    torch_module = ModuleType("torch")
    torch_module.cuda = SimpleNamespace(is_available=lambda: False)
    observed: dict[str, object] = {}
    target = tmp_path / "predictions.jsonl"
    tasks = {"HumanEval/0": {"prompt": "def f():"}}

    class _Decoder:
        def __init__(self, **kwargs: object) -> None:
            observed["decoder_kwargs"] = kwargs

    def _codegen(**kwargs: object) -> None:
        observed["dataset"] = kwargs["dataset"]
        target.write_text(
            json.dumps({"task_id": "HumanEval/0", "solution": "def f(): return 1"}) + "\n",
            encoding="utf-8",
        )
        target.with_name("predictions.raw.jsonl").write_text("raw output", encoding="utf-8")

    codegen_module.codegen = _codegen
    hf_module.HuggingFaceDecoder = _Decoder
    monkeypatch.setitem(sys.modules, "evalplus", evalplus_module)
    monkeypatch.setitem(sys.modules, "evalplus.provider", provider_module)
    monkeypatch.setitem(sys.modules, "evalplus.codegen", codegen_module)
    monkeypatch.setitem(sys.modules, "evalplus.provider.hf", hf_module)
    monkeypatch.setitem(sys.modules, "torch", torch_module)

    result = _generate_predictions(tmp_path, target, tasks, max_new_tokens=8)

    assert observed["dataset"] == "humaneval"
    assert observed["decoder_kwargs"]["force_base_prompt"] is True
    assert result["sample_count"] == 1
    assert not target.with_name("predictions.raw.jsonl").exists()
