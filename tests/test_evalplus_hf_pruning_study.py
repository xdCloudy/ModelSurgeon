from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
from tools.run_evalplus_hf_pruning_study import (
    _decoder_with_task_local_stopping_criteria,
    _docker_wrapper,
    _generate_predictions,
    _low_magnitude_channels,
    _prepare_workspace,
    _repair_prediction_checkpoint,
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
    assert b"shutil.copyfile(sample_path, evaluation_sample_path)" in wrapper
    assert b"_eval_results.json" in wrapper
    assert b"output_file" not in wrapper
    compile(wrapper, "evalplus-docker-wrapper.py", "exec")
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
            self.model = SimpleNamespace(_get_stopping_criteria=lambda: [])

    def _codegen(**kwargs: object) -> None:
        observed["dataset"] = kwargs["dataset"]
        observed["resume"] = kwargs["resume"]
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
    assert observed["resume"] is True
    assert observed["decoder_kwargs"]["force_base_prompt"] is True
    assert result["sample_count"] == 1
    assert not target.with_name("predictions.raw.jsonl").exists()


def test_decoder_restores_stopping_criteria_after_each_task() -> None:
    class _Model:
        def __init__(self) -> None:
            self.calls = 0

        def _get_stopping_criteria(self) -> list[object]:
            self.calls += 1
            return []

    class _Decoder:
        def __init__(self) -> None:
            self.model = _Model()

        def codegen(self) -> list[object]:
            original = self.model._get_stopping_criteria

            def wrapped() -> list[object]:
                return original()

            self.model._get_stopping_criteria = wrapped
            return self.model._get_stopping_criteria()

    decoder_type = _decoder_with_task_local_stopping_criteria(_Decoder)
    decoder = decoder_type()
    original = decoder.model._get_stopping_criteria

    decoder.codegen()
    assert decoder.model._get_stopping_criteria == original
    decoder.codegen()
    assert decoder.model._get_stopping_criteria == original
    assert decoder.model.calls == 2


def test_workspace_records_settings_and_rejects_mismatched_resume(tmp_path) -> None:
    args = SimpleNamespace(
        model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        model_revision="revision-a",
        remove_channels=256,
        max_new_tokens=256,
        evalplus_image="modelsurgeon-evalplus:test",
    )

    workspace = _prepare_workspace(tmp_path / "study", args)

    assert workspace == tmp_path / "study"
    manifest = json.loads((workspace / ".modelsurgeon-evalplus-study.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["stop_criteria_policy"] == "restore-after-each-task-v1"
    assert _prepare_workspace(workspace, args) == workspace
    args.remove_channels = 128
    with pytest.raises(RuntimeError, match="settings do not match"):
        _prepare_workspace(workspace, args)


def test_workspace_refuses_unmarked_nonempty_directory(tmp_path) -> None:
    workspace = tmp_path / "study"
    workspace.mkdir()
    (workspace / "unrelated.json").write_text("{}", encoding="utf-8")
    args = SimpleNamespace(
        model="model",
        model_revision="revision-a",
        remove_channels=256,
        max_new_tokens=256,
        evalplus_image="modelsurgeon-evalplus:test",
    )

    with pytest.raises(RuntimeError, match="without a study manifest"):
        _prepare_workspace(workspace, args)


def test_checkpoint_recovery_discards_only_an_incomplete_final_record(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    complete = b'{"task_id":"HumanEval/0","solution":"def f(): return 0"}\n'
    path.write_bytes(complete + b'{"task_id":"HumanEval/1","solution":"unfinished')

    _repair_prediction_checkpoint(path)

    assert path.read_bytes() == complete


def test_checkpoint_recovery_terminates_a_complete_final_record(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    complete_record = b'{"task_id":"HumanEval/0","solution":"def f(): return 0"}'
    path.write_bytes(complete_record)

    _repair_prediction_checkpoint(path)

    assert path.read_bytes() == complete_record + b"\n"


def test_checkpoint_recovery_drops_a_truncated_only_record(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_bytes(b'{"task_id":"HumanEval/0","solution":')

    _repair_prediction_checkpoint(path)

    assert path.read_bytes() == b""
