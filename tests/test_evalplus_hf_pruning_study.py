from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from tools.run_evalplus_hf_pruning_study import (
    _activation_output_energy_channels,
    _activation_weighted_channels,
    _decoder_with_task_local_stopping_criteria,
    _docker_wrapper,
    _evaluate_in_docker,
    _generate_predictions,
    _low_magnitude_channels,
    _prepare_workspace,
    _repair_prediction_checkpoint,
    _summarize_generation_metrics,
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


def test_activation_weighted_selection_ranks_coupled_channels_by_calibration_rms() -> None:
    torch = pytest.importorskip("torch")
    identity = torch.eye(3).tolist()
    model = _toy_model(identity, identity, identity)

    assert _activation_weighted_channels(model, 1, (torch.tensor([0.1, 0.2, 1.0]),)) == (0,)


def test_activation_output_energy_scores_the_post_gated_down_projection_contribution() -> None:
    torch = pytest.importorskip("torch")
    model = _toy_model(
        [[100.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[100.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.1, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]],
    )
    activation_rms = (torch.tensor([0.01, 0.01, 0.02]),)

    assert _activation_weighted_channels(model, 1, activation_rms) == (2,)
    assert _activation_output_energy_channels(model, 1, activation_rms) == (0,)


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


def test_evalplus_docker_command_uses_numeric_default_task_timeout(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.run_evalplus_hf_pruning_study as study

    samples = tmp_path / "arm" / "samples.jsonl"
    samples.parent.mkdir()
    samples.write_text("{}\n", encoding="utf-8")
    dataset = tmp_path / "HumanEvalPlus.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    (samples.parent / dataset.name).write_bytes(dataset.read_bytes())
    per_task = {
        f"HumanEval/{index}": {"base_status": "pass", "plus_status": "pass"} for index in range(164)
    }
    report = {
        "schema_version": 1,
        "evalplus_version": "0.3.1",
        "task_count": 164,
        "per_task": per_task,
        "evalplus_dataset_hash": "dataset-hash",
        "dataset_sha256": "dataset-sha256",
        "base_passes": 164,
        "plus_passes": 164,
        "base_pass_at_1": 1.0,
        "plus_pass_at_1": 1.0,
    }
    observed: dict[str, object] = {}

    def _run(command: list[str], *_args: object, **_kwargs: object) -> SimpleNamespace:
        observed["command"] = command
        return SimpleNamespace(
            stdout=("MODELSURGEON_EVALPLUS_REPORT=" + json.dumps(report) + "\n").encode(),
            stderr=b"",
            returncode=0,
            timed_out=False,
            output_exceeded=False,
        )

    monkeypatch.setattr(study, "run_bounded_process", _run)

    result = _evaluate_in_docker(
        samples,
        dataset,
        docker="docker",
        image="evalplus:test",
        image_identity={"revision": "revision", "image_id": "image-id"},
        timeout_seconds=30,
    )

    command = observed["command"]
    assert isinstance(command, list)
    assert "EVALPLUS_TIMEOUT_PER_TASK=30" not in command
    assert result["plus_pass_at_1"] == 1.0


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
            self.model = _Model()

        def codegen(self, *_args: object, **_kwargs: object) -> object:
            return self.model.generate(_Tensor((1, 3)))

    class _Tensor:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.shape = shape

    class _Model:
        def __init__(self) -> None:
            self._get_stopping_criteria = lambda: []

        def generate(self, input_ids: _Tensor) -> _Tensor:
            return _Tensor((1, input_ids.shape[-1] + 2))

    def _codegen(**kwargs: object) -> None:
        observed["dataset"] = kwargs["dataset"]
        observed["resume"] = kwargs["resume"]
        decoder = kwargs["model"]
        observed["decoder"] = decoder
        decoder.codegen("prompt")
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
    assert result["runtime"]["generation_calls"] == 1
    assert result["runtime"]["generated_tokens"] == 2
    decoder = observed["decoder"]
    assert "generate" not in decoder.model.__dict__
    assert decoder.model._get_stopping_criteria == decoder._modelsurgeon_base_get_stopping_criteria
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


def test_generation_metrics_report_synchronized_latency_and_effective_throughput() -> None:
    result = _summarize_generation_metrics(
        [
            {"seconds": 1.0, "prompt_tokens": 3, "generated_tokens": 10},
            {"seconds": 2.0, "prompt_tokens": 4, "generated_tokens": 20},
        ]
    )

    assert result["generation_calls"] == 2
    assert result["prompt_tokens"] == 7
    assert result["generated_tokens"] == 30
    assert result["generate_wall_seconds"] == 3.0
    assert result["generated_tokens_per_second"] == 10.0
    assert result["median_generation_latency_seconds"] == 1.5
    assert result["p95_generation_latency_seconds"] == 2.0


def test_workspace_records_settings_and_rejects_mismatched_resume(tmp_path) -> None:
    args = SimpleNamespace(
        model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        model_revision="revision-a",
        remove_channels=256,
        max_new_tokens=256,
        evalplus_image="modelsurgeon-evalplus:test",
        channel_scorer="magnitude",
        calibration_tokens=2048,
    )

    workspace = _prepare_workspace(tmp_path / "study", args)

    assert workspace == tmp_path / "study"
    manifest = json.loads((workspace / ".modelsurgeon-evalplus-study.json").read_text())
    assert manifest["schema_version"] == 4
    assert manifest["stop_criteria_policy"] == "restore-after-each-task-v1"
    assert manifest["runtime_measurement_policy"] == "cuda-synchronized-generate-wall-time-v1"
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


def test_activation_weighted_study_comparison_is_paired_and_provenanced() -> None:
    root = Path(__file__).resolve().parents[1]
    report = json.loads(
        (
            root / "reports/benchmarks/qwen25_coder_05b_humanevalplus_mlp_scorer_comparison.json"
        ).read_text(encoding="utf-8")
    )

    assert report["paired_controls"]["source_samples_identical"] is True
    assert report["paired_controls"]["torch_version"] == "2.11.0+cu128"
    assert report["results"]["activation_weighted"]["plus_passes"] == 47
    assert report["results"]["activation_output_energy"]["plus_passes"] == 50
    assert report["results"]["magnitude"]["plus_passes"] == 50
    assert report["results"]["activation_output_energy_minus_magnitude"]["plus_passes"] == 0
    assert (
        report["results"]["activation_output_energy_minus_magnitude"]["removed_channel_overlap"]
        == 12
    )
    assert (
        report["results"]["activation_output_energy"]["generated_tokens_per_second"]
        < report["results"]["magnitude"]["generated_tokens_per_second"]
    )
    assert (
        report["results"]["activation_weighted"]["artifact_bytes"]
        == report["results"]["magnitude"]["artifact_bytes"]
    )
    paired = report["paired_inference"]
    assert paired["comparisons"][4]["mcnemar_exact_two_sided_p_value"] == 1.0
    assert paired["comparisons"][4]["left_minus_right_percentage_points"] == 0.0
    paired_report = root / paired["report_path"]
    normalized_paired_report = paired_report.read_bytes().replace(b"\r\n", b"\n")
    assert (
        hashlib.sha256(normalized_paired_report).hexdigest()
        == paired["report_sha256_lf_normalized"]
    )
    sensitivity = report["calibration_sensitivity"]
    assert sensitivity["run_metadata"]["output_energy_2048"]["candidate_plus_passes"] == 50
    assert sensitivity["run_metadata"]["output_energy_8192"]["candidate_plus_passes"] == 50
    sensitivity_comparison = sensitivity["comparisons"][2]
    assert sensitivity_comparison["left_minus_right_percentage_points"] == 0.0
    assert sensitivity_comparison["mcnemar_exact_two_sided_p_value"] == 1.0
    assert sensitivity_comparison["removed_channel_selection"]["overlap_count"] == 164
    sensitivity_report = root / sensitivity["report_path"]
    normalized_sensitivity_report = sensitivity_report.read_bytes().replace(b"\r\n", b"\n")
    assert (
        hashlib.sha256(normalized_sensitivity_report).hexdigest()
        == sensitivity["report_sha256_lf_normalized"]
    )
    assert "No third-party source code copied" in report["method_provenance"]["reuse"]
    for item in report["source_reports"]:
        source = root / item["path"]
        normalized = source.read_bytes().replace(b"\r\n", b"\n")
        digest = hashlib.sha256(normalized).hexdigest()
        assert digest == item["sha256_lf_normalized"]


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
