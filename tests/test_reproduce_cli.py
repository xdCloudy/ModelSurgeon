"""End-to-end persisted-run reconstruction and metric comparison tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import modelsurgeon.cli.app as app_module
from modelsurgeon.cli.reproduce import (
    ReproductionCommandError,
    ReproductionPlan,
    prepare_reproduction,
    run_reproduction,
)
from modelsurgeon.experiments import (
    ContentAddressedArtifactStore,
    CPUInventory,
    CUDAInventory,
    DatasetTarget,
    DiskInventory,
    ExperimentIdentitySpec,
    ExperimentMetadataStore,
    ExperimentOutcome,
    ExperimentOutcomeKind,
    ExperimentRecord,
    GitRevision,
    HardwareInventory,
    LockDigest,
    MemoryInventory,
    MetricObservation,
    MetricState,
    MetricTolerance,
    ModelTarget,
    SeedContext,
    SoftwareInventory,
    VersionContext,
    capture_reproducibility_manifest,
    derive_experiment_identity,
    derive_run_identity,
    publish_reproducibility_manifest,
)
from modelsurgeon.graph import ComponentId
from modelsurgeon.surgery.contracts import (
    MutationDelta,
    MutationKind,
    MutationPlan,
    MutationRequest,
)
from modelsurgeon.surgery.serialization import (
    MutationOutcome,
    MutationOutcomeStatus,
    MutationProvenance,
    MutationRunRecord,
)


def _hardware() -> HardwareInventory:
    return HardwareInventory(
        "Windows",
        "11",
        "build",
        CPUInventory("AMD64", "test-cpu", 8),
        MemoryInventory(64 * 1024**3, 32 * 1024**3),
        DiskInventory("C:/original", 1024**4, 512 * 1024**3),
        CUDAInventory(False, None, (), ()),
        SoftwareInventory("3.12.14", "CPython", "0.0.1", None),
    )


def _record() -> tuple[ExperimentRecord, dict[str, object]]:
    config: dict[str, object] = {"calibration": {"samples": 8}, "seed": 7}
    model = ModelTarget("tiny/model", "model-rev", "llama", "safetensors", 128)
    dataset = DatasetTarget(
        "tiny/data",
        "data-rev",
        "validation",
        "manifest-1",
        "tiny/tokenizer",
        "tokenizer-rev",
    )
    seeds = SeedContext(7, 8, 9)
    identity = derive_experiment_identity(
        ExperimentIdentitySpec(model, dataset, config, seeds, "tool-rev", "eval-v1", 1, 1)
    )
    run = derive_run_identity(identity.experiment_id)
    component = ComponentId.parse("model.layers.0.mlp.up_proj")
    request = MutationRequest(MutationKind.MASK, (component,))
    delta = MutationDelta(parameters=-1)
    mutation = MutationRunRecord(
        MutationPlan(request, (component,), (), delta),
        MutationProvenance(model.revision, "tool-rev"),
        MutationOutcome(MutationOutcomeStatus.ROLLED_BACK, delta),
    )
    return (
        ExperimentRecord(
            run.run_id,
            identity.experiment_id,
            "attempt-1",
            model,
            dataset,
            (component,),
            mutation,
            (MetricObservation("perplexity", MetricState.MEASURED, 10.0),),
            (MetricObservation("perplexity", MetricState.MEASURED, 10.25),),
            (MetricObservation("perplexity", MetricState.MEASURED, 0.25),),
            ExperimentOutcome(ExperimentOutcomeKind.SUCCEEDED),
            _hardware(),
            VersionContext("tool-rev", identity.config_digest, "eval-v1", 1, 1),
            seeds,
        ),
        config,
    )


def _persisted_recipe(tmp_path: Path) -> tuple[str, Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    record, config = _record()
    metadata_path = tmp_path / "metadata.sqlite3"
    artifact_root = tmp_path / "artifacts"
    lock_path = tmp_path / "uv.lock"
    lock_path.write_text("locked\n", encoding="utf-8")
    lock = LockDigest("uv.lock", "b" * 64)
    manifest = capture_reproducibility_manifest(
        record,
        git=GitRevision("a" * 40, True),
        lock=lock,
        resolved_config=config,
        command=("modelsurgeon", "experiment", "mutation.json", "--runtime", "fixture:run"),
        metric_tolerances=(
            MetricTolerance("baseline:perplexity", absolute=0.1),
            MetricTolerance("delta:perplexity", relative=0.1),
            MetricTolerance("post:perplexity", absolute=0.1),
        ),
    )
    with ExperimentMetadataStore(metadata_path) as metadata:
        persisted = metadata.persist_experiment(record)
        publish_reproducibility_manifest(
            manifest,
            persisted,
            artifact_store=ContentAddressedArtifactStore(artifact_root),
            metadata_store=metadata,
        )
    return record.run_id, metadata_path, artifact_root, lock_path


def _plan(tmp_path: Path, *, git: GitRevision | None = None) -> ReproductionPlan:
    run_id, metadata, artifacts, lock = _persisted_recipe(tmp_path)
    return prepare_reproduction(
        run_id,
        metadata_path=metadata,
        artifact_root=artifacts,
        repository_root=tmp_path,
        lock_path=lock,
        current_git=git or GitRevision("a" * 40, True),
        current_lock=LockDigest("uv.lock", "b" * 64),
        current_hardware=_hardware(),
    )


class _Executor:
    def __init__(self, metrics: dict[str, float]) -> None:
        self.metrics = metrics
        self.plan: ReproductionPlan | None = None

    def execute(self, plan: ReproductionPlan) -> dict[str, float]:
        self.plan = plan
        return self.metrics


def test_dry_plan_reconstructs_exact_inputs_command_and_environment(tmp_path: Path) -> None:
    plan = _plan(tmp_path)

    assert plan.executable
    assert plan.resolved_config == {"calibration": {"samples": 8}, "seed": 7}
    assert plan.command == (
        "modelsurgeon",
        "experiment",
        "mutation.json",
        "--runtime",
        "fixture:run",
    )
    assert dict(plan.original_metrics) == {
        "baseline:perplexity": 10.0,
        "delta:perplexity": 0.25,
        "post:perplexity": 10.25,
    }
    assert plan.to_record()["original_run_id"] == plan.run_id


def test_rerun_metrics_link_to_original_and_apply_declared_tolerances(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    executor = _Executor(
        {
            "baseline:perplexity": 10.05,
            "delta:perplexity": 0.26,
            "post:perplexity": 10.20,
        }
    )

    result = run_reproduction(plan, executor)

    assert result.passed
    assert result.run_id == plan.run_id
    assert result.candidate_id == plan.candidate_id
    assert result.reproduction_id.startswith("reproduction_")
    assert executor.plan == plan
    assert all(item.passed for item in result.comparisons)


def test_environment_mismatch_and_metric_drift_fail_closed(tmp_path: Path) -> None:
    mismatch = _plan(tmp_path / "mismatch", git=GitRevision("c" * 40, True))
    assert not mismatch.executable
    assert mismatch.mismatches[0].code == "git_revision"
    with pytest.raises(ReproductionCommandError, match="environment does not match"):
        run_reproduction(mismatch, _Executor({}))

    plan = _plan(tmp_path / "drift")
    result = run_reproduction(
        plan,
        _Executor(
            {
                "baseline:perplexity": 11.0,
                "delta:perplexity": 0.25,
                "post:perplexity": 10.25,
            }
        ),
    )
    assert not result.passed
    failed = [item for item in result.comparisons if not item.passed]
    assert [(item.metric, item.absolute_delta, item.allowed_delta) for item in failed] == [
        ("baseline:perplexity", 1.0, 0.1)
    ]


def test_cli_dry_run_prints_canonical_plan_without_loading_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan(tmp_path)
    monkeypatch.setattr(app_module, "prepare_reproduction", None, raising=False)
    monkeypatch.setattr("modelsurgeon.cli.reproduce.prepare_reproduction", lambda *a, **k: plan)

    result = CliRunner().invoke(
        app_module.app,
        [
            "reproduce",
            plan.run_id,
            "--metadata",
            str(tmp_path / "unused.sqlite3"),
            "--artifacts",
            str(tmp_path / "unused-artifacts"),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["record_type"] == "reproduction_plan"
    assert payload["exact_command"] == list(plan.command)
    assert payload["exact_inputs"]["model"]["revision"] == "model-rev"


def test_missing_read_paths_fail_without_creating_metadata_or_artifacts(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "missing.sqlite3"
    artifacts = tmp_path / "missing-artifacts"

    with pytest.raises(ReproductionCommandError, match="existing regular SQLite"):
        prepare_reproduction(
            "run_" + "a" * 64,
            metadata_path=metadata,
            artifact_root=artifacts,
            repository_root=tmp_path,
            lock_path=tmp_path / "uv.lock",
        )

    assert not metadata.exists()
    assert not artifacts.exists()
