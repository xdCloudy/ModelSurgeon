"""Tests for the WAL-backed SQLite experiment metadata store."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from modelsurgeon.experiments import (
    CPUInventory,
    CUDAInventory,
    DatasetTarget,
    DiskInventory,
    ExperimentIdentitySpec,
    ExperimentMetadataStore,
    ExperimentOutcome,
    ExperimentOutcomeKind,
    ExperimentRecord,
    ExperimentStoreError,
    HardwareInventory,
    MemoryInventory,
    MetricObservation,
    MetricPhase,
    MetricState,
    ModelTarget,
    SeedContext,
    SoftwareInventory,
    StageTiming,
    VersionContext,
    derive_experiment_identity,
    derive_run_identity,
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
        "Linux",
        "test",
        "test-version",
        CPUInventory("x86_64", "test-cpu", 8),
        MemoryInventory(4096, 2048),
        DiskInventory("/tmp", 8192, 4096),
        CUDAInventory(False, None, (), ()),
        SoftwareInventory("3.12", "CPython", "0.0.1", None),
    )


def _record(*, attempt_id: str = "attempt-1") -> ExperimentRecord:
    model = ModelTarget("tiny/model", "model-rev-1", "llama", "safetensors", 128)
    dataset = DatasetTarget(
        "tiny/data",
        "data-rev-1",
        "validation",
        "manifest-1",
        "tiny/tokenizer",
        "tokenizer-rev-1",
    )
    seeds = SeedContext(1, 2, 3)
    identity = derive_experiment_identity(
        ExperimentIdentitySpec(
            model,
            dataset,
            {"evaluation": {"max_tier": 2}},
            seeds,
            "tool",
            "1",
            1,
            1,
        )
    )
    run = derive_run_identity(identity.experiment_id)
    component = ComponentId.parse("model.layers.0.mlp.up_proj")
    request = MutationRequest(MutationKind.MASK, (component,))
    delta = MutationDelta(parameters=-8, flops=-16)
    mutation = MutationRunRecord(
        MutationPlan(request, (component,), (), delta),
        MutationProvenance(model.revision, "tool"),
        MutationOutcome(MutationOutcomeStatus.ROLLED_BACK, delta),
    )
    return ExperimentRecord(
        run.run_id,
        identity.experiment_id,
        attempt_id,
        model,
        dataset,
        (component,),
        mutation,
        (
            MetricObservation("loss", MetricState.MEASURED, 1.0),
            MetricObservation("perplexity", MetricState.MEASURED, 2.0),
        ),
        (MetricObservation("perplexity", MetricState.MEASURED, 2.1),),
        (MetricObservation("perplexity_delta", MetricState.MEASURED, 0.1),),
        ExperimentOutcome(ExperimentOutcomeKind.SUCCEEDED),
        _hardware(),
        VersionContext("tool", identity.config_digest, "1", 1, 1),
        seeds,
        (StageTiming("evaluation", 0.25, 0.2, tokens=32, candidates=1),),
    )


def test_persist_query_and_trace_full_immutable_context(tmp_path: Path) -> None:
    database = tmp_path / "experiments.sqlite3"
    record = _record()
    with ExperimentMetadataStore(database) as store:
        persisted = store.persist_experiment(record)
        assert store.journal_mode == "wal"
        assert store.persist_experiment(record) == persisted

        stored_input = store.get_input(persisted.input_id)
        assert stored_input is not None
        assert stored_input.model_revision == "model-rev-1"
        assert stored_input.config_digest == record.versions.config_digest

        stored_run = store.get_run(record.run_id)
        assert stored_run is not None
        assert stored_run.input_id == persisted.input_id
        assert stored_run.experiment_id == record.experiment_id

        candidate = store.get_candidate(persisted.candidate_id)
        assert candidate is not None
        assert candidate.run_id == record.run_id
        assert candidate.affected_components == ("model.layers.0.mlp.up_proj",)

        metrics = store.list_metrics(persisted.candidate_id)
        assert [(item.phase, item.name) for item in metrics] == [
            (MetricPhase.BASELINE, "loss"),
            (MetricPhase.BASELINE, "perplexity"),
            (MetricPhase.DELTA, "perplexity_delta"),
            (MetricPhase.POST, "perplexity"),
        ]
        trace = store.trace_candidate(persisted.candidate_id)
        assert trace is not None
        assert trace.model_revision == "model-rev-1"
        assert trace.dataset_manifest_id == "manifest-1"
        assert trace.config_digest == record.versions.config_digest


def test_states_are_append_only_and_artifacts_keep_foreign_keys(tmp_path: Path) -> None:
    with ExperimentMetadataStore(tmp_path / "metadata.sqlite3") as store:
        persisted = store.persist_experiment(_record())
        first = store.append_state(persisted.candidate_id, "queued")
        second = store.append_state(persisted.candidate_id, "evaluating", "Tier 1")
        assert (first.sequence, second.sequence) == (0, 1)
        assert [item.state for item in store.list_states(persisted.candidate_id)] == [
            "queued",
            "evaluating",
        ]

        artifact = store.add_artifact_reference(
            persisted.candidate_id,
            role="evaluation-log",
            digest="sha256:abc",
            metadata={"format": "jsonl", "bytes": 42},
        )
        assert store.add_artifact_reference(
            persisted.candidate_id,
            role="evaluation-log",
            digest="sha256:abc",
            metadata={"bytes": 42, "format": "jsonl"},
        ) == artifact
        assert store.list_artifact_references(persisted.candidate_id) == (artifact,)

        with pytest.raises(ExperimentStoreError, match="unknown candidate"):
            store.add_artifact_reference(
                "cand_missing",
                role="evaluation-log",
                digest="sha256:def",
            )
        with pytest.raises(ExperimentStoreError, match="unknown experiment candidate"):
            store.append_state("cand_missing", "queued")


def test_conflicting_immutable_run_identity_fails_closed(tmp_path: Path) -> None:
    with ExperimentMetadataStore(tmp_path / "metadata.sqlite3") as store:
        store.persist_experiment(_record())
        with pytest.raises(ExperimentStoreError, match="conflicts"):
            store.persist_experiment(_record(attempt_id="attempt-2"))


def test_wal_reader_remains_available_while_writer_lock_is_held(tmp_path: Path) -> None:
    database = tmp_path / "metadata.sqlite3"
    with ExperimentMetadataStore(database) as store:
        persisted = store.persist_experiment(_record())
        writer = sqlite3.connect(database, timeout=0.1)
        try:
            assert str(writer.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"
            writer.execute("BEGIN IMMEDIATE")
            writer.execute(
                "UPDATE experiment_runs SET attempt_id = attempt_id WHERE run_id = ?",
                (persisted.run_id,),
            )
            with store.reader() as reader:
                assert str(reader.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"
                row = reader.execute(
                    "SELECT attempt_id FROM experiment_runs WHERE run_id = ?",
                    (persisted.run_id,),
                ).fetchone()
                assert row is not None
                assert row[0] == "attempt-1"
            writer.rollback()
        finally:
            writer.close()


def test_closed_store_rejects_new_readers(tmp_path: Path) -> None:
    store = ExperimentMetadataStore(tmp_path / "metadata.sqlite3")
    store.close()
    with pytest.raises(ExperimentStoreError, match="closed"), store.reader():
        pass
