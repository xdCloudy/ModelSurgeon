"""Tests for immutable experiment reproducibility manifests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

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
    HardwareInventory,
    MemoryInventory,
    ModelTarget,
    SeedContext,
    SoftwareInventory,
    VersionContext,
    derive_experiment_identity,
    derive_run_identity,
)
from modelsurgeon.experiments.reproducibility import (
    REPRODUCIBILITY_ARTIFACT_ROLE,
    GitRevision,
    LockDigest,
    ReproducibilityError,
    ReproducibilityIssueCode,
    capture_reproducibility_manifest,
    digest_lock_file,
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
        "Linux",
        "test",
        "test-version",
        CPUInventory("x86_64", "test-cpu", 4),
        MemoryInventory(1024, 512),
        DiskInventory("/tmp", 4096, 2048),
        CUDAInventory(True, "12.8", ("590.1",), ()),
        SoftwareInventory("3.12.14", "CPython", "0.0.1", "2.9.0"),
    )


def _record() -> ExperimentRecord:
    model = ModelTarget("tiny/model", "model-rev", "llama", "safetensors", 128)
    dataset = DatasetTarget(
        "tiny/data",
        "data-rev",
        "validation",
        "manifest-1",
        "tiny/tokenizer",
        "tokenizer-rev",
    )
    seeds = SeedContext(11, 22, 33)
    identity = derive_experiment_identity(
        ExperimentIdentitySpec(model, dataset, {"batch": 4}, seeds, "tool", "eval-1", 1, 1)
    )
    run = derive_run_identity(identity.experiment_id)
    component = ComponentId.parse("model.layers.0.mlp.up_proj")
    request = MutationRequest(MutationKind.MASK, (component,))
    delta = MutationDelta(parameters=-1)
    mutation = MutationRunRecord(
        MutationPlan(request, (component,), (), delta),
        MutationProvenance(model.revision, "tool"),
        MutationOutcome(MutationOutcomeStatus.ROLLED_BACK, delta),
    )
    return ExperimentRecord(
        run.run_id,
        identity.experiment_id,
        "attempt-1",
        model,
        dataset,
        (component,),
        mutation,
        (),
        (),
        (),
        ExperimentOutcome(ExperimentOutcomeKind.SUCCEEDED),
        _hardware(),
        VersionContext("tool", identity.config_digest, "eval-1", 1, 1),
        seeds,
    )


def _git(*, clean: bool = True) -> GitRevision:
    return GitRevision("a" * 40, clean)


def _lock() -> LockDigest:
    return LockDigest("uv.lock", "b" * 64)


def test_complete_manifest_is_reproducible_and_canonical() -> None:
    manifest = capture_reproducibility_manifest(_record(), git=_git(), lock=_lock())

    assert manifest.reproducible
    assert manifest.issues == ()
    manifest.require_reproducible()
    assert manifest.manifest_id.startswith("repro_")
    record = manifest.to_record()
    assert record["git"] == {"commit": "a" * 40, "clean": True}
    assert record["lock"] == {
        "name": "uv.lock",
        "algorithm": "sha256",
        "hexadecimal": "b" * 64,
    }
    assert record["hardware"] == _hardware().to_record()
    assert record["seeds"] == {"experiment_seed": 11, "data_seed": 22, "mutation_seed": 33}
    assert manifest.canonical_json() == manifest.canonical_json()


def test_missing_git_or_lock_and_dirty_tree_block_reproducible_status() -> None:
    manifest = capture_reproducibility_manifest(
        _record(),
        git=GitRevision(None, False),
        lock=LockDigest("uv.lock", None),
    )

    assert not manifest.reproducible
    codes = {item.code for item in manifest.issues}
    assert codes == {
        ReproducibilityIssueCode.MISSING_GIT_REVISION,
        ReproducibilityIssueCode.DIRTY_GIT_WORKTREE,
        ReproducibilityIssueCode.MISSING_LOCK_DIGEST,
    }
    with pytest.raises(ReproducibilityError, match="run is not reproducible"):
        manifest.require_reproducible()


def test_dirty_worktree_blocks_reproduction_even_with_exact_commit() -> None:
    manifest = capture_reproducibility_manifest(_record(), git=_git(clean=False), lock=_lock())
    assert not manifest.reproducible
    assert tuple(item.code for item in manifest.issues) == (
        ReproducibilityIssueCode.DIRTY_GIT_WORKTREE,
    )


def test_lock_digest_streams_exact_file_and_rejects_symlink(tmp_path: Path) -> None:
    lock_file = tmp_path / "uv.lock"
    payload = (b"locked-dependency\n" * 100_000) + b"end"
    lock_file.write_bytes(payload)

    identity = digest_lock_file(lock_file)
    assert identity.name == "uv.lock"
    assert identity.hexadecimal == hashlib.sha256(payload).hexdigest()

    link = tmp_path / "lock-link"
    try:
        link.symlink_to(lock_file)
    except OSError:
        pytest.skip("symlinks are not available on this platform")
    with pytest.raises(ReproducibilityError, match="regular file"):
        digest_lock_file(link)


def test_git_and_lock_identity_validation_fail_closed() -> None:
    with pytest.raises(ReproducibilityError, match="full SHA"):
        GitRevision("abc", True)
    with pytest.raises(ReproducibilityError, match="lowercase"):
        GitRevision("A" * 40, True)
    with pytest.raises(ReproducibilityError, match="64 lowercase"):
        LockDigest("uv.lock", "xyz")
    with pytest.raises(ReproducibilityError, match="non-empty name"):
        LockDigest("", "b" * 64)


def test_manifest_publication_is_immutable_and_linked_to_persisted_run(
    tmp_path: Path,
) -> None:
    record = _record()
    manifest = capture_reproducibility_manifest(record, git=_git(), lock=_lock())
    artifact_store = ContentAddressedArtifactStore(tmp_path / "artifacts")
    with ExperimentMetadataStore(tmp_path / "metadata.sqlite3") as metadata_store:
        persisted = metadata_store.persist_experiment(record)
        first = publish_reproducibility_manifest(
            manifest,
            persisted,
            artifact_store=artifact_store,
            metadata_store=metadata_store,
        )
        second = publish_reproducibility_manifest(
            manifest,
            persisted,
            artifact_store=artifact_store,
            metadata_store=metadata_store,
        )

        assert first.artifact.metadata.digest == second.artifact.metadata.digest
        assert first.reference == second.reference
        assert first.reference.role == REPRODUCIBILITY_ARTIFACT_ROLE
        assert first.reference.metadata["manifest_id"] == manifest.manifest_id
        assert first.reference.metadata["run_id"] == record.run_id
        stored = first.artifact.data_path.read_text(encoding="utf-8")
        assert stored == manifest.canonical_json() + "\n"
        references = metadata_store.list_artifact_references(persisted.candidate_id)
        assert references == (first.reference,)


def test_manifest_cannot_be_linked_to_a_different_persisted_run(tmp_path: Path) -> None:
    record = _record()
    manifest = capture_reproducibility_manifest(record, git=_git(), lock=_lock())
    artifact_store = ContentAddressedArtifactStore(tmp_path / "artifacts")
    with ExperimentMetadataStore(tmp_path / "metadata.sqlite3") as metadata_store:
        persisted = metadata_store.persist_experiment(record)
        wrong = type(persisted)(persisted.input_id, "run_" + "f" * 64, persisted.candidate_id)
        with pytest.raises(ReproducibilityError, match="persisted run does not match"):
            publish_reproducibility_manifest(
                manifest,
                wrong,
                artifact_store=artifact_store,
                metadata_store=metadata_store,
            )
