"""Tests for immutable content-addressed experiment artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from modelsurgeon.experiments import (
    ArtifactDigest,
    ArtifactStoreError,
    CPUInventory,
    CUDAInventory,
    ContentAddressedArtifactStore,
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
        CUDAInventory(False, None, (), ()),
        SoftwareInventory("3.12", "CPython", "0.0.1", None),
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
    seeds = SeedContext(1, 2, 3)
    identity = derive_experiment_identity(
        ExperimentIdentitySpec(model, dataset, {}, seeds, "tool", "1", 1, 1)
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
        VersionContext("tool", identity.config_digest, "1", 1, 1),
        seeds,
    )


def test_duplicate_content_reuses_one_immutable_publication(tmp_path: Path) -> None:
    store = ContentAddressedArtifactStore(tmp_path / "artifacts", chunk_bytes=3)
    first = store.put_bytes(b"duplicate-content")
    first_mtime = first.metadata_path.stat().st_mtime_ns
    second = store.put_bytes(b"duplicate-content")

    assert first.metadata.digest == second.metadata.digest
    assert first.directory == second.directory
    assert second.data_path.read_bytes() == b"duplicate-content"
    assert second.metadata_path.stat().st_mtime_ns == first_mtime
    assert store.exists(first.metadata.digest)


def test_expected_digest_mismatch_is_rejected_without_publication(tmp_path: Path) -> None:
    store = ContentAddressedArtifactStore(tmp_path / "artifacts")
    wrong = ArtifactDigest("sha256", "0" * 64)
    with pytest.raises(ArtifactStoreError, match="digest mismatch"):
        store.put_bytes(b"actual-content", expected_digest=wrong)

    assert not store.exists(wrong)
    staging = store.root / ".staging"
    assert not staging.exists() or not tuple(staging.iterdir())


def test_incomplete_or_corrupt_final_content_fails_closed(tmp_path: Path) -> None:
    payload = b"known-content"
    digest_hex = hashlib.sha256(payload).hexdigest()
    digest = ArtifactDigest("sha256", digest_hex)
    store = ContentAddressedArtifactStore(tmp_path / "artifacts")
    directory = store.algorithm_root / digest_hex[:2] / digest_hex[2:]
    directory.mkdir(parents=True)
    (directory / "data").write_bytes(payload)

    with pytest.raises(ArtifactStoreError, match="incomplete"):
        store.get(digest)
    with pytest.raises(ArtifactStoreError, match="incomplete"):
        store.put_bytes(payload)

    (directory / "metadata.json").write_text(
        '{"digest":"sha256:' + digest_hex + '","schema_version":1,"size_bytes":1}',
        encoding="utf-8",
    )
    with pytest.raises(ArtifactStoreError, match="size"):
        store.get(digest)


def test_file_streaming_uses_bounded_chunks_and_verifies_digest(tmp_path: Path) -> None:
    source = tmp_path / "large.bin"
    payload = bytes(range(256)) * 4096
    source.write_bytes(payload)
    expected = ArtifactDigest("sha256", hashlib.sha256(payload).hexdigest())
    store = ContentAddressedArtifactStore(tmp_path / "artifacts", chunk_bytes=1024)

    artifact = store.put_file(source, expected_digest=expected)
    assert artifact.metadata.size_bytes == len(payload)
    assert artifact.metadata.digest == expected
    assert store.get(expected).data_path.read_bytes() == payload


def test_candidate_publication_tracks_reference_in_sqlite(tmp_path: Path) -> None:
    with ExperimentMetadataStore(tmp_path / "metadata.sqlite3") as metadata_store:
        persisted = metadata_store.persist_experiment(_record())
        artifact_store = ContentAddressedArtifactStore(tmp_path / "artifacts")
        published = artifact_store.publish_for_candidate(
            metadata_store,
            persisted.candidate_id,
            role="evaluation-log",
            payload=b'{"event":"done"}\n',
            reference_metadata={"format": "jsonl", "records": 1},
        )

        references = metadata_store.list_artifact_references(persisted.candidate_id)
        assert references == (published.reference,)
        assert published.reference.digest == str(published.artifact.metadata.digest)
        assert published.reference.metadata == {"format": "jsonl", "records": 1}


def test_digest_parser_and_symlink_sources_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ArtifactStoreError, match="64 hexadecimal"):
        ArtifactDigest.parse("sha256:abc")
    with pytest.raises(ArtifactStoreError, match="algorithm"):
        ArtifactDigest.parse("md5:" + "0" * 64)

    source = tmp_path / "source.bin"
    source.write_bytes(b"data")
    link = tmp_path / "source-link.bin"
    try:
        link.symlink_to(source)
    except OSError:
        pytest.skip("symlinks are not available on this platform")
    store = ContentAddressedArtifactStore(tmp_path / "artifacts")
    with pytest.raises(ArtifactStoreError, match="regular file"):
        store.put_file(link)
