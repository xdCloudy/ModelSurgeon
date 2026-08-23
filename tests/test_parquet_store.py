"""Tests for partitioned feature/example Parquet publication."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

import pytest

import modelsurgeon.datasets.parquet_store as parquet_module
from modelsurgeon.datasets import (
    ParquetDependencyError,
    ParquetPartition,
    ParquetStoreError,
    PartitionedParquetStore,
    PartitionKind,
    PartitionPredicate,
    PyArrowParquetBackend,
)
from modelsurgeon.experiments import (
    CPUInventory,
    CUDAInventory,
    DatasetTarget,
    DiskInventory,
    ExperimentOutcome,
    ExperimentOutcomeKind,
    HardwareInventory,
    MemoryInventory,
    ModelTarget,
    MutationExampleRecord,
    SeedContext,
    SoftwareInventory,
    VersionContext,
)
from modelsurgeon.features.schema import (
    FeatureKind,
    FeatureRecord,
    PrecisionProvenance,
    PrecisionSource,
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


class JsonTestBackend:
    """Deterministic backend used to test store logic without optional PyArrow."""

    def __init__(self, *, fail_write: bool = False) -> None:
        self.fail_write = fail_write
        self.read_paths: list[Path] = []
        self.write_paths: list[Path] = []

    def write_rows(self, path: Path, rows: Sequence[Mapping[str, object]]) -> None:
        self.write_paths.append(path)
        path.write_text(
            json.dumps([dict(row) for row in rows], sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        if self.fail_write:
            raise RuntimeError("injected backend failure")

    def read_rows(self, path: Path) -> tuple[dict[str, object], ...]:
        self.read_paths.append(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(value, list)
        return tuple(dict(row) for row in value)


def _feature(component: str, name: str) -> FeatureRecord:
    return FeatureRecord(
        ComponentId.parse(component),
        name,
        FeatureKind.SCALAR,
        1.25,
        "float64",
        "test_extractor",
        "1",
        PrecisionProvenance(PrecisionSource.HIGH_PRECISION, "float32", "float64"),
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


def _example() -> MutationExampleRecord:
    model = ModelTarget("model/a", "rev-a", "llama", "safetensors", 128)
    dataset = DatasetTarget(
        "dataset/a",
        "data-rev",
        "validation",
        "manifest-a",
        "tokenizer/a",
        "tokenizer-rev",
    )
    component = ComponentId.parse("model.layers.0.mlp.up_proj")
    request = MutationRequest(MutationKind.MASK, (component,))
    delta = MutationDelta(parameters=-1)
    mutation = MutationRunRecord(
        MutationPlan(request, (component,), (), delta),
        MutationProvenance(model.revision, "tool"),
        MutationOutcome(MutationOutcomeStatus.ROLLED_BACK, delta),
    )
    return MutationExampleRecord(
        "example-a",
        "experiment-a",
        model,
        dataset,
        (component,),
        mutation,
        (_feature(str(component), "weight_mean"),),
        (),
        (),
        (),
        ExperimentOutcome(ExperimentOutcomeKind.SUCCEEDED),
        _hardware(),
        VersionContext("tool", "config", "1", 1, 1),
        SeedContext(1, 2, 3),
    )


def test_predicate_reads_prune_unrelated_partitions(tmp_path: Path) -> None:
    backend = JsonTestBackend()
    store = PartitionedParquetStore(tmp_path, backend)
    first_partition = ParquetPartition(
        PartitionKind.FEATURES,
        1,
        "model/a",
        "rev-a",
        "campaign-a",
    )
    second_partition = ParquetPartition(
        PartitionKind.FEATURES,
        1,
        "model/b",
        "rev-b",
        "campaign-b",
    )
    first = store.write_features(
        first_partition,
        (_feature("model.layers.0.mlp.up_proj", "weight_mean"),),
    )
    second = store.write_features(
        second_partition,
        (_feature("model.layers.1.mlp.up_proj", "weight_variance"),),
    )
    backend.read_paths.clear()

    rows = store.read_rows(
        PartitionPredicate(
            kind=PartitionKind.FEATURES,
            model_identifier="model/a",
            campaign_id="campaign-a",
        )
    )

    expected_path = store.root / Path(*PurePosixPath(first.relative_path).parts)
    assert len(rows) == 1
    assert rows[0]["component_id"] == "model.layers.0.mlp.up_proj"
    assert backend.read_paths == [expected_path]
    scanned = {path.relative_to(store.root).as_posix() for path in backend.read_paths}
    assert second.relative_path not in scanned
    assert len(store.load_manifest().entries) == 2


def test_partial_backend_writes_are_never_visible(tmp_path: Path) -> None:
    backend = JsonTestBackend(fail_write=True)
    store = PartitionedParquetStore(tmp_path, backend)
    partition = ParquetPartition(
        PartitionKind.FEATURES,
        1,
        "model/a",
        "rev-a",
        "campaign-a",
    )
    with pytest.raises(RuntimeError, match="injected"):
        store.write_features(
            partition,
            (_feature("model.layers.0.mlp.up_proj", "weight_mean"),),
        )

    assert store.load_manifest().entries == ()
    assert store.read_rows() == ()
    assert not tuple(store.root.rglob("*.partial"))


def test_unmanifested_parquet_files_are_excluded_from_reads(tmp_path: Path) -> None:
    backend = JsonTestBackend()
    store = PartitionedParquetStore(tmp_path, backend)
    orphan = store.root / "schema=1" / "orphan.parquet"
    orphan.parent.mkdir(parents=True)
    orphan.write_text('[{"record_type":"orphan"}]', encoding="utf-8")

    assert store.read_rows() == ()
    assert backend.read_paths == []


def test_example_partition_preserves_example_identity_and_payload(tmp_path: Path) -> None:
    backend = JsonTestBackend()
    store = PartitionedParquetStore(tmp_path, backend)
    example = _example()
    partition = ParquetPartition(
        PartitionKind.EXAMPLES,
        example.schema_version,
        example.model.identifier,
        example.model.revision,
        "campaign-a",
    )
    entry = store.write_examples(partition, (example,))

    rows = store.read_rows(PartitionPredicate(kind=PartitionKind.EXAMPLES))
    assert entry.row_count == 1
    assert rows[0]["example_id"] == example.example_id
    assert rows[0]["mutation_id"] == example.mutation_id
    payload = json.loads(str(rows[0]["payload_json"]))
    assert payload["model"]["revision"] == "rev-a"


def test_partition_schema_and_model_mismatches_fail_closed(tmp_path: Path) -> None:
    store = PartitionedParquetStore(tmp_path, JsonTestBackend())
    with pytest.raises(ParquetStoreError, match="schema versions"):
        store.write_features(
            ParquetPartition(PartitionKind.FEATURES, 2, "model/a", "rev-a", "campaign"),
            (_feature("model.layers.0.mlp.up_proj", "weight_mean"),),
        )

    example = _example()
    with pytest.raises(ParquetStoreError, match="model identity"):
        store.write_examples(
            ParquetPartition(PartitionKind.EXAMPLES, 1, "other", "rev-a", "campaign"),
            (example,),
        )


def test_lazy_pyarrow_backend_reports_actionable_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fail_import(name: str) -> object:
        raise ImportError(name)

    monkeypatch.setattr(parquet_module, "import_module", fail_import)
    backend = PyArrowParquetBackend()
    with pytest.raises(ParquetDependencyError, match="pyarrow"):
        backend.write_rows(tmp_path / "rows.parquet", ({"value": 1},))
