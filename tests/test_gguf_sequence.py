"""Contract tests for cumulative native GGUF surgery coordination."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from modelsurgeon.surgery import (
    GGUFCumulativeError,
    GGUFEdit,
    GGUFResourceLimits,
    GGUFSourceState,
    GGUFStageMeasurement,
    run_gguf_cumulative_sequence,
)


def _sequence() -> tuple[GGUFSourceState, tuple[GGUFEdit, ...]]:
    names = ("tensor.a", "tensor.b", "tensor.c")
    payloads = {name: hashlib.sha256(name.encode()).hexdigest() for name in names}
    current_parameters = 300
    current_storage = 1000

    def make_edit(index: int, name: str) -> GGUFEdit:
        nonlocal current_parameters, current_storage

        def execute(source: Path, destination: Path) -> GGUFStageMeasurement:
            nonlocal current_parameters, current_storage
            destination.write_bytes(source.read_bytes()[:-10])
            current_parameters -= 10
            current_storage -= 10
            payloads[name] = hashlib.sha256(destination.read_bytes() + name.encode()).hexdigest()
            return GGUFStageMeasurement(
                current_parameters,
                current_storage,
                hashlib.sha256(str(current_parameters).encode()).hexdigest(),
                tuple(sorted(payloads.items())),
                (name,),
                1024 + index,
                2048 + index,
                True,
                True,
            )

        return GGUFEdit(f"edit-{index}", f"native-{name}", execute)

    source_state = GGUFSourceState(
        300,
        1000,
        hashlib.sha256(b"source").hexdigest(),
        tuple(sorted(payloads.items())),
    )
    return source_state, tuple(make_edit(index, name) for index, name in enumerate(names))


def test_three_native_stages_preserve_untouched_payloads_and_reconcile(tmp_path: Path) -> None:
    source_state, edits = _sequence()
    source = tmp_path / "source.gguf"
    source.write_bytes(b"x" * 1000)

    run = run_gguf_cumulative_sequence(
        source,
        source_state,
        edits,
        output_root=tmp_path / "children",
        source_outcome_id="source-gguf-outcome",
        limits=GGUFResourceLimits(4096, 4096),
    )

    assert len(run.stages) == 3
    assert run.failed_index is None
    assert run.reconciliation.valid
    assert run.reconciliation.cumulative_parameter_delta == -30
    assert run.reconciliation.cumulative_storage_delta == -30
    assert all(stage.artifact.is_file() for stage in run.stages)
    assert run.stages[0].unchanged_tensor_names == ("tensor.b", "tensor.c")


def test_memory_limit_failure_removes_unpublished_child(tmp_path: Path) -> None:
    source_state, edits = _sequence()
    source = tmp_path / "source.gguf"
    source.write_bytes(b"x" * 1000)

    with pytest.raises(GGUFCumulativeError, match="working-memory"):
        run_gguf_cumulative_sequence(
            source,
            source_state,
            edits,
            output_root=tmp_path / "children",
            source_outcome_id="source-gguf-outcome",
            limits=GGUFResourceLimits(512, 4096),
        )

    assert not list((tmp_path / "children").glob("*.gguf"))


def test_changed_untouched_payload_is_rejected_and_cleaned(tmp_path: Path) -> None:
    source_state, _ = _sequence()
    source = tmp_path / "source.gguf"
    source.write_bytes(b"x" * 1000)
    bad = GGUFEdit(
        "bad-edit",
        "bad-payload",
        lambda source_path, destination: (
            destination.write_bytes(source_path.read_bytes()[:-10]),
            GGUFStageMeasurement(
                290,
                990,
                hashlib.sha256(b"bad").hexdigest(),
                tuple(
                    (name, hashlib.sha256(b"changed").hexdigest())
                    for name, _ in source_state.tensor_payload_sha256
                ),
                ("tensor.a",),
                1024,
                2048,
                True,
                True,
            ),
        )[1],
    )

    with pytest.raises(GGUFCumulativeError, match="untouched GGUF tensor"):
        run_gguf_cumulative_sequence(
            source,
            source_state,
            (bad,),
            output_root=tmp_path / "children",
            source_outcome_id="source-gguf-outcome",
            limits=GGUFResourceLimits(4096, 4096),
        )
    assert not list((tmp_path / "children").glob("*.gguf"))
