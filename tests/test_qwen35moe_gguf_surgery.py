"""Exact layout and encoded-slice tests for the narrow Qwen3.5 MoE path."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import pytest

from modelsurgeon.adapters.gguf import (
    GGUFDiskEstimate,
    GGUFValueType,
    GGUFWriteMetadata,
    GGUFWriteTensor,
    Qwen35MoeSurgeryError,
    build_qwen35moe_gguf_layout,
    execute_qwen35moe_expert_pruning,
    open_gguf,
    plan_gguf_output,
    preflight_gguf_disk,
    rank_qwen35moe_experts_by_router_norm,
    select_qwen35moe_experts_by_scores,
    write_gguf_transactionally,
)


def _metadata(experts: int = 4, block_count: int = 2) -> tuple[GGUFWriteMetadata, ...]:
    return (
        GGUFWriteMetadata("general.architecture", GGUFValueType.STRING, "qwen35moe"),
        GGUFWriteMetadata("general.quantization_version", GGUFValueType.UINT32, 2),
        GGUFWriteMetadata("qwen35moe.block_count", GGUFValueType.UINT32, block_count),
        GGUFWriteMetadata("qwen35moe.expert_count", GGUFValueType.UINT32, experts),
        GGUFWriteMetadata("qwen35moe.expert_used_count", GGUFValueType.UINT32, 2),
        GGUFWriteMetadata("qwen35moe.embedding_length", GGUFValueType.UINT32, 256),
        GGUFWriteMetadata("qwen35moe.expert_feed_forward_length", GGUFValueType.UINT32, 512),
    )


def _floats(values: tuple[float, ...]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def _fixture_tensors() -> tuple[GGUFWriteTensor, ...]:
    result: list[GGUFWriteTensor] = []
    for layer in range(2):
        prefix = f"blk.{layer}."
        router = tuple(
            float(layer * 100 + expert * 10 + index) for expert in range(4) for index in range(256)
        )
        result.extend(
            (
                GGUFWriteTensor(prefix + "ffn_gate_inp.weight", (256, 4), 0, (_floats(router),)),
                GGUFWriteTensor(
                    prefix + "ffn_gate_exps.weight",
                    (256, 512, 4),
                    12,
                    (b"".join(bytes([layer * 10 + expert]) * 73_728 for expert in range(4)),),
                ),
                GGUFWriteTensor(
                    prefix + "ffn_up_exps.weight",
                    (256, 512, 4),
                    12,
                    (b"".join(bytes([layer * 10 + 20 + expert]) * 73_728 for expert in range(4)),),
                ),
                GGUFWriteTensor(
                    prefix + "ffn_down_exps.weight",
                    (512, 256, 4),
                    14,
                    (b"".join(bytes([layer * 10 + 40 + expert]) * 107_520 for expert in range(4)),),
                ),
                GGUFWriteTensor(
                    prefix + "ffn_gate_inp_shexp.weight", (256,), 0, (_floats((1.0,) * 256),)
                ),
                GGUFWriteTensor(prefix + "ssm_conv1d.weight", (256,), 0, (_floats((3.0,) * 256),)),
            )
        )
    return tuple(result)


def _write(path: Path, metadata: tuple[GGUFWriteMetadata, ...] | None = None) -> None:
    selected_metadata = metadata or _metadata()
    tensors = _fixture_tensors()
    layout = plan_gguf_output(selected_metadata, tensors)
    disk = preflight_gguf_disk(
        path,
        path.parent,
        GGUFDiskEstimate(layout.total_bytes, 0, layout.alignment),
    )
    write_gguf_transactionally(path, selected_metadata, tensors, disk)


def test_qwen35moe_layout_and_pruning_preserve_unrelated_tensors(tmp_path: Path) -> None:
    source_path = tmp_path / "source.gguf"
    output_path = tmp_path / "candidate.gguf"
    _write(source_path)

    with open_gguf(source_path) as source:
        source_layout = build_qwen35moe_gguf_layout(source.container)
        assert rank_qwen35moe_experts_by_router_norm(source, source_layout) == (
            (0, 1, 2, 3),
            (0, 1, 2, 3),
        )
        disk = preflight_gguf_disk(
            output_path,
            tmp_path,
            GGUFDiskEstimate(source_path.stat().st_size, 0),
        )
        result = execute_qwen35moe_expert_pruning(
            source,
            {0: (0, 2, 3), 1: (0, 1, 3)},
            output_path,
            disk,
        )

    assert source_layout.expert_count == 4
    assert result.output_layout.expert_count == 3
    assert result.output_layout.expert_used_count == 2
    assert len(result.unchanged_tensor_sha256) == 4
    with open_gguf(output_path) as output:
        for layer, retained in ((0, (0, 2, 3)), (1, (0, 1, 3))):
            router = output.container.tensor(f"blk.{layer}.ffn_gate_inp.weight")
            assert router is not None
            assert router.dimensions == (256, 3)
            actual = output.raw_bytes(router.data_offset, router.byte_size)
            expected = _floats(
                tuple(
                    float(layer * 100 + expert * 10 + index)
                    for expert in retained
                    for index in range(256)
                )
            )
            assert actual == expected
            gate = output.container.tensor(f"blk.{layer}.ffn_gate_exps.weight")
            assert gate is not None
            assert gate.dimensions == (256, 512, 3)
            gate_bytes = output.raw_bytes(gate.data_offset, gate.byte_size)
            assert gate_bytes == b"".join(
                bytes([layer * 10 + expert]) * 73_728 for expert in retained
            )
            down = output.container.tensor(f"blk.{layer}.ffn_down_exps.weight")
            assert down is not None
            assert down.dimensions == (512, 256, 3)
            down_bytes = output.raw_bytes(down.data_offset, down.byte_size)
            assert down_bytes == b"".join(
                bytes([layer * 10 + 40 + expert]) * 107_520 for expert in retained
            )
        assert output.container.metadata_entry("qwen35moe.expert_count").value == 3
        assert output.container.metadata_entry("qwen35moe.expert_used_count").value == 2


def test_qwen35moe_saliency_selection_feeds_layerwise_gguf_slicing(tmp_path: Path) -> None:
    source_path = tmp_path / "source.gguf"
    output_path = tmp_path / "candidate.gguf"
    _write(source_path)
    scores = ((0.1, 0.9, None, 0.2), (0.8, 0.1, 0.7, None))

    with open_gguf(source_path) as source:
        layout = build_qwen35moe_gguf_layout(source.container)
        retained = select_qwen35moe_experts_by_scores(layout, scores, retained_count=3)
        assert retained == {0: (1, 2, 3), 1: (0, 2, 3)}
        tied = select_qwen35moe_experts_by_scores(
            layout,
            ((0.5, 0.5, 0.1, None), (0.1, 0.2, 0.3, 0.4)),
            retained_count=3,
        )
        assert tied[0] == (0, 1, 3)
        disk = preflight_gguf_disk(
            output_path,
            tmp_path,
            GGUFDiskEstimate(source_path.stat().st_size, 0),
        )
        result = execute_qwen35moe_expert_pruning(source, retained, output_path, disk)

    assert result.output_layout.expert_count == 3
    with open_gguf(output_path) as output:
        for layer, expected in ((0, (1, 2, 3)), (1, (0, 2, 3))):
            router = output.container.tensor(f"blk.{layer}.ffn_gate_inp.weight")
            assert router is not None
            assert router.dimensions == (256, 3)
            data = output.raw_bytes(router.data_offset, router.byte_size)
            assert data == _floats(
                tuple(
                    float(layer * 100 + expert * 10 + index)
                    for expert in expected
                    for index in range(256)
                )
            )


def test_qwen35moe_saliency_selection_refuses_unsupported_or_incomplete_scores(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.gguf"
    _write(source_path)
    with open_gguf(source_path) as source:
        layout = build_qwen35moe_gguf_layout(source.container)
        with pytest.raises(Qwen35MoeSurgeryError, match="every block"):
            select_qwen35moe_experts_by_scores(layout, ((0.1, 0.2, 0.3, 0.4),), 3)
        with pytest.raises(Qwen35MoeSurgeryError, match="more unseen"):
            select_qwen35moe_experts_by_scores(
                layout,
                ((None, None, None, None), (0.1, 0.2, 0.3, 0.4)),
                3,
            )
        with pytest.raises(Qwen35MoeSurgeryError, match="finite and nonnegative"):
            select_qwen35moe_experts_by_scores(
                layout,
                ((0.1, float("nan"), 0.3, 0.4), (0.1, 0.2, 0.3, 0.4)),
                3,
            )
        with pytest.raises(Qwen35MoeSurgeryError, match="must be finite"):
            select_qwen35moe_experts_by_scores(
                layout,
                ((10**1000, 0.2, 0.3, 0.4), (0.1, 0.2, 0.3, 0.4)),
                3,
            )
        with pytest.raises(Qwen35MoeSurgeryError, match="retained_count"):
            select_qwen35moe_experts_by_scores(
                layout,
                ((0.1, 0.2, 0.3, 0.4), (0.1, 0.2, 0.3, 0.4)),
                True,
            )


@pytest.mark.parametrize(
    "retained",
    (
        {0: (0, 1, 2), 1: (0, 1)},
        {0: (0,), 1: (0,)},
        {0: (0, 1, 1), 1: (0, 1, 2)},
        {0: (0, 1, 2), 1: (0, 1, 4)},
        {0: (True, 2, 3), 1: (0, 1, 3)},
        {0: (0, 2.0, 3), 1: (0, 1, 3)},
        {0: ("0", 2, 3), 1: (0, 1, 3)},
    ),
)
def test_qwen35moe_selection_must_be_uniform_valid_and_keep_top_k(
    tmp_path: Path, retained: dict[int, tuple[int, ...]]
) -> None:
    source_path = tmp_path / "source.gguf"
    output_path = tmp_path / "candidate.gguf"
    _write(source_path)
    with open_gguf(source_path) as source:
        disk = preflight_gguf_disk(
            output_path,
            tmp_path,
            GGUFDiskEstimate(source_path.stat().st_size, 0),
        )
        with pytest.raises(Qwen35MoeSurgeryError):
            execute_qwen35moe_expert_pruning(source, retained, output_path, disk)


def test_qwen35moe_layout_rejects_wrong_router_shape(tmp_path: Path) -> None:
    path = tmp_path / "wrong.gguf"
    tensors = list(_fixture_tensors())
    tensors[0] = GGUFWriteTensor(
        "blk.0.ffn_gate_inp.weight", (4, 256), 0, (_floats((0.0,) * 1024),)
    )
    layout = plan_gguf_output(_metadata(), tuple(tensors))
    disk = preflight_gguf_disk(path, tmp_path, GGUFDiskEstimate(layout.total_bytes, 0))
    write_gguf_transactionally(path, _metadata(), tuple(tensors), disk)

    with open_gguf(path) as source, pytest.raises(Qwen35MoeSurgeryError, match="dimensions"):
        build_qwen35moe_gguf_layout(source.container)


def test_qwen35moe_layout_rejects_malformed_block_tensor_name(tmp_path: Path) -> None:
    path = tmp_path / "malformed-name.gguf"
    metadata = _metadata()
    tensors = (*_fixture_tensors(), GGUFWriteTensor("blk.", (1,), 0, (_floats((1.0,)),)))
    layout = plan_gguf_output(metadata, tensors)
    disk = preflight_gguf_disk(path, tmp_path, GGUFDiskEstimate(layout.total_bytes, 0))
    write_gguf_transactionally(path, metadata, tensors, disk)

    with open_gguf(path) as source, pytest.raises(Qwen35MoeSurgeryError, match="malformed"):
        build_qwen35moe_gguf_layout(source.container)


def test_qwen35moe_layout_rejects_block_count_larger_than_tensor_inventory(
    tmp_path: Path,
) -> None:
    path = tmp_path / "excessive-block-count.gguf"
    metadata = _metadata(block_count=1_000_000_000)
    _write(path, metadata)

    with (
        open_gguf(path) as source,
        pytest.raises(Qwen35MoeSurgeryError, match="block_count exceeds"),
    ):
        build_qwen35moe_gguf_layout(source.container)


def test_qwen35moe_failed_staging_validation_does_not_publish_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import modelsurgeon.adapters.gguf.qwen35moe_surgery as surgery

    source_path = tmp_path / "source.gguf"
    output_path = tmp_path / "candidate.gguf"
    _write(source_path)
    original_validate = surgery.build_qwen35moe_gguf_layout
    calls = 0

    def reject_staged_output(container: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise Qwen35MoeSurgeryError("injected staged semantic validation failure")
        return original_validate(container)

    monkeypatch.setattr(surgery, "build_qwen35moe_gguf_layout", reject_staged_output)
    with open_gguf(source_path) as source:
        disk = preflight_gguf_disk(
            output_path,
            tmp_path,
            GGUFDiskEstimate(source_path.stat().st_size, 0),
        )
        with pytest.raises(Qwen35MoeSurgeryError, match="injected staged"):
            execute_qwen35moe_expert_pruning(
                source,
                {0: (0, 2, 3), 1: (0, 1, 3)},
                output_path,
                disk,
            )

    assert not output_path.exists()
    assert not tuple(tmp_path.glob(".candidate.gguf.modelsurgeon-*.tmp"))
