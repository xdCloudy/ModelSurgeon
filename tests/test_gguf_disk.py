"""Tests for conservative GGUF disk preflight and monitoring."""

from __future__ import annotations

from pathlib import Path

import pytest

from modelsurgeon.adapters.gguf import (
    DiskProbe,
    GGUFDiskEstimate,
    GGUFDiskSpaceError,
    monitor_gguf_disk,
    preflight_gguf_disk,
)


def _probe(devices: dict[Path, DiskProbe]):
    def probe(path: Path) -> DiskProbe:
        return devices[path.resolve()]

    return probe


def test_same_filesystem_combines_output_scratch_alignment_and_one_margin(
    tmp_path: Path,
) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    probe = _probe(
        {
            tmp_path.resolve(): DiskProbe("disk-a", 2000),
            scratch: DiskProbe("disk-a", 2000),
        }
    )
    estimate = GGUFDiskEstimate(1001, 500, alignment_bytes=32, safety_margin_bytes=100)

    plan = preflight_gguf_disk(tmp_path / "out.gguf", scratch, estimate, probe=probe)

    assert estimate.aligned_output_bytes == 1024
    assert estimate.alignment_padding_bytes == 23
    assert len(plan.filesystems) == 1
    assert plan.filesystems[0].required_bytes == 1024 + 500 + 100


def test_separate_filesystems_each_keep_safety_margin(tmp_path: Path) -> None:
    output = tmp_path / "output"
    scratch = tmp_path / "scratch"
    output.mkdir()
    scratch.mkdir()
    probe = _probe(
        {
            output: DiskProbe("output-disk", 2000),
            scratch: DiskProbe("scratch-disk", 1000),
        }
    )

    plan = preflight_gguf_disk(
        output / "model.gguf",
        scratch,
        GGUFDiskEstimate(1000, 500, safety_margin_bytes=100),
        probe=probe,
    )

    by_device = {item.device_id: item for item in plan.filesystems}
    assert by_device["output-disk"].required_bytes == 1024 + 100
    assert by_device["scratch-disk"].required_bytes == 500 + 100


def test_insufficient_space_fails_before_any_output_is_created(tmp_path: Path) -> None:
    output = tmp_path / "model.gguf"
    source = tmp_path / "source.gguf"
    source.write_bytes(b"unchanged")
    probe = _probe({tmp_path.resolve(): DiskProbe("disk-a", 100)})

    with pytest.raises(GGUFDiskSpaceError, match="requires 352 bytes"):
        preflight_gguf_disk(
            output,
            tmp_path,
            GGUFDiskEstimate(200, 100, safety_margin_bytes=28),
            probe=probe,
        )

    assert not output.exists()
    assert source.read_bytes() == b"unchanged"


def test_monitor_rechecks_remaining_space_and_fails_closed(tmp_path: Path) -> None:
    initial = _probe({tmp_path.resolve(): DiskProbe("disk-a", 1000)})
    plan = preflight_gguf_disk(
        tmp_path / "out.gguf",
        tmp_path,
        GGUFDiskEstimate(400, 200, safety_margin_bytes=50),
        probe=initial,
    )
    reduced = _probe({tmp_path.resolve(): DiskProbe("disk-a", 200)})

    with pytest.raises(GGUFDiskSpaceError, match="requires 300 bytes"):
        monitor_gguf_disk(
            plan,
            output_remaining_bytes=150,
            scratch_remaining_bytes=100,
            probe=reduced,
        )


@pytest.mark.parametrize("alignment", [0, 3, 48])
def test_invalid_alignment_fails(alignment: int) -> None:
    with pytest.raises(ValueError, match="alignment"):
        GGUFDiskEstimate(1, 1, alignment_bytes=alignment)
