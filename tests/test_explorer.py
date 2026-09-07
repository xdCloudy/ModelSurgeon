"""Offline explorer projection and safety tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from modelsurgeon.explain import (
    ExplorerCell,
    ExplorerCellStatus,
    ExplorerError,
    generate_explorer,
    pareto_front,
    write_explorer,
)


def _cells() -> tuple[ExplorerCell, ...]:
    return (
        ExplorerCell(
            "cell-a",
            ExplorerCellStatus.SUPPORTED,
            "artifact-a",
            (("cost", 1.0), ("quality", 0.90)),
            ("run-a",),
            {"width": {"before": 1024, "after": 768}},
        ),
        ExplorerCell(
            "cell-b",
            ExplorerCellStatus.SUPPORTED,
            "artifact-b",
            (("cost", 2.0), ("quality", 0.95)),
            ("run-b",),
        ),
        ExplorerCell(
            "cell-c",
            ExplorerCellStatus.UNSUPPORTED,
            "artifact-c",
            (("cost", None), ("quality", None)),
            failure_reason="runtime capability is unavailable",
        ),
        ExplorerCell(
            "cell-d",
            ExplorerCellStatus.NEGATIVE,
            "artifact-d",
            (("cost", 3.0), ("quality", 0.80)),
            failure_reason="held-out quality target was not reached",
        ),
    )


def test_pareto_front_and_static_projection_are_deterministic() -> None:
    cells = _cells()
    assert pareto_front(cells, cost_metric="cost", quality_metric="quality") == (
        "cell-a",
        "cell-b",
    )
    first = generate_explorer(cells, title="Fixture explorer")
    second = generate_explorer(cells, title="Fixture explorer")
    assert first.html_text == second.html_text
    assert first.html_sha256 == second.html_sha256
    assert "unsupported" in first.html_text
    assert "runtime capability is unavailable" in first.html_text
    assert 'data-status="negative"' in first.html_text
    assert "artifact-a" in first.html_text
    assert "Architecture diff" in first.html_text
    assert "http://" not in first.html_text


def test_html_escapes_untrusted_text_and_embedded_data() -> None:
    cell = ExplorerCell(
        "cell-x",
        ExplorerCellStatus.SUPPORTED,
        "source-x",
        (("quality", 1.0),),
        failure_reason="<script>alert('x')</script>",
    )
    artifact = generate_explorer((cell,), title="<img src=x onerror=alert(1)>")
    assert "<img src=x" not in artifact.html_text
    assert "<script>alert" not in artifact.html_text
    assert "&lt;script&gt;alert" in artifact.html_text


def test_invalid_and_unsorted_inputs_are_rejected() -> None:
    with pytest.raises(ExplorerError, match="sorted"):
        ExplorerCell("cell", ExplorerCellStatus.SUPPORTED, "source", (("z", 1), ("a", 2)))
    with pytest.raises(ExplorerError, match="reason"):
        ExplorerCell("cell", ExplorerCellStatus.UNKNOWN, "source")
    with pytest.raises(ExplorerError, match="sorted"):
        generate_explorer(tuple(reversed(_cells())))


def test_explorer_writes_without_overwrite(tmp_path: Path) -> None:
    artifact = generate_explorer(_cells())
    path = tmp_path / "explorer.html"
    write_explorer(path, artifact)
    assert path.read_text(encoding="utf-8") == artifact.html_text
    with pytest.raises(ExplorerError, match="overwrite"):
        write_explorer(path, artifact)

