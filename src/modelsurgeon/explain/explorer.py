"""Deterministic offline benchmark, Pareto, and lineage explorer artifacts."""

from __future__ import annotations

import hashlib
import html
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from modelsurgeon.experiments.identity import canonical_identity_json

EXPLORER_SCHEMA_VERSION = 1


class ExplorerError(ValueError):
    """Raised when explorer input is incomplete, unsafe, or non-canonical."""


class ExplorerCellStatus(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"
    NEGATIVE = "negative"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExplorerError(f"{label} must be non-empty text")
    return value


@dataclass(frozen=True, slots=True)
class ExplorerCell:
    """One immutable evidence projection shown by the explorer."""

    cell_id: str
    status: ExplorerCellStatus
    source_id: str
    metrics: tuple[tuple[str, float | None], ...] = ()
    lineage: tuple[str, ...] = ()
    architecture_diff: Mapping[str, object] = field(default_factory=dict)
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        _text(self.cell_id, "cell ID")
        _text(self.source_id, "source ID")
        if any(character in self.source_id for character in "\\/"):
            raise ExplorerError("source IDs cannot contain local path separators")
        names = tuple(name for name, _ in self.metrics)
        if names != tuple(sorted(set(names))) or any(not name.strip() for name in names):
            raise ExplorerError("cell metric names must be sorted and unique")
        for _, value in self.metrics:
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ExplorerError("cell metrics must be finite numbers or unknown")
        if self.lineage != tuple(sorted(set(self.lineage))):
            raise ExplorerError("cell lineage must be sorted and unique")
        if (
            self.status in {ExplorerCellStatus.FAILED, ExplorerCellStatus.UNKNOWN}
            and not self.failure_reason
        ):
            raise ExplorerError("failed and unknown cells require a reason")

    def to_record(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "status": self.status.value,
            "source_id": self.source_id,
            "metrics": {name: value for name, value in self.metrics},
            "lineage": list(self.lineage),
            "architecture_diff": dict(self.architecture_diff),
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True, slots=True)
class ExplorerArtifact:
    """Self-contained static HTML plus its canonical source projection."""

    html_text: str
    data: Mapping[str, object]
    cell_count: int
    pareto_cell_ids: tuple[str, ...]
    html_sha256: str

    @property
    def artifact_id(self) -> str:
        return f"explorer_{self.html_sha256}"

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "offline_explorer",
            "schema_version": EXPLORER_SCHEMA_VERSION,
            "artifact_id": self.artifact_id,
            "cell_count": self.cell_count,
            "pareto_cell_ids": list(self.pareto_cell_ids),
            "html_sha256": self.html_sha256,
            "data": dict(self.data),
        }


def pareto_front(
    cells: Sequence[ExplorerCell],
    *,
    cost_metric: str,
    quality_metric: str,
) -> tuple[str, ...]:
    """Return deterministic supported/negative cells not dominated by cost/quality."""

    eligible = [
        cell
        for cell in cells
        if cell.status in {ExplorerCellStatus.SUPPORTED, ExplorerCellStatus.NEGATIVE}
        and dict(cell.metrics).get(cost_metric) is not None
        and dict(cell.metrics).get(quality_metric) is not None
    ]
    front: list[ExplorerCell] = []
    for candidate in eligible:
        candidate_metrics = dict(candidate.metrics)
        candidate_cost = candidate_metrics[cost_metric]
        candidate_quality = candidate_metrics[quality_metric]
        assert candidate_cost is not None and candidate_quality is not None
        dominated = False
        for other in eligible:
            if other.cell_id == candidate.cell_id:
                continue
            other_metrics = dict(other.metrics)
            other_cost = other_metrics[cost_metric]
            other_quality = other_metrics[quality_metric]
            assert other_cost is not None and other_quality is not None
            if other_cost <= candidate_cost and other_quality >= candidate_quality and (
                other_cost < candidate_cost or other_quality > candidate_quality
            ):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return tuple(sorted(item.cell_id for item in front))


def _safe_json_script(value: object) -> str:
    return (
        canonical_identity_json(value)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _row(cell: ExplorerCell, pareto_ids: set[str]) -> str:
    metrics = " ".join(
        f"<span class=metric><b>{html.escape(name)}</b>: "
        f"{html.escape('unknown' if value is None else str(value))}</span>"
        for name, value in cell.metrics
    )
    lineage = ", ".join(html.escape(item) for item in cell.lineage) or "none"
    reason = "" if cell.failure_reason is None else html.escape(cell.failure_reason)
    frontier = "yes" if cell.cell_id in pareto_ids else "no"
    architecture_diff = html.escape(canonical_identity_json(cell.architecture_diff))
    return (
        f'<tr data-status="{html.escape(cell.status.value)}">'
        f'<th scope="row">{html.escape(cell.cell_id)}</th>'
        f'<td><span class="status status-{html.escape(cell.status.value)}">'
        f"{html.escape(cell.status.value)}</span></td>"
        f'<td><a href="#source-{html.escape(cell.source_id)}">'
        f"{html.escape(cell.source_id)}</a></td>"
        f"<td>{metrics or 'none'}</td><td>{lineage}</td>"
        f"<td><code>{architecture_diff}</code></td>"
        f"<td>{frontier}</td>"
        f"<td>{reason}</td></tr>"
    )


def generate_explorer(
    cells: Sequence[ExplorerCell],
    *,
    title: str = "ModelSurgeon evidence explorer",
    source_report_id: str = "canonical-evidence",
    cost_metric: str = "cost",
    quality_metric: str = "quality",
    max_cells: int = 100_000,
) -> ExplorerArtifact:
    """Generate an offline static explorer from canonical evidence cells."""

    if not cells:
        raise ExplorerError("explorer requires at least one evidence cell")
    if len(cells) > max_cells or max_cells <= 0:
        raise ExplorerError("explorer cell count exceeds its declared generation bound")
    _text(title, "explorer title")
    _text(source_report_id, "source report ID")
    cell_ids = tuple(cell.cell_id for cell in cells)
    if cell_ids != tuple(sorted(cell_ids)) or len(cell_ids) != len(set(cell_ids)):
        raise ExplorerError("explorer cells must be sorted by unique cell ID")
    pareto_ids = pareto_front(cells, cost_metric=cost_metric, quality_metric=quality_metric)
    data = {
        "schema_version": EXPLORER_SCHEMA_VERSION,
        "source_report_id": source_report_id,
        "title": title,
        "cost_metric": cost_metric,
        "quality_metric": quality_metric,
        "cells": [cell.to_record() for cell in cells],
        "pareto_cell_ids": list(pareto_ids),
    }
    embedded = _safe_json_script(data)
    rows = "".join(_row(cell, set(pareto_ids)) for cell in cells)
    page = (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>{html.escape(title)}</title><style>"
        ":root{font:16px system-ui,sans-serif;color:#172033;background:#f7f9fc}"
        "body{max-width:1500px;margin:2rem auto;padding:0 1rem}"
        "table{border-collapse:collapse;width:100%;background:#fff}"
        "th,td{border:1px solid #ccd3df;padding:.5rem;text-align:left;vertical-align:top}"
        "th{background:#e9eef6}.status{font-weight:700}.status-supported{color:#126b35}"
        ".status-negative{color:#895b00}.status-failed{color:#a21b1b}.status-unknown{color:#5e4a9a}"
        ".status-unsupported{color:#596273}.metric{display:inline-block;margin-right:.75rem}"
        "button,select{font:inherit;padding:.35rem;margin:.25rem 0}"
        "</style></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        f"<p>Canonical source: <code>{html.escape(source_report_id)}</code>; "
        f"cells: {len(cells)}; Pareto cells: {len(pareto_ids)}</p>"
        "<label for=\"status-filter\">Filter status</label> "
        "<select id=\"status-filter\"><option value=\"all\">all</option>"
        "<option>supported</option><option>negative</option><option>unsupported</option>"
        "<option>failed</option><option>unknown</option></select>"
        "<table><caption>Canonical benchmark evidence cells</caption><thead><tr>"
        "<th scope=\"col\">Cell</th><th scope=\"col\">Status</th>"
        "<th scope=\"col\">Immutable source</th><th scope=\"col\">Metrics</th>"
        "<th scope=\"col\">Lineage</th><th scope=\"col\">Architecture diff</th>"
        "<th scope=\"col\">Pareto</th>"
        "<th scope=\"col\">Failure or limitation</th></tr></thead><tbody>"
        f"{rows}</tbody></table>"
        f'<script type="application/json" id="modelsurgeon-explorer-data">{embedded}</script>'
        "<script>(function(){const s=document.getElementById('status-filter');"
        "const rows=[...document.querySelectorAll('tbody tr')];"
        "s.addEventListener('change',function(){rows.forEach(function(r){"
        "r.hidden=s.value!=='all'&&r.dataset.status!==s.value;});});})();</script>"
        "</body></html>"
    )
    digest = hashlib.sha256(page.encode("utf-8")).hexdigest()
    return ExplorerArtifact(page, data, len(cells), pareto_ids, digest)


def write_explorer(
    path: str | Path, artifact: ExplorerArtifact, *, allow_overwrite: bool = False
) -> Path:
    """Write one static explorer with exclusive-create semantics by default."""

    destination = Path(path)
    if destination.exists() and not allow_overwrite:
        raise ExplorerError(f"refusing to overwrite explorer: {destination}")
    destination.write_text(artifact.html_text, encoding="utf-8", newline="\n")
    return destination


__all__ = [
    "EXPLORER_SCHEMA_VERSION",
    "ExplorerArtifact",
    "ExplorerCell",
    "ExplorerCellStatus",
    "ExplorerError",
    "generate_explorer",
    "pareto_front",
    "write_explorer",
]
