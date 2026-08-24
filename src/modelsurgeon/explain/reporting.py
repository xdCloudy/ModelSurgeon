"""Deterministic self-contained JSON and HTML experiment reports."""

from __future__ import annotations

import hashlib
import html
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

REPORT_SCHEMA_VERSION: Final[int] = 1
_REDACTED = "<redacted>"
_REDACTED_PATH = "<redacted-path>"


class ReportGenerationError(ValueError):
    """Raised when report data is unsafe, incomplete, or non-deterministic."""


type JSONValue = bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"] | None


@dataclass(frozen=True, slots=True)
class ReportRedaction:
    path_prefixes: tuple[str, ...] = ()
    secret_keys: tuple[str, ...] = (
        "api_key",
        "authorization",
        "password",
        "secret",
        "token",
    )

    def __post_init__(self) -> None:
        if self.path_prefixes != tuple(sorted(set(self.path_prefixes))):
            raise ReportGenerationError("report redaction paths must be unique and canonical")
        if self.secret_keys != tuple(sorted(set(self.secret_keys))):
            raise ReportGenerationError("report secret keys must be unique and canonical")
        if any(not value for value in (*self.path_prefixes, *self.secret_keys)):
            raise ReportGenerationError("report redaction values cannot be blank")


@dataclass(frozen=True, slots=True)
class ReportLink:
    label: str
    kind: str
    immutable_id: str

    def __post_init__(self) -> None:
        if not self.label or not self.kind or not self.immutable_id:
            raise ReportGenerationError("report links require label, kind, and immutable ID")

    def to_record(self) -> dict[str, str]:
        return {"label": self.label, "kind": self.kind, "immutable_id": self.immutable_id}


@dataclass(frozen=True, slots=True)
class ReportFailure:
    stage: str
    code: str
    message: str
    recoverable: bool

    def __post_init__(self) -> None:
        if not self.stage or not self.code or not self.message:
            raise ReportGenerationError("report failures require stage, code, and message")

    def to_record(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "code": self.code,
            "message": self.message,
            "recoverable": self.recoverable,
        }


@dataclass(frozen=True, slots=True)
class ReportPlot:
    name: str
    x_label: str
    y_label: str
    points: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        if not self.name or not self.x_label or not self.y_label or not self.points:
            raise ReportGenerationError("report plots require identity, axes, and points")
        if any(
            not math.isfinite(value)
            for point in self.points
            for value in point
        ):
            raise ReportGenerationError("report plot points must be finite")

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "x_label": self.x_label,
            "y_label": self.y_label,
            "points": [[x, y] for x, y in self.points],
        }


@dataclass(frozen=True, slots=True)
class ReportInput:
    subject_kind: str
    subject_id: str
    generated_at: str | None
    resolved_config: Mapping[str, object]
    lineage: tuple[Mapping[str, object], ...]
    metrics: tuple[tuple[str, float | None], ...]
    plots: tuple[ReportPlot, ...]
    failures: tuple[ReportFailure, ...]
    hardware: Mapping[str, object]
    links: tuple[ReportLink, ...]
    redaction: ReportRedaction = ReportRedaction()

    def __post_init__(self) -> None:
        if self.subject_kind not in {"run", "campaign", "search"} or not self.subject_id:
            raise ReportGenerationError("report subject kind or identity is invalid")
        if self.generated_at is not None and not self.generated_at:
            raise ReportGenerationError("declared report timestamp cannot be blank")
        metric_names = tuple(name for name, _ in self.metrics)
        if metric_names != tuple(sorted(set(metric_names))) or any(
            not name for name in metric_names
        ):
            raise ReportGenerationError("report metric names must be unique and canonical")
        if any(value is not None and not math.isfinite(value) for _, value in self.metrics):
            raise ReportGenerationError("report metrics must be finite or unknown")
        plot_names = tuple(item.name for item in self.plots)
        if plot_names != tuple(sorted(set(plot_names))):
            raise ReportGenerationError("report plots must be unique and canonical")
        link_ids = tuple(item.immutable_id for item in self.links)
        if link_ids != tuple(sorted(set(link_ids))):
            raise ReportGenerationError("report links must use unique canonical identities")


@dataclass(frozen=True, slots=True)
class GeneratedReport:
    record: dict[str, JSONValue]
    json_text: str
    html_text: str
    json_sha256: str
    html_sha256: str


def _json_value(value: object, label: str) -> JSONValue:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ReportGenerationError(f"{label} contains a non-finite value")
        return value
    if isinstance(value, Mapping):
        output: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ReportGenerationError(f"{label} contains an invalid object key")
            output[key] = _json_value(item, f"{label}.{key}")
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item, label) for item in value]
    raise ReportGenerationError(f"{label} contains unsupported value {type(value).__name__}")


def _redact_path(value: str, prefixes: tuple[str, ...]) -> str:
    normalized = value.replace("\\", "/")
    for prefix in prefixes:
        candidate = prefix.replace("\\", "/").rstrip("/")
        if normalized.casefold() == candidate.casefold():
            return _REDACTED_PATH
        marker = candidate + "/"
        if normalized.casefold().startswith(marker.casefold()):
            return _REDACTED_PATH + normalized[len(candidate) :]
    return value


def _redact(value: JSONValue, config: ReportRedaction, *, key: str | None = None) -> JSONValue:
    if key is not None and key.casefold() in {item.casefold() for item in config.secret_keys}:
        return _REDACTED
    if isinstance(value, str):
        return _redact_path(value, config.path_prefixes)
    if isinstance(value, list):
        return [_redact(item, config) for item in value]
    if isinstance(value, dict):
        return {name: _redact(item, config, key=name) for name, item in value.items()}
    return value


def _anchor(value: str) -> str:
    return "identity-" + hashlib.sha256(value.encode()).hexdigest()[:16]


def _table(rows: Sequence[tuple[str, str]]) -> str:
    return "<table>" + "".join(
        f"<tr><th>{html.escape(name)}</th><td>{html.escape(value)}</td></tr>"
        for name, value in rows
    ) + "</table>"


def _display(value: JSONValue) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return format(value, ".12g")
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _svg(plot: ReportPlot, redaction: ReportRedaction) -> str:
    width, height, margin = 640.0, 220.0, 28.0
    xs = tuple(point[0] for point in plot.points)
    ys = tuple(point[1] for point in plot.points)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max_x - min_x or 1.0
    span_y = max_y - min_y or 1.0
    coordinates = " ".join(
        f"{margin + (x - min_x) / span_x * (width - 2 * margin):.3f},"
        f"{height - margin - (y - min_y) / span_y * (height - 2 * margin):.3f}"
        for x, y in plot.points
    )
    name = _redact_path(plot.name, redaction.path_prefixes)
    x_label = _redact_path(plot.x_label, redaction.path_prefixes)
    y_label = _redact_path(plot.y_label, redaction.path_prefixes)
    return (
        f'<figure><figcaption>{html.escape(name)}</figcaption>'
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        f'aria-label="{html.escape(name)}">'
        f'<polyline points="{coordinates}" fill="none" stroke="#2563eb" stroke-width="2"/>'
        f'<text x="{width / 2:.0f}" y="{height - 3:.0f}">{html.escape(x_label)}</text>'
        f'<text x="3" y="14">{html.escape(y_label)}</text></svg></figure>'
    )


def _html(record: dict[str, JSONValue], source: ReportInput) -> str:
    metrics = record["metrics"]
    config = record["resolved_config"]
    hardware = record["hardware"]
    assert isinstance(metrics, dict) and isinstance(config, dict) and isinstance(hardware, dict)
    embedded = json.dumps(record, sort_keys=True, separators=(",", ":")).replace("<", "\\u003c")
    link_records = record["links"]
    assert isinstance(link_records, list)
    links = "".join(
        f'<li id="{_anchor(_display(item["immutable_id"]))}">'
        f'<strong>{html.escape(_display(item["label"]))}</strong> '
        f'({html.escape(_display(item["kind"]))}): '
        f'<code>{html.escape(_display(item["immutable_id"]))}</code></li>'
        for item in link_records
        if isinstance(item, dict)
    ) or "<li>none</li>"
    lineage = record["lineage"]
    failures = record["failures"]
    assert isinstance(lineage, list) and isinstance(failures, list)
    lineage_html = "".join(
        f"<li><code>{html.escape(_display(item))}</code></li>" for item in lineage
    ) or "<li>none</li>"
    failure_html = "".join(
        f"<li><code>{html.escape(_display(item))}</code></li>" for item in failures
    ) or "<li>none</li>"
    plots = "".join(_svg(item, source.redaction) for item in source.plots) or "<p>none</p>"
    title = f"ModelSurgeon {source.subject_kind} report: {source.subject_id}"
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" "
        "content=\"width=device-width,initial-scale=1\"><title>"
        + html.escape(title)
        + "</title><style>body{font:15px system-ui;max-width:960px;margin:2rem auto;"
        "padding:0 1rem;color:#172033}table{border-collapse:collapse;width:100%}th,td{"
        "border:1px solid #d7deea;padding:.45rem;text-align:left}th{width:32%;background:#f5f7fb}"
        "code{overflow-wrap:anywhere}svg{width:100%;background:#fafcff;border:1px solid #d7deea}"
        "section{margin:1.5rem 0}</style></head><body><h1>"
        + html.escape(title)
        + f"</h1><p>Generated at: {html.escape(source.generated_at or 'not declared')}</p>"
        + "<section><h2>Metrics</h2>"
        + _table(tuple((name, _display(value)) for name, value in sorted(metrics.items())))
        + "</section><section><h2>Resolved configuration</h2>"
        + _table(tuple((name, _display(value)) for name, value in sorted(config.items())))
        + "</section><section><h2>Hardware</h2>"
        + _table(tuple((name, _display(value)) for name, value in sorted(hardware.items())))
        + f"</section><section><h2>Lineage</h2><ol>{lineage_html}</ol></section>"
        + f"<section><h2>Plots</h2>{plots}</section>"
        + f"<section><h2>Failures</h2><ul>{failure_html}</ul></section>"
        + f"<section><h2>Immutable identities</h2><ul>{links}</ul></section>"
        + f'<script type="application/json" id="modelsurgeon-report">{embedded}</script>'
        + "</body></html>"
    )


def generate_report(source: ReportInput) -> GeneratedReport:
    """Render canonical JSON and dependency-free HTML from one redacted record."""

    raw: dict[str, JSONValue] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "subject": {"kind": source.subject_kind, "id": source.subject_id},
        "generated_at": source.generated_at,
        "resolved_config": _json_value(source.resolved_config, "resolved config"),
        "lineage": [_json_value(item, "lineage") for item in source.lineage],
        "metrics": {name: value for name, value in source.metrics},
        "plots": [_json_value(item.to_record(), "plot") for item in source.plots],
        "failures": [
            _json_value(item.to_record(), "failure") for item in source.failures
        ],
        "hardware": _json_value(source.hardware, "hardware"),
        "links": [_json_value(item.to_record(), "link") for item in source.links],
    }
    redacted = _redact(raw, source.redaction)
    assert isinstance(redacted, dict)
    record = redacted
    json_text = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    html_text = _html(record, source)
    return GeneratedReport(
        record,
        json_text,
        html_text,
        hashlib.sha256(json_text.encode()).hexdigest(),
        hashlib.sha256(html_text.encode()).hexdigest(),
    )
