"""Audit the bounded v2.5 negotiation decision-quality boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from modelsurgeon.conversation import (
    NegotiationPolicy,
    load_and_run_negotiation_study,
)


class NegotiationQualityAuditError(ValueError):
    """Raised when the v2.5 study boundary or evidence drifts."""


_DEPENDENCIES = {
    "v25-infeasibility": (458, "c6f5059bdba31087d691c25dfdf2aade0c15699a"),
    "v25-pareto": (459, "5f405827649ffb43c47f2431c1bb7887ff5195f2"),
    "v25-amendments": (460, "45daba27316ce01af116d18585ef0523eea72435"),
}


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise NegotiationQualityAuditError(f"{label} must be an object")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NegotiationQualityAuditError(f"{label} must be non-empty text")
    return value


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise NegotiationQualityAuditError(f"{label} must be an array")
    return value


def _files(root: Path, value: object, label: str) -> None:
    for index, raw in enumerate(_array(value, label)):
        path = Path(_text(raw, f"{label}[{index}]"))
        if path.is_absolute() or ".." in path.parts or not (root / path).is_file():
            raise NegotiationQualityAuditError(f"{label}[{index}] references a missing file")


def audit_release(root: Path = Path("."), *, manifest: Path | None = None) -> None:
    """Validate the checked-in protocol, evidence, and release decision."""

    root = root.resolve()
    manifest_path = manifest or (
        root / "docs" / "research" / "v2.5-negotiation-decision-quality-v1.json"
    )
    try:
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NegotiationQualityAuditError("could not read v2.5 research record") from error
    if record.get("record_type") != "bounded_negotiation_decision_quality":
        raise NegotiationQualityAuditError("unexpected v2.5 record type")
    if record.get("schema_version") != 1:
        raise NegotiationQualityAuditError("unsupported v2.5 record schema")
    if record.get("protocol_revision") != "modelsurgeon-v2.5-negotiation-decision-quality-v1":
        raise NegotiationQualityAuditError("unexpected v2.5 protocol revision")
    if record.get("status") != "bounded_release_boundary":
        raise NegotiationQualityAuditError("v2.5 status must remain bounded_release_boundary")

    dependencies = _array(record.get("dependency_evidence"), "dependency_evidence")
    seen: set[str] = set()
    for index, raw in enumerate(dependencies):
        item = _object(raw, f"dependency_evidence[{index}]")
        key = _text(item.get("key"), f"dependency_evidence[{index}].key")
        if key in seen or key not in _DEPENDENCIES:
            raise NegotiationQualityAuditError(f"unexpected or duplicate dependency {key}")
        seen.add(key)
        if item.get("status") != "merged":
            raise NegotiationQualityAuditError(f"dependency {key} is not merged")
        issue, commit = _DEPENDENCIES[key]
        if item.get("issue") != issue or item.get("commit") != commit:
            raise NegotiationQualityAuditError(f"dependency {key} identity drifted")
        _files(root, item.get("artifacts"), f"dependency {key}.artifacts")
        _files(root, item.get("tests"), f"dependency {key}.tests")
    if seen != set(_DEPENDENCIES):
        raise NegotiationQualityAuditError("dependency evidence is incomplete")

    protocol = _object(record.get("protocol"), "protocol")
    _files(root, [protocol.get("fixture")], "protocol.fixture")
    if protocol.get("protocol_id") != "v25-negotiation-decision-quality-v1":
        raise NegotiationQualityAuditError("protocol ID drifted")
    if protocol.get("scenarios") != 4 or protocol.get("seeds_per_scenario") != 3:
        raise NegotiationQualityAuditError("scenario or seed bound drifted")
    if protocol.get("fixed_evaluation_count") != 4 or protocol.get("retained_cell_count") != 36:
        raise NegotiationQualityAuditError("fixed evaluation or retained cell count drifted")
    if protocol.get("negative_and_inconclusive_retained") is not True:
        raise NegotiationQualityAuditError("negative and inconclusive retention is missing")
    if (
        protocol.get("live_provider_benchmarks") != "not_run"
        or protocol.get("live_model_quality") != "not_run"
    ):
        raise NegotiationQualityAuditError("live evidence must remain not_run")

    study_path = root / _text(protocol["fixture"], "protocol.fixture")
    run = load_and_run_negotiation_study(study_path)
    if run.run_id != protocol.get("run_id") or len(run.results) != protocol.get(
        "retained_cell_count"
    ):
        raise NegotiationQualityAuditError("replayed study identity or size drifted")
    if not run.passed:
        raise NegotiationQualityAuditError("measured-Pareto study no longer passes")
    measured = next(
        item for item in run.metrics if item.policy is NegotiationPolicy.MEASURED_PARETO
    )
    expected_metrics = _object(record.get("measured_pareto_metrics"), "measured_pareto_metrics")
    for key, value in expected_metrics.items():
        if measured.to_record().get(key) != value:
            raise NegotiationQualityAuditError(f"measured-Pareto metric drifted: {key}")

    control = _object(record.get("prediction_only_control"), "prediction_only_control")
    if control.get("shippable") is not False or control.get("decision") != "never_shippable":
        raise NegotiationQualityAuditError("prediction-only control became shippable")
    prediction = next(
        item for item in run.metrics if item.policy is NegotiationPolicy.PREDICTION_ONLY
    )
    if prediction.misleading_claim_rate != control.get("misleading_claim_rate"):
        raise NegotiationQualityAuditError("prediction-only misleading-claim rate drifted")
    if prediction.measured_alternative_grounding_rate != control.get(
        "measured_alternative_grounding_rate"
    ):
        raise NegotiationQualityAuditError("prediction-only grounding status drifted")

    unsupported = _array(record.get("unsupported_cells"), "unsupported_cells")
    if len(unsupported) != 4:
        raise NegotiationQualityAuditError("unsupported-cell boundary drifted")
    for index, raw in enumerate(unsupported):
        item = _object(raw, f"unsupported_cells[{index}]")
        _text(item.get("cell"), f"unsupported_cells[{index}].cell")
        _text(item.get("status"), f"unsupported_cells[{index}].status")
        _text(item.get("reason"), f"unsupported_cells[{index}].reason")

    authority = _object(record.get("authority_and_identity"), "authority_and_identity")
    for key in ("hard_constraints_preserved", "source_model_immutable", "provenance_retained"):
        if authority.get(key) is not True:
            raise NegotiationQualityAuditError(f"{key} is not recorded")
    if authority.get("prediction_only_shippable") is not False:
        raise NegotiationQualityAuditError("prediction-only authority boundary drifted")
    _files(root, record.get("documentation"), "documentation")
    if not _array(record.get("reproduction_commands"), "reproduction_commands"):
        raise NegotiationQualityAuditError("reproduction commands are missing")
    quality_gate = _object(record.get("quality_gate"), "quality_gate")
    if not _array(quality_gate.get("commands"), "quality_gate.commands"):
        raise NegotiationQualityAuditError("quality gate commands are missing")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    audit_release(args.root, manifest=args.manifest)
    print("v2.5 negotiation decision-quality boundary verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
