"""Audit the bounded v2.8 Pareto-selection explanation contract."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class ParetoSelectionAuditError(ValueError):
    """Raised when the v2.8 selection-explanation evidence drifts."""


_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DEPENDENCIES = {
    "v20-selection-evidence": (428, "e40517169573b497b32f3e70dffff31b31013548"),
    "v25-pareto": (459, "5f405827649ffb43c47f2431c1bb7887ff5195f2"),
    "v25-amendments": (460, "45daba27316ce01af116d18585ef0523eea72435"),
    "v28-evidence-query": (475, "e0b4940a0b3a9d6dda468b46bcc02de303a492bb"),
}


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ParetoSelectionAuditError(f"{label} must be an object")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ParetoSelectionAuditError(f"{label} must be non-empty text")
    return value


def _files(root: Path, value: object, label: str) -> None:
    if not isinstance(value, list) or not value:
        raise ParetoSelectionAuditError(f"{label} must be a non-empty array")
    for index, raw in enumerate(value):
        path = Path(_text(raw, f"{label}[{index}]"))
        if path.is_absolute() or ".." in path.parts or not (root / path).is_file():
            raise ParetoSelectionAuditError(f"{label}[{index}] references a missing file")


def audit_release(
    root: Path = Path("."), *, manifest: Path | None = None
) -> None:
    """Validate the v2.8 selection explanation and its dependency evidence."""

    root = root.resolve()
    manifest_path = manifest or (
        root / "docs" / "research" / "v2.8-pareto-selection-explanation-v1.json"
    )
    try:
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ParetoSelectionAuditError("could not read selection audit manifest") from error
    record = _object(record, "selection audit manifest")
    if record.get("record_type") != "bounded_pareto_selection_explanation":
        raise ParetoSelectionAuditError("unexpected selection audit record type")
    if record.get("schema_version") != 1 or record.get("protocol_revision") != (
        "modelsurgeon-v2.8-pareto-selection-explanation-v1"
    ):
        raise ParetoSelectionAuditError("selection audit schema or protocol drifted")

    dependencies = record.get("dependency_evidence")
    if not isinstance(dependencies, list):
        raise ParetoSelectionAuditError("dependency evidence must be an array")
    seen: set[str] = set()
    for index, raw in enumerate(dependencies):
        item = _object(raw, f"dependency_evidence[{index}]")
        key = _text(item.get("key"), f"dependency_evidence[{index}].key")
        if key in seen or key not in _DEPENDENCIES:
            raise ParetoSelectionAuditError(f"unexpected or duplicate dependency {key}")
        seen.add(key)
        issue, commit = _DEPENDENCIES[key]
        if item.get("issue") != issue or item.get("status") != "merged":
            raise ParetoSelectionAuditError(f"dependency {key} is not recorded as merged")
        actual_commit = _text(item.get("commit"), f"dependency {key}.commit")
        if _COMMIT.fullmatch(actual_commit) is None or actual_commit != commit:
            raise ParetoSelectionAuditError(f"dependency {key} identity drifted")
        _files(root, item.get("artifacts"), f"dependency {key}.artifacts")
    if seen != set(_DEPENDENCIES):
        raise ParetoSelectionAuditError("dependency evidence is incomplete")

    boundary = _object(record.get("boundary"), "boundary")
    expected = {
        "measured_only_frontier": True,
        "hard_constraints_before_soft_tradeoffs": True,
        "selection_reproduces_from_canonical_evidence": True,
        "selection_outside_frontier": "rejected",
        "uncertainty_and_ties_visible": True,
        "negative_evidence_retained": True,
        "amendment_identity_preserved": True,
    }
    for key, value in expected.items():
        if boundary.get(key) != value:
            raise ParetoSelectionAuditError(f"selection boundary drifted: {key}")

    _files(root, record.get("api_and_design"), "api_and_design")
    _files(root, record.get("examples"), "examples")
    _files(root, record.get("tests"), "tests")
    if record.get("status") != "bounded_evidence_boundary":
        raise ParetoSelectionAuditError("selection audit status drifted")


def main() -> None:
    audit_release()
    print("Pareto selection explanation audit passed")


if __name__ == "__main__":
    main()
