"""Audit the bounded v2.8 evidence-grounding factuality boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from modelsurgeon.evaluation.evidence_factuality_study import (
    EvidenceFactualityMethod,
    load_and_run_evidence_factuality_study,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "docs" / "research" / "v2.8-evidence-factuality-study-v1.json"


class EvidenceFactualityAuditError(ValueError):
    """Raised when protocol, replay, or release-threshold evidence drifts."""


_DEPENDENCIES = {
    "v28-claim-renderer": (476, "5558cbcfaab7de1e61c0ac35a500dd7aff5d172d"),
    "v28-negative-results": (477, "e2dec02c1747e4023ba56d1ba6a3a384447c0022"),
    "v28-pareto-selection": (478, "6b09bf0c7cfa0e6fe35dbbc4700c33f431c42694"),
    "v28-evidence-snapshots": (475, "e0b4940a0b3a9d6dda468b46bcc02de303a492bb"),
}


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceFactualityAuditError(f"{label} must be an object")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceFactualityAuditError(f"{label} must be non-empty text")
    return value


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise EvidenceFactualityAuditError(f"{label} must be an array")
    return value


def _files(root: Path, value: object, label: str) -> None:
    for index, raw in enumerate(_array(value, label)):
        relative = Path(_text(raw, f"{label}[{index}]"))
        if relative.is_absolute() or ".." in relative.parts or not (root / relative).is_file():
            raise EvidenceFactualityAuditError(f"{label}[{index}] references a missing file")


def audit_release(root: Path = Path("."), *, manifest: Path | None = None) -> None:
    """Validate the checked-in protocol and replay its release decision."""

    root = root.resolve()
    manifest_path = manifest or (root / DEFAULT_MANIFEST.relative_to(ROOT))
    try:
        record = _object(json.loads(manifest_path.read_text(encoding="utf-8")), "manifest")
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceFactualityAuditError("could not read evidence-factuality manifest") from error

    if record.get("record_type") != "bounded_evidence_factuality_study":
        raise EvidenceFactualityAuditError("unexpected evidence-factuality record type")
    if record.get("schema_version") != 1:
        raise EvidenceFactualityAuditError("unsupported evidence-factuality manifest schema")
    if record.get("protocol_revision") != "modelsurgeon-v2.8-evidence-factuality-v1":
        raise EvidenceFactualityAuditError("evidence-factuality protocol revision drifted")
    if record.get("status") != "bounded_release_threshold_decision":
        raise EvidenceFactualityAuditError("evidence-factuality status drifted")

    dependencies = _array(record.get("dependency_evidence"), "dependency_evidence")
    seen: set[str] = set()
    for index, raw in enumerate(dependencies):
        item = _object(raw, f"dependency_evidence[{index}]")
        key = _text(item.get("key"), f"dependency_evidence[{index}].key")
        if key in seen or key not in _DEPENDENCIES:
            raise EvidenceFactualityAuditError(f"unexpected or duplicate dependency {key}")
        seen.add(key)
        issue, commit = _DEPENDENCIES[key]
        if item.get("issue") != issue or item.get("status") != "merged":
            raise EvidenceFactualityAuditError(f"dependency {key} is not the merged dependency")
        if item.get("commit") != commit:
            raise EvidenceFactualityAuditError(f"dependency {key} identity drifted")
        _files(root, item.get("artifacts"), f"dependency_evidence[{index}].artifacts")
        _files(root, item.get("tests"), f"dependency_evidence[{index}].tests")
    if seen != set(_DEPENDENCIES):
        raise EvidenceFactualityAuditError("dependency evidence is incomplete")

    protocol = _object(record.get("protocol"), "protocol")
    fixture = root / _text(protocol.get("fixture"), "protocol.fixture")
    _files(root, [protocol.get("fixture")], "protocol.fixture")
    if protocol.get("protocol_id") != "v28-evidence-factuality-v1":
        raise EvidenceFactualityAuditError("protocol ID drifted")
    expected_protocol = {
        "case_count": 6,
        "seeds_per_case": 3,
        "method_count": 3,
        "retained_cell_count": 54,
        "negative_and_inconclusive_retained": True,
        "live_provider_replay": "not_run",
        "fixed_cpu_bound_seconds": 10.0,
        "critical_escape_policy": "fail_closed_for_shippability",
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise EvidenceFactualityAuditError(f"protocol boundary drifted: {key}")

    run = load_and_run_evidence_factuality_study(fixture)
    if run.run_id != record.get("run_id"):
        raise EvidenceFactualityAuditError("replayed run identity drifted")
    if len(run.results) != protocol["retained_cell_count"]:
        raise EvidenceFactualityAuditError("retained cell count drifted")
    if run.stop_reason != "corpus_complete;critical_escapes_fail_closed_for_shippability":
        raise EvidenceFactualityAuditError("stop/fail-closed policy drifted")

    metrics = {item.method.value: item for item in run.metrics}
    expected_metrics = _object(record.get("metrics"), "metrics")
    for method_name, expected_values in expected_metrics.items():
        if method_name not in metrics:
            raise EvidenceFactualityAuditError(f"missing method metric: {method_name}")
        expected_method = _object(expected_values, f"metrics.{method_name}")
        actual = metrics[method_name].to_record()
        for key, expected in expected_method.items():
            if actual.get(key) != expected:
                raise EvidenceFactualityAuditError(f"metric drifted: {method_name}.{key}")

    if not run.policy_passed(EvidenceFactualityMethod.GROUNDED_RENDERER):
        raise EvidenceFactualityAuditError("grounded renderer no longer meets thresholds")
    if not run.policy_passed(EvidenceFactualityMethod.TEMPLATE_ONLY):
        raise EvidenceFactualityAuditError(
            "template-only safety baseline no longer meets thresholds"
        )
    if run.policy_passed(EvidenceFactualityMethod.UNCONSTRAINED_TEXT):
        raise EvidenceFactualityAuditError("unconstrained text control became shippable")

    decision = _object(record.get("release_threshold_decision"), "release_threshold_decision")
    if decision.get("renderer") != "ship_bounded_grounded_renderer":
        raise EvidenceFactualityAuditError("renderer release decision drifted")
    if decision.get("unconstrained_text") != "never_shippable":
        raise EvidenceFactualityAuditError("unconstrained-text decision drifted")
    unsupported = _array(
        record.get("retained_failures_and_inconclusive"), "retained_failures_and_inconclusive"
    )
    if len(unsupported) < 2:
        raise EvidenceFactualityAuditError(
            "negative and inconclusive retention evidence is incomplete"
        )
    for index, raw in enumerate(unsupported):
        item = _object(raw, f"retained_failures_and_inconclusive[{index}]")
        _text(item.get("cell"), f"retained_failures_and_inconclusive[{index}].cell")
        _text(item.get("disposition"), f"retained_failures_and_inconclusive[{index}].disposition")

    authority = _object(record.get("authority_boundary"), "authority_boundary")
    for key in (
        "canonical_evidence_authoritative",
        "source_model_immutable",
        "provider_independent",
        "unconstrained_text_shippable",
    ):
        expected = key != "unconstrained_text_shippable"
        if authority.get(key) is not expected:
            raise EvidenceFactualityAuditError(f"authority boundary drifted: {key}")
    _files(root, record.get("documentation"), "documentation")
    _files(root, record.get("examples"), "examples")
    commands = _array(record.get("reproduction_commands"), "reproduction_commands")
    for required in (
        "run_evidence_factuality_study.py",
        "audit_evidence_factuality_study.py",
        "docs/examples/evidence_factuality_study.py",
    ):
        if not any(required in str(command) for command in commands):
            raise EvidenceFactualityAuditError(f"reproduction commands are missing {required}")
    quality = _object(record.get("quality_gate"), "quality_gate")
    if quality.get("status") != "passed":
        raise EvidenceFactualityAuditError("quality gate is not recorded as passed")
    for required in (
        "git diff --check",
        "ruff check src tests",
        "mypy src/modelsurgeon",
        "pytest -q",
    ):
        if not any(
            required in str(command)
            for command in _array(quality.get("commands"), "quality_gate.commands")
        ):
            raise EvidenceFactualityAuditError(f"quality gate is missing {required}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    audit_release(args.root, manifest=args.manifest)
    print("v2.8 evidence-grounding factuality study audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
