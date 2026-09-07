"""Audit the closed v2.8 evidence-grounded explanations release boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from tools.audit_evidence_factuality_study import audit_release as audit_factuality_release
except ModuleNotFoundError:  # pragma: no cover - direct ``python tools/...`` execution
    from audit_evidence_factuality_study import audit_release as audit_factuality_release


class EvidenceGroundingReleaseAuditError(ValueError):
    """Raised when the v2.8 release boundary is incomplete or overclaims."""


_DEPENDENCIES = {
    "v28-evidence-query": (475, "e0b4940a0b3a9d6dda468b46bcc02de303a492bb"),
    "v28-claim-renderer": (476, "5558cbcfaab7de1e61c0ac35a500dd7aff5d172d"),
    "v28-negative-results": (477, "e2dec02c1747e4023ba56d1ba6a3a384447c0022"),
    "v28-pareto-selection": (478, "6b09bf0c7cfa0e6fe35dbbc4700c33f431c42694"),
    "v28-factuality": (479, "8c26280fa595c66fac26bd0b6789f922d4233fae"),
}

_REQUIRED_DOCS = {
    "ROADMAP.md",
    "docs/goal.md",
    "ARCHITECTURE.md",
    "README.md",
    "CHANGELOG.md",
    "docs/design/conversational-evidence-queries.md",
    "docs/design/claim-to-evidence-renderer.md",
    "docs/design/negative-evidence.md",
    "docs/design/pareto-selection-explanations.md",
    "docs/design/evidence-factuality-study.md",
    "docs/design/evidence-grounding-release.md",
    "docs/release/v2.8-evidence-grounding-boundary.md",
}

_REQUIRED_TESTS = {
    "tests/test_v28_evidence_grounding_release.py",
    "tests/test_evidence_query.py",
    "tests/test_claim_evidence.py",
    "tests/test_negative_evidence.py",
    "tests/test_pareto_selection.py",
    "tests/test_evidence_factuality_study.py",
}

_REQUIRED_EXAMPLES = {
    "docs/examples/evidence_query.py",
    "docs/examples/claim_evidence_renderer.py",
    "docs/examples/pareto_selection.py",
    "docs/examples/evidence_factuality_study.py",
}

_REQUIRED_OUTCOMES = {
    "measured",
    "predicted",
    "rejected",
    "rolled_back",
    "failed",
    "unsupported",
    "unknown",
    "inconclusive",
}

_REQUIRED_DEFERRED_TYPES = {
    "optimizer_optimality_or_proof",
    "live_provider_or_external_model_quality",
    "unmeasured_hardware_or_latency_explanations",
    "free_form_unbounded_narrative_explanations",
    "cross_model_generalization_without_held_out_evidence",
    "causal_attribution_without_canonical_intervention_evidence",
}


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise EvidenceGroundingReleaseAuditError(f"{label} must be an object")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceGroundingReleaseAuditError(f"{label} must be non-empty text")
    return value


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise EvidenceGroundingReleaseAuditError(f"{label} must be an array")
    return value


def _files(root: Path, value: object, label: str) -> None:
    values = _array(value, label)
    if not values:
        raise EvidenceGroundingReleaseAuditError(f"{label} must not be empty")
    for index, raw in enumerate(values):
        relative = Path(_text(raw, f"{label}[{index}]"))
        if relative.is_absolute() or ".." in relative.parts or not (root / relative).is_file():
            raise EvidenceGroundingReleaseAuditError(f"{label}[{index}] references a missing file")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), label)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceGroundingReleaseAuditError(f"could not read {label}") from error


def _require_texts(value: object, label: str) -> set[str]:
    result = {_text(item, f"{label}[{index}]") for index, item in enumerate(_array(value, label))}
    if not result:
        raise EvidenceGroundingReleaseAuditError(f"{label} must not be empty")
    return result


def _check_dependencies(root: Path, record: dict[str, Any]) -> None:
    seen: set[str] = set()
    for index, raw in enumerate(_array(record.get("dependency_evidence"), "dependency_evidence")):
        item = _object(raw, f"dependency_evidence[{index}]")
        key = _text(item.get("key"), f"dependency_evidence[{index}].key")
        if key in seen or key not in _DEPENDENCIES:
            raise EvidenceGroundingReleaseAuditError(f"unexpected or duplicate dependency {key}")
        seen.add(key)
        issue, commit = _DEPENDENCIES[key]
        if item.get("issue") != issue or item.get("status") != "merged":
            raise EvidenceGroundingReleaseAuditError(
                f"dependency {key} is not the merged dependency"
            )
        if item.get("commit") != commit:
            raise EvidenceGroundingReleaseAuditError(f"dependency {key} identity drifted")
        _files(root, item.get("artifacts"), f"dependency {key}.artifacts")
        _files(root, item.get("tests"), f"dependency {key}.tests")
    if seen != set(_DEPENDENCIES):
        raise EvidenceGroundingReleaseAuditError("v2.8 dependency evidence is incomplete")


def _check_authority(record: dict[str, Any]) -> None:
    authority = _object(record.get("authority_boundary"), "authority_boundary")
    expected = {
        "canonical_query_api": "modelsurgeon.conversation.direct_evidence_report",
        "canonical_query_engine": "modelsurgeon.conversation.EvidenceQueryEngine",
        "claim_renderer_input": "typed EvidenceQueryResponse only",
        "negative_report_projection": "modelsurgeon.explain.direct_negative_evidence_report",
        "pareto_report_projection": "modelsurgeon.explain.build_pareto_selection_explanation",
        "provider_text": "not_authoritative",
        "transcript_text": "not_authoritative",
        "prediction_values": "not_measurements",
        "source_model": "immutable",
    }
    for key, value in expected.items():
        if authority.get(key) != value:
            raise EvidenceGroundingReleaseAuditError(f"authority boundary drifted: {key}")


def _check_policy(record: dict[str, Any]) -> None:
    policy = _object(record.get("claim_policy"), "claim_policy")
    expected = {
        "source_traceability": (
            "every factual claim carries an evidence ID and source digest or is marked unavailable"
        ),
        "qualification": (
            "measured values retain units and uncertainty; unmeasured decisions are prediction-only"
        ),
        "negative_evidence": (
            "rejected, rolled_back, failed, unsupported, unknown, and inconclusive "
            "rows remain retained"
        ),
        "unknown_evidence": "missing or incomplete fields remain unknown and are never inferred",
        "optimizer_proof": "forbidden",
        "unavailable_measurements": "forbidden",
    }
    for key, value in expected.items():
        if policy.get(key) != value:
            raise EvidenceGroundingReleaseAuditError(f"claim policy drifted: {key}")
    if _require_texts(
        policy.get("outcome_vocabulary"), "claim_policy.outcome_vocabulary"
    ) != _REQUIRED_OUTCOMES:
        raise EvidenceGroundingReleaseAuditError("outcome vocabulary is incomplete or drifted")


def _check_factuality(root: Path, record: dict[str, Any]) -> None:
    factuality = _object(record.get("factuality_evidence"), "factuality_evidence")
    manifest = root / _text(factuality.get("manifest"), "factuality_evidence.manifest")
    _files(root, [factuality.get("manifest")], "factuality_evidence.manifest")
    audit_factuality_release(root, manifest=manifest)
    if factuality.get("grounded_renderer") != "passed":
        raise EvidenceGroundingReleaseAuditError(
            "grounded renderer factuality decision is not passed"
        )
    if factuality.get("template_only") != "passed":
        raise EvidenceGroundingReleaseAuditError("template-only factuality decision is not passed")
    if factuality.get("unconstrained_text") != "retained_never_shippable_negative_control":
        raise EvidenceGroundingReleaseAuditError("unconstrained text control policy drifted")
    thresholds = _object(factuality.get("thresholds"), "factuality_evidence.thresholds")
    expected_thresholds = {
        "claim_source_precision": 1.0,
        "claim_source_recall": 1.0,
        "source_id_correctness": 1.0,
        "negative_evidence_coverage": 1.0,
        "uncertainty_disclosure": 1.0,
        "uncertainty_calibration": 1.0,
        "deterministic_ids": 1.0,
        "measured_predicted_confusion_rate": 0.0,
        "unsupported_claim_rate": 0.0,
        "critical_escapes": 0,
    }
    if thresholds != expected_thresholds:
        raise EvidenceGroundingReleaseAuditError("factuality thresholds drifted")
    if factuality.get("negative_and_inconclusive_retained") is not True:
        raise EvidenceGroundingReleaseAuditError("negative/inconclusive retention was not recorded")


def _check_limits(record: dict[str, Any]) -> None:
    limitations = _object(record.get("limitations"), "limitations")
    for key in ("supported", "unsupported", "unknown"):
        _require_texts(limitations.get(key), f"limitations.{key}")
    deferred = _require_texts(
        limitations.get("deferred_explanation_types"),
        "limitations.deferred_explanation_types",
    )
    if not _REQUIRED_DEFERRED_TYPES.issubset(deferred):
        raise EvidenceGroundingReleaseAuditError("deferred explanation types are incomplete")
    if limitations.get("optimizer_proof_claim") != "not_supported":
        raise EvidenceGroundingReleaseAuditError("optimizer proof limitation drifted")
    if limitations.get("unavailable_measurement_claim") != "not_supported":
        raise EvidenceGroundingReleaseAuditError("unavailable measurement limitation drifted")


def _check_duplicate_scope(record: dict[str, Any]) -> None:
    review = _object(record.get("duplicate_scope_review"), "duplicate_scope_review")
    if review.get("status") != "passed" or review.get("no_duplicate_authority") is not True:
        raise EvidenceGroundingReleaseAuditError("duplicate scope review did not pass")
    issues = _array(review.get("reviewed_issues"), "duplicate_scope_review.reviewed_issues")
    by_issue = {
        item.get("issue"): _object(item, "reviewed duplicate-scope issue") for item in issues
    }
    for issue in (420, 411, 429):
        item = by_issue.get(issue)
        if item is None or item.get("boundary") != {
            420: "offline explorer/report projection remains the explorer authority",
            411: "signed immutable evidence bundles remain the evidence-package authority",
            429: "optimizer benchmark/reference artifacts remain the measured benchmark authority",
        }[issue]:
            raise EvidenceGroundingReleaseAuditError(f"duplicate scope for #{issue} drifted")


def audit_release(root: Path = Path("."), *, manifest: Path | None = None) -> None:
    """Validate and replay the complete v2.8 milestone boundary."""

    root = root.resolve()
    manifest_path = manifest or (
        root / "docs" / "research" / "v2.8-evidence-grounding-release-v1.json"
    )
    record = _read_json(manifest_path, "v2.8 evidence-grounding release manifest")
    if record.get("record_type") != "bounded_evidence_grounding_release":
        raise EvidenceGroundingReleaseAuditError("unexpected v2.8 release record type")
    if record.get("schema_version") != 1:
        raise EvidenceGroundingReleaseAuditError("unsupported v2.8 release schema")
    if record.get("protocol_revision") != "modelsurgeon-v2.8-evidence-grounding-release-v1":
        raise EvidenceGroundingReleaseAuditError("v2.8 release protocol revision drifted")
    if record.get("milestone") != "v2.8" or record.get("status") != "closed":
        raise EvidenceGroundingReleaseAuditError("v2.8 milestone status is not closed")

    _check_dependencies(root, record)
    _check_authority(record)
    _check_policy(record)
    _check_factuality(root, record)
    _check_limits(record)
    _check_duplicate_scope(record)
    _files(root, record.get("documentation"), "documentation")
    if not _REQUIRED_DOCS.issubset(set(record["documentation"])):
        raise EvidenceGroundingReleaseAuditError("v2.8 documentation set is incomplete")
    _files(root, record.get("tests"), "tests")
    if not _REQUIRED_TESTS.issubset(set(record["tests"])):
        raise EvidenceGroundingReleaseAuditError("v2.8 acceptance test set is incomplete")
    _files(root, record.get("examples"), "examples")
    if not _REQUIRED_EXAMPLES.issubset(set(record["examples"])):
        raise EvidenceGroundingReleaseAuditError("v2.8 representative examples are incomplete")

    quality = _object(record.get("quality_gate"), "quality_gate")
    if quality.get("status") != "passed":
        raise EvidenceGroundingReleaseAuditError("quality gate is not recorded as passed")
    required_commands = (
        "git diff --check",
        "ruff check src tests",
        "mypy src/modelsurgeon",
        "pytest -q",
        "audit_v28_evidence_grounding_release.py",
        "docs/examples/evidence_query.py",
        "docs/examples/claim_evidence_renderer.py",
        "docs/examples/pareto_selection.py",
        "docs/examples/evidence_factuality_study.py",
    )
    commands = _array(quality.get("commands"), "quality_gate.commands")
    for required in required_commands:
        if not any(required in str(command) for command in commands):
            raise EvidenceGroundingReleaseAuditError(f"quality gate is missing {required}")
    if _text(quality.get("revision"), "quality_gate.revision") != (
        "issue-480-evidence-grounding-release"
    ):
        raise EvidenceGroundingReleaseAuditError("quality-gate revision drifted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    audit_release(args.root, manifest=args.manifest)
    print("v2.8 evidence-grounding release audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
