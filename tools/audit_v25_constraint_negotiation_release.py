"""Audit the bounded v2.5 constraint-negotiation release boundary."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from modelsurgeon.conversation import (
    NegotiationPolicy,
    load_and_run_negotiation_study,
    load_negotiation_study,
)

try:
    from tools.audit_v25_negotiation_quality import audit_release as audit_quality_release
except ModuleNotFoundError:  # pragma: no cover - direct ``python tools/...`` execution
    from audit_v25_negotiation_quality import audit_release as audit_quality_release


class ConstraintNegotiationReleaseAuditError(ValueError):
    """Raised when the v2.5 release boundary or evidence drifts."""


_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DEPENDENCIES = {
    "v20-approval-boundary": (422, "f6775eff7ed9bc69f5b4e5ab830c7ffc4880888b"),
    "v25-infeasibility": (458, "c6f5059bdba31087d691c25dfdf2aade0c15699a"),
    "v25-pareto": (459, "5f405827649ffb43c47f2431c1bb7887ff5195f2"),
    "v25-amendments": (460, "45daba27316ce01af116d18585ef0523eea72435"),
    "v25-decision-quality": (461, "c4b6cd1027823b65f869bd60d86ee416138cbbc8"),
}
_CANDIDATE_STATUSES = {"measured", "predicted", "unsupported", "failed", "unknown", "inconclusive"}
_REQUIRED_OUTCOMES = {
    "supported",
    "unsupported",
    "failed",
    "unknown",
    "negative",
    "inconclusive",
    "prediction_only",
}


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConstraintNegotiationReleaseAuditError(f"{label} must be an object")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConstraintNegotiationReleaseAuditError(f"{label} must be non-empty text")
    return value


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConstraintNegotiationReleaseAuditError(f"{label} must be an array")
    return value


def _relative_file(root: Path, value: object, label: str) -> None:
    raw = _text(value, label)
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts or not (root / path).is_file():
        raise ConstraintNegotiationReleaseAuditError(f"{label} references a missing file")


def _files(root: Path, value: object, label: str) -> None:
    for index, item in enumerate(_array(value, label)):
        _relative_file(root, item, f"{label}[{index}]")


def _require_strings(value: object, label: str) -> None:
    values = _array(value, label)
    if not values:
        raise ConstraintNegotiationReleaseAuditError(f"{label} must not be empty")
    for index, item in enumerate(values):
        _text(item, f"{label}[{index}]")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConstraintNegotiationReleaseAuditError(f"could not read {label}") from error
    return _object(value, label)


def audit_release(root: Path = Path("."), *, manifest: Path | None = None) -> None:
    """Validate the v2.5 milestone boundary and replay its evidence."""

    root = root.resolve()
    manifest_path = manifest or (
        root / "docs" / "research" / "v2.5-constraint-negotiation-release-v1.json"
    )
    record = _read_json(manifest_path, "v2.5 constraint-negotiation release record")

    if record.get("record_type") != "bounded_constraint_negotiation_release":
        raise ConstraintNegotiationReleaseAuditError("unexpected v2.5 release record type")
    if record.get("schema_version") != 1:
        raise ConstraintNegotiationReleaseAuditError("unsupported v2.5 release schema")
    if record.get("protocol_revision") != "modelsurgeon-v2.5-constraint-negotiation-release-v1":
        raise ConstraintNegotiationReleaseAuditError("unexpected v2.5 release protocol revision")
    if record.get("milestone") != "v2.5" or record.get("status") != "bounded_release_boundary":
        raise ConstraintNegotiationReleaseAuditError("v2.5 release status or milestone drifted")

    dependencies = _array(record.get("dependency_evidence"), "dependency_evidence")
    seen: set[str] = set()
    for index, raw in enumerate(dependencies):
        item = _object(raw, f"dependency_evidence[{index}]")
        key = _text(item.get("key"), f"dependency_evidence[{index}].key")
        if key in seen or key not in _DEPENDENCIES:
            raise ConstraintNegotiationReleaseAuditError(
                f"unexpected or duplicate dependency {key}"
            )
        seen.add(key)
        issue, commit = _DEPENDENCIES[key]
        if item.get("issue") != issue or item.get("status") != "merged":
            raise ConstraintNegotiationReleaseAuditError(
                f"dependency {key} is not the merged dependency"
            )
        actual_commit = _text(item.get("commit"), f"dependency {key}.commit")
        if _COMMIT.fullmatch(actual_commit) is None or actual_commit != commit:
            raise ConstraintNegotiationReleaseAuditError(f"dependency {key} identity drifted")
        _files(root, item.get("artifacts"), f"dependency {key}.artifacts")
        _files(root, item.get("tests"), f"dependency {key}.tests")
    if seen != set(_DEPENDENCIES):
        raise ConstraintNegotiationReleaseAuditError("dependency evidence is incomplete")

    boundary = _object(record.get("release_boundary"), "release_boundary")
    expected_boundary = {
        "automatic_constraint_relaxation": "not_supported",
        "hard_constraints": "preserved_until_explicit_approved_amendment",
        "source_model": "immutable",
        "pareto_evidence": "measured_only",
        "amendments": "user_approved_immutable_reproducible",
        "prediction_only": "retained_never_shippable",
        "negative_and_inconclusive": "retained",
    }
    for key, expected in expected_boundary.items():
        if boundary.get(key) != expected:
            raise ConstraintNegotiationReleaseAuditError(f"release boundary drifted: {key}")

    v20 = _object(record.get("v20_approval_boundary"), "v20_approval_boundary")
    _relative_file(root, v20.get("release_record"), "v20_approval_boundary.release_record")
    if v20.get("require_execution_approval") is not True:
        raise ConstraintNegotiationReleaseAuditError("v2.0 execution approval is not required")
    if v20.get("require_custom_plugin_approval") is not True:
        raise ConstraintNegotiationReleaseAuditError("v2.0 custom-plugin approval is not required")
    if v20.get("approval_scope") != "exact_visible_diff":
        raise ConstraintNegotiationReleaseAuditError(
            "approval scope is not bound to the visible diff"
        )
    if v20.get("approval_expiry") != "enforced" or v20.get("stale_replay") != "fail_closed":
        raise ConstraintNegotiationReleaseAuditError(
            "v2.0 approval expiry or stale replay boundary drifted"
        )
    v20_record = _read_json(
        root / _text(v20["release_record"], "v2.0 release record"), "v2.0 release record"
    )
    if v20_record.get("status") != "bounded_release_boundary":
        raise ConstraintNegotiationReleaseAuditError("v2.0 release record is not bounded")

    infeasibility = _object(record.get("infeasibility"), "infeasibility")
    if infeasibility.get("non_destructive") is not True:
        raise ConstraintNegotiationReleaseAuditError("infeasibility is not non-destructive")
    if infeasibility.get("hard_constraint_relaxation") != "never_automatic":
        raise ConstraintNegotiationReleaseAuditError("infeasibility permits automatic relaxation")
    if infeasibility.get("source_model_immutable") is not True:
        raise ConstraintNegotiationReleaseAuditError(
            "infeasibility does not preserve source identity"
        )
    statuses = set(
        _array(infeasibility.get("candidate_statuses"), "infeasibility.candidate_statuses")
    )
    if statuses != _CANDIDATE_STATUSES:
        raise ConstraintNegotiationReleaseAuditError("candidate status retention boundary drifted")
    outcomes = set(
        _array(infeasibility.get("outcomes_retained"), "infeasibility.outcomes_retained")
    )
    if outcomes != _REQUIRED_OUTCOMES:
        raise ConstraintNegotiationReleaseAuditError("outcome retention boundary drifted")
    _files(root, infeasibility.get("api_and_design"), "infeasibility.api_and_design")

    pareto = _object(record.get("pareto"), "pareto")
    if pareto.get("measured_only") is not True or pareto.get("relaxes_constraints") is not False:
        raise ConstraintNegotiationReleaseAuditError(
            "Pareto boundary is not measured-only and read-only"
        )
    if pareto.get("retains_terminal_evidence") is not True:
        raise ConstraintNegotiationReleaseAuditError("Pareto terminal evidence is not retained")
    _files(root, pareto.get("api_and_design"), "pareto.api_and_design")

    amendments = _object(record.get("amendments"), "amendments")
    expected_amendments = {
        "approval_required": True,
        "immutable": True,
        "reproducible": True,
        "original_contract_retained": True,
        "original_evidence_retained": True,
        "material_change_new_campaign": True,
        "direct_api_history": "append_only",
    }
    for key, expected in expected_amendments.items():
        if amendments.get(key) != expected:
            raise ConstraintNegotiationReleaseAuditError(f"amendment boundary drifted: {key}")
    _files(root, amendments.get("api_and_design"), "amendments.api_and_design")

    decision = _object(record.get("decision_quality_evidence"), "decision_quality_evidence")
    study_record_path = root / _text(
        decision.get("research_record"), "decision_quality.research_record"
    )
    _relative_file(root, decision.get("research_record"), "decision_quality.research_record")
    _relative_file(root, decision.get("fixture"), "decision_quality.fixture")
    audit_quality_release(
        root, manifest=root / "docs" / "research" / "v2.5-negotiation-decision-quality-v1.json"
    )
    study = load_negotiation_study(root / _text(decision["fixture"], "decision_quality.fixture"))
    run = load_and_run_negotiation_study(
        root / _text(decision["fixture"], "decision_quality.fixture")
    )
    expected_run_id = _text(decision.get("run_id"), "decision_quality.run_id")
    if run.run_id != expected_run_id or len(run.results) != decision.get("retained_cell_count"):
        raise ConstraintNegotiationReleaseAuditError(
            "decision-quality replay identity or cell count drifted"
        )
    if study.corpus_revision != _text(
        decision.get("corpus_revision"), "decision_quality.corpus_revision"
    ):
        raise ConstraintNegotiationReleaseAuditError("decision-quality corpus revision drifted")
    if not run.passed:
        raise ConstraintNegotiationReleaseAuditError(
            "measured decision-quality replay no longer passes"
        )
    measured = next(
        item for item in run.metrics if item.policy is NegotiationPolicy.MEASURED_PARETO
    )
    if measured.to_record() != _object(
        decision.get("measured_pareto_metrics"), "measured_pareto_metrics"
    ):
        raise ConstraintNegotiationReleaseAuditError("measured decision-quality metrics drifted")
    prediction = next(
        item for item in run.metrics if item.policy is NegotiationPolicy.PREDICTION_ONLY
    )
    prediction_control = _object(decision.get("prediction_only_control"), "prediction_only_control")
    if (
        prediction_control.get("shippable") is not False
        or prediction_control.get("decision") != "never_shippable"
    ):
        raise ConstraintNegotiationReleaseAuditError("prediction-only control became shippable")
    if prediction.misleading_claim_rate != prediction_control.get("misleading_claim_rate"):
        raise ConstraintNegotiationReleaseAuditError("prediction-only metric drifted")
    all_statuses = {
        candidate.status.value for scenario in study.scenarios for candidate in scenario.candidates
    }
    if all_statuses != _CANDIDATE_STATUSES:
        raise ConstraintNegotiationReleaseAuditError("study dropped a retained candidate status")
    if not all(item.negative for item in run.results) or not any(
        item.inconclusive for item in run.results
    ):
        raise ConstraintNegotiationReleaseAuditError("negative or inconclusive cells were dropped")
    _files(root, decision.get("documentation"), "decision_quality.documentation")

    direct_api = _object(record.get("direct_api_compatibility"), "direct_api_compatibility")
    if (
        direct_api.get("preserved") is not True
        or direct_api.get("objective_history") != "preserved"
    ):
        raise ConstraintNegotiationReleaseAuditError("direct API compatibility boundary drifted")
    _files(root, direct_api.get("tests"), "direct_api_compatibility.tests")

    duplicate_review = _object(record.get("duplicate_scope_review"), "duplicate_scope_review")
    if (
        duplicate_review.get("status") != "passed"
        or duplicate_review.get("no_duplicate_authority") is not True
    ):
        raise ConstraintNegotiationReleaseAuditError("duplicate-scope review did not pass")
    _require_strings(
        duplicate_review.get("reviewed_boundaries"), "duplicate_scope_review.reviewed_boundaries"
    )

    _files(root, record.get("documentation"), "documentation")
    _require_strings(record.get("reproduction_commands"), "reproduction_commands")
    quality = _object(record.get("quality_gate"), "quality_gate")
    if quality.get("status") != "passed":
        raise ConstraintNegotiationReleaseAuditError(
            "release quality gate is not recorded as passed"
        )
    _require_strings(quality.get("commands"), "quality_gate.commands")
    _text(quality.get("environment"), "quality_gate.environment")
    _text(quality.get("revision"), "quality_gate.revision")
    _require_strings(quality.get("outcomes"), "quality_gate.outcomes")
    _require_strings(quality.get("known_skips"), "quality_gate.known_skips")
    _object(record.get("resource_budgets"), "resource_budgets")
    _text(study_record_path.as_posix(), "study record path")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    audit_release(args.root, manifest=args.manifest)
    print("v2.5 constraint-negotiation release boundary verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
