"""Audit the bounded v2.4 clarification release boundary."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from modelsurgeon.conversation import (
    CLARIFICATION_SCHEMA_VERSION,
    ClarificationStatus,
)


class ClarificationReleaseAuditError(ValueError):
    """Raised when the v2.4 release record is incomplete or overclaims."""


_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DEPENDENCIES = {
    "v24-clarification-state": (453, "f6fd437b3ebf7d7282d936c88711aeb51c2d8ceb"),
    "v24-contradictions": (454, "09241f4f9bc8997debeb7d07681b68e03144c598"),
    "v24-measurable-targets": (455, "94781c862df2a41e0594299da25ecf4e5c71acb8"),
    "v24-research": (456, "4b836287d76e4839a9882dccfa38a8a0ac2f0cc0"),
}
_EXPECTED_STATES = {item.value for item in ClarificationStatus}
_EXPECTED_MEASURES = {
    "disk_size",
    "latency",
    "memory",
    "parameter_count",
    "perplexity",
    "quality",
    "throughput",
}
_EXPECTED_UNSUPPORTED_CELLS = {
    "unknown_metric",
    "budget_field_in_v2_objective",
    "allowed_operation_field_in_v2_objective",
    "deployment_target_execution",
    "raw_provider_text_answer",
    "arbitrary_language_or_locale",
    "autonomous_negotiation",
}


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ClarificationReleaseAuditError(f"{label} must be an object")
    return value


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise ClarificationReleaseAuditError(f"{label} must be a non-empty array")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ClarificationReleaseAuditError(f"{label} must be non-empty text")
    return value


def _relative_file(root: Path, value: object, label: str) -> None:
    raw = _text(value, label)
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ClarificationReleaseAuditError(f"{label} must be repository-relative")
    if not (root / path).is_file():
        raise ClarificationReleaseAuditError(f"{label} references missing file {raw!r}")


def _files(root: Path, value: object, label: str) -> None:
    for index, item in enumerate(_array(value, label)):
        _relative_file(root, item, f"{label}[{index}]")


def _require_strings(value: object, label: str) -> list[str]:
    result = []
    for index, item in enumerate(_array(value, label)):
        result.append(_text(item, f"{label}[{index}]"))
    return result


def audit_release(
    root: Path = Path("."), *, manifest: Path | None = None
) -> None:
    """Validate the checked-in v2.4 release record and evidence paths."""

    root = root.resolve()
    manifest_path = manifest or (
        root / "docs" / "research" / "v2.4-clarification-release-v1.json"
    )
    try:
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ClarificationReleaseAuditError(
            f"could not read release record {manifest_path}"
        ) from error

    if record.get("record_type") != "bounded_clarification_release":
        raise ClarificationReleaseAuditError("unexpected v2.4 release record type")
    if record.get("schema_version") != 1:
        raise ClarificationReleaseAuditError("unsupported v2.4 release schema version")
    if record.get("protocol_revision") != "modelsurgeon-v2.4-clarification-release-v1":
        raise ClarificationReleaseAuditError("unexpected v2.4 protocol revision")
    if record.get("milestone") != "v2.4" or record.get("status") != "bounded_release_boundary":
        raise ClarificationReleaseAuditError("release must remain a bounded v2.4 boundary")

    dependencies = _array(record.get("dependency_evidence"), "dependency_evidence")
    seen: set[str] = set()
    for index, raw in enumerate(dependencies):
        dependency = _object(raw, f"dependency_evidence[{index}]")
        key = _text(dependency.get("key"), f"dependency_evidence[{index}].key")
        if key in seen or key not in _DEPENDENCIES:
            raise ClarificationReleaseAuditError(f"unexpected or duplicate dependency {key}")
        seen.add(key)
        issue, commit = _DEPENDENCIES[key]
        if dependency.get("issue") != issue or dependency.get("status") != "merged":
            raise ClarificationReleaseAuditError(f"dependency {key} is not the merged issue")
        actual_commit = _text(dependency.get("commit"), f"dependency {key}.commit")
        if _COMMIT.fullmatch(actual_commit) is None or actual_commit != commit:
            raise ClarificationReleaseAuditError(f"dependency {key} has an unexpected commit")
        _files(root, dependency.get("artifacts"), f"dependency {key}.artifacts")
        _files(root, dependency.get("tests"), f"dependency {key}.tests")
    if seen != set(_DEPENDENCIES):
        raise ClarificationReleaseAuditError("dependency evidence is incomplete")

    contract = _object(record.get("clarification_contract"), "clarification_contract")
    if contract.get("schema_version") != CLARIFICATION_SCHEMA_VERSION:
        raise ClarificationReleaseAuditError("clarification schema version drifted")
    states = set(_require_strings(contract.get("states"), "clarification_contract.states"))
    if states != _EXPECTED_STATES:
        raise ClarificationReleaseAuditError("clarification state vocabulary drifted")
    if contract.get("outcome_precedence") != [
        "refused",
        "unsupported",
        "clarification_required",
        "executable",
    ]:
        raise ClarificationReleaseAuditError("fail-closed outcome precedence drifted")
    if contract.get("typed_answers_only") is not True:
        raise ClarificationReleaseAuditError("typed-answer requirement is missing")
    if contract.get("hard_constraints_mutable_by_answer") is not False:
        raise ClarificationReleaseAuditError("hard-constraint immutability is missing")
    if contract.get("contradictory_hard_constraints_receive_questions") is not False:
        raise ClarificationReleaseAuditError(
            "contradictory hard constraints must not be questioned"
        )
    for field in ("general_natural_language_understanding", "autonomous_objective_negotiation"):
        if contract.get(field) != "not_claimed":
            raise ClarificationReleaseAuditError(f"{field} must remain not_claimed")

    targets = _object(record.get("target_policy"), "target_policy")
    recognized_measures = set(
        _require_strings(targets.get("recognized_measures"), "target_policy.recognized_measures")
    )
    if recognized_measures != _EXPECTED_MEASURES:
        raise ClarificationReleaseAuditError("recognized measure vocabulary drifted")
    if targets.get("complete_requires") != [
        "metric",
        "direction",
        "unit",
        "typed_threshold_for_hard_constraints",
        "executable_policy",
    ]:
        raise ClarificationReleaseAuditError("complete-target requirements drifted")
    if targets.get("unknown_metric_outcome") != "unsupported":
        raise ClarificationReleaseAuditError("unknown metrics must remain unsupported")
    if targets.get("default_threshold") is not None or targets.get("default_baseline") is not None:
        raise ClarificationReleaseAuditError("target defaults must remain absent")
    if targets.get("deployment_compatibility_claim") != "not_inferred":
        raise ClarificationReleaseAuditError("deployment compatibility must not be inferred")

    evidence = _object(record.get("evidence"), "evidence")
    _relative_file(root, evidence.get("protocol_record"), "evidence.protocol_record")
    _relative_file(root, evidence.get("source_corpus"), "evidence.source_corpus")
    if evidence.get("retained_case_count") != 11 or evidence.get("retained_cell_count") != 44:
        raise ClarificationReleaseAuditError("retained clarification study size drifted")
    if evidence.get("negative_and_inconclusive_retained") is not True:
        raise ClarificationReleaseAuditError("negative and inconclusive evidence must be retained")
    if evidence.get("live_provider_benchmarks") != "not_run":
        raise ClarificationReleaseAuditError("live provider benchmarks must remain not_run")
    if evidence.get("live_model_quality") != "not_run":
        raise ClarificationReleaseAuditError("live model-quality evidence must remain not_run")
    if evidence.get("provider_replay_seeds") != []:
        raise ClarificationReleaseAuditError("provider replay seeds must remain empty")
    metrics = _object(evidence.get("schema_driven_metrics"), "schema_driven_metrics")
    expected_metrics = {
        "necessary_question_recall": 1.0,
        "unnecessary_question_rate": 0.0,
        "silent_constraint_invention_rate": 0.0,
        "executable_spec_precision": 1.0,
        "executable_spec_recall": 1.0,
        "exact_spec_equivalence_rate": 1.0,
        "refusal_correctness_rate": 1.0,
    }
    if metrics != expected_metrics:
        raise ClarificationReleaseAuditError("schema-driven study metrics drifted")
    _text(evidence.get("run_id"), "evidence.run_id")

    limits = _object(record.get("limits"), "limits")
    if limits != {
        "max_cases": 64,
        "max_questions_per_case": 16,
        "max_provider_replay_seeds": 3,
        "hostile_process_containment": "not_claimed",
    }:
        raise ClarificationReleaseAuditError("clarification resource limits drifted")

    authority = _object(record.get("authority_and_identity"), "authority_and_identity")
    if authority.get("hard_constraints_preserved") is not True:
        raise ClarificationReleaseAuditError("hard-constraint preservation is not recorded")
    if authority.get("source_model_immutable") is not True:
        raise ClarificationReleaseAuditError("source-model immutability is not recorded")
    if authority.get("provenance_retained") is not True:
        raise ClarificationReleaseAuditError("provenance retention is not recorded")
    if authority.get("execution_authority") != "deterministic_modelsurgeon_engine":
        raise ClarificationReleaseAuditError("clarification cannot become execution authority")
    if authority.get("canonical_identity_fields") != [
        "question_id",
        "answer_digest",
        "state_id",
        "decision_id",
        "result_id",
        "run_id",
    ]:
        raise ClarificationReleaseAuditError("canonical identity fields drifted")

    cells = _array(record.get("unsupported_cells"), "unsupported_cells")
    seen_cells: set[str] = set()
    for index, raw in enumerate(cells):
        cell = _object(raw, f"unsupported_cells[{index}]")
        name = _text(cell.get("cell"), f"unsupported_cells[{index}].cell")
        if name in seen_cells or name not in _EXPECTED_UNSUPPORTED_CELLS:
            raise ClarificationReleaseAuditError(f"unexpected or duplicate unsupported cell {name}")
        seen_cells.add(name)
        if cell.get("status") not in {"unsupported", "not_claimed"}:
            raise ClarificationReleaseAuditError(f"unsupported cell {name} has invalid status")
        _text(cell.get("reason"), f"unsupported cell {name}.reason")
    if seen_cells != _EXPECTED_UNSUPPORTED_CELLS:
        raise ClarificationReleaseAuditError("unsupported-cell evidence is incomplete")

    direct_api = _object(record.get("direct_api_compatibility"), "direct_api_compatibility")
    if direct_api.get("preserved") is not True:
        raise ClarificationReleaseAuditError("direct API preservation is not recorded")
    _files(root, direct_api.get("tests"), "direct_api_compatibility.tests")

    _files(root, record.get("documentation"), "documentation")
    _require_strings(record.get("reproduction_commands"), "reproduction_commands")
    quality = _object(record.get("quality_gate"), "quality_gate")
    _require_strings(quality.get("commands"), "quality_gate.commands")
    _text(quality.get("runtime"), "quality_gate.runtime")
    _require_strings(quality.get("known_skips"), "quality_gate.known_skips")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    audit_release(args.root, manifest=args.manifest)
    print("v2.4 clarification release boundary verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
