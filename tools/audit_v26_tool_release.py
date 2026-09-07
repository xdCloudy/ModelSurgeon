"""Audit the bounded v2.6 conversational tool-calling release boundary."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from modelsurgeon.conversation import (
    DEFAULT_TOOL_CATALOG,
    TOOL_RESULT_SCHEMA_VERSION,
    TOOL_SCHEMA_VERSION,
    ToolAccess,
    ToolOutcome,
)


class ToolReleaseAuditError(ValueError):
    """Raised when the v2.6 tool release record is incomplete or unsafe."""


_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DEPENDENCIES = {
    "v26-tool-schema": (463, "dee384ba2fd785c75dfa4022e890caf5d5e27d8e"),
    "v26-allowlist": (464, "cf6940494ac9a10ff6c7a5309da37ae17f3097ae"),
    "v26-transaction-boundary": (465, "fc35a4c8b8484fc8fe2b2bba02a4383be40feb09"),
    "v26-grounding": (466, "75208c2f4dc9366c7b66606dc642c1b0ac248504"),
    "v26-adversarial": (467, "f1d99d1435fb05352d3db3cab134c49b608219e8"),
}
_EXPECTED_TOOLS = {
    "inspect_model": ToolAccess.READ_ONLY,
    "preview_plan": ToolAccess.READ_ONLY,
    "query_evidence": ToolAccess.READ_ONLY,
    "execute_approved_plan": ToolAccess.CONSEQUENTIAL,
}
_FORBIDDEN_AUTHORITY = (
    "shell",
    "python",
    "filesystem",
    "network",
    "command",
    "callback",
    "executor",
    "tensor",
    "remove",
    "delete",
)
_UNSUPPORTED_CELLS = {
    "arbitrary_shell_tool",
    "arbitrary_python_tool",
    "filesystem_tool",
    "network_tool",
    "provider_selector",
    "model_session_tool",
    "direct_tensor_removal",
    "hostile_process_containment",
}
_CORE_MODULES = (
    "src/modelsurgeon/config.py",
    "src/modelsurgeon/config_io.py",
    "src/modelsurgeon/optimization.py",
    "src/modelsurgeon/optimization_orchestrator.py",
    "src/modelsurgeon/provider_kind.py",
)


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ToolReleaseAuditError(f"{label} must be an object")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ToolReleaseAuditError(f"{label} must be non-empty text")
    return value


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ToolReleaseAuditError(f"{label} must be an array")
    return value


def _relative_file(root: Path, value: object, label: str) -> None:
    raw = _text(value, label)
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise ToolReleaseAuditError(f"{label} must be a repository-relative path")
    if not (root / path).is_file():
        raise ToolReleaseAuditError(f"{label} references missing file {raw!r}")


def _files(root: Path, value: object, label: str) -> None:
    values = _array(value, label)
    if not values:
        raise ToolReleaseAuditError(f"{label} must not be empty")
    for index, item in enumerate(values):
        _relative_file(root, item, f"{label}[{index}]")


def _require_strings(value: object, label: str) -> list[str]:
    values = _array(value, label)
    if not values:
        raise ToolReleaseAuditError(f"{label} must not be empty")
    result = []
    for index, item in enumerate(values):
        result.append(_text(item, f"{label}[{index}]"))
    return result


def _check_core_import_isolation(root: Path) -> None:
    for relative_path in _CORE_MODULES:
        path = root / relative_path
        if not path.is_file():
            raise ToolReleaseAuditError(f"core module is missing: {relative_path}")
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as error:
            raise ToolReleaseAuditError(f"could not parse core module {relative_path}") from error
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)
        if any(
            name == "modelsurgeon.conversation"
            or name.startswith("modelsurgeon.conversation.")
            for name in imported
        ):
            raise ToolReleaseAuditError(f"{relative_path} imports the conversational boundary")


def _check_tool_surface() -> None:
    definitions = DEFAULT_TOOL_CATALOG.definitions
    names = {definition.name.value for definition in definitions}
    if names != set(_EXPECTED_TOOLS):
        raise ToolReleaseAuditError(f"default tool surface drifted: {names}")
    for definition in definitions:
        expected_access = _EXPECTED_TOOLS[definition.name.value]
        if definition.access is not expected_access:
            raise ToolReleaseAuditError(f"tool {definition.name.value} has the wrong access class")
        if definition.capability.value != definition.name.value:
            raise ToolReleaseAuditError(f"tool {definition.name.value} is not capability-scoped")
        if definition.budget.max_wall_seconds <= 0 or definition.budget.max_memory_bytes <= 0:
            raise ToolReleaseAuditError(f"tool {definition.name.value} has no positive budget")
        if definition.budget.max_evaluation_count <= 0 or definition.budget.max_output_bytes <= 0:
            raise ToolReleaseAuditError(f"tool {definition.name.value} has no complete budget")
        if definition.access is ToolAccess.CONSEQUENTIAL and not definition.approval_required:
            raise ToolReleaseAuditError(f"tool {definition.name.value} bypasses approval")
        encoded = json.dumps(definition.to_record(), sort_keys=True).lower()
        if any(term in encoded for term in _FORBIDDEN_AUTHORITY):
            raise ToolReleaseAuditError(f"tool {definition.name.value} exposes forbidden authority")
    if TOOL_SCHEMA_VERSION != 1 or TOOL_RESULT_SCHEMA_VERSION != 2:
        raise ToolReleaseAuditError("tool schema versions drifted")
    expected_outcomes = {
        ToolOutcome.SUPPORTED,
        ToolOutcome.UNSUPPORTED,
        ToolOutcome.FAILED,
        ToolOutcome.UNKNOWN,
        ToolOutcome.REFUSED,
        ToolOutcome.TIMEOUT,
        ToolOutcome.CANCELLED,
    }
    if not expected_outcomes.issubset(set(ToolOutcome)):
        raise ToolReleaseAuditError("explicit tool outcomes are incomplete")


def audit_release(root: Path = Path("."), *, manifest: Path | None = None) -> None:
    """Validate the checked-in v2.6 release record and repository evidence."""

    root = root.resolve()
    manifest_path = manifest or (
        root / "docs" / "research" / "v2.6-bounded-conversational-tool-release-v1.json"
    )
    try:
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ToolReleaseAuditError(f"could not read release record {manifest_path}") from error

    if record.get("record_type") != "bounded_conversational_tool_release":
        raise ToolReleaseAuditError("unexpected v2.6 tool release record type")
    if record.get("schema_version") != 1:
        raise ToolReleaseAuditError("unsupported v2.6 tool release schema version")
    if record.get("protocol_revision") != "modelsurgeon-v2.6-bounded-tool-release-v1":
        raise ToolReleaseAuditError("unexpected v2.6 tool release protocol revision")
    if record.get("status") != "bounded_release_boundary":
        raise ToolReleaseAuditError("release status must remain bounded_release_boundary")

    dependencies = _array(record.get("dependency_evidence"), "dependency_evidence")
    seen: set[str] = set()
    for index, raw in enumerate(dependencies):
        dependency = _object(raw, f"dependency_evidence[{index}]")
        key = _text(dependency.get("key"), f"dependency_evidence[{index}].key")
        if key in seen or key not in _DEPENDENCIES:
            raise ToolReleaseAuditError(f"unexpected or duplicate dependency {key}")
        seen.add(key)
        if dependency.get("status") != "merged":
            raise ToolReleaseAuditError(f"dependency {key} is not marked merged")
        commit = _text(dependency.get("commit"), f"dependency {key}.commit")
        if _COMMIT.fullmatch(commit) is None or commit != _DEPENDENCIES[key][1]:
            raise ToolReleaseAuditError(f"dependency {key} has an unexpected commit")
        if dependency.get("issue") != _DEPENDENCIES[key][0]:
            raise ToolReleaseAuditError(f"dependency {key} has an unexpected issue number")
        _files(root, dependency.get("artifacts"), f"dependency {key}.artifacts")
        _files(root, dependency.get("tests"), f"dependency {key}.tests")
    if seen != set(_DEPENDENCIES):
        raise ToolReleaseAuditError("dependency evidence is incomplete")

    contract = _object(record.get("tool_contract"), "tool_contract")
    if contract.get("request_schema_version") != TOOL_SCHEMA_VERSION:
        raise ToolReleaseAuditError("request schema version is not the implementation version")
    if contract.get("result_schema_version") != TOOL_RESULT_SCHEMA_VERSION:
        raise ToolReleaseAuditError("result schema version is not the implementation version")
    if set(_require_strings(contract.get("outcomes"), "tool_contract.outcomes")) != {
        item.value for item in ToolOutcome
    }:
        raise ToolReleaseAuditError("tool outcome vocabulary is incomplete")
    forbidden_authority = _require_strings(
        contract.get("forbidden_authority"), "tool_contract.forbidden_authority"
    )
    if set(forbidden_authority) != set(_FORBIDDEN_AUTHORITY):
        raise ToolReleaseAuditError("forbidden authority policy drifted")
    _check_tool_surface()
    declared_tools = _object(contract.get("tools"), "tool_contract.tools")
    if set(declared_tools) != set(_EXPECTED_TOOLS):
        raise ToolReleaseAuditError("release record tool list is incomplete")
    for name, access in _EXPECTED_TOOLS.items():
        if declared_tools.get(name) != access.value:
            raise ToolReleaseAuditError(f"release record has the wrong access for {name}")

    transaction = _object(record.get("transaction_boundary"), "transaction_boundary")
    if transaction.get("consequential_requires_approval") is not True:
        raise ToolReleaseAuditError("consequential approval requirement is not recorded")
    if transaction.get("read_only_has_participant") is not False:
        raise ToolReleaseAuditError("read-only transaction isolation is not recorded")
    _require_strings(transaction.get("failure_outcomes"), "transaction_boundary.failure_outcomes")

    adversarial = _object(record.get("adversarial_evidence"), "adversarial_evidence")
    _files(root, adversarial.get("corpus"), "adversarial_evidence.corpus")
    _files(root, adversarial.get("tests"), "adversarial_evidence.tests")
    if adversarial.get("negative_results_retained") is not True:
        raise ToolReleaseAuditError("negative adversarial results must be retained")
    if adversarial.get("process_isolation") != "not_claimed":
        raise ToolReleaseAuditError("process isolation limitation must remain explicit")

    evidence = _object(record.get("evidence_policy"), "evidence_policy")
    for field in ("live_provider_benchmarks", "live_campaigns", "live_model_quality"):
        if evidence.get(field) != "not_run":
            raise ToolReleaseAuditError(f"release record must not claim {field}")
    _text(evidence.get("negative_and_inconclusive"), "evidence_policy.negative_and_inconclusive")
    cells = _array(evidence.get("unsupported_cells"), "evidence_policy.unsupported_cells")
    seen_cells: set[str] = set()
    for index, raw_cell in enumerate(cells):
        cell = _object(raw_cell, f"evidence_policy.unsupported_cells[{index}]")
        name = _text(cell.get("cell"), f"unsupported cell {index}.cell")
        if name in seen_cells or name not in _UNSUPPORTED_CELLS:
            raise ToolReleaseAuditError(f"unexpected or duplicate unsupported cell {name}")
        seen_cells.add(name)
        if cell.get("status") not in {"unsupported", "not_claimed"}:
            raise ToolReleaseAuditError(f"unsupported cell {name} has an invalid status")
        _text(cell.get("reason"), f"unsupported cell {name}.reason")
    if seen_cells != _UNSUPPORTED_CELLS:
        raise ToolReleaseAuditError("unsupported-cell evidence is incomplete")

    direct_api = _object(record.get("direct_api_compatibility"), "direct_api_compatibility")
    if direct_api.get("preserved") is not True:
        raise ToolReleaseAuditError("direct API preservation is not recorded")
    _files(root, direct_api.get("tests"), "direct_api_compatibility.tests")
    _check_core_import_isolation(root)

    _files(root, record.get("schema_and_api"), "schema_and_api")
    _files(root, record.get("documentation"), "documentation")
    _require_strings(record.get("reproduction_commands"), "reproduction_commands")
    quality = _object(record.get("quality_gate"), "quality_gate")
    _require_strings(quality.get("commands"), "quality_gate.commands")
    _text(quality.get("runtime"), "quality_gate.runtime")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    audit_release(args.root, manifest=args.manifest)
    print("v2.6 bounded conversational tool release boundary verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
