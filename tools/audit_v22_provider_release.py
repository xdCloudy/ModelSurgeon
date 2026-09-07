"""Audit the bounded v2.2 replaceable-provider release boundary."""

from __future__ import annotations

import ast
import json
import re
from itertools import product
from pathlib import Path
from typing import Any


class ProviderReleaseAuditError(ValueError):
    """Raised when the v2.2 provider release record is incomplete or unsafe."""


_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")
_DEPENDENCY_KEYS = {
    "v22-provider-contract",
    "v22-local-provider",
    "v22-remote-provider",
    "v22-direct-api-config",
    "v22-conformance",
}
_DEPENDENCY_ISSUES = {
    "v22-provider-contract": 442,
    "v22-local-provider": 443,
    "v22-remote-provider": 444,
    "v22-direct-api-config": 445,
    "v22-conformance": 446,
}
_PROVIDER_KINDS = {"local", "compatible_endpoint", "hosted", "none"}
_CAPABILITIES = {
    "structured_output",
    "refusal",
    "timeout",
    "cancellation",
    "token_budget",
    "resource_budget",
    "provenance",
}
_CORE_MODULES = (
    "src/modelsurgeon/config.py",
    "src/modelsurgeon/config_io.py",
    "src/modelsurgeon/optimization.py",
    "src/modelsurgeon/optimization_orchestrator.py",
    "src/modelsurgeon/provider_kind.py",
)
_FORBIDDEN_CORE_IMPORTS = ("modelsurgeon.conversation", "modelsurgeon.providers")


def _require_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProviderReleaseAuditError(f"{label} must be an object")
    return value


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProviderReleaseAuditError(f"{label} must be non-empty text")
    return value


def _require_list(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ProviderReleaseAuditError(f"{label} must be an array")
    return value


def _check_paths(root: Path, values: object, label: str) -> None:
    for raw_path in _require_list(values, label):
        path = root / _require_string(raw_path, f"{label} entry")
        if not path.is_file():
            raise ProviderReleaseAuditError(f"{label} references missing file {raw_path!r}")


def _check_core_import_isolation(root: Path) -> None:
    for relative_path in _CORE_MODULES:
        path = root / relative_path
        if not path.is_file():
            raise ProviderReleaseAuditError(f"core module is missing: {relative_path}")
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as error:
            raise ProviderReleaseAuditError(
                f"could not parse core module {relative_path}"
            ) from error
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported.add(node.module)
        for module in _FORBIDDEN_CORE_IMPORTS:
            if any(name == module or name.startswith(f"{module}.") for name in imported):
                raise ProviderReleaseAuditError(
                    f"{relative_path} imports provider-specific module {module}"
                )


def audit_release(root: Path, *, manifest: Path | None = None) -> None:
    """Validate the checked-in v2.2 release record and its repository evidence."""

    manifest_path = manifest or root / "docs" / "research" / "v2.2-provider-layer-release-v1.json"
    try:
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProviderReleaseAuditError(f"could not read release record {manifest_path}") from error

    if record.get("record_type") != "provider_layer_release_boundary":
        raise ProviderReleaseAuditError("unexpected v2.2 provider release record type")
    if record.get("schema_version") != 1:
        raise ProviderReleaseAuditError("unsupported v2.2 provider release schema version")
    if record.get("protocol_revision") != "modelsurgeon-v2.2-provider-layer-release-v1":
        raise ProviderReleaseAuditError("unexpected v2.2 provider release protocol revision")
    if record.get("status") != "bounded_release_boundary":
        raise ProviderReleaseAuditError("release status must remain bounded_release_boundary")

    dependencies = _require_list(record.get("dependency_evidence"), "dependency_evidence")
    seen_keys: set[str] = set()
    for index, raw_dependency in enumerate(dependencies):
        dependency = _require_mapping(raw_dependency, f"dependency_evidence[{index}]")
        key = _require_string(dependency.get("key"), f"dependency_evidence[{index}].key")
        seen_keys.add(key)
        if dependency.get("status") != "merged":
            raise ProviderReleaseAuditError(f"dependency {key} is not marked merged")
        commit = _require_string(dependency.get("commit"), f"dependency {key} commit")
        if _COMMIT.fullmatch(commit) is None:
            raise ProviderReleaseAuditError(f"dependency {key} has an invalid commit")
        if dependency.get("issue") != _DEPENDENCY_ISSUES.get(key):
            raise ProviderReleaseAuditError(f"dependency {key} has an unexpected issue number")
        _check_paths(root, dependency.get("artifacts"), f"dependency {key} artifacts")
        _check_paths(root, dependency.get("tests"), f"dependency {key} tests")
    if seen_keys != _DEPENDENCY_KEYS:
        raise ProviderReleaseAuditError(
            f"dependency evidence keys must be exactly {_DEPENDENCY_KEYS}, got {seen_keys}"
        )

    matrix = _require_mapping(record.get("conformance_matrix"), "conformance_matrix")
    kinds = _require_list(matrix.get("provider_kinds"), "conformance_matrix.provider_kinds")
    capabilities = _require_list(matrix.get("capabilities"), "conformance_matrix.capabilities")
    if set(kinds) != _PROVIDER_KINDS or len(kinds) != len(_PROVIDER_KINDS):
        raise ProviderReleaseAuditError("provider conformance kinds are incomplete or duplicated")
    if set(capabilities) != _CAPABILITIES or len(capabilities) != len(_CAPABILITIES):
        raise ProviderReleaseAuditError(
            "provider conformance capabilities are incomplete or duplicated"
        )

    cells = _require_list(matrix.get("cells"), "conformance_matrix.cells")
    expected = set(product(_PROVIDER_KINDS, _CAPABILITIES))
    actual: set[tuple[object, object]] = set()
    unsupported: set[tuple[object, object]] = set()
    for index, raw_cell in enumerate(cells):
        cell = _require_mapping(raw_cell, f"conformance_matrix.cells[{index}]")
        kind = cell.get("kind")
        capability = cell.get("capability")
        status = cell.get("status")
        rationale = _require_string(cell.get("rationale"), f"conformance cell {index} rationale")
        key = (kind, capability)
        if key in actual:
            raise ProviderReleaseAuditError(f"duplicate conformance cell {key}")
        actual.add(key)
        if status not in {"supported", "unsupported"}:
            raise ProviderReleaseAuditError(f"conformance cell {key} has invalid status")
        if status == "unsupported":
            if "unsupported" not in rationale.lower():
                raise ProviderReleaseAuditError(
                    f"unsupported conformance cell {key} needs a rationale"
                )
            unsupported.add(key)
    if actual != expected:
        raise ProviderReleaseAuditError("conformance matrix does not enumerate every provider cell")
    expected_unsupported = {
        ("none", capability)
        for capability in ("structured_output", "timeout", "token_budget", "resource_budget")
    }
    if unsupported != expected_unsupported:
        raise ProviderReleaseAuditError(
            f"no-LLM unsupported cells changed unexpectedly: {unsupported}"
        )

    direct_api = _require_mapping(record.get("direct_api_no_llm"), "direct_api_no_llm")
    if direct_api.get("default_provider_kind") != "none":
        raise ProviderReleaseAuditError("direct API default provider must be none")
    _check_paths(root, direct_api.get("evidence_tests"), "direct_api_no_llm.evidence_tests")

    isolation = _require_mapping(record.get("core_execution_isolation"), "core_execution_isolation")
    if tuple(isolation.get("modules", ())) != _CORE_MODULES:
        raise ProviderReleaseAuditError("core execution isolation module list drifted")
    _check_core_import_isolation(root)

    evidence_policy = _require_mapping(record.get("evidence_policy"), "evidence_policy")
    if evidence_policy.get("live_provider_benchmarks") != "not_run":
        raise ProviderReleaseAuditError("release record must not claim live provider benchmarks")
    _require_string(evidence_policy.get("negative_and_inconclusive"), "negative_and_inconclusive")
    _check_paths(root, record.get("documentation"), "documentation")
    quality_gate = _require_mapping(record.get("quality_gate"), "quality_gate")
    if not _require_list(quality_gate.get("commands"), "quality_gate.commands"):
        raise ProviderReleaseAuditError("quality gate must declare at least one command")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    audit_release(args.root, manifest=args.manifest)
    print("v2.2 provider release boundary verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
