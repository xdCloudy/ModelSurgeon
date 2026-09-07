"""Audit the bounded v2.0 autonomous optimizer release contract."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


class V20ReleaseAuditError(ValueError):
    """Raised when the v2.0 release record is incomplete or unsafe."""


_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DEPENDENCIES = {
    "v20-reference-benchmark": (429, "a50f26f05b7ced018cb0ec166c54b204442a70b7"),
    "v19-user-workflows": (421, "a767d7cc56c261a16e05cd576d5c31f54e0ac8fb"),
    "v20-objective-contract": (422, "f6775eff7ed9bc69f5b4e5ab830c7ffc4880888b"),
}
_CAPABILITY_STATES = {"verified", "experimental", "unsupported", "unknown"}
_REQUIRED_OUTCOMES = {"supported", "unsupported", "failed", "unknown", "negative"}


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise V20ReleaseAuditError(f"{label} must be an object")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise V20ReleaseAuditError(f"{label} must be non-empty text")
    return value


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise V20ReleaseAuditError(f"{label} must be an array")
    return value


def _relative_file(root: Path, value: object, label: str) -> None:
    raw = _text(value, label)
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts:
        raise V20ReleaseAuditError(f"{label} must be a repository-relative path")
    if not (root / path).is_file():
        raise V20ReleaseAuditError(f"{label} references missing file {raw!r}")


def _files(root: Path, value: object, label: str) -> None:
    for index, item in enumerate(_array(value, label)):
        _relative_file(root, item, f"{label}[{index}]")


def _require_strings(value: object, label: str) -> None:
    values = _array(value, label)
    if not values:
        raise V20ReleaseAuditError(f"{label} must not be empty")
    for index, item in enumerate(values):
        _text(item, f"{label}[{index}]")


def audit_release(
    root: Path = Path("."), *, manifest: Path | None = None
) -> None:
    """Validate the checked-in v2.0 release record and its evidence paths."""

    root = root.resolve()
    manifest_path = manifest or (
        root / "docs" / "research" / "v2.0-autonomous-optimizer-release-v1.json"
    )
    try:
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise V20ReleaseAuditError(f"could not read release record {manifest_path}") from error

    if record.get("record_type") != "v2_autonomous_optimizer_release":
        raise V20ReleaseAuditError("unexpected v2.0 release record type")
    if record.get("schema_version") != 1:
        raise V20ReleaseAuditError("unsupported v2.0 release schema version")
    if record.get("protocol_revision") != "modelsurgeon-v2.0-autonomous-optimizer-release-v1":
        raise V20ReleaseAuditError("unexpected v2.0 release protocol revision")
    if record.get("status") != "bounded_release_boundary":
        raise V20ReleaseAuditError("release status must remain bounded_release_boundary")

    dependencies = _array(record.get("dependency_evidence"), "dependency_evidence")
    seen_dependencies: set[str] = set()
    for index, raw in enumerate(dependencies):
        dependency = _object(raw, f"dependency_evidence[{index}]")
        key = _text(dependency.get("key"), f"dependency_evidence[{index}].key")
        if key in seen_dependencies:
            raise V20ReleaseAuditError(f"duplicate dependency {key}")
        seen_dependencies.add(key)
        expected = _DEPENDENCIES.get(key)
        if expected is None:
            raise V20ReleaseAuditError(f"unexpected dependency {key}")
        if dependency.get("status") != "merged":
            raise V20ReleaseAuditError(f"dependency {key} is not marked merged")
        commit = _text(dependency.get("commit"), f"dependency {key} commit")
        if _COMMIT.fullmatch(commit) is None or commit != expected[1]:
            raise V20ReleaseAuditError(f"dependency {key} has an unexpected commit")
        if dependency.get("issue") != expected[0]:
            raise V20ReleaseAuditError(f"dependency {key} has an unexpected issue number")
        _files(root, dependency.get("artifacts"), f"dependency {key} artifacts")
        _files(root, dependency.get("tests"), f"dependency {key} tests")
    if seen_dependencies != set(_DEPENDENCIES):
        raise V20ReleaseAuditError("dependency evidence is incomplete")

    capabilities = _array(record.get("capabilities"), "capabilities")
    seen_capabilities: set[str] = set()
    states: set[str] = set()
    for index, raw in enumerate(capabilities):
        capability = _object(raw, f"capabilities[{index}]")
        capability_id = _text(capability.get("id"), f"capabilities[{index}].id")
        if capability_id in seen_capabilities:
            raise V20ReleaseAuditError(f"duplicate capability {capability_id}")
        seen_capabilities.add(capability_id)
        state = _text(capability.get("state"), f"capability {capability_id} state")
        if state not in _CAPABILITY_STATES:
            raise V20ReleaseAuditError(f"capability {capability_id} has invalid state {state!r}")
        states.add(state)
        for field in ("surface", "format", "runtime", "evidence", "limitation"):
            _text(capability.get(field), f"capability {capability_id}.{field}")
        _require_strings(capability.get("outcomes"), f"capability {capability_id}.outcomes")
        if not set(capability["outcomes"]) <= _REQUIRED_OUTCOMES:
            raise V20ReleaseAuditError(f"capability {capability_id} has an invalid outcome")
    if states != _CAPABILITY_STATES:
        raise V20ReleaseAuditError(
            "capabilities must include verified, experimental, unsupported, and unknown states"
        )

    evidence = _object(record.get("evidence_policy"), "evidence_policy")
    if evidence.get("live_benchmark") != "not_run":
        raise V20ReleaseAuditError("release record must not claim a live benchmark")
    if evidence.get("signed_packages") != "not_built":
        raise V20ReleaseAuditError("release record must not claim signed packages")
    if evidence.get("published_reference_artifacts") is not False:
        raise V20ReleaseAuditError("reference artifacts must be explicitly unpublished")
    if evidence.get("untested_claims") != "not_claimed":
        raise V20ReleaseAuditError("untested claims must remain unclaimed")

    research = _object(record.get("research_protocol"), "research_protocol")
    for field in (
        "model_families_and_revisions",
        "datasets_tasks_splits_and_licenses",
        "baselines_and_ablations",
        "hardware_and_budgets",
        "metrics_and_thresholds",
        "seeds_and_repetitions",
        "negative_and_unsupported_cells",
        "provenance_requirements",
    ):
        _require_strings(research.get(field), f"research_protocol.{field}")
    _files(root, research.get("report"), "research_protocol.report")

    compatibility = _object(record.get("compatibility"), "compatibility")
    _files(root, compatibility.get("matrix"), "compatibility.matrix")
    _files(root, compatibility.get("migration_guidance"), "compatibility.migration_guidance")
    _text(compatibility.get("policy"), "compatibility.policy")

    audit = _object(record.get("independent_audit_replay"), "independent_audit_replay")
    if audit.get("audit_status") != "passed" or audit.get("replay_status") != "passed":
        raise V20ReleaseAuditError("independent audit and replay must pass")
    if audit.get("critical_findings") != []:
        raise V20ReleaseAuditError("critical audit findings must be empty")
    _files(root, audit.get("tests"), "independent_audit_replay.tests")

    quality = _object(record.get("quality_gate"), "quality_gate")
    if quality.get("status") != "passed":
        raise V20ReleaseAuditError("quality gate must be recorded as passed")
    _require_strings(quality.get("commands"), "quality_gate.commands")
    results = _array(quality.get("results"), "quality_gate.results")
    if len(results) != len(quality["commands"]):
        raise V20ReleaseAuditError("quality gate results must match declared commands")
    for index, raw in enumerate(results):
        result = _object(raw, f"quality_gate.results[{index}]")
        if result.get("status") != "passed":
            raise V20ReleaseAuditError(f"quality gate command {index} did not pass")
        _text(result.get("environment"), f"quality_gate.results[{index}].environment")
        _text(result.get("revision"), f"quality_gate.results[{index}].revision")
        _text(result.get("outcome"), f"quality_gate.results[{index}].outcome")
    _require_strings(quality.get("known_skips"), "quality_gate.known_skips")

    _files(root, record.get("documentation"), "documentation")
    _files(root, record.get("schema_and_api"), "schema_and_api")
    _require_strings(record.get("reproduction_commands"), "reproduction_commands")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    audit_release(args.root, manifest=args.manifest)
    print("v2.0 autonomous optimizer release boundary verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
