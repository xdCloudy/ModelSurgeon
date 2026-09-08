"""Audit the bounded v3.0 conversational product release manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


class V30ReleaseAuditError(ValueError):
    """Raised when the v3.0 release manifest is incomplete or overclaims."""


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "research" / "v3.0-product-release-v1.json"

REQUIRED_STATES = {"verified", "experimental", "unsupported", "unknown", "deferred_v3_1"}
REQUIRED_DEPENDENCIES = {430, 485, 486, 487, 488, 489, 490}


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise V30ReleaseAuditError(f"{label} must be an object")
    return value


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list) or not value:
        raise V30ReleaseAuditError(f"{label} must be a non-empty array")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise V30ReleaseAuditError(f"{label} must be non-empty text")
    return value


def _path(root: Path, value: object, label: str) -> Path:
    raw = _text(value, label)
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts or not (root / path).is_file():
        raise V30ReleaseAuditError(f"{label} references a missing or unsafe file: {raw}")
    return path


def _files(root: Path, value: object, label: str) -> None:
    paths = _array(value, label)
    seen: set[str] = set()
    for index, item in enumerate(paths):
        path = _path(root, item, f"{label}[{index}]")
        if str(path) in seen:
            raise V30ReleaseAuditError(f"{label} contains a duplicate: {path}")
        seen.add(str(path))


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), label)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise V30ReleaseAuditError(f"could not read {label}: {path}") from error


def _check_evidence(root: Path, record: dict[str, Any]) -> None:
    evidence = _array(record.get("p0_exit_evidence"), "p0_exit_evidence")
    seen: set[str] = set()
    for index, raw in enumerate(evidence):
        item = _object(raw, f"p0_exit_evidence[{index}]")
        key = _text(item.get("key"), f"p0_exit_evidence[{index}].key")
        if key in seen:
            raise V30ReleaseAuditError(f"duplicate P0 evidence key: {key}")
        seen.add(key)
        issue = item.get("issue")
        if not isinstance(issue, int) or issue <= 0:
            raise V30ReleaseAuditError(f"{key} has an invalid issue")
        _text(item.get("status"), f"{key}.status")
        _files(root, item.get("artifacts"), f"{key}.artifacts")
        _files(root, item.get("tests"), f"{key}.tests")
        _files(root, item.get("audits"), f"{key}.audits")
        if item.get("negative_evidence_retained") is not True:
            raise V30ReleaseAuditError(f"{key} must retain negative evidence")
    expected = {
        "v2.0-autonomous-optimizer",
        "v2.1-intent-contract",
        "v2.2-provider-boundary",
        "v2.3-chat-slice",
        "v2.4-clarification",
        "v2.5-negotiation",
        "v2.6-tools",
        "v2.7-stateful-campaigns",
        "v2.8-evidence-grounding",
        "v2.9-security",
        "v3.0-setup-provider-journey",
        "v3.0-migration",
        "v3.0-acceptance",
    }
    if seen != expected:
        raise V30ReleaseAuditError(
            f"P0 evidence inventory drifted: expected {expected}, got {seen}"
        )


def _check_dependencies(record: dict[str, Any]) -> None:
    dependencies = _array(record.get("dependency_issues"), "dependency_issues")
    seen: set[int] = set()
    for index, raw in enumerate(dependencies):
        item = _object(raw, f"dependency_issues[{index}]")
        issue = item.get("issue")
        if not isinstance(issue, int) or issue in seen:
            raise V30ReleaseAuditError("dependency issues must be unique positive integers")
        seen.add(issue)
        if item.get("state") != "closed":
            raise V30ReleaseAuditError(f"dependency #{issue} is not recorded as closed")
        _text(item.get("claim"), f"dependency_issues[{index}].claim")
    if seen != REQUIRED_DEPENDENCIES:
        raise V30ReleaseAuditError(
            f"dependency issue inventory drifted: expected {REQUIRED_DEPENDENCIES}, got {seen}"
        )


def _check_states(record: dict[str, Any]) -> None:
    states = _object(record.get("capability_states"), "capability_states")
    if set(states) != REQUIRED_STATES:
        raise V30ReleaseAuditError(
            f"capability states must be exactly {REQUIRED_STATES}, got {set(states)}"
        )
    for state, values in states.items():
        _array(values, f"capability_states.{state}")
        for index, value in enumerate(values):
            _text(value, f"capability_states.{state}[{index}]")
    verified = "direct CLI/Python automation remains supported"
    if not any(verified.lower() in value.lower() for value in states["verified"]):
        raise V30ReleaseAuditError("verified state must preserve direct CLI/Python automation")
    if not any("meta-surgeon" in value.lower() for value in states["deferred_v3_1"]):
        raise V30ReleaseAuditError("deferred_v3_1 must name the learned Meta-Surgeon work")


def _check_boundaries(record: dict[str, Any]) -> None:
    boundary = _object(record.get("authority_boundary"), "authority_boundary")
    expected_false = (
        "text_llm_is_optimizer_authority",
        "text_llm_is_evidence_authority",
        "optimization_spec_is_free_form",
        "learned_meta_surgeon_is_measurement_authority",
        "chat_history_is_canonical_state",
        "product_release_claims_optimizer_optimality",
    )
    for key in expected_false:
        if boundary.get(key) is not False:
            raise V30ReleaseAuditError(f"authority boundary weakened: {key}")
    for key in ("text_llm", "optimization_spec", "deterministic_engine", "learned_meta_surgeon"):
        _text(boundary.get(key), f"authority_boundary.{key}")


def audit_release(root: Path = ROOT, *, manifest: Path | None = None) -> None:
    """Validate the release manifest and every referenced repository artifact."""

    root = root.resolve()
    record = _read_json(manifest or root / MANIFEST.relative_to(ROOT), "v3.0 product release")
    if record.get("record_type") != "v30_conversational_product_release":
        raise V30ReleaseAuditError("unexpected v3.0 release record type")
    if record.get("schema_version") != 1:
        raise V30ReleaseAuditError("unsupported v3.0 release schema version")
    if record.get("protocol_revision") != "modelsurgeon-v3.0-product-release-v1":
        raise V30ReleaseAuditError("unexpected v3.0 release protocol revision")
    if record.get("issue") != 491 or record.get("milestone") != "v3.0":
        raise V30ReleaseAuditError("release issue or milestone is incorrect")
    if record.get("status") != "bounded_product_release_candidate":
        raise V30ReleaseAuditError("release status must remain bounded_product_release_candidate")
    _check_dependencies(record)
    _check_evidence(root, record)
    _check_states(record)
    _check_boundaries(record)
    _files(root, record.get("documentation"), "documentation")
    _files(root, record.get("release_checklist"), "release_checklist")
    _files(root, record.get("quality_gate_tests"), "quality_gate_tests")
    risks = _array(record.get("residual_risks"), "residual_risks")
    for index, raw in enumerate(risks):
        item = _object(raw, f"residual_risks[{index}]")
        _text(item.get("id"), f"residual_risks[{index}].id")
        _text(item.get("status"), f"residual_risks[{index}].status")
        _text(item.get("mitigation"), f"residual_risks[{index}].mitigation")
    _files(root, record.get("release_candidate_observations"), "release_candidate_observations")
    audit_commands = _array(record.get("audit_commands"), "audit_commands")
    for index, raw in enumerate(audit_commands):
        item = _object(raw, f"audit_commands[{index}]")
        _text(item.get("name"), f"audit_commands[{index}].name")
        command = item.get("command")
        if not isinstance(command, list) or not command or any(
            not isinstance(part, str) for part in command
        ):
            raise V30ReleaseAuditError(
                f"audit_commands[{index}].command must be a non-empty string array"
            )


def _run(root: Path, command: list[str], timeout: int) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "status": "failed", "error": str(error)}
    output = (result.stdout + result.stderr).strip()
    return {
        "command": command,
        "status": "passed" if result.returncode == 0 else "failed",
        "exit_code": result.returncode,
        "output_tail": output[-4000:],
    }


def run_audit(root: Path = ROOT, *, timeout: int = 900) -> dict[str, Any]:
    """Run the manifest's focused audits and return machine-readable results."""

    root = root.resolve()
    audit_release(root)
    record = _read_json(root / MANIFEST.relative_to(ROOT), "v3.0 product release")
    results = [
        _run(root, [str(part) for part in item["command"]], timeout)
        for item in record["audit_commands"]
    ]
    return {
        "record_type": "v30_product_release_audit",
        "schema_version": 1,
        "protocol_revision": "modelsurgeon-v3.0-product-release-audit-v1",
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "status": "passed" if all(item["status"] == "passed" for item in results) else "failed",
        "commands": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--run", action="store_true", help="run the declared focused audits")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.run:
        result = run_audit(args.root, timeout=args.timeout)
        if args.output:
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "passed" else 1
    audit_release(args.root, manifest=args.manifest)
    print("v3.0 conversational product release manifest verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
