"""Audit the bounded v2.9 conversational security and approval gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from tools.audit_adversarial_resistance import audit_corpus
except ModuleNotFoundError:  # pragma: no cover - direct ``python tools/...`` execution
    from audit_adversarial_resistance import audit_corpus


class ConversationalSecurityReleaseAuditError(ValueError):
    """Raised when the v2.9 security gate is incomplete or overclaims."""


_DEPENDENCIES = {
    "v29-scoped-approvals": (481, "708d3a884f721a4d9951566e9c524f30a45686b7"),
    "v29-adversarial-resistance": (482, "0af28a2f5f9a3095575ace7741fe257e3854dc92"),
    "v29-provider-tool-isolation": (483, "f8d29139430a56e65dfbed03e7bd43023294c4a9"),
    "v29-policy-precedence": (484, "5c54f08cdbae4906e2d089c89b907dd54852ccf2"),
}

_REQUIRED_SUITES = {
    "approval": {
        "tests/test_scoped_approvals.py",
        "tests/test_replanning.py",
        "tests/test_objective_amendments.py",
        "tests/test_optimization_orchestrator.py",
    },
    "adversarial": {
        "tests/test_adversarial_resistance.py",
        "tests/test_conversational_adversarial_corpus.py",
        "tests/test_policy_precedence.py",
    },
    "isolation": {
        "tests/test_provider_interface.py",
        "tests/test_conversational_tools.py",
        "tests/test_conversational_dispatcher.py",
        "tests/test_endpoint_adapters.py",
        "tests/test_provider_conformance.py",
    },
    "replay": {
        "tests/test_replay_environment.py",
        "tests/test_campaign_recovery.py",
        "tests/test_v27_campaign_recovery.py",
        "tests/test_reproduce_cli.py",
        "tests/test_search_resume.py",
    },
    "direct_api": {
        "tests/test_public_api_contract.py",
        "tests/test_optimization.py",
        "tests/test_optimization_package.py",
        "tests/test_evidence_query.py",
        "tests/test_claim_evidence.py",
    },
}

_REQUIRED_DOCS = {
    "ROADMAP.md",
    "docs/goal.md",
    "ARCHITECTURE.md",
    "README.md",
    "SECURITY.md",
    "CHANGELOG.md",
    "docs/design/conversational-control-plane.md",
    "docs/design/scoped-approvals.md",
    "docs/design/provider-tool-isolation.md",
    "docs/design/policy-precedence.md",
    "docs/testing/adversarial-resistance.md",
    "docs/release/v2.9-conversational-security-boundary.md",
}


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConversationalSecurityReleaseAuditError(f"{label} must be an object")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConversationalSecurityReleaseAuditError(f"{label} must be non-empty text")
    return value


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConversationalSecurityReleaseAuditError(f"{label} must be an array")
    return value


def _relative_file(root: Path, value: object, label: str) -> None:
    raw = _text(value, label)
    path = Path(raw)
    if path.is_absolute() or ".." in path.parts or not (root / path).is_file():
        raise ConversationalSecurityReleaseAuditError(f"{label} references a missing file")


def _files(root: Path, value: object, label: str) -> set[str]:
    values = _array(value, label)
    if not values:
        raise ConversationalSecurityReleaseAuditError(f"{label} must not be empty")
    result = set()
    for index, item in enumerate(values):
        raw = _text(item, f"{label}[{index}]")
        _relative_file(root, raw, f"{label}[{index}]")
        if raw in result:
            raise ConversationalSecurityReleaseAuditError(f"{label} contains a duplicate")
        result.add(raw)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), "release manifest")
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConversationalSecurityReleaseAuditError("could not read release manifest") from error


def _check_dependencies(root: Path, record: dict[str, Any]) -> None:
    seen: set[str] = set()
    for index, raw in enumerate(_array(record.get("dependency_evidence"), "dependency_evidence")):
        item = _object(raw, f"dependency_evidence[{index}]")
        key = _text(item.get("key"), f"dependency_evidence[{index}].key")
        if key in seen or key not in _DEPENDENCIES:
            raise ConversationalSecurityReleaseAuditError(
                f"unexpected or duplicate dependency {key}"
            )
        seen.add(key)
        issue, commit = _DEPENDENCIES[key]
        if item.get("issue") != issue or item.get("status") != "merged":
            raise ConversationalSecurityReleaseAuditError(f"dependency {key} is not merged")
        if item.get("commit") != commit:
            raise ConversationalSecurityReleaseAuditError(f"dependency {key} identity drifted")
        _files(root, item.get("artifacts"), f"dependency {key}.artifacts")
        _files(root, item.get("tests"), f"dependency {key}.tests")
    if seen != set(_DEPENDENCIES):
        raise ConversationalSecurityReleaseAuditError("v2.9 dependency evidence is incomplete")


def _check_contracts(record: dict[str, Any]) -> None:
    approvals = _object(record.get("approval_contract"), "approval_contract")
    expected_approvals = {
        "scope": "exact operation and capability scope",
        "plan_binding": "canonical plan ID and digest",
        "material_diff_binding": "content-addressed material diff ID and changed paths",
        "expiry": "expired approvals fail closed",
        "reuse": "one_time or reusable with bounded consumption",
        "actor_binding": "operator identity and context",
        "secret_policy": "redacted immutable audit evidence",
    }
    for key, value in expected_approvals.items():
        if approvals.get(key) != value:
            raise ConversationalSecurityReleaseAuditError(f"approval contract drifted: {key}")

    policy = _object(record.get("policy_contract"), "policy_contract")
    if policy.get("trusted_precedence") != [
        "hard_constraints",
        "validated_spec",
        "approval_policy",
        "tool_capability",
        "evidence_status",
    ]:
        raise ConversationalSecurityReleaseAuditError("trusted policy precedence drifted")
    for key in ("unknown", "contradictory", "missing_approval", "isolation_failure"):
        if policy.get(key) != "non_executable_fail_closed":
            raise ConversationalSecurityReleaseAuditError(f"policy fail-closed rule drifted: {key}")
    if policy.get("prompt_or_provider_authority") != "never_authoritative":
        raise ConversationalSecurityReleaseAuditError("untrusted authority policy drifted")

    isolation = _object(record.get("isolation_contract"), "isolation_contract")
    if isolation.get("trusted_zone") != "trusted_engine":
        raise ConversationalSecurityReleaseAuditError("trusted isolation zone drifted")
    if set(isolation.get("untrusted_zones", [])) != {"untrusted_provider", "untrusted_tool"}:
        raise ConversationalSecurityReleaseAuditError("untrusted isolation zones are incomplete")
    if isolation.get("secret_values_in_canonical_records") is not False:
        raise ConversationalSecurityReleaseAuditError("secret retention policy drifted")
    if isolation.get("hostile_process_containment") != "not_claimed":
        raise ConversationalSecurityReleaseAuditError("process-containment limitation was weakened")


def _check_corpus(root: Path, record: dict[str, Any]) -> None:
    corpus = _object(record.get("adversarial_evidence"), "adversarial_evidence")
    corpus_path = root / _text(corpus.get("fixture"), "adversarial_evidence.fixture")
    _relative_file(root, corpus.get("fixture"), "adversarial_evidence.fixture")
    report = audit_corpus(corpus_path)
    if report["failed"] != 0 or report["all_failures_retained"] is not True:
        raise ConversationalSecurityReleaseAuditError(
            "adversarial corpus did not pass retention gate"
        )
    expected = {
        "revision": "v2.9-adversarial-resistance-v1",
        "case_count": 18,
        "variant_count": 69,
        "unresolved_cases_retained": True,
        "hostile_process_containment": "not_claimed",
    }
    for key, value in expected.items():
        if corpus.get(key) != value:
            raise ConversationalSecurityReleaseAuditError(f"adversarial evidence drifted: {key}")


def audit_release(root: Path = Path("."), *, manifest: Path | None = None) -> None:
    """Validate the checked-in v2.9 gate and replay its deterministic corpus."""

    root = root.resolve()
    manifest_path = manifest or (
        root / "docs" / "research" / "v2.9-conversational-security-release-v1.json"
    )
    record = _read_json(manifest_path)
    if record.get("record_type") != "conversational_security_release":
        raise ConversationalSecurityReleaseAuditError("unexpected v2.9 release record type")
    if record.get("schema_version") != 1:
        raise ConversationalSecurityReleaseAuditError("unsupported v2.9 release schema")
    if record.get("protocol_revision") != "modelsurgeon-v2.9-conversational-security-release-v1":
        raise ConversationalSecurityReleaseAuditError("v2.9 release protocol revision drifted")
    if record.get("milestone") != "v2.9" or record.get("status") != "closed_bounded_gate":
        raise ConversationalSecurityReleaseAuditError("v2.9 gate status is not closed")

    _check_dependencies(root, record)
    _check_contracts(record)
    _check_corpus(root, record)

    suites = _object(record.get("test_suites"), "test_suites")
    for suite_name, required in _REQUIRED_SUITES.items():
        declared = _files(root, suites.get(suite_name), f"test_suites.{suite_name}")
        if not required.issubset(declared):
            raise ConversationalSecurityReleaseAuditError(f"{suite_name} suite is incomplete")

    docs = _files(root, record.get("documentation"), "documentation")
    if not _REQUIRED_DOCS.issubset(docs):
        raise ConversationalSecurityReleaseAuditError("v2.9 documentation set is incomplete")

    limits = _object(record.get("residual_risk_and_limits"), "residual_risk_and_limits")
    for key in ("supported", "unsupported", "residual_risks", "known_skips"):
        values = _array(limits.get(key), f"residual_risk_and_limits.{key}")
        if not values or any(not isinstance(item, str) or not item.strip() for item in values):
            raise ConversationalSecurityReleaseAuditError(f"{key} limits must be explicit")
    if limits.get("overclaim_policy") != "do_not_promote_unmeasured_or_uncontained_behavior":
        raise ConversationalSecurityReleaseAuditError("residual-risk overclaim policy drifted")

    clean = _object(record.get("clean_environment_evidence"), "clean_environment_evidence")
    if clean.get("status") != "passed" or clean.get("locked_dependencies") is not True:
        raise ConversationalSecurityReleaseAuditError("clean-environment evidence is incomplete")
    _text(clean.get("runtime"), "clean_environment_evidence.runtime")
    _text(clean.get("base_revision"), "clean_environment_evidence.base_revision")
    _array(clean.get("commands"), "clean_environment_evidence.commands")

    quality = _object(record.get("quality_gate"), "quality_gate")
    if quality.get("status") != "passed":
        raise ConversationalSecurityReleaseAuditError("quality gate is not recorded as passed")
    commands = _array(quality.get("commands"), "quality_gate.commands")
    required_commands = (
        "git diff --check",
        "ruff check src tests",
        "mypy src/modelsurgeon",
        "pytest -q",
        "audit_adversarial_resistance.py",
        "audit_v29_conversational_security_release.py",
    )
    for required in required_commands:
        if not any(required in str(command) for command in commands):
            raise ConversationalSecurityReleaseAuditError(f"quality gate is missing {required}")
    _text(quality.get("revision"), "quality_gate.revision")


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    audit_release(args.root, manifest=args.manifest)
    print("v2.9 conversational security and approval gate verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
