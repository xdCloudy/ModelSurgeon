"""Approval decisions, deterministic decision replay, and final run packages.

This module is deliberately independent of model execution.  It records the
plan that a human saw, binds every approval to that exact plan and diff, and
packages the resulting decision evidence with the existing offline-verifiable
evidence bundle format.  Package contents are public evidence only; operator
context is restricted to non-secret scalar metadata.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final

from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.experiments.signed_evidence import (
    EvidenceBundleExternalReference,
    EvidenceBundleIndex,
    EvidenceBundleLineage,
    EvidenceBundleMember,
    EvidenceBundleRedaction,
    EvidenceBundleSignature,
    EvidenceKeyRecord,
    EvidenceKeyStatus,
    EvidenceVerification,
    build_evidence_bundle,
    render_evidence_bundle_index,
    sign_evidence_bundle,
    verify_evidence_bundle,
)

OPTIMIZATION_PACKAGE_SCHEMA_VERSION: Final[int] = 1
APPROVAL_SCHEMA_VERSION: Final[int] = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SECRET_KEY = re.compile(
    r"(?:secret|token|password|passwd|credential|api[_-]?key|private[_-]?key)", re.I
)


class OptimizationPackageError(ValueError):
    """Raised when approval, replay, or package evidence is unsafe."""


class ApprovalDecisionKind(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PackageVerificationStatus(StrEnum):
    VERIFIED = "verified"
    INCOMPLETE = "incomplete"
    UNSIGNED = "unsigned"
    FAILED = "failed"


def _canonical(value: object) -> str:
    try:
        return canonical_identity_json(value)
    except (TypeError, ValueError) as error:
        raise OptimizationPackageError("evidence must be canonical JSON data") from error


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OptimizationPackageError(f"{label} must be non-empty text")
    return value


def _digest(value: object, label: str) -> str:
    result = _text(value, label)
    if _SHA256.fullmatch(result) is None:
        raise OptimizationPackageError(f"{label} must be a lowercase SHA-256")
    return result


def _timestamp(value: object, label: str) -> datetime:
    text = _text(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise OptimizationPackageError(f"{label} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise OptimizationPackageError(f"{label} must include a timezone")
    return parsed.astimezone(UTC)


def _now() -> datetime:
    return datetime.now(UTC)


def _record(value: object, label: str) -> dict[str, object]:
    if hasattr(value, "to_record"):
        value = value.to_record()
    if not isinstance(value, Mapping):
        raise OptimizationPackageError(f"{label} must be an object")
    return {str(key): item for key, item in value.items()}


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key in sorted(value, key=str):
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(value[key], path))
        return result
    if isinstance(value, list):
        result = {}
        for index, item in enumerate(value):
            result.update(_flatten(item, f"{prefix}[{index}]"))
        return result
    return {prefix: value}


def _approval_plan_record(value: object) -> dict[str, object]:
    record = _record(value, "plan")
    # These fields are derived from the content and are not approval scope.
    for key in ("plan_id", "config_digest", "resume_token", "executable"):
        record.pop(key, None)
    lineage = record.get("artifact_lineage")
    if isinstance(lineage, Mapping):
        lineage = dict(lineage)
        lineage.pop("plan_id", None)
        lineage.pop("resolved_config_digest", None)
        record["artifact_lineage"] = lineage
    return record


@dataclass(frozen=True, slots=True)
class PlanDiff:
    """Content-addressed comparison between the plan shown before and after a change."""

    before_plan_id: str
    after_plan_id: str
    before_digest: str
    after_digest: str
    changed_paths: tuple[str, ...]
    material: bool

    def __post_init__(self) -> None:
        _text(self.before_plan_id, "previous plan ID")
        _text(self.after_plan_id, "current plan ID")
        _digest(self.before_digest, "previous plan digest")
        _digest(self.after_digest, "current plan digest")
        if self.changed_paths != tuple(sorted(set(self.changed_paths))):
            raise OptimizationPackageError("plan diff paths must be sorted and unique")
        if self.material != bool(self.changed_paths):
            raise OptimizationPackageError("plan diff materiality does not match changed paths")

    @property
    def diff_id(self) -> str:
        return "plan_diff_" + _sha256(self.to_record(include_id=False))

    def to_record(self, *, include_id: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": APPROVAL_SCHEMA_VERSION,
            "before_plan_id": self.before_plan_id,
            "after_plan_id": self.after_plan_id,
            "before_digest": self.before_digest,
            "after_digest": self.after_digest,
            "changed_paths": list(self.changed_paths),
            "material": self.material,
        }
        if include_id:
            record["diff_id"] = self.diff_id
        return record


def diff_plans(before: object, after: object) -> PlanDiff:
    """Return the exact, deterministic diff used by a new approval decision."""

    before_record = _approval_plan_record(before)
    after_record = _approval_plan_record(after)
    before_id = str(_record(before, "previous plan").get("plan_id", "unknown"))
    after_id = str(_record(after, "current plan").get("plan_id", "unknown"))
    before_digest = _sha256(before_record)
    after_digest = _sha256(after_record)
    before_flat = _flatten(before_record)
    after_flat = _flatten(after_record)
    paths = tuple(
        sorted(
            path
            for path in set(before_flat) | set(after_flat)
            if before_flat.get(path) != after_flat.get(path)
        )
    )
    return PlanDiff(before_id, after_id, before_digest, after_digest, paths, bool(paths))


def plan_digest(plan: object) -> str:
    """Compute the digest approvals bind to, excluding derived identity fields."""

    return _sha256(_approval_plan_record(plan))


def _safe_context(context: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    items: list[tuple[str, str]] = []
    for key, value in context.items():
        name = _text(key, "operator context key")
        if _SECRET_KEY.search(name):
            raise OptimizationPackageError("operator context cannot contain secret fields")
        scalar = _text(value, f"operator context {name}")
        if _SECRET_KEY.search(scalar):
            raise OptimizationPackageError("operator context cannot contain secret markers")
        items.append((name, scalar))
    return tuple(sorted(items))


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    code: str
    plan_id: str
    plan_digest: str
    scope: tuple[str, ...]
    diff_id: str
    requested_at: str
    expires_at: str
    operator_id: str
    operator_context: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _text(self.code, "approval code")
        _text(self.plan_id, "approval plan ID")
        _digest(self.plan_digest, "approval plan digest")
        _text(self.diff_id, "approval diff ID")
        if self.scope != tuple(sorted(set(self.scope))) or any(
            not item.strip() for item in self.scope
        ):
            raise OptimizationPackageError("approval scope must be sorted and unique")
        requested = _timestamp(self.requested_at, "approval requested_at")
        expires = _timestamp(self.expires_at, "approval expires_at")
        if expires <= requested:
            raise OptimizationPackageError("approval expiry must be after its request")
        _text(self.operator_id, "operator identity")
        _safe_context(dict(self.operator_context))

    @property
    def request_id(self) -> str:
        return "approval_request_" + _sha256(self.to_record(include_id=False))

    def is_expired(self, at: datetime | None = None) -> bool:
        return (
            _now() >= _timestamp(self.expires_at, "approval expires_at")
            if at is None
            else at >= _timestamp(self.expires_at, "approval expires_at")
        )

    def to_record(self, *, include_id: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": APPROVAL_SCHEMA_VERSION,
            "code": self.code,
            "plan_id": self.plan_id,
            "plan_digest": self.plan_digest,
            "scope": list(self.scope),
            "diff_id": self.diff_id,
            "requested_at": self.requested_at,
            "expires_at": self.expires_at,
            "operator_id": self.operator_id,
            "operator_context": {key: value for key, value in self.operator_context},
        }
        if include_id:
            record["request_id"] = self.request_id
        return record


@dataclass(frozen=True, slots=True)
class ApprovalDecision:
    request_id: str
    code: str
    plan_id: str
    plan_digest: str
    diff_id: str
    decision: ApprovalDecisionKind
    decided_at: str
    operator_id: str
    reason: str

    def __post_init__(self) -> None:
        _text(self.request_id, "approval request ID")
        _text(self.code, "approval code")
        _text(self.plan_id, "approval plan ID")
        _digest(self.plan_digest, "decision plan digest")
        _text(self.diff_id, "decision diff ID")
        _timestamp(self.decided_at, "approval decided_at")
        _text(self.operator_id, "decision operator identity")
        _text(self.reason, "approval decision reason")

    @property
    def decision_id(self) -> str:
        return "approval_decision_" + _sha256(self.to_record(include_id=False))

    def to_record(self, *, include_id: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": APPROVAL_SCHEMA_VERSION,
            "request_id": self.request_id,
            "code": self.code,
            "plan_id": self.plan_id,
            "plan_digest": self.plan_digest,
            "diff_id": self.diff_id,
            "decision": self.decision.value,
            "decided_at": self.decided_at,
            "operator_id": self.operator_id,
            "reason": self.reason,
        }
        if include_id:
            record["decision_id"] = self.decision_id
        return record


def validate_approval(
    request: ApprovalRequest,
    decision: ApprovalDecision,
    *,
    current_plan_id: str,
    current_plan_digest: str,
    at: datetime | None = None,
) -> None:
    """Fail closed unless a decision covers the exact current plan and diff."""

    if decision.request_id != request.request_id or decision.code != request.code:
        raise OptimizationPackageError("approval decision does not match its request")
    if decision.plan_id != current_plan_id or decision.plan_digest != current_plan_digest:
        raise OptimizationPackageError("approval decision does not cover the current plan")
    if decision.diff_id != request.diff_id:
        raise OptimizationPackageError("approval decision does not cover the current plan diff")
    if request.is_expired(at) or _timestamp(
        decision.decided_at, "approval decided_at"
    ) >= _timestamp(request.expires_at, "approval expires_at"):
        raise OptimizationPackageError("approval is expired")
    if decision.decision is not ApprovalDecisionKind.APPROVED:
        raise OptimizationPackageError(f"approval was {decision.decision.value}")


@dataclass(frozen=True, slots=True)
class DecisionReplay:
    evidence_digest: str
    strategy_decision_id: str
    selected_candidate_id: str | None
    candidate_ids: tuple[str, ...]
    deterministic: bool

    def __post_init__(self) -> None:
        _digest(self.evidence_digest, "decision evidence digest")
        _text(self.strategy_decision_id, "strategy decision ID")
        if self.candidate_ids != tuple(sorted(set(self.candidate_ids))):
            raise OptimizationPackageError("replay candidate IDs must be sorted and unique")
        if (
            self.selected_candidate_id is not None
            and self.selected_candidate_id not in self.candidate_ids
        ):
            raise OptimizationPackageError("replay selection is not in the evidence")

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "decision_replay",
            "schema_version": OPTIMIZATION_PACKAGE_SCHEMA_VERSION,
            "evidence_digest": self.evidence_digest,
            "strategy_decision_id": self.strategy_decision_id,
            "selected_candidate_id": self.selected_candidate_id,
            "candidate_ids": list(self.candidate_ids),
            "deterministic": self.deterministic,
        }


def _candidate_records(evidence: object) -> tuple[dict[str, object], ...]:
    if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes, bytearray)):
        return ()
    records = []
    for item in evidence:
        if isinstance(item, Mapping) and isinstance(item.get("candidate_id"), str):
            records.append({str(key): value for key, value in item.items()})
    return tuple(records)


def replay_decision_evidence(
    evidence: object,
    *,
    selected_candidate_id: str | None = None,
) -> DecisionReplay:
    """Replay a canonical candidate evidence sequence without model execution.

    A candidate with measured, complete, and constraint-passing evidence is
    preferred.  Numeric ``score`` values are minimized, with candidate ID as
    the stable tie-breaker.  If the original selection is supplied, it must be
    present in the evidence; this turns the function into an integrity check
    rather than trusting a copied final-candidate field.
    """

    evidence_digest = _sha256(evidence)
    records = _candidate_records(evidence)
    candidate_ids = tuple(sorted({str(item["candidate_id"]) for item in records}))
    if selected_candidate_id is not None and selected_candidate_id not in candidate_ids:
        raise OptimizationPackageError("selected candidate is absent from decision evidence")
    if selected_candidate_id is None and records:
        feasible = [
            item
            for item in records
            if item.get("measured") is True
            and item.get("complete") is True
            and item.get("constraints_passed") is True
        ]
        if feasible:

            def key(item: Mapping[str, object]) -> tuple[float, str]:
                score = item.get("score", 0.0)
                numeric = (
                    float(score)
                    if isinstance(score, (int, float)) and not isinstance(score, bool)
                    else 0.0
                )
                if not math.isfinite(numeric):
                    raise OptimizationPackageError("candidate scores must be finite")
                return numeric, str(item["candidate_id"])

            selected_candidate_id = min(feasible, key=key)["candidate_id"]  # type: ignore[assignment]
    decision_payload = {
        "schema_version": OPTIMIZATION_PACKAGE_SCHEMA_VERSION,
        "evidence_digest": evidence_digest,
        "candidate_ids": list(candidate_ids),
        "selected_candidate_id": selected_candidate_id,
    }
    return DecisionReplay(
        evidence_digest,
        "strategy_replay_" + _sha256(decision_payload),
        selected_candidate_id,
        candidate_ids,
        True,
    )


def replay_optimize_run(run: object) -> DecisionReplay:
    """Replay the persisted optimize-stage evidence without invoking a runtime."""

    record = _record(run, "optimize run")
    stages = record.get("stages")
    if not isinstance(stages, list):
        raise OptimizationPackageError("optimize run stages must be an array")
    evidence = [
        item["result"]
        for item in stages
        if isinstance(item, Mapping) and isinstance(item.get("result"), Mapping)
    ]
    selected = None
    for item in reversed(evidence):
        if isinstance(item, Mapping) and isinstance(item.get("candidate_id"), str):
            selected = str(item["candidate_id"])
            break
    return replay_decision_evidence(evidence, selected_candidate_id=selected)


@dataclass(frozen=True, slots=True)
class PackageVerification:
    status: PackageVerificationStatus
    package_id: str | None
    issues: tuple[str, ...]
    evidence: EvidenceVerification | None

    @property
    def verified(self) -> bool:
        return self.status is PackageVerificationStatus.VERIFIED

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "reproducibility_package_verification",
            "status": self.status.value,
            "verified": self.verified,
            "package_id": self.package_id,
            "issues": list(self.issues),
            "evidence": None if self.evidence is None else self.evidence.to_record(),
        }


@dataclass(frozen=True, slots=True)
class ReproducibilityPackage:
    package_id: str
    destination: Path
    manifest: Mapping[str, object]
    evidence_index: EvidenceBundleIndex

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "reproducibility_package",
            "schema_version": OPTIMIZATION_PACKAGE_SCHEMA_VERSION,
            "package_id": self.package_id,
            "manifest": dict(self.manifest),
            "evidence_index": self.evidence_index.to_record(),
        }


def _external_record(value: object) -> EvidenceBundleExternalReference:
    if not isinstance(value, Mapping):
        raise OptimizationPackageError("external artifact references must be objects")
    try:
        return EvidenceBundleExternalReference(
            str(value["path"]),
            str(value["digest"]),
            int(value["size_bytes"]),
            str(value["locator"]),
            str(value.get("availability", "external")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise OptimizationPackageError("external artifact reference is malformed") from error


def _write_json(path: Path, value: object) -> None:
    path.write_text(_canonical(value) + "\n", encoding="utf-8", newline="\n")


def write_reproducibility_package(
    destination: str | Path,
    *,
    plan: object,
    run: object,
    decision_evidence: object,
    source_commit: str = "unavailable",
    data_revisions: tuple[str, ...] = ("unavailable",),
    hardware_contexts: tuple[str, ...] = ("unavailable",),
    external_artifacts: tuple[EvidenceBundleExternalReference, ...] = (),
    nondeterministic_tolerances: tuple[Mapping[str, object], ...] = (),
    signing_key: EvidenceKeyRecord | None = None,
    signing_key_material: bytes | None = None,
) -> ReproducibilityPackage:
    """Write an immutable package directory and its offline-verifiable index."""

    target = Path(destination).absolute().resolve(strict=False)
    if target.exists() or target.is_symlink():
        raise OptimizationPackageError(f"refusing to overwrite package: {target}")
    plan_record = _record(plan, "plan")
    run_record = _record(run, "run")
    plan_id = _text(plan_record.get("plan_id"), "package plan ID")
    run_id = _text(run_record.get("run_id"), "package run ID")
    run_plan_id = run_record.get("plan_id")
    if run_plan_id != plan_id:
        raise OptimizationPackageError("package run and plan identities do not match")
    replay = replay_decision_evidence(decision_evidence)
    external = tuple(sorted(external_artifacts, key=lambda item: item.path))
    if len({item.path for item in external}) != len(external):
        raise OptimizationPackageError("external artifact paths must be unique")
    tolerances = tuple(
        sorted(
            (dict(item) for item in nondeterministic_tolerances),
            key=lambda item: str(item.get("name", "")),
        )
    )
    for item in tolerances:
        if not _text(item.get("name"), "non-deterministic tolerance name") or not _text(
            item.get("reason"), "non-deterministic tolerance reason"
        ):
            raise OptimizationPackageError("non-deterministic tolerances require name and reason")
    manifest = {
        "record_type": "reproducibility_package_manifest",
        "schema_version": OPTIMIZATION_PACKAGE_SCHEMA_VERSION,
        "run_id": run_id,
        "plan_id": plan_id,
        "plan_digest": plan_digest(plan),
        "run_digest": _sha256(run_record),
        "decision_evidence_digest": replay.evidence_digest,
        "decision_replay": replay.to_record(),
        "external_artifacts": [item.to_record() for item in external],
        "unavailable_external_artifacts": [
            item.path for item in external if item.availability != "available"
        ],
        "nondeterministic_tolerances": list(tolerances),
        "claim_limit": "bounded_reproducibility" if external or tolerances else "full",
    }
    parent_digests = []
    for key in ("source_artifact_digest", "accepted_artifact_digest"):
        value = run_record.get(key)
        if isinstance(value, str) and value.startswith("sha256:"):
            parent_digests.append(value[7:])
    if not parent_digests:
        parent_digests.append(_sha256({"plan_id": plan_id, "run_id": run_id}))
    config_digest = str(plan_record.get("config_digest", _sha256(plan_record)))
    if config_digest.startswith("sha256:"):
        config_digest = config_digest[7:]
    _digest(config_digest, "package config digest")
    staging_parent = target.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.partial-", dir=staging_parent))
    try:
        contents = staging / "contents"
        contents.mkdir()
        _write_json(contents / "manifest.json", manifest)
        _write_json(contents / "plan.json", plan_record)
        _write_json(contents / "run.json", run_record)
        _write_json(contents / "decision-evidence.json", decision_evidence)
        lineage = EvidenceBundleLineage(
            _text(source_commit, "package source commit"),
            config_digest,
            tuple(sorted(set(parent_digests))),
            tuple(sorted(set(data_revisions))),
            tuple(sorted(set(hardware_contexts))),
        )
        index = build_evidence_bundle(
            contents,
            bundle_kind="modelsurgeon.reproducibility.package",
            lineage=lineage,
            external_references=external,
        )
        if signing_key is not None:
            if signing_key_material is None:
                raise OptimizationPackageError("a signing key requires key material")
            index = sign_evidence_bundle(index, key=signing_key, key_material=signing_key_material)
        elif signing_key_material is not None:
            raise OptimizationPackageError("signing key metadata is required with key material")
        (staging / "index.json").write_text(
            render_evidence_bundle_index(index), encoding="utf-8", newline="\n"
        )
        os.replace(staging, target)
        staging = target
        return ReproducibilityPackage(index.bundle_id, target, manifest, index)
    except Exception:
        if staging.exists() and staging != target:
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _load_index(payload: object) -> EvidenceBundleIndex:
    if not isinstance(payload, Mapping):
        raise OptimizationPackageError("package index must be an object")
    try:
        members = tuple(
            EvidenceBundleMember(
                str(item["path"]), str(item["digest"]), int(item["size_bytes"]), str(item["role"])
            )
            for item in payload["members"]
        )
        lineage_raw = payload["lineage"]
        if not isinstance(lineage_raw, Mapping):
            raise TypeError
        lineage = EvidenceBundleLineage(
            str(lineage_raw["source_commit"]),
            str(lineage_raw["config_digest"]),
            tuple(str(item) for item in lineage_raw["parent_artifact_digests"]),
            tuple(str(item) for item in lineage_raw["data_revisions"]),
            tuple(str(item) for item in lineage_raw["hardware_contexts"]),
        )
        external = tuple(_external_record(item) for item in payload.get("external_references", []))
        redactions = tuple(
            EvidenceBundleRedaction(
                str(item["path"]),
                str(item["reason"]),
                str(item.get("claim_limit", "bounded_reproducibility")),
            )
            for item in payload.get("redactions", [])
        )
        keys = tuple(
            EvidenceKeyRecord(
                str(item["key_id"]),
                str(item["fingerprint"]),
                EvidenceKeyStatus(str(item["status"])),
                str(item.get("algorithm", "hmac-sha256")),
                item.get("predecessor_key_id"),
                item.get("successor_key_id"),
                item.get("revocation_reason"),
            )
            for item in payload.get("key_records", [])
        )
        signature_raw = payload.get("signature")
        signature = (
            None
            if signature_raw is None
            else EvidenceBundleSignature(
                str(signature_raw["key_id"]),
                str(signature_raw["algorithm"]),
                str(signature_raw["signed_payload_digest"]),
                str(signature_raw["signature"]),
            )
        )
        return EvidenceBundleIndex(
            str(payload["bundle_kind"]),
            members,
            lineage,
            str(payload["merkle_root"]),
            redactions,
            external,
            keys,
            signature,
            int(payload.get("schema_version", 1)),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise OptimizationPackageError("package evidence index is malformed") from error


def verify_reproducibility_package(
    root: str | Path,
    *,
    key_material: Mapping[str, bytes] | None = None,
) -> PackageVerification:
    """Verify package manifest, content hashes, Merkle evidence, and signatures offline."""

    target = Path(root).absolute().resolve(strict=False)
    issues: list[str] = []
    if not target.is_dir() or target.is_symlink():
        return PackageVerification(
            PackageVerificationStatus.FAILED, None, ("package root is missing",), None
        )
    try:
        index = _load_index(json.loads((target / "index.json").read_text(encoding="utf-8")))
        manifest = json.loads((target / "contents" / "manifest.json").read_text(encoding="utf-8"))
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("record_type") != "reproducibility_package_manifest"
        ):
            issues.append("package manifest is malformed")
        elif manifest.get("decision_evidence_digest") != _sha256(
            json.loads((target / "contents" / "decision-evidence.json").read_text(encoding="utf-8"))
        ):
            issues.append("decision evidence digest does not reconcile")
        evidence = verify_evidence_bundle(target / "contents", index, key_material=key_material)
        issues.extend(evidence.issues)
        expected_external_issue = "external references are unavailable to this offline verifier"
        expected_unsigned_issue = "detached signature is missing"
        unexpected = [
            issue
            for issue in issues
            if issue not in {expected_external_issue, expected_unsigned_issue}
        ]
        if unexpected:
            status = PackageVerificationStatus.FAILED
        elif index.external_references or evidence.status.value == "incomplete":
            status = PackageVerificationStatus.INCOMPLETE
        elif evidence.status.value == "unsigned":
            status = PackageVerificationStatus.UNSIGNED
        else:
            status = PackageVerificationStatus.VERIFIED
        return PackageVerification(status, index.bundle_id, tuple(issues), evidence)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, OptimizationPackageError) as error:
        return PackageVerification(PackageVerificationStatus.FAILED, None, (str(error),), None)


__all__ = [
    "APPROVAL_SCHEMA_VERSION",
    "OPTIMIZATION_PACKAGE_SCHEMA_VERSION",
    "ApprovalDecision",
    "ApprovalDecisionKind",
    "ApprovalRequest",
    "DecisionReplay",
    "OptimizationPackageError",
    "PackageVerification",
    "PackageVerificationStatus",
    "PlanDiff",
    "ReproducibilityPackage",
    "diff_plans",
    "plan_digest",
    "replay_decision_evidence",
    "replay_optimize_run",
    "validate_approval",
    "verify_reproducibility_package",
    "write_reproducibility_package",
]
