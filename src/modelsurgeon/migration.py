"""Explicit v2.0-to-current compatibility and migration contracts.

Migration is a data-boundary operation, not an execution shortcut.  The
supported adapters validate the complete source identity, preserve canonical
payloads and terminal outcomes, and return a new record only after the target
schema can validate it.  Unknown, future, or semantically ambiguous records
are rejected before a caller can resume a campaign or publish an artifact.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from modelsurgeon.config import Settings
from modelsurgeon.conversation.campaign_state import (
    CAMPAIGN_STATE_SCHEMA_VERSION,
    CampaignEvidence,
    CampaignOutcome,
    CampaignStateError,
)
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.optimization_orchestrator import (
    ORCHESTRATOR_SCHEMA_VERSION,
    OptimizeOrchestratorError,
    _run_from_record,
)

V20_CONFIG_SCHEMA_VERSION = 1
V20_CAMPAIGN_SCHEMA_VERSION = 2
V20_EVIDENCE_SCHEMA_VERSION = 1
MIGRATION_CONTRACT_SCHEMA_VERSION = 1


class MigrationRefusal(ValueError):
    """Raised when a record cannot be migrated without guessing."""


class MigrationKind(StrEnum):
    """Supported record families at the migration boundary."""

    CONFIG = "config"
    CAMPAIGN = "campaign"
    EVIDENCE = "evidence"


@dataclass(frozen=True, slots=True)
class MigrationResult:
    """Canonical migration output plus an auditable change summary."""

    kind: MigrationKind
    source_record_type: str
    target_record_type: str
    source_schema_version: int
    target_schema_version: int
    record: Mapping[str, object]
    changed: bool
    changes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.source_schema_version < 0 or self.target_schema_version <= 0:
            raise MigrationRefusal("migration schema versions are invalid")
        if self.changes != tuple(sorted(set(self.changes))):
            raise MigrationRefusal("migration changes must be sorted and unique")
        if self.warnings != tuple(sorted(set(self.warnings))):
            raise MigrationRefusal("migration warnings must be sorted and unique")
        if not self.changed and self.changes:
            raise MigrationRefusal("unchanged records cannot report migration changes")
        if not isinstance(self.record, Mapping) or not all(
            isinstance(key, str) for key in self.record
        ):
            raise MigrationRefusal("migration output must be a JSON object")

    def to_record(self) -> dict[str, object]:
        """Return the migrated payload without adding non-schema metadata."""

        return cast(dict[str, object], json.loads(canonical_identity_json(self.record)))

    def report(self) -> dict[str, object]:
        """Return machine-readable audit metadata for a migration operation."""

        return {
            "record_type": "modelsurgeon_migration_report",
            "schema_version": MIGRATION_CONTRACT_SCHEMA_VERSION,
            "kind": self.kind.value,
            "source_record_type": self.source_record_type,
            "target_record_type": self.target_record_type,
            "source_schema_version": self.source_schema_version,
            "target_schema_version": self.target_schema_version,
            "changed": self.changed,
            "changes": list(self.changes),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class CompatibilityCell:
    """One declared source/target capability in the migration matrix."""

    kind: MigrationKind
    source: str
    target: str
    status: str
    guarantee: str

    def to_record(self) -> dict[str, str]:
        return {
            "kind": self.kind.value,
            "source": self.source,
            "target": self.target,
            "status": self.status,
            "guarantee": self.guarantee,
        }


MIGRATION_CAPABILITY_MATRIX: tuple[CompatibilityCell, ...] = (
    CompatibilityCell(
        MigrationKind.CONFIG,
        "v2.0 settings schema 1",
        "current settings schema 1",
        "supported",
        "Defaults only the optional conversational provider to none; validates every other field.",
    ),
    CompatibilityCell(
        MigrationKind.CAMPAIGN,
        "v2.0 autonomous_optimize_run schema 2",
        "current autonomous_optimize_run schema 3",
        "supported",
        "Preserves plan identity, source digest, stage evidence, approvals, outcomes, and reasons.",
    ),
    CompatibilityCell(
        MigrationKind.EVIDENCE,
        "v2.0_campaign_evidence schema 1",
        "current canonical_campaign_evidence schema 1",
        "supported",
        "Preserves evidence identity, source/artifact digests, provenance, measurements, "
        "and terminal outcome.",
    ),
    CompatibilityCell(
        MigrationKind.CAMPAIGN,
        "conversational campaign state",
        "v2.7 canonical campaign state",
        "read-only-compatible",
        "Already-current state is validated and returned unchanged; no v2.0 guess is applied.",
    ),
    CompatibilityCell(
        MigrationKind.EVIDENCE,
        "stage-only evidence without source artifact digest",
        "canonical campaign evidence",
        "unsupported",
        "Refuse because source lineage cannot be reconstructed safely.",
    ),
    CompatibilityCell(
        MigrationKind.CONFIG,
        "non-none conversational provider in a v2.0 config",
        "current provider configuration",
        "unsupported",
        "Provider configuration is a later control-plane contract and is never "
        "invented during v2.0 migration.",
    ),
    CompatibilityCell(
        MigrationKind.CONFIG,
        "unknown or future schema",
        "any current schema",
        "unsupported",
        "Fail closed; no field dropping, renaming, or semantic guessing is permitted.",
    ),
)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise MigrationRefusal(f"{label} must be a JSON object")
    return {str(key): item for key, item in value.items()}


def _schema_version(record: Mapping[str, object], *, supported: set[int], label: str) -> int:
    value = record.get("schema_version")
    if isinstance(value, bool) or not isinstance(value, int):
        raise MigrationRefusal(f"{label} must declare an integer schema_version")
    if value not in supported:
        supported_text = ", ".join(str(item) for item in sorted(supported))
        raise MigrationRefusal(
            f"{label} schema {value} is unsupported; supported source schemas: {supported_text}"
        )
    return value


def _changes(values: list[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


def _reject_unknown_fields(
    value: Mapping[str, object], allowed: set[str], label: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise MigrationRefusal(f"{label} contains unknown fields: {', '.join(unknown)}")


def migrate_v20_config(value: object) -> MigrationResult:
    """Validate and canonicalize a v2.0 settings record.

    v2.0 did not require conversational-provider configuration.  A missing
    provider block therefore becomes the explicit ``none`` provider, which
    preserves the direct CLI/Python path without importing or calling an LLM.
    """

    raw = _mapping(value, "v2.0 config")
    version = _schema_version(raw, supported={V20_CONFIG_SCHEMA_VERSION}, label="config")
    record_type = raw.get("record_type")
    if record_type is not None and record_type != "modelsurgeon_config":
        raise MigrationRefusal("config record has an unknown record_type")
    if "config" in raw:
        if record_type != "modelsurgeon_config":
            raise MigrationRefusal("wrapped config records must declare modelsurgeon_config")
        payload = _mapping(raw["config"], "wrapped config payload")
    else:
        payload = dict(raw)
        payload.pop("record_type", None)

    allowed = set(Settings.model_fields)
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise MigrationRefusal(f"config contains unknown fields: {', '.join(unknown)}")
    changes: list[str] = []
    if "provider" not in payload:
        payload["provider"] = {"kind": "none"}
        changes.append("provider defaulted to explicit none")
    provider = payload.get("provider")
    provider_mapping = _mapping(provider, "config provider")
    if provider_mapping.get("kind", "none") != "none":
        raise MigrationRefusal(
            "v2.0 config migration refuses non-none conversational provider configuration"
        )
    try:
        settings = Settings.model_validate(payload)
    except (TypeError, ValueError) as error:
        raise MigrationRefusal(f"v2.0 config does not validate: {error}") from error
    record = settings.canonical_dict()
    changed = bool(changes) or record != payload
    if changed and not changes:
        changes.append("canonicalized settings field ordering and defaults")
    return MigrationResult(
        MigrationKind.CONFIG,
        "modelsurgeon_config" if record_type is not None else "settings",
        "settings",
        version,
        V20_CONFIG_SCHEMA_VERSION,
        record,
        changed,
        _changes(changes),
    )


def migrate_v20_campaign(value: object) -> MigrationResult:
    """Upgrade an autonomous v2.0 run cursor to the current run schema."""

    raw = _mapping(value, "campaign")
    if raw.get("record_type") != "autonomous_optimize_run":
        raise MigrationRefusal("campaign record is not an autonomous optimize run")
    _reject_unknown_fields(
        raw,
        {
            "record_type",
            "schema_version",
            "run_id",
            "plan_id",
            "plan_digest",
            "plan_record",
            "source_artifact_digest",
            "status",
            "outcome",
            "cursor",
            "stages",
            "approvals",
            "overrides",
            "accepted_artifact_digest",
            "alternatives",
            "reasons",
            "approval_audit",
        },
        "campaign",
    )
    version = _schema_version(
        raw,
        supported={V20_CAMPAIGN_SCHEMA_VERSION, ORCHESTRATOR_SCHEMA_VERSION},
        label="campaign",
    )
    if version == ORCHESTRATOR_SCHEMA_VERSION:
        try:
            current = _run_from_record(raw).to_record()
        except (OptimizeOrchestratorError, TypeError, ValueError) as error:
            raise MigrationRefusal(f"current campaign record is invalid: {error}") from error
        return MigrationResult(
            MigrationKind.CAMPAIGN,
            "autonomous_optimize_run",
            "autonomous_optimize_run",
            version,
            ORCHESTRATOR_SCHEMA_VERSION,
            current,
            False,
        )

    current = dict(raw)
    current["schema_version"] = ORCHESTRATOR_SCHEMA_VERSION
    current.setdefault("approval_audit", [])
    try:
        migrated = _run_from_record(current).to_record()
    except (OptimizeOrchestratorError, TypeError, ValueError) as error:
        raise MigrationRefusal(
            f"v2.0 campaign cannot be validated for current schema: {error}"
        ) from error
    changes = ["schema_version 2 upgraded to 3"]
    if "approval_audit" not in raw:
        changes.append("approval_audit defaulted to an empty append-only history")
    return MigrationResult(
        MigrationKind.CAMPAIGN,
        "autonomous_optimize_run",
        "autonomous_optimize_run",
        version,
        ORCHESTRATOR_SCHEMA_VERSION,
        migrated,
        True,
        _changes(changes),
        ("Approval history absent from the v2.0 record remains absent; no approval is inferred.",),
    )


def _legacy_evidence(value: Mapping[str, object]) -> MigrationResult:
    version = _schema_version(
        value, supported={V20_EVIDENCE_SCHEMA_VERSION}, label="evidence"
    )
    expected_fields = {
        "record_type",
        "schema_version",
        "evidence_id",
        "source_artifact_digest",
        "outcome",
        "detail",
        "provenance",
        "measured",
        "complete",
        "constraints_passed",
        "artifact_digest",
    }
    unknown = sorted(set(value) - expected_fields)
    missing = sorted(expected_fields - set(value))
    if unknown or missing:
        detail = []
        if missing:
            detail.append(f"missing fields: {', '.join(missing)}")
        if unknown:
            detail.append(f"unknown fields: {', '.join(unknown)}")
        raise MigrationRefusal("v2.0 evidence shape mismatch (" + "; ".join(detail) + ")")
    if value["record_type"] != "v2.0_campaign_evidence":
        raise MigrationRefusal("evidence record is not a supported v2.0 campaign evidence record")
    try:
        outcome = CampaignOutcome(str(value["outcome"]))
    except ValueError as error:
        raise MigrationRefusal("v2.0 evidence has an unknown terminal outcome") from error
    measured = value["measured"]
    complete = value["complete"]
    constraints_passed = value["constraints_passed"]
    if not all(isinstance(item, bool) for item in (measured, complete, constraints_passed)):
        raise MigrationRefusal("v2.0 evidence measurement flags must be boolean")
    expected_inconclusive = outcome is CampaignOutcome.UNKNOWN or not measured or not complete
    provenance = _mapping(value["provenance"], "v2.0 evidence provenance")
    try:
        evidence = CampaignEvidence(
            str(value["evidence_id"]),
            str(value["source_artifact_digest"]),
            outcome,
            str(value["detail"]),
            provenance,
            None if value["artifact_digest"] is None else str(value["artifact_digest"]),
            expected_inconclusive,
        )
    except (CampaignStateError, TypeError, ValueError) as error:
        raise MigrationRefusal(f"v2.0 evidence cannot be validated: {error}") from error
    return MigrationResult(
        MigrationKind.EVIDENCE,
        "v2.0_campaign_evidence",
        "canonical_campaign_evidence",
        version,
        CAMPAIGN_STATE_SCHEMA_VERSION,
        evidence.to_record(),
        True,
        ("record_type upgraded to canonical_campaign_evidence",),
        ()
        if constraints_passed
        else ("hard constraints were not passed; evidence remains non-promotable.",),
    )


def migrate_v20_evidence(value: object) -> MigrationResult:
    """Upgrade a v2.0 evidence record while retaining negative evidence."""

    raw = _mapping(value, "evidence")
    record_type = raw.get("record_type")
    if record_type == "canonical_campaign_evidence":
        version = _schema_version(
            raw, supported={CAMPAIGN_STATE_SCHEMA_VERSION}, label="evidence"
        )
        try:
            current = CampaignEvidence.from_record(raw).to_record()
        except (CampaignStateError, TypeError, ValueError) as error:
            raise MigrationRefusal(f"current evidence record is invalid: {error}") from error
        return MigrationResult(
            MigrationKind.EVIDENCE,
            "canonical_campaign_evidence",
            "canonical_campaign_evidence",
            version,
            CAMPAIGN_STATE_SCHEMA_VERSION,
            current,
            False,
        )
    if record_type == "v2.0_campaign_evidence":
        return _legacy_evidence(raw)
    raise MigrationRefusal(
        "evidence record type is unsupported; source artifact lineage is required"
    )


def detect_migration_kind(value: object) -> MigrationKind:
    """Detect a supported family without accepting an ambiguous record."""

    raw = _mapping(value, "migration input")
    record_type = raw.get("record_type")
    if record_type in {"modelsurgeon_config"}:
        return MigrationKind.CONFIG
    if record_type == "autonomous_optimize_run":
        return MigrationKind.CAMPAIGN
    if record_type in {"canonical_campaign_evidence", "v2.0_campaign_evidence"}:
        return MigrationKind.EVIDENCE
    if "model" in raw and "calibration" in raw and "constraints" in raw:
        return MigrationKind.CONFIG
    raise MigrationRefusal("migration input has no unambiguous supported record family")


def migrate_record(value: object, *, kind: MigrationKind | str | None = None) -> MigrationResult:
    """Migrate one supported record, refusing ambiguous or incompatible input."""

    selected = detect_migration_kind(value) if kind is None else MigrationKind(str(kind))
    if selected is MigrationKind.CONFIG:
        return migrate_v20_config(value)
    if selected is MigrationKind.CAMPAIGN:
        return migrate_v20_campaign(value)
    return migrate_v20_evidence(value)


def migration_matrix_records() -> tuple[dict[str, str], ...]:
    """Return the stable machine-readable capability matrix."""

    return tuple(cell.to_record() for cell in MIGRATION_CAPABILITY_MATRIX)


__all__ = [
    "MIGRATION_CAPABILITY_MATRIX",
    "MIGRATION_CONTRACT_SCHEMA_VERSION",
    "V20_CAMPAIGN_SCHEMA_VERSION",
    "V20_CONFIG_SCHEMA_VERSION",
    "V20_EVIDENCE_SCHEMA_VERSION",
    "CompatibilityCell",
    "MigrationKind",
    "MigrationRefusal",
    "MigrationResult",
    "detect_migration_kind",
    "migrate_record",
    "migrate_v20_campaign",
    "migrate_v20_config",
    "migrate_v20_evidence",
    "migration_matrix_records",
]
