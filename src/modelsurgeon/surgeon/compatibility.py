"""Explicit model lineage and fail-closed surgeon compatibility decisions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from modelsurgeon.adapters.family import ModelFamily

SURGEON_COMPATIBILITY_SCHEMA_VERSION: Final[int] = 1
SURGEON_COMPATIBILITY_PROTOCOL_REVISION: Final[str] = "surgeon-compatibility-v1"


class CompatibilityError(ValueError):
    """Raised when explicit lineage or compatibility evidence is malformed."""


class CompatibilityLevel(StrEnum):
    EXACT = "exact"
    COMPATIBLE = "compatible"
    ADAPTABLE = "adaptable"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class CompatibilityDimension(StrEnum):
    ANCESTRY = "ancestry"
    ARCHITECTURE = "architecture"
    FEATURES = "features"
    TARGETS = "targets"
    MUTATIONS = "mutations"
    CODEC = "codec"
    HARDWARE = "hardware"
    STATE = "state"
    OPERATIONS = "operations"


class CompatibilityOperation(StrEnum):
    TRAIN = "train"
    INFER = "infer"
    REPLAY = "replay"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CompatibilityError(f"{label} is required")
    return value


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class ModelLineageNode:
    model_id: str
    revision: str
    family: ModelFamily | None
    architecture_version: str
    aliases: tuple[str, ...] = ()
    parent_model_ids: tuple[str, ...] = ()
    derivation: str = "source"

    def __post_init__(self) -> None:
        for label, value in (
            ("model ID", self.model_id),
            ("revision", self.revision),
            ("architecture version", self.architecture_version),
            ("derivation", self.derivation),
        ):
            _text(value, label)
        if self.model_id in self.aliases or self.model_id in self.parent_model_ids:
            raise CompatibilityError("lineage node cannot alias or parent itself")
        if self.aliases != tuple(sorted(set(self.aliases))) or any(
            not item for item in self.aliases
        ):
            raise CompatibilityError("lineage aliases must be unique and canonical")
        if self.parent_model_ids != tuple(sorted(set(self.parent_model_ids))) or any(
            not item for item in self.parent_model_ids
        ):
            raise CompatibilityError("lineage parents must be unique and canonical")

    def to_record(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "family": None if self.family is None else self.family.value,
            "architecture_version": self.architecture_version,
            "aliases": list(self.aliases),
            "parent_model_ids": list(self.parent_model_ids),
            "derivation": self.derivation,
        }


@dataclass(frozen=True, slots=True)
class ModelLineageGraph:
    nodes: tuple[ModelLineageNode, ...]
    schema_version: int = SURGEON_COMPATIBILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        ids = tuple(node.model_id for node in self.nodes)
        if not ids or ids != tuple(sorted(set(ids))):
            raise CompatibilityError("lineage graph model IDs must be non-empty and canonical")
        aliases: list[str] = []
        for node in self.nodes:
            aliases.extend(node.aliases)
            for parent in node.parent_model_ids:
                if parent not in ids:
                    raise CompatibilityError(f"lineage parent {parent!r} is absent")
        if len(aliases) != len(set(aliases)) or set(aliases) & set(ids):
            raise CompatibilityError("lineage aliases must be globally unique")
        self._assert_acyclic()

    def _assert_acyclic(self) -> None:
        parents = {node.model_id: node.parent_model_ids for node in self.nodes}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(model_id: str) -> None:
            if model_id in visiting:
                raise CompatibilityError("model ancestry contains a cycle")
            if model_id in visited:
                return
            visiting.add(model_id)
            for parent in parents[model_id]:
                visit(parent)
            visiting.remove(model_id)
            visited.add(model_id)

        for model_id in parents:
            visit(model_id)

    def node(self, model_id_or_alias: str) -> ModelLineageNode:
        matches = tuple(
            node
            for node in self.nodes
            if node.model_id == model_id_or_alias or model_id_or_alias in node.aliases
        )
        if len(matches) != 1:
            raise CompatibilityError(f"lineage identifier is not unique: {model_id_or_alias!r}")
        return matches[0]

    def closure(self, model_id_or_alias: str) -> frozenset[str]:
        origin = self.node(model_id_or_alias).model_id
        parents = {node.model_id: node.parent_model_ids for node in self.nodes}
        found: set[str] = set()

        def collect(model_id: str) -> None:
            if model_id in found:
                return
            found.add(model_id)
            for parent in parents[model_id]:
                collect(parent)

        collect(origin)
        return frozenset(found)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_revision": SURGEON_COMPATIBILITY_PROTOCOL_REVISION,
            "nodes": [node.to_record() for node in self.nodes],
        }


@dataclass(frozen=True, slots=True)
class CompatibilityContext:
    model_id: str
    revision: str
    family: ModelFamily | None
    architecture_version: str | None
    feature_schema_id: str | None
    feature_schema_version: int | None
    target_schema_id: str | None
    mutation_schema_version: int | None
    codec: str | None
    hardware_profile: str | None
    state_schema_version: int | None
    operations: tuple[CompatibilityOperation, ...]

    def __post_init__(self) -> None:
        _text(self.model_id, "compatibility model ID")
        _text(self.revision, "compatibility revision")
        for label, value in (
            ("architecture version", self.architecture_version),
            ("feature schema ID", self.feature_schema_id),
            ("target schema ID", self.target_schema_id),
            ("codec", self.codec),
            ("hardware profile", self.hardware_profile),
        ):
            if value is not None:
                _text(value, label)
        for label, version_value in (
            ("feature schema version", self.feature_schema_version),
            ("mutation schema version", self.mutation_schema_version),
            ("state schema version", self.state_schema_version),
        ):
            if version_value is not None and (
                isinstance(version_value, bool)
                or not isinstance(version_value, int)
                or version_value <= 0
            ):
                raise CompatibilityError(f"{label} must be positive when present")
        if not self.operations or self.operations != tuple(
            sorted(set(self.operations), key=lambda item: item.value)
        ):
            raise CompatibilityError("compatibility operations must be non-empty and canonical")

    def to_record(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "family": None if self.family is None else self.family.value,
            "architecture_version": self.architecture_version,
            "feature_schema_id": self.feature_schema_id,
            "feature_schema_version": self.feature_schema_version,
            "target_schema_id": self.target_schema_id,
            "mutation_schema_version": self.mutation_schema_version,
            "codec": self.codec,
            "hardware_profile": self.hardware_profile,
            "state_schema_version": self.state_schema_version,
            "operations": [item.value for item in self.operations],
        }


@dataclass(frozen=True, slots=True)
class CompatibilityReason:
    code: str
    dimension: CompatibilityDimension
    level: CompatibilityLevel
    detail: str

    def __post_init__(self) -> None:
        for label, value in (("reason code", self.code), ("reason detail", self.detail)):
            _text(value, label)

    def to_record(self) -> dict[str, str]:
        return {
            "code": self.code,
            "dimension": self.dimension.value,
            "level": self.level.value,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class CompatibilityDecision:
    source_model_id: str
    target_model_id: str
    operation: CompatibilityOperation
    level: CompatibilityLevel
    allowed: bool
    reasons: tuple[CompatibilityReason, ...]
    decision_id: str
    schema_version: int = SURGEON_COMPATIBILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _text(self.source_model_id, "source model ID")
        _text(self.target_model_id, "target model ID")
        if not self.reasons or not self.decision_id:
            raise CompatibilityError("compatibility decisions require reasons and identity")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "protocol_revision": SURGEON_COMPATIBILITY_PROTOCOL_REVISION,
            "source_model_id": self.source_model_id,
            "target_model_id": self.target_model_id,
            "operation": self.operation.value,
            "level": self.level.value,
            "allowed": self.allowed,
            "reasons": [item.to_record() for item in self.reasons],
            "decision_id": self.decision_id,
        }


def validate_heldout_lineage(
    graph: ModelLineageGraph,
    train_model_ids: Sequence[str],
    heldout_model_ids: Sequence[str],
) -> None:
    """Reject ancestry or alias overlap across a training/held-out boundary."""

    train_closure = set().union(*(graph.closure(model_id) for model_id in train_model_ids))
    heldout_closure = set().union(*(graph.closure(model_id) for model_id in heldout_model_ids))
    overlap = sorted(train_closure & heldout_closure)
    if overlap:
        raise CompatibilityError("model ancestry crosses held-out boundary: " + ", ".join(overlap))


def _reason(
    code: str, dimension: CompatibilityDimension, level: CompatibilityLevel, detail: str
) -> CompatibilityReason:
    return CompatibilityReason(code, dimension, level, detail)


def decide_compatibility(
    source: CompatibilityContext,
    target: CompatibilityContext,
    *,
    operation: CompatibilityOperation,
    lineage: ModelLineageGraph | None = None,
    allow_adaptation: bool = False,
) -> CompatibilityDecision:
    """Compare every required contract dimension without inferring from names."""

    reasons: list[CompatibilityReason] = []
    if lineage is not None:
        source_node = lineage.node(source.model_id)
        target_node = lineage.node(target.model_id)
        source_closure = lineage.closure(source_node.model_id)
        if target_node.model_id in source_closure:
            reasons.append(
                _reason(
                    "derived-target",
                    CompatibilityDimension.ANCESTRY,
                    CompatibilityLevel.COMPATIBLE,
                    "target is an explicit descendant of the source",
                )
            )
        else:
            reasons.append(
                _reason(
                    "independent-target",
                    CompatibilityDimension.ANCESTRY,
                    CompatibilityLevel.ADAPTABLE,
                    "source and target have no explicit ancestry link",
                )
            )
    else:
        reasons.append(
            _reason(
                "missing-lineage",
                CompatibilityDimension.ANCESTRY,
                CompatibilityLevel.UNKNOWN,
                "explicit lineage graph is required",
            )
        )

    if source.family is None or target.family is None:
        reasons.append(
            _reason(
                "missing-family",
                CompatibilityDimension.ARCHITECTURE,
                CompatibilityLevel.UNKNOWN,
                "architecture family evidence is missing",
            )
        )
    elif (
        source.family is target.family
        and source.architecture_version == target.architecture_version
    ):
        reasons.append(
            _reason(
                "architecture-exact",
                CompatibilityDimension.ARCHITECTURE,
                CompatibilityLevel.EXACT,
                "family and architecture revision match",
            )
        )
    elif source.family is target.family:
        reasons.append(
            _reason(
                "architecture-revision",
                CompatibilityDimension.ARCHITECTURE,
                CompatibilityLevel.COMPATIBLE,
                "family matches but architecture revisions differ",
            )
        )
    else:
        reasons.append(
            _reason(
                "architecture-family-adaptation",
                CompatibilityDimension.ARCHITECTURE,
                CompatibilityLevel.ADAPTABLE,
                "explicit cross-family feature adaptation is required",
            )
        )

    _compare_optional(
        reasons,
        CompatibilityDimension.FEATURES,
        "feature-schema",
        source.feature_schema_id,
        target.feature_schema_id,
        source.feature_schema_version,
        target.feature_schema_version,
    )
    _compare_optional(
        reasons,
        CompatibilityDimension.TARGETS,
        "target-schema",
        source.target_schema_id,
        target.target_schema_id,
        None,
        None,
    )
    _compare_version(
        reasons,
        CompatibilityDimension.MUTATIONS,
        "mutation-schema",
        source.mutation_schema_version,
        target.mutation_schema_version,
    )
    _compare_optional(
        reasons, CompatibilityDimension.CODEC, "codec", source.codec, target.codec, None, None
    )
    _compare_optional(
        reasons,
        CompatibilityDimension.HARDWARE,
        "hardware",
        source.hardware_profile,
        target.hardware_profile,
        None,
        None,
    )
    _compare_version(
        reasons,
        CompatibilityDimension.STATE,
        "state-schema",
        source.state_schema_version,
        target.state_schema_version,
    )
    if operation not in target.operations:
        reasons.append(
            _reason(
                "operation-missing",
                CompatibilityDimension.OPERATIONS,
                CompatibilityLevel.UNSUPPORTED,
                f"target does not declare {operation.value}",
            )
        )
    else:
        reasons.append(
            _reason(
                "operation-supported",
                CompatibilityDimension.OPERATIONS,
                CompatibilityLevel.EXACT,
                f"target declares {operation.value}",
            )
        )

    levels = [item.level for item in reasons]
    if CompatibilityLevel.UNKNOWN in levels:
        overall = CompatibilityLevel.UNKNOWN
    elif CompatibilityLevel.UNSUPPORTED in levels:
        overall = CompatibilityLevel.UNSUPPORTED
    elif CompatibilityLevel.ADAPTABLE in levels:
        overall = CompatibilityLevel.ADAPTABLE
    elif CompatibilityLevel.COMPATIBLE in levels:
        overall = CompatibilityLevel.COMPATIBLE
    else:
        overall = CompatibilityLevel.EXACT
    allowed = overall in {CompatibilityLevel.EXACT, CompatibilityLevel.COMPATIBLE} or (
        overall is CompatibilityLevel.ADAPTABLE and allow_adaptation
    )
    identity = {
        "source": source.to_record(),
        "target": target.to_record(),
        "operation": operation.value,
        "reasons": [item.to_record() for item in reasons],
        "allow_adaptation": allow_adaptation,
    }
    decision_id = "sha256:" + hashlib.sha256(_canonical(identity).encode("utf-8")).hexdigest()
    return CompatibilityDecision(
        source.model_id, target.model_id, operation, overall, allowed, tuple(reasons), decision_id
    )


def _compare_optional(
    reasons: list[CompatibilityReason],
    dimension: CompatibilityDimension,
    code: str,
    source_value: str | None,
    target_value: str | None,
    source_version: int | None,
    target_version: int | None,
) -> None:
    if source_value is None or target_value is None:
        reasons.append(
            _reason(
                f"missing-{code}",
                dimension,
                CompatibilityLevel.UNKNOWN,
                f"{code} evidence is missing",
            )
        )
    elif source_value == target_value and (
        source_version is None or source_version == target_version
    ):
        reasons.append(
            _reason(f"{code}-exact", dimension, CompatibilityLevel.EXACT, f"{code} matches")
        )
    elif (
        source_version is not None
        and target_version is not None
        and source_version != target_version
    ):
        reasons.append(
            _reason(
                f"{code}-version",
                dimension,
                CompatibilityLevel.UNSUPPORTED,
                f"{code} versions are incompatible",
            )
        )
    else:
        reasons.append(
            _reason(
                f"{code}-adaptable",
                dimension,
                CompatibilityLevel.ADAPTABLE,
                f"{code} adaptation is required",
            )
        )


def _compare_version(
    reasons: list[CompatibilityReason],
    dimension: CompatibilityDimension,
    code: str,
    source_value: int | None,
    target_value: int | None,
) -> None:
    if source_value is None or target_value is None:
        reasons.append(
            _reason(
                f"missing-{code}",
                dimension,
                CompatibilityLevel.UNKNOWN,
                f"{code} evidence is missing",
            )
        )
    elif source_value == target_value:
        reasons.append(
            _reason(f"{code}-exact", dimension, CompatibilityLevel.EXACT, f"{code} matches")
        )
    else:
        reasons.append(
            _reason(
                f"{code}-version",
                dimension,
                CompatibilityLevel.UNSUPPORTED,
                f"{code} versions are incompatible",
            )
        )
