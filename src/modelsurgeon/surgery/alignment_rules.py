"""Fail-closed hardware-specific legality and preference rules."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modelsurgeon.evaluation.microbenchmarks import MicrobenchmarkPartition

ALIGNMENT_RULE_SCHEMA_VERSION = 1


class AlignmentRuleError(ValueError):
    """Raised when a hardware-specific alignment rule is unsafe or incomplete."""


class AlignmentAxis(StrEnum):
    MLP_WIDTH = "mlp_width"
    QUERY_HEAD_GROUP = "query_head_group"
    KV_HEAD_GROUP = "kv_head_group"
    HIDDEN_DIMENSION = "hidden_dimension"
    LAYER_COUNT = "layer_count"


class AlignmentRuleOutcome(StrEnum):
    MEASURED = "measured"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


class AlignmentDecisionOutcome(StrEnum):
    LEGAL = "legal"
    ILLEGAL = "illegal"
    UNKNOWN = "unknown"


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AlignmentRuleError(f"{label} is required")
    return value


def _positive(value: int, label: str) -> None:
    if isinstance(value, bool) or value <= 0:
        raise AlignmentRuleError(f"{label} must be positive")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class AlignmentConstraint:
    """Hard structural/codec legality; it never encodes empirical preference."""

    axis: AlignmentAxis
    format: str
    codec: str | None
    graph_multiple: int
    codec_multiple: int
    source_revision: str

    def __post_init__(self) -> None:
        for label, value in (("format", self.format), ("source revision", self.source_revision)):
            _text(value, label)
        if self.codec is not None:
            _text(self.codec, "codec")
        _positive(self.graph_multiple, "graph multiple")
        _positive(self.codec_multiple, "codec multiple")

    @property
    def legal_multiple(self) -> int:
        return math.lcm(self.graph_multiple, self.codec_multiple)

    def is_legal(self, granularity: int) -> bool:
        return granularity > 0 and granularity % self.legal_multiple == 0

    def to_record(self) -> dict[str, object]:
        return {
            "axis": self.axis.value,
            "format": self.format,
            "codec": self.codec,
            "graph_multiple": self.graph_multiple,
            "codec_multiple": self.codec_multiple,
            "legal_multiple": self.legal_multiple,
            "source_revision": self.source_revision,
        }


@dataclass(frozen=True, slots=True)
class AlignmentEvidence:
    """One measured microbenchmark link used by a preference rule."""

    partition_id: str
    result_id: str
    profile_id: str
    shape_id: str
    phase: str
    speedup_lower: float
    speedup_upper: float

    def __post_init__(self) -> None:
        for label, value in (
            ("partition ID", self.partition_id),
            ("result ID", self.result_id),
            ("profile ID", self.profile_id),
            ("shape ID", self.shape_id),
            ("evidence phase", self.phase),
        ):
            _text(value, label)
        if self.phase not in {"training", "validation"}:
            raise AlignmentRuleError("evidence phase must be training or validation")
        if not math.isfinite(self.speedup_lower) or not math.isfinite(self.speedup_upper):
            raise AlignmentRuleError("evidence speedup bounds must be finite")
        if self.speedup_lower > self.speedup_upper:
            raise AlignmentRuleError("evidence speedup bounds must be ordered")

    def to_record(self) -> dict[str, object]:
        return {
            "partition_id": self.partition_id,
            "result_id": self.result_id,
            "profile_id": self.profile_id,
            "shape_id": self.shape_id,
            "phase": self.phase,
            "speedup_lower": self.speedup_lower,
            "speedup_upper": self.speedup_upper,
        }


@dataclass(frozen=True, slots=True)
class AlignmentPreference:
    """Empirical preference that cannot weaken hard legality constraints."""

    axis: AlignmentAxis
    format: str
    codec: str | None
    hardware_profile_id: str
    preferred_granularity: int | None
    outcome: AlignmentRuleOutcome
    training_evidence: tuple[AlignmentEvidence, ...]
    validation_evidence: tuple[AlignmentEvidence, ...]
    false_preference_rate: float | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        _text(self.format, "preference format")
        _text(self.hardware_profile_id, "preference hardware profile ID")
        if self.codec is not None:
            _text(self.codec, "preference codec")
        if self.outcome is AlignmentRuleOutcome.MEASURED:
            if self.preferred_granularity is None:
                raise AlignmentRuleError("measured preferences require a granularity")
            _positive(self.preferred_granularity, "preferred granularity")
            if not self.training_evidence or not self.validation_evidence:
                raise AlignmentRuleError(
                    "measured preferences require training and validation evidence"
                )
            if self.false_preference_rate is None or not 0 <= self.false_preference_rate <= 1:
                raise AlignmentRuleError("measured preferences require a false-preference rate")
            if self.reason is not None:
                raise AlignmentRuleError("measured preferences cannot carry a reason")
        else:
            if self.preferred_granularity is not None:
                raise AlignmentRuleError("non-measured preferences cannot publish a granularity")
            _text(self.reason or "", "preference outcome reason")
            if self.training_evidence or self.validation_evidence:
                raise AlignmentRuleError("non-measured preferences cannot carry evidence")

    def to_record(self) -> dict[str, object]:
        return {
            "axis": self.axis.value,
            "format": self.format,
            "codec": self.codec,
            "hardware_profile_id": self.hardware_profile_id,
            "preferred_granularity": self.preferred_granularity,
            "outcome": self.outcome.value,
            "training_evidence": [item.to_record() for item in self.training_evidence],
            "validation_evidence": [item.to_record() for item in self.validation_evidence],
            "false_preference_rate": self.false_preference_rate,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class AlignmentDecision:
    outcome: AlignmentDecisionOutcome
    legal: bool | None
    preferred_granularity: int | None
    reason: str

    def to_record(self) -> dict[str, object]:
        return {
            "outcome": self.outcome.value,
            "legal": self.legal,
            "preferred_granularity": self.preferred_granularity,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class AlignmentRuleSet:
    """Profile-specific rules with conservative lookup for unseen shapes."""

    hardware_profile_id: str
    constraints: tuple[AlignmentConstraint, ...]
    preferences: tuple[AlignmentPreference, ...]
    source_partition_id: str
    schema_version: int = ALIGNMENT_RULE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _text(self.hardware_profile_id, "rule-set hardware profile ID")
        _text(self.source_partition_id, "rule-set source partition ID")
        if self.schema_version != ALIGNMENT_RULE_SCHEMA_VERSION:
            raise AlignmentRuleError("unsupported alignment rule schema version")
        constraint_keys = tuple((item.axis, item.format, item.codec) for item in self.constraints)
        if constraint_keys != tuple(sorted(set(constraint_keys), key=str)):
            raise AlignmentRuleError("alignment constraints must be unique and canonically ordered")
        preference_keys = tuple((item.axis, item.format, item.codec) for item in self.preferences)
        if preference_keys != tuple(sorted(set(preference_keys), key=str)):
            raise AlignmentRuleError("alignment preferences must be unique and canonically ordered")
        for preference in self.preferences:
            if preference.hardware_profile_id != self.hardware_profile_id:
                raise AlignmentRuleError("preference profile does not match rule-set profile")

    @property
    def ruleset_id(self) -> str:
        digest = hashlib.sha256(_canonical(self.to_record(include_id=False)).encode()).hexdigest()
        return f"alignment_rules_{digest}"

    def to_record(self, *, include_id: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": self.schema_version,
            "hardware_profile_id": self.hardware_profile_id,
            "source_partition_id": self.source_partition_id,
            "constraints": [item.to_record() for item in self.constraints],
            "preferences": [item.to_record() for item in self.preferences],
        }
        if include_id:
            record["ruleset_id"] = self.ruleset_id
        return record

    def decide(
        self,
        axis: AlignmentAxis,
        format: str,
        codec: str | None,
        granularity: int,
    ) -> AlignmentDecision:
        constraint = next(
            (
                item
                for item in self.constraints
                if (item.axis, item.format, item.codec) == (axis, format, codec)
            ),
            None,
        )
        if constraint is None:
            return AlignmentDecision(
                AlignmentDecisionOutcome.UNKNOWN,
                None,
                None,
                "no measured or declared legality rule covers this shape",
            )
        legal = constraint.is_legal(granularity)
        if not legal:
            return AlignmentDecision(
                AlignmentDecisionOutcome.ILLEGAL,
                False,
                None,
                "requested granularity violates graph or codec alignment",
            )
        preference = next(
            (
                item
                for item in self.preferences
                if (item.axis, item.format, item.codec) == (axis, format, codec)
            ),
            None,
        )
        if preference is None or preference.outcome is not AlignmentRuleOutcome.MEASURED:
            return AlignmentDecision(
                AlignmentDecisionOutcome.UNKNOWN,
                True,
                None,
                "granularity is legal but empirical preference is unknown",
            )
        return AlignmentDecision(
            AlignmentDecisionOutcome.LEGAL,
            True,
            preference.preferred_granularity,
            "granularity satisfies declared legality; preference is evidence-backed",
        )


def build_alignment_rule_set(
    partition: MicrobenchmarkPartition,
    *,
    constraints: tuple[AlignmentConstraint, ...],
    preferences: tuple[AlignmentPreference, ...],
) -> AlignmentRuleSet:
    """Validate preference links against measured results in one microbenchmark partition."""

    result_by_id = {result.result_id: result for result in partition.results}
    for preference in preferences:
        for evidence in (*preference.training_evidence, *preference.validation_evidence):
            if evidence.partition_id != partition.partition_id:
                raise AlignmentRuleError("preference evidence references another partition")
            result = result_by_id.get(evidence.result_id)
            if result is None or result.outcome.value != "measured":
                raise AlignmentRuleError("preference evidence must reference a measured result")
            if (
                evidence.profile_id != partition.profile_id
                or evidence.shape_id != result.key.shape_id
            ):
                raise AlignmentRuleError("preference evidence provenance does not match result")
    return AlignmentRuleSet(
        partition.profile_id,
        tuple(
            sorted(constraints, key=lambda item: (item.axis.value, item.format, item.codec or ""))
        ),
        tuple(
            sorted(preferences, key=lambda item: (item.axis.value, item.format, item.codec or ""))
        ),
        partition.partition_id,
    )


def build_unknown_preference(
    axis: AlignmentAxis,
    format: str,
    codec: str | None,
    hardware_profile_id: str,
    reason: str,
) -> AlignmentPreference:
    """Create the explicit fail-closed result for an unseen or conflicting rule."""

    return AlignmentPreference(
        axis,
        format,
        codec,
        hardware_profile_id,
        None,
        AlignmentRuleOutcome.UNKNOWN,
        (),
        (),
        reason=reason,
    )
