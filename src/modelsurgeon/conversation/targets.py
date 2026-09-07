"""Bounded elicitation of measurable conversational targets.

This module classifies an already typed intent.  It does not infer a default
threshold, baseline, deployment compatibility claim, or quality metric from
provider text.  Vague requests receive one deterministic necessary question;
unsupported metrics remain explicit refusals.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from .clarification import ClarificationQuestion
from .intent import IntentField, IntentOutcome, IntentRecord

if TYPE_CHECKING:
    from modelsurgeon.search.intent_policy import IntentPolicyDecision

MEASURABLE_TARGET_SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_SUPPORTED_MEASURES = (
    "disk_size",
    "latency",
    "memory",
    "parameter_count",
    "perplexity",
    "quality",
    "throughput",
)
_VAGUE_MARKERS: tuple[tuple[str, str], ...] = (
    ("faster", "latency"),
    ("speed", "latency"),
    ("latency", "latency"),
    ("throughput", "throughput"),
    ("fit on", "memory"),
    ("fit in", "memory"),
    ("gpu", "memory"),
    ("memory", "memory"),
    ("quality", "quality"),
    ("accurate", "quality"),
    ("deploy", "deployment"),
    ("production", "deployment"),
    ("serve", "deployment"),
)


class MeasurableTargetStatus(StrEnum):
    COMPLETE = "complete"
    NEEDS_CLARIFICATION = "needs_clarification"
    UNSUPPORTED = "unsupported"


class MeasurableTargetError(ValueError):
    """Raised for malformed target-assessment inputs."""


def _canonical(value: object) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise MeasurableTargetError("target assessment is not canonical JSON") from error


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise MeasurableTargetError(f"{label} must be a canonical identifier")
    return value


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _kind(field: IntentField) -> str | None:
    if not isinstance(field.value, Mapping):
        return None
    raw = field.value.get("kind", field.value.get("type", field.value.get("declaration")))
    if not isinstance(raw, str):
        return None
    return {
        "constraint": "hard_constraint",
        "hard_constraint": "hard_constraint",
        "objective": "soft_objective",
        "soft_objective": "soft_objective",
        "preference": "soft_objective",
    }.get(raw)


def _metric(field: IntentField) -> str | None:
    if not isinstance(field.value, Mapping):
        return None
    value = field.value.get("metric")
    return value if isinstance(value, str) else None


def _question(
    intent: IntentRecord, category: str, measure: str, prompt: str
) -> ClarificationQuestion:
    field_id = f"target.{measure}"
    identity = {
        "intent_id": intent.intent_id,
        "category": category,
        "field_id": field_id,
        "measure": measure,
        "prompt": prompt,
    }
    question_id = "target_question_" + _digest(identity)[len("sha256:") :]
    return ClarificationQuestion(
        question_id,
        field_id,
        category,
        prompt,
        (),
        True,
        (),
        ("measurable-target-required",),
    )


@dataclass(frozen=True, slots=True)
class MeasurableTargetAssessment:
    """Canonical answer to whether an intent has an actionable target."""

    intent_id: str
    status: MeasurableTargetStatus
    questions: tuple[ClarificationQuestion, ...]
    supported_measures: tuple[str, ...]
    diagnostics: tuple[str, ...]
    schema_version: int = MEASURABLE_TARGET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _identifier(self.intent_id, "intent ID")
        if self.schema_version != MEASURABLE_TARGET_SCHEMA_VERSION:
            raise MeasurableTargetError("unsupported measurable-target schema version")
        if self.supported_measures != tuple(sorted(set(self.supported_measures))):
            raise MeasurableTargetError("supported measures must be sorted and unique")
        if self.diagnostics != tuple(sorted(set(self.diagnostics))):
            raise MeasurableTargetError("target diagnostics must be sorted and unique")
        if tuple(item.question_id for item in self.questions) != tuple(
            sorted({item.question_id for item in self.questions})
        ):
            raise MeasurableTargetError("target questions must be sorted and unique")
        if self.status is MeasurableTargetStatus.COMPLETE and self.questions:
            raise MeasurableTargetError("complete target assessment cannot ask questions")
        if self.status is MeasurableTargetStatus.NEEDS_CLARIFICATION and not self.questions:
            raise MeasurableTargetError("target clarification requires a question")

    @property
    def assessment_id(self) -> str:
        return "target_assessment_" + _digest(
            self.to_record(include_identity=False)
        )[len("sha256:") :]

    def to_record(self, *, include_identity: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "schema_version": self.schema_version,
            "intent_id": self.intent_id,
            "status": self.status.value,
            "questions": [item.to_record() for item in self.questions],
            "supported_measures": list(self.supported_measures),
            "diagnostics": list(self.diagnostics),
        }
        if include_identity:
            record["assessment_id"] = self.assessment_id
        return record


def _vague_measure(request: str) -> str | None:
    lowered = request.casefold()
    for marker, measure in _VAGUE_MARKERS:
        if marker in lowered:
            return measure
    return None


def _prompt(measure: str) -> tuple[str, str]:
    prompts = {
        "latency": (
            "missing-latency-target",
            "What measurable latency target should be optimized? Provide a value, "
            "unit (for example milliseconds), direction, and whether it is a hard "
            "constraint. No default threshold is assumed.",
        ),
        "throughput": (
            "missing-throughput-target",
            "What measurable throughput target should be optimized? Provide a value, "
            "unit (for example items/second), direction, and whether it is a hard "
            "constraint. No default threshold is assumed.",
        ),
        "memory": (
            "missing-memory-target",
            "What measurable memory limit should be enforced? Provide a value, unit "
            "(bytes or GiB), runtime/device context, and whether it is a hard "
            "constraint. No compatibility claim is inferred.",
        ),
        "quality": (
            "missing-quality-target",
            "Which quality metric and minimum acceptable target should be preserved? "
            "Provide the metric, value, unit, and baseline if relative comparison "
            "is intended. No quality threshold is invented.",
        ),
        "deployment": (
            "missing-deployment-target",
            "Which verified deployment target should be supported, and what measurable "
            "runtime, memory, latency, or artifact-size budget applies? Compatibility "
            "is unsupported until declared and checked.",
        ),
    }
    return prompts.get(
        measure,
        (
            "missing-measurable-target",
            "Provide one supported measurable target with its metric, value, unit, "
            "direction, and hard or soft semantics. No threshold or baseline is "
            "inferred.",
        ),
    )


def assess_measurable_targets(
    intent: IntentRecord,
    policy: IntentPolicyDecision | None = None,
    *,
    supported_measures: Sequence[str] = _SUPPORTED_MEASURES,
) -> MeasurableTargetAssessment:
    """Classify whether an intent has a safe, measurable target.

    Existing executable specs pass without a question. Vague or incomplete
    requests receive one deterministic question tied to the request wording;
    unknown metrics are explicit unsupported outcomes.
    """

    if not isinstance(intent, IntentRecord):
        raise MeasurableTargetError("target assessment requires an IntentRecord")
    measures = tuple(sorted(set(supported_measures)))
    if any(not isinstance(item, str) or not item for item in measures):
        raise MeasurableTargetError("supported measures must be non-empty strings")
    if policy is None:
        from modelsurgeon.search.intent_policy import evaluate_intent_policy

        policy = evaluate_intent_policy(intent)
    if policy.intent_id != intent.intent_id:
        raise MeasurableTargetError("target policy belongs to another intent")

    fields = tuple(intent.fields)
    declared_metrics = tuple(
        metric for field in fields if (metric := _metric(field)) is not None
    )
    unsupported = tuple(
        sorted(set(metric for metric in declared_metrics if metric not in measures))
    )
    if unsupported:
        return MeasurableTargetAssessment(
            intent.intent_id,
            MeasurableTargetStatus.UNSUPPORTED,
            (),
            measures,
            tuple(f"unsupported metric {metric!r}" for metric in unsupported),
        )

    complete_fields = True
    for field in fields:
        value = field.value
        kind = _kind(field)
        if kind is None or not isinstance(value, Mapping) or _metric(field) is None:
            complete_fields = False
            break
        required = {"metric", "direction", "unit"}
        if kind == "hard_constraint":
            required.add("threshold")
        if not required.issubset(value):
            complete_fields = False
            break
    if intent.outcome is IntentOutcome.EXECUTABLE and complete_fields and policy.executable:
        return MeasurableTargetAssessment(
            intent.intent_id, MeasurableTargetStatus.COMPLETE, (), measures, ()
        )

    measure = _vague_measure(intent.original_request)
    if measure is None:
        if any(_kind(field) == "soft_objective" for field in fields):
            measure = (
                "quality"
                if not any(_kind(field) == "hard_constraint" for field in fields)
                else "latency"
            )
        else:
            measure = "quality"
    category, prompt = _prompt(measure)
    question = _question(intent, category, measure, prompt)
    diagnostics = tuple(
        sorted(set((*intent.diagnostics, *(item.code for item in policy.diagnostics))))
    )
    return MeasurableTargetAssessment(
        intent.intent_id,
        MeasurableTargetStatus.NEEDS_CLARIFICATION,
        (question,),
        measures,
        diagnostics or ("measurable-target-required",),
    )


__all__ = [
    "MEASURABLE_TARGET_SCHEMA_VERSION",
    "MeasurableTargetAssessment",
    "MeasurableTargetError",
    "MeasurableTargetStatus",
    "assess_measurable_targets",
]
