"""Fail-closed preview and confirmation boundary for conversational specs.

The preview is a read-only projection of the existing objective contract.  It
does not create a plan, select a tensor, inspect a model, or call an executor.
The only object that can be submitted to a later execution adapter is the
explicitly confirmed :class:`SpecSubmission`, which carries the exact
canonical spec and its identity.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, cast

from modelsurgeon.conversation import IntentField, IntentOutcome, IntentRecord
from modelsurgeon.policy import PolicyDecision

from .intent_policy import IntentPolicyDecision, PolicyDiagnostic, evaluate_intent_policy

if TYPE_CHECKING:
    from .intent_compiler import IntentCompilation


SPEC_PREVIEW_SCHEMA_VERSION = 1
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")


class SpecPreviewError(ValueError):
    """Raised when a preview cannot be safely rendered or confirmed."""


class SpecDiffKind(StrEnum):
    INITIAL = "initial"
    MATERIAL = "material"
    NON_MATERIAL = "non_material"
    UNRESOLVED = "unresolved"


def _canonical(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise SpecPreviewError(f"{label} is not a canonical identifier")
    return value


def _field_kind(field: IntentField) -> str | None:
    if not isinstance(field.value, Mapping):
        return None
    raw = field.value.get("kind", field.value.get("type", field.value.get("declaration")))
    if not isinstance(raw, str):
        return None
    return raw.replace("-", "_").lower()


def _field_records(fields: Sequence[IntentField], kinds: set[str]) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "field_id": field.field_id,
            "value": field.value,
            "unit": field.unit,
            "required": field.required,
            "source_span_ids": list(field.source_span_ids),
        }
        for field in fields
        if (_field_kind(field) or "") in kinds
    )


def _unresolved_fields(
    intent: IntentRecord,
    decision: IntentPolicyDecision,
) -> tuple[str, ...]:
    unresolved = {
        item.field_id
        for item in decision.ambiguities
        if item.required
    }
    unresolved.update(
        item.field_id for item in decision.diagnostics if item.field_id is not None
    )
    supported_kinds = {
        "constraint",
        "hard_constraint",
        "objective",
        "soft_objective",
        "preference",
        "task_quality",
    }
    unresolved.update(
        field.field_id
        for field in intent.fields
        if (_field_kind(field) or "") not in supported_kinds
    )
    if decision.outcome is not IntentOutcome.EXECUTABLE and not unresolved:
        unresolved.add("intent")
    return tuple(sorted(unresolved))


def _changed_paths(left: object, right: object, path: str = "spec") -> set[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        paths: set[str] = set()
        keys = set(left) | set(right)
        for key in keys:
            child = f"{path}.{key}"
            if key not in left or key not in right:
                paths.add(child)
            else:
                paths.update(_changed_paths(left[key], right[key], child))
        return paths
    if isinstance(left, list) and isinstance(right, list):
        paths = set()
        for index in range(max(len(left), len(right))):
            child = f"{path}[{index}]"
            if index >= len(left) or index >= len(right):
                paths.add(child)
            else:
                paths.update(_changed_paths(left[index], right[index], child))
        return paths
    return set() if left == right else {path}


@dataclass(frozen=True, slots=True)
class SpecDiff:
    """Canonical, visible difference between two interpreted specs."""

    kind: SpecDiffKind
    from_spec_identity: str | None
    to_spec_identity: str | None
    from_spec_digest: str | None
    to_spec_digest: str | None
    changed_paths: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        if self.changed_paths != tuple(sorted(set(self.changed_paths))):
            raise SpecPreviewError("spec diff paths must be sorted and unique")
        if not self.reason.strip():
            raise SpecPreviewError("spec diff requires a reason")

    @property
    def material(self) -> bool:
        return self.kind is SpecDiffKind.MATERIAL

    def to_record(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "from_spec_identity": self.from_spec_identity,
            "to_spec_identity": self.to_spec_identity,
            "from_spec_digest": self.from_spec_digest,
            "to_spec_digest": self.to_spec_digest,
            "changed_paths": list(self.changed_paths),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class SpecSubmission:
    """Exact spec payload released after explicit confirmation and approval."""

    preview_id: str
    intent_id: str
    spec_identity: str
    spec_digest: str
    spec: Mapping[str, object]
    approval_id: str
    confirmation_id: str
    schema_version: int = SPEC_PREVIEW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SPEC_PREVIEW_SCHEMA_VERSION:
            raise SpecPreviewError("unsupported spec submission schema version")
        _identifier(self.preview_id, "preview ID")
        _identifier(self.intent_id, "intent ID")
        _identifier(self.spec_identity, "spec identity")
        _identifier(self.approval_id, "approval ID")
        if not self.spec_digest.startswith("sha256:"):
            raise SpecPreviewError("submission spec digest is invalid")
        if _digest(dict(self.spec)) != self.spec_digest:
            raise SpecPreviewError("submission spec digest does not match the exact spec")

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "optimization_spec_submission",
            "schema_version": self.schema_version,
            "preview_id": self.preview_id,
            "intent_id": self.intent_id,
            "spec_identity": self.spec_identity,
            "spec_digest": self.spec_digest,
            "spec": dict(self.spec),
            "approval_id": self.approval_id,
            "confirmation_id": self.confirmation_id,
        }

    def canonical_json(self) -> str:
        return _canonical(self.to_record())

    def submit(self, executor: Callable[[Mapping[str, object]], object]) -> object:
        """Submit only this confirmed payload to a trusted future adapter."""

        if not callable(executor):
            raise SpecPreviewError("submission executor must be callable")
        return executor(dict(self.spec))


@dataclass(frozen=True, slots=True)
class SpecPreview:
    """Canonical interpreted-spec preview with no execution authority."""

    intent_id: str
    outcome: IntentOutcome
    spec_identity: str | None
    spec_digest: str | None
    spec: Mapping[str, object] | None
    objectives: tuple[Mapping[str, object], ...]
    hard_constraints: tuple[Mapping[str, object], ...]
    preferences: tuple[Mapping[str, object], ...]
    budgets: tuple[Mapping[str, object], ...]
    allowed_operations: tuple[Mapping[str, object], ...]
    unresolved_fields: tuple[str, ...]
    diagnostics: tuple[PolicyDiagnostic, ...]
    provenance: Mapping[str, object]
    approval_required: bool
    diff: SpecDiff | None = None
    schema_version: int = SPEC_PREVIEW_SCHEMA_VERSION
    policy_decision: PolicyDecision | None = None

    def __post_init__(self) -> None:
        if self.schema_version != SPEC_PREVIEW_SCHEMA_VERSION:
            raise SpecPreviewError("unsupported spec preview schema version")
        _identifier(self.intent_id, "intent ID")
        if self.outcome is IntentOutcome.EXECUTABLE:
            if self.spec is None or self.spec_identity is None or self.spec_digest is None:
                raise SpecPreviewError("executable preview requires an exact spec identity")
            if self.unresolved_fields:
                raise SpecPreviewError("executable preview cannot retain unresolved fields")
            if _digest(dict(self.spec)) != self.spec_digest:
                raise SpecPreviewError("preview spec digest does not match the exact spec")
        elif (
            self.spec is not None
            or self.spec_identity is not None
            or self.spec_digest is not None
        ):
            raise SpecPreviewError("non-executable preview cannot expose a spec")
        if self.unresolved_fields != tuple(sorted(set(self.unresolved_fields))):
            raise SpecPreviewError("unresolved fields must be sorted and unique")
        if self.policy_decision is not None:
            if self.policy_decision.operation != "intent-compilation":
                raise SpecPreviewError("preview policy decision has the wrong operation")
            if self.outcome is IntentOutcome.EXECUTABLE and not self.policy_decision.executable:
                raise SpecPreviewError("executable preview requires an allow policy decision")

    @property
    def executable(self) -> bool:
        return self.outcome is IntentOutcome.EXECUTABLE and not self.unresolved_fields

    @property
    def preview_id(self) -> str:
        return "spec_preview_" + hashlib.sha256(
            self.canonical_json(include_identity=False).encode("utf-8")
        ).hexdigest()

    def to_record(self, *, include_identity: bool = True) -> dict[str, object]:
        record: dict[str, object] = {
            "record_type": "optimization_spec_preview",
            "schema_version": self.schema_version,
            "intent_id": self.intent_id,
            "outcome": self.outcome.value,
            "spec_identity": self.spec_identity,
            "spec_digest": self.spec_digest,
            "spec": None if self.spec is None else dict(self.spec),
            "objectives": [dict(item) for item in self.objectives],
            "hard_constraints": [dict(item) for item in self.hard_constraints],
            "preferences": [dict(item) for item in self.preferences],
            "budgets": [dict(item) for item in self.budgets],
            "allowed_operations": [dict(item) for item in self.allowed_operations],
            "unresolved_fields": list(self.unresolved_fields),
            "diagnostics": [item.to_record() for item in self.diagnostics],
            "provenance": dict(self.provenance),
            "approval": {
                "required": self.approval_required,
                "status": "pending",
            },
            "diff": None if self.diff is None else self.diff.to_record(),
        }
        if include_identity:
            record["preview_id"] = self.preview_id
        return record

    def canonical_json(self, *, include_identity: bool = True) -> str:
        return _canonical(self.to_record(include_identity=include_identity))

    def render(self) -> str:
        """Return stable pretty JSON for human preview output."""

        return json.dumps(self.to_record(), ensure_ascii=False, sort_keys=True, indent=2)

    def confirm(self, *, approval_id: str | None = None) -> SpecSubmission:
        """Release the exact spec only after explicit confirmation and approval."""

        if not self.executable or self.spec is None:
            raise SpecPreviewError(
                "execution is blocked until the preview is executable and all fields are resolved"
            )
        if self.approval_required and approval_id is None:
            raise SpecPreviewError("explicit execution approval is required for this spec")
        selected_approval = approval_id or "approval_not_required"
        _identifier(selected_approval, "approval ID")
        confirmation_id = "spec_confirmation_" + _digest(
            {
                "preview_id": self.preview_id,
                "spec_identity": self.spec_identity,
                "spec_digest": self.spec_digest,
                "approval_id": selected_approval,
            }
        )[len("sha256:") :]
        return SpecSubmission(
            self.preview_id,
            self.intent_id,
            cast(str, self.spec_identity),
            cast(str, self.spec_digest),
            dict(self.spec),
            selected_approval,
            confirmation_id,
        )

    def recompile(
        self,
        intent: IntentRecord,
        compilation: IntentCompilation | None = None,
    ) -> SpecPreview:
        """Recompile an edited intent and retain a deterministic visible diff."""

        return build_spec_preview(intent, compilation=compilation, previous=self)

    edit = recompile


def diff_spec_previews(previous: SpecPreview, current: SpecPreview) -> SpecDiff:
    """Build the visible diff used when a preview is edited or recompiled."""

    if not isinstance(previous, SpecPreview) or not isinstance(current, SpecPreview):
        raise SpecPreviewError("spec diff requires two SpecPreview values")
    left = None if previous.spec is None else dict(previous.spec)
    right = None if current.spec is None else dict(current.spec)
    paths = tuple(sorted(_changed_paths(left, right)))
    if previous.outcome is not current.outcome and not paths:
        paths = ("outcome",)
    if paths:
        kind = SpecDiffKind.MATERIAL
        reason = "interpreted OptimizationSpec changed; prior confirmation is invalid"
    elif previous.unresolved_fields != current.unresolved_fields:
        kind = SpecDiffKind.UNRESOLVED
        reason = "resolution state changed; execution remains bound to the current preview"
    else:
        kind = SpecDiffKind.NON_MATERIAL
        reason = "no material OptimizationSpec change; provenance or diagnostics changed only"
    return SpecDiff(
        kind,
        previous.spec_identity,
        current.spec_identity,
        previous.spec_digest,
        current.spec_digest,
        paths,
        reason,
    )


def build_spec_preview(
    intent: IntentRecord,
    *,
    decision: IntentPolicyDecision | None = None,
    compilation: IntentCompilation | None = None,
    previous: SpecPreview | None = None,
) -> SpecPreview:
    """Evaluate and render one exact interpreted spec without executing it."""

    if not isinstance(intent, IntentRecord):
        raise SpecPreviewError("spec preview requires an IntentRecord")
    selected = evaluate_intent_policy(intent, compilation) if decision is None else decision
    if selected.intent_id != intent.intent_id:
        raise SpecPreviewError("policy decision belongs to a different intent")
    contract = selected.contract
    spec = None if contract is None else contract.to_record()
    if spec is not None and intent.emitted_spec is not None:
        task_quality = intent.emitted_spec.get("task_quality")
        if task_quality is not None:
            spec["task_quality"] = task_quality
    objectives = () if contract is None else tuple(
        dict(item.to_record()) for item in contract.objectives
    )
    constraints = () if contract is None else tuple(
        dict(item.to_record()) for item in contract.constraints
    )
    budgets = _field_records(intent.fields, {"budget", "resource_budget"})
    allowed_operations = _field_records(
        intent.fields,
        {"allowed_operation", "allowed_operations", "operation", "operation_allowlist"},
    )
    provenance = {
        **intent.provenance.to_record(),
        "intent_id": intent.intent_id,
        "request_digest": intent.request_digest,
        "decision_id": selected.decision_id,
    }
    current = SpecPreview(
        intent.intent_id,
        selected.outcome,
        None if contract is None else contract.contract_id,
        None if spec is None else _digest(spec),
        spec,
        objectives,
        constraints,
        objectives,
        budgets,
        allowed_operations,
        _unresolved_fields(intent, selected),
        selected.diagnostics,
        provenance,
        False if contract is None else contract.approval_policy.require_execution_approval,
        policy_decision=selected.precedence,
    )
    if previous is None:
        return current
    return SpecPreview(
        current.intent_id,
        current.outcome,
        current.spec_identity,
        current.spec_digest,
        current.spec,
        current.objectives,
        current.hard_constraints,
        current.preferences,
        current.budgets,
        current.allowed_operations,
        current.unresolved_fields,
        current.diagnostics,
        current.provenance,
        current.approval_required,
        diff_spec_previews(previous, current),
        policy_decision=current.policy_decision,
    )


def preview_intent(
    intent: IntentRecord,
    *,
    compilation: IntentCompilation | None = None,
) -> SpecPreview:
    """Public descriptive alias for :func:`build_spec_preview`."""

    return build_spec_preview(intent, compilation=compilation)


__all__ = [
    "SPEC_PREVIEW_SCHEMA_VERSION",
    "SpecDiff",
    "SpecDiffKind",
    "SpecPreview",
    "SpecPreviewError",
    "SpecSubmission",
    "build_spec_preview",
    "diff_spec_previews",
    "preview_intent",
]
