"""Fail-closed compilation of mixed architecture mutation sequences."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum

from modelsurgeon.graph import ComponentId, ComponentIdentityMapping, ComponentIdentityRemap
from modelsurgeon.surgery.alignment_rules import AlignmentDecision
from modelsurgeon.surgery.contracts import MutationDelta, MutationPlan

from .deployable_state import ArchitectureAxis, DeployableArchitectureState

MULTI_AXIS_SEQUENCE_SCHEMA_VERSION = 1


class MultiAxisCompilationError(ValueError):
    """Raised when a mixed sequence would violate a current-state invariant."""


class MultiAxisCompilationOutcome(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


_ALIGNMENT_REQUIRED = frozenset(
    {
        ArchitectureAxis.DEPTH,
        ArchitectureAxis.LAYER_WIDTHS,
        ArchitectureAxis.QUERY_HEADS,
        ArchitectureAxis.KV_HEADS,
        ArchitectureAxis.HIDDEN_SIZE,
        ArchitectureAxis.EMBEDDING_SIZE,
    }
)


def _delta(left: MutationDelta, right: MutationDelta) -> MutationDelta:
    return MutationDelta(
        left.parameters + right.parameters,
        left.flops + right.flops,
        left.memory_bytes + right.memory_bytes,
        left.storage_bytes + right.storage_bytes,
    )


def _axis_value(state: DeployableArchitectureState, axis: ArchitectureAxis) -> object:
    return {
        ArchitectureAxis.DEPTH: state.depth,
        ArchitectureAxis.LAYER_WIDTHS: state.layer_widths,
        ArchitectureAxis.QUERY_HEADS: state.query_heads,
        ArchitectureAxis.KV_HEADS: state.kv_heads,
        ArchitectureAxis.HIDDEN_SIZE: state.hidden_size,
        ArchitectureAxis.EMBEDDING_SIZE: state.embedding_size,
        ArchitectureAxis.LOW_RANK_FACTORS: state.low_rank_factors,
        ArchitectureAxis.SPARSITY: state.sparsity,
        ArchitectureAxis.QUANTIZATION: state.quantization,
        ArchitectureAxis.PLACEMENT: state.placement,
    }[axis]


@dataclass(frozen=True, slots=True)
class ArchitectureMutationRequest:
    """One requested axis transition with its already-compiled physical plan."""

    axis: ArchitectureAxis
    source_state_id: str
    target_state: DeployableArchitectureState
    plan: MutationPlan
    identity_remap: ComponentIdentityRemap
    alignment: AlignmentDecision | None = None

    def __post_init__(self) -> None:
        if not self.source_state_id.startswith("state_"):
            raise MultiAxisCompilationError("mutation requests require a source state ID")
        sources = {mapping.source for mapping in self.identity_remap.mappings}
        if sources != set(self.plan.affected_components):
            raise MultiAxisCompilationError(
                "identity remap must cover exactly the physical plan affected components"
            )
        if self.alignment is not None and not self.alignment.reason.strip():
            raise MultiAxisCompilationError("alignment decisions require a reason")

    @property
    def mutation_id(self) -> str:
        return self.plan.request.mutation_id


@dataclass(frozen=True, slots=True)
class CompiledArchitectureStep:
    sequence_index: int
    axis: ArchitectureAxis
    mutation_id: str
    source_state_id: str
    target_state_id: str
    plan: MutationPlan
    identity_remap: ComponentIdentityRemap
    alignment: AlignmentDecision | None
    cumulative_delta: MutationDelta

    def to_record(self) -> dict[str, object]:
        return {
            "sequence_index": self.sequence_index,
            "axis": self.axis.value,
            "mutation_id": self.mutation_id,
            "source_state_id": self.source_state_id,
            "target_state_id": self.target_state_id,
            "expected_delta": {
                "parameters": self.plan.expected_delta.parameters,
                "flops": self.plan.expected_delta.flops,
                "memory_bytes": self.plan.expected_delta.memory_bytes,
                "storage_bytes": self.plan.expected_delta.storage_bytes,
            },
            "identity_remap": self.identity_remap.to_record(),
            "alignment": None if self.alignment is None else self.alignment.to_record(),
            "cumulative_delta": {
                "parameters": self.cumulative_delta.parameters,
                "flops": self.cumulative_delta.flops,
                "memory_bytes": self.cumulative_delta.memory_bytes,
                "storage_bytes": self.cumulative_delta.storage_bytes,
            },
        }


@dataclass(frozen=True, slots=True)
class CompiledArchitectureSequence:
    """Current sequence state with active/invalidated identities and reconciled deltas."""

    root_state: DeployableArchitectureState
    current_state: DeployableArchitectureState
    root_components: tuple[ComponentId, ...]
    active_components: tuple[ComponentId, ...]
    invalidated_components: tuple[ComponentId, ...]
    steps: tuple[CompiledArchitectureStep, ...]
    cumulative_delta: MutationDelta
    schema_version: int = MULTI_AXIS_SEQUENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MULTI_AXIS_SEQUENCE_SCHEMA_VERSION:
            raise MultiAxisCompilationError("unsupported multi-axis sequence schema")
        for name, values in (
            ("root", self.root_components),
            ("active", self.active_components),
            ("invalidated", self.invalidated_components),
        ):
            if values != tuple(sorted(set(values))):
                raise MultiAxisCompilationError(f"{name} identities must be canonical")
        if not self.root_components or not self.active_components:
            raise MultiAxisCompilationError("sequences require non-empty component states")
        if set(self.active_components) & set(self.invalidated_components):
            raise MultiAxisCompilationError("active and invalidated identities must be disjoint")
        if self.current_state.mutation_order != tuple(step.mutation_id for step in self.steps):
            raise MultiAxisCompilationError("state mutation order does not match compiled steps")
        if self.root_state.state_id == self.current_state.state_id and self.steps:
            raise MultiAxisCompilationError("non-empty sequence did not change architecture state")

    @classmethod
    def initial(
        cls,
        root_state: DeployableArchitectureState,
        root_components: tuple[ComponentId, ...],
    ) -> CompiledArchitectureSequence:
        canonical = tuple(sorted(set(root_components)))
        if not canonical or canonical != root_components:
            raise MultiAxisCompilationError("root components must be non-empty and canonical")
        return cls(root_state, root_state, canonical, canonical, (), (), MutationDelta())

    @property
    def sequence_id(self) -> str:
        digest = hashlib.sha256(self._identity_record_bytes()).hexdigest()
        return f"architecture_sequence_{digest}"

    def _identity_record_bytes(self) -> bytes:
        import json

        record = {
            "schema_version": self.schema_version,
            "root_state_id": self.root_state.state_id,
            "current_state_id": self.current_state.state_id,
            "root_components": [str(item) for item in self.root_components],
            "active_components": [str(item) for item in self.active_components],
            "invalidated_components": [str(item) for item in self.invalidated_components],
            "steps": [item.to_record() for item in self.steps],
            "cumulative_delta": self._delta_record(),
        }
        return json.dumps(record, sort_keys=True, separators=(",", ":")).encode()

    def _delta_record(self) -> dict[str, int]:
        return {
            "parameters": self.cumulative_delta.parameters,
            "flops": self.cumulative_delta.flops,
            "memory_bytes": self.cumulative_delta.memory_bytes,
            "storage_bytes": self.cumulative_delta.storage_bytes,
        }

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "sequence_id": self.sequence_id,
            "root_state": self.root_state.to_record(),
            "current_state": self.current_state.to_record(),
            "root_components": [str(item) for item in self.root_components],
            "active_components": [str(item) for item in self.active_components],
            "invalidated_components": [str(item) for item in self.invalidated_components],
            "steps": [item.to_record() for item in self.steps],
            "cumulative_delta": self._delta_record(),
        }

    def extend(self, request: ArchitectureMutationRequest) -> CompiledArchitectureSequence:
        """Append one step only after revalidating current state and all identities."""

        if request.source_state_id != self.current_state.state_id:
            raise MultiAxisCompilationError("mutation was compiled for a stale architecture state")
        if request.mutation_id in {step.mutation_id for step in self.steps}:
            raise MultiAxisCompilationError("mutation identity was already used in this sequence")
        if request.target_state.model_family != self.current_state.model_family:
            raise MultiAxisCompilationError("mixed sequences cannot change model family")
        if request.target_state.model_revision != self.current_state.model_revision:
            raise MultiAxisCompilationError("mixed sequences cannot change model revision")
        expected_order = (*self.current_state.mutation_order, request.mutation_id)
        if request.target_state.mutation_order != expected_order:
            raise MultiAxisCompilationError("target state mutation order is stale or incomplete")
        changed = tuple(
            axis
            for axis in ArchitectureAxis
            if _axis_value(self.current_state, axis) != _axis_value(request.target_state, axis)
        )
        if changed != (request.axis,):
            raise MultiAxisCompilationError(
                "one mutation must change exactly its declared axis; "
                + ", ".join(axis.value for axis in changed)
            )
        if request.target_state.axis_status(request.axis).value in {"unknown", "unsupported"}:
            raise MultiAxisCompilationError(
                f"requested axis {request.axis.value} is unknown or unsupported"
            )
        affected = set(request.plan.affected_components)
        active = set(self.active_components)
        if not affected <= active:
            raise MultiAxisCompilationError("mutation plan reuses removed or unknown identities")
        if request.axis in _ALIGNMENT_REQUIRED and (
            request.alignment is None or request.alignment.legal is not True
        ):
            raise MultiAxisCompilationError(
                f"{request.axis.value} requires a legal hardware alignment decision"
            )
        explicit = {mapping.source: mapping for mapping in request.identity_remap.mappings}
        expanded = ComponentIdentityRemap.build(
            tuple(
                explicit.get(
                    component,
                    ComponentIdentityMapping(
                        component, (component,), "unaffected sequence component"
                    ),
                )
                for component in self.active_components
            )
        )
        targets = {target for mapping in expanded.mappings for target in mapping.targets}
        reused = targets & set(self.invalidated_components)
        if reused:
            raise MultiAxisCompilationError(
                "mutation targets reuse invalidated component identities"
            )
        if not targets:
            raise MultiAxisCompilationError("a sequence cannot remove every active component")
        if self.root_state.parameter_count is None or request.target_state.parameter_count is None:
            raise MultiAxisCompilationError(
                "parameter counts are required for sequence reconciliation"
            )
        if self.root_state.storage_bytes is None or request.target_state.storage_bytes is None:
            raise MultiAxisCompilationError(
                "storage bytes are required for sequence reconciliation"
            )
        cumulative = _delta(self.cumulative_delta, request.plan.expected_delta)
        if (
            request.target_state.parameter_count
            != self.root_state.parameter_count + cumulative.parameters
        ):
            raise MultiAxisCompilationError(
                "final parameter count does not reconcile cumulative deltas"
            )
        if (
            request.target_state.storage_bytes
            != self.root_state.storage_bytes + cumulative.storage_bytes
        ):
            raise MultiAxisCompilationError(
                "final storage bytes do not reconcile cumulative deltas"
            )
        step = CompiledArchitectureStep(
            len(self.steps),
            request.axis,
            request.mutation_id,
            self.current_state.state_id,
            request.target_state.state_id,
            request.plan,
            expanded,
            request.alignment,
            cumulative,
        )
        return CompiledArchitectureSequence(
            self.root_state,
            request.target_state,
            self.root_components,
            tuple(sorted(targets)),
            tuple(sorted(set(self.invalidated_components) | (active - targets))),
            (*self.steps, step),
            cumulative,
        )


@dataclass(frozen=True, slots=True)
class ArchitectureSequenceCompilation:
    outcome: MultiAxisCompilationOutcome
    sequence: CompiledArchitectureSequence | None
    rejected_index: int | None
    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise MultiAxisCompilationError("compilation results require a reason")
        if self.outcome is MultiAxisCompilationOutcome.ACCEPTED and self.sequence is None:
            raise MultiAxisCompilationError("accepted compilation requires a sequence")
        if self.outcome is not MultiAxisCompilationOutcome.ACCEPTED and self.sequence is not None:
            raise MultiAxisCompilationError("rejected compilation cannot publish a sequence")

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": MULTI_AXIS_SEQUENCE_SCHEMA_VERSION,
            "outcome": self.outcome.value,
            "sequence": None if self.sequence is None else self.sequence.to_record(),
            "rejected_index": self.rejected_index,
            "reason": self.reason,
        }


def compile_multi_axis_sequence(
    root_state: DeployableArchitectureState,
    root_components: tuple[ComponentId, ...],
    requests: tuple[ArchitectureMutationRequest, ...],
) -> ArchitectureSequenceCompilation:
    """Compile a sequence atomically, returning the first named rejection."""

    if not requests:
        return ArchitectureSequenceCompilation(
            MultiAxisCompilationOutcome.UNKNOWN,
            None,
            None,
            "a multi-axis sequence requires at least one mutation",
        )
    try:
        sequence = CompiledArchitectureSequence.initial(root_state, root_components)
        for index, request in enumerate(requests):
            try:
                sequence = sequence.extend(request)
            except MultiAxisCompilationError as error:
                return ArchitectureSequenceCompilation(
                    MultiAxisCompilationOutcome.REJECTED,
                    None,
                    index,
                    str(error),
                )
        return ArchitectureSequenceCompilation(
            MultiAxisCompilationOutcome.ACCEPTED,
            sequence,
            None,
            "all mutation steps reconcile current state, identity, delta, and alignment",
        )
    except MultiAxisCompilationError as error:
        return ArchitectureSequenceCompilation(
            MultiAxisCompilationOutcome.UNKNOWN,
            None,
            None,
            str(error),
        )
