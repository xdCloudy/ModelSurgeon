"""Graph-identity-safe sequences of compatible mutation plans."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.graph import (
    ComponentId,
    ComponentIdentityMapping,
    ComponentIdentityRemap,
)
from modelsurgeon.surgery.contracts import MutationDelta, MutationPlan

MUTATION_SEQUENCE_SCHEMA_VERSION = 1


class MutationSequenceError(ValueError):
    """Raised when a sequence uses stale, conflicting, or reused identities."""


def _delta(left: MutationDelta, right: MutationDelta) -> MutationDelta:
    return MutationDelta(
        left.parameters + right.parameters,
        left.flops + right.flops,
        left.memory_bytes + right.memory_bytes,
        left.storage_bytes + right.storage_bytes,
    )


@dataclass(frozen=True, slots=True)
class SequenceMutationPlan:
    """A graph-valid plan compiled specifically for one sequence state."""

    source_state_id: str
    plan: MutationPlan
    identity_remap: ComponentIdentityRemap
    commutativity_proof: str | None = None

    def __post_init__(self) -> None:
        if not self.source_state_id.startswith("state_"):
            raise MutationSequenceError("sequence plans require their source state ID")
        if self.commutativity_proof is not None and not self.commutativity_proof.strip():
            raise MutationSequenceError("commutativity proof names cannot be blank")
        sources = {mapping.source for mapping in self.identity_remap.mappings}
        if sources != set(self.plan.affected_components):
            raise MutationSequenceError(
                "sequence identity remap must cover exactly the affected components"
            )

    @property
    def mutation_id(self) -> str:
        return self.plan.request.mutation_id

    @property
    def retained_only(self) -> bool:
        return all(mapping.targets == (mapping.source,) for mapping in self.identity_remap.mappings)

    def to_record(self) -> dict[str, object]:
        return {
            "source_state_id": self.source_state_id,
            "mutation_id": self.mutation_id,
            "affected_components": [str(item) for item in self.plan.affected_components],
            "expected_delta": {
                "parameters": self.plan.expected_delta.parameters,
                "flops": self.plan.expected_delta.flops,
                "memory_bytes": self.plan.expected_delta.memory_bytes,
                "storage_bytes": self.plan.expected_delta.storage_bytes,
            },
            "identity_remap": self.identity_remap.to_record(),
            "commutativity_proof": self.commutativity_proof,
        }


@dataclass(frozen=True, slots=True)
class MutationSequenceState:
    root_components: tuple[ComponentId, ...]
    active_components: tuple[ComponentId, ...]
    invalidated_components: tuple[ComponentId, ...]
    root_to_current: ComponentIdentityRemap
    steps: tuple[SequenceMutationPlan, ...]
    cumulative_delta: MutationDelta
    schema_version: int = MUTATION_SEQUENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MUTATION_SEQUENCE_SCHEMA_VERSION:
            raise MutationSequenceError("unsupported mutation sequence schema")
        for name, values in (
            ("root", self.root_components),
            ("active", self.active_components),
            ("invalidated", self.invalidated_components),
        ):
            if values != tuple(sorted(set(values))):
                raise MutationSequenceError(f"{name} component identities must be canonical")
        if not self.root_components or not self.active_components:
            raise MutationSequenceError("mutation sequences require non-empty model states")
        if set(self.active_components) & set(self.invalidated_components):
            raise MutationSequenceError(
                "active and invalidated component identities must be disjoint"
            )
        if {mapping.source for mapping in self.root_to_current.mappings} != set(
            self.root_components
        ):
            raise MutationSequenceError("root remap must cover every original component")

    @classmethod
    def initial(cls, components: tuple[ComponentId, ...]) -> MutationSequenceState:
        canonical = tuple(sorted(set(components)))
        if canonical != components:
            raise MutationSequenceError("initial components must be unique and canonical")
        return cls(
            canonical,
            canonical,
            (),
            ComponentIdentityRemap.retained(canonical, reason="sequence root"),
            (),
            MutationDelta(),
        )

    @property
    def state_id(self) -> str:
        payload = canonical_identity_json(self.to_record()).encode()
        return f"state_{hashlib.sha256(payload).hexdigest()}"

    @property
    def equivalence_id(self) -> str:
        mutation_ids = [step.mutation_id for step in self.steps]
        commutative = self._all_steps_proven_commutative()
        if commutative:
            mutation_ids.sort()
        payload = {
            "schema_version": self.schema_version,
            "root_components": [str(item) for item in self.root_components],
            "equivalence_rule": (
                f"commutative:{self.steps[0].commutativity_proof}" if commutative else "ordered"
            ),
            "mutation_ids": mutation_ids,
        }
        return f"sequence_{hashlib.sha256(canonical_identity_json(payload).encode()).hexdigest()}"

    def _all_steps_proven_commutative(self) -> bool:
        if len(self.steps) < 2:
            return False
        proof = self.steps[0].commutativity_proof
        if proof is None or any(
            step.commutativity_proof != proof or not step.retained_only for step in self.steps
        ):
            return False
        affected = [set(step.plan.affected_components) for step in self.steps]
        return all(
            affected[left].isdisjoint(affected[right])
            for left in range(len(affected))
            for right in range(left + 1, len(affected))
        )

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "root_components": [str(item) for item in self.root_components],
            "active_components": [str(item) for item in self.active_components],
            "invalidated_components": [str(item) for item in self.invalidated_components],
            "root_to_current": self.root_to_current.to_record(),
            "steps": [step.to_record() for step in self.steps],
            "cumulative_delta": {
                "parameters": self.cumulative_delta.parameters,
                "flops": self.cumulative_delta.flops,
                "memory_bytes": self.cumulative_delta.memory_bytes,
                "storage_bytes": self.cumulative_delta.storage_bytes,
            },
        }

    def extend(self, step: SequenceMutationPlan) -> MutationSequenceState:
        """Extend this exact state, rejecting stale or historically reused IDs."""

        if step.source_state_id != self.state_id:
            raise MutationSequenceError("sequence mutation plan was compiled for a stale state")
        active = set(self.active_components)
        if not set(step.plan.affected_components) <= active:
            raise MutationSequenceError("mutation plan affects inactive or unknown components")
        explicit = {mapping.source: mapping for mapping in step.identity_remap.mappings}
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
            raise MutationSequenceError(
                "invalidated component identities cannot be reused: "
                + ", ".join(str(item) for item in sorted(reused))
            )
        new_active = tuple(sorted(targets))
        if not new_active:
            raise MutationSequenceError("a mutation sequence cannot remove the entire model state")
        newly_invalidated = active - targets
        invalidated = tuple(sorted(set(self.invalidated_components) | newly_invalidated))
        return MutationSequenceState(
            self.root_components,
            new_active,
            invalidated,
            self.root_to_current.compose(expanded),
            (*self.steps, step),
            _delta(self.cumulative_delta, step.plan.expected_delta),
        )
