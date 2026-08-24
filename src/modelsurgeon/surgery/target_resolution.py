"""Deterministic mutation target and structural coupling closure."""

from __future__ import annotations

from dataclasses import dataclass

from modelsurgeon.graph import (
    ComponentGraph,
    ComponentId,
    EdgeKind,
    GraphEdge,
    GraphValidationError,
    MutationConstraint,
    validate_component_graph,
)
from modelsurgeon.surgery.contracts import (
    MutationContractError,
    MutationDelta,
    MutationPlan,
    MutationPrecondition,
    MutationRequest,
)


class MutationTargetResolutionError(MutationContractError):
    """Raised when requested graph targets cannot form one complete plan."""


@dataclass(frozen=True, slots=True)
class ResolvedMutationTarget:
    component_id: ComponentId
    requested: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.reasons or tuple(sorted(set(self.reasons))) != self.reasons:
            raise MutationTargetResolutionError(
                "resolved target reasons must be non-empty, unique, and canonical"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "component_id": str(self.component_id),
            "requested": self.requested,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class ResolvedMutationTargets:
    request: MutationRequest
    targets: tuple[ResolvedMutationTarget, ...]
    constraint_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        ids = tuple(item.component_id for item in self.targets)
        if not ids or ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise MutationTargetResolutionError(
                "resolved targets must be non-empty, unique, and canonical"
            )
        if not set(self.request.targets).issubset(ids):
            raise MutationTargetResolutionError("resolved closure omitted a requested target")
        if self.constraint_ids != tuple(sorted(set(self.constraint_ids))):
            raise MutationTargetResolutionError("constraint IDs must be unique and canonical")

    @property
    def affected_components(self) -> tuple[ComponentId, ...]:
        return tuple(item.component_id for item in self.targets)

    def to_plan(
        self,
        *,
        preconditions: tuple[MutationPrecondition, ...],
        expected_delta: MutationDelta,
    ) -> MutationPlan:
        return MutationPlan(
            self.request,
            self.affected_components,
            preconditions,
            expected_delta,
        )

    def to_record(self) -> dict[str, object]:
        return {
            "mutation_id": self.request.mutation_id,
            "targets": [target.to_record() for target in self.targets],
            "constraint_ids": list(self.constraint_ids),
        }


class MutationTargetResolver:
    """Validate and index one immutable graph for repeated target resolution."""

    def __init__(self, graph: ComponentGraph) -> None:
        try:
            validate_component_graph(graph).raise_for_errors()
        except GraphValidationError as error:
            raise MutationTargetResolutionError(
                f"component graph is invalid for mutation planning: {error}"
            ) from error
        self._known = frozenset(node.component_id for node in graph.nodes)
        constraints: dict[ComponentId, list[MutationConstraint]] = {}
        for constraint in graph.constraints:
            for member in constraint.members:
                constraints.setdefault(member, []).append(constraint)
        self._constraints_by_member = {
            member: tuple(items) for member, items in constraints.items()
        }
        coupled: dict[ComponentId, list[GraphEdge]] = {}
        for edge in graph.edges:
            if edge.kind is not EdgeKind.COUPLED:
                continue
            coupled.setdefault(edge.source, []).append(edge)
            coupled.setdefault(edge.target, []).append(edge)
        self._coupled_by_member = {member: tuple(items) for member, items in coupled.items()}

    def resolve(self, request: MutationRequest) -> ResolvedMutationTargets:
        missing = tuple(sorted(set(request.targets) - self._known))
        if missing:
            raise MutationTargetResolutionError(
                "requested mutation targets are absent from the component graph: "
                + ", ".join(map(str, missing))
            )

        closure = set(request.targets)
        reasons: dict[ComponentId, set[str]] = {
            target: {f"requested:{target}"} for target in request.targets
        }
        activated_constraints: set[str] = set()
        activated_edges: set[tuple[ComponentId, ComponentId]] = set()
        pending = list(request.targets)
        cursor = 0
        while cursor < len(pending):
            component = pending[cursor]
            cursor += 1
            for constraint in self._constraints_by_member.get(component, ()):
                if constraint.constraint_id in activated_constraints:
                    continue
                activated_constraints.add(constraint.constraint_id)
                reason = f"constraint:{constraint.constraint_id}:{constraint.kind.value}"
                for member in constraint.members:
                    if member not in closure:
                        closure.add(member)
                        pending.append(member)
                    reasons.setdefault(member, set()).add(reason)
            for edge in self._coupled_by_member.get(component, ()):
                key = (edge.source, edge.target)
                if key in activated_edges:
                    continue
                activated_edges.add(key)
                reason = f"coupled:{edge.source}:{edge.target}"
                for endpoint in key:
                    if endpoint not in closure:
                        closure.add(endpoint)
                        pending.append(endpoint)
                    reasons.setdefault(endpoint, set()).add(reason)

        targets = tuple(
            ResolvedMutationTarget(
                component_id,
                component_id in request.targets,
                tuple(sorted(reasons[component_id])),
            )
            for component_id in sorted(closure)
        )
        return ResolvedMutationTargets(
            request,
            targets,
            tuple(sorted(activated_constraints)),
        )


def resolve_mutation_targets(
    request: MutationRequest,
    graph: ComponentGraph,
) -> ResolvedMutationTargets:
    """Resolve one request, validating and indexing the graph for this call."""

    return MutationTargetResolver(graph).resolve(request)
