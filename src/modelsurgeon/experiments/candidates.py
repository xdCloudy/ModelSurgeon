"""Canonical seeded enumeration of single-target mutation candidates."""

from __future__ import annotations

import hashlib
import heapq
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum

from modelsurgeon.experiments.identity import derive_candidate_identity
from modelsurgeon.graph import ComponentGraph, ComponentId, GraphNode
from modelsurgeon.surgery.contracts import MutationKind, MutationPrimitive, MutationRequest
from modelsurgeon.surgery.target_resolution import (
    MutationTargetResolutionError,
    MutationTargetResolver,
)

CANDIDATE_ENUMERATOR_VERSION = "1"
MAX_ENUMERATED_CANDIDATES = 100_000


class CandidateEnumerationError(ValueError):
    """Raised when candidate enumeration inputs cannot produce a safe deterministic set."""


class CandidateScope(StrEnum):
    ATTENTION_HEAD = "head"
    MLP_CHANNEL = "channel"
    COMPONENT = "component"
    TRANSFORMER_LAYER = "layer"


_ALL_SCOPES = tuple(CandidateScope)
_COMPONENT_KINDS = frozenset(
    {
        "attention",
        "projection",
        "mlp",
        "embedding",
        "normalization",
        "moe_expert",
        "moe_router",
    }
)
_SCOPE_BY_LOGICAL_KIND = {
    "attention_head": CandidateScope.ATTENTION_HEAD,
    "mlp_channel": CandidateScope.MLP_CHANNEL,
    "transformer_layer": CandidateScope.TRANSFORMER_LAYER,
}


@dataclass(frozen=True, slots=True)
class CandidateFilter:
    scopes: tuple[CandidateScope, ...] = _ALL_SCOPES
    include_kinds: tuple[str, ...] = ()
    exclude_kinds: tuple[str, ...] = ()
    include_prefixes: tuple[ComponentId, ...] = ()
    exclude_prefixes: tuple[ComponentId, ...] = ()
    layer_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.scopes or len(self.scopes) != len(set(self.scopes)):
            raise CandidateEnumerationError("candidate scopes must be non-empty and unique")
        for label, kinds in (
            ("include", self.include_kinds),
            ("exclude", self.exclude_kinds),
        ):
            if any(not kind for kind in kinds) or len(kinds) != len(set(kinds)):
                raise CandidateEnumerationError(f"{label} kinds must be non-empty and unique")
        if set(self.include_kinds) & set(self.exclude_kinds):
            raise CandidateEnumerationError("candidate include/exclude kinds cannot overlap")
        for label, prefixes in (
            ("include", self.include_prefixes),
            ("exclude", self.exclude_prefixes),
        ):
            if len(prefixes) != len(set(prefixes)):
                raise CandidateEnumerationError(f"{label} prefixes must be unique")
        if self.layer_indices != tuple(sorted(set(self.layer_indices))):
            raise CandidateEnumerationError("layer indices must be unique and canonical")
        if any(index < 0 for index in self.layer_indices):
            raise CandidateEnumerationError("layer indices cannot be negative")


@dataclass(frozen=True, slots=True)
class CandidateEnumeratorConfig:
    seed: int
    filters: CandidateFilter = CandidateFilter()
    max_candidates: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or self.seed < 0 or self.seed >= 1 << 64:
            raise CandidateEnumerationError("candidate seed must be an unsigned 64-bit integer")
        if self.max_candidates is not None and (
            isinstance(self.max_candidates, bool) or self.max_candidates <= 0
        ):
            raise CandidateEnumerationError("max_candidates must be positive when set")
        if self.max_candidates is not None and self.max_candidates > MAX_ENUMERATED_CANDIDATES:
            raise CandidateEnumerationError("max_candidates cannot exceed 100000")


@dataclass(frozen=True, slots=True)
class MutationCandidate:
    candidate_id: str
    scope: CandidateScope
    component_id: ComponentId
    node_kind: str
    layer_index: int | None
    request: MutationRequest
    affected_components: tuple[ComponentId, ...]
    constraint_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.candidate_id.startswith("cand_"):
            raise CandidateEnumerationError(
                "candidate identity must use the canonical cand_ prefix"
            )
        if self.request.targets != (self.component_id,):
            raise CandidateEnumerationError(
                "enumerated candidates require exactly one requested target"
            )
        if self.component_id not in self.affected_components:
            raise CandidateEnumerationError(
                "candidate coupling closure omitted its requested target"
            )

    @property
    def mutation_id(self) -> str:
        return self.request.mutation_id

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "scope": self.scope.value,
            "component_id": str(self.component_id),
            "node_kind": self.node_kind,
            "layer_index": self.layer_index,
            "mutation_id": self.mutation_id,
            "request": self.request.to_record(),
            "affected_components": [str(item) for item in self.affected_components],
            "constraint_ids": list(self.constraint_ids),
        }


@dataclass(frozen=True, slots=True)
class CandidateExclusionCount:
    reason: str
    count: int

    def __post_init__(self) -> None:
        if not self.reason or self.count <= 0:
            raise CandidateEnumerationError("candidate exclusion counts require reason and count")


@dataclass(frozen=True, slots=True)
class CandidateEnumerationReport:
    version: str
    seed: int
    graph_node_count: int
    eligible_count: int
    candidates: tuple[MutationCandidate, ...]
    exclusions: tuple[CandidateExclusionCount, ...]

    def __post_init__(self) -> None:
        if self.version != CANDIDATE_ENUMERATOR_VERSION:
            raise CandidateEnumerationError(
                f"unsupported candidate enumerator version {self.version}"
            )
        if self.graph_node_count < 0 or self.eligible_count < 0:
            raise CandidateEnumerationError("candidate enumeration counts cannot be negative")
        ids = tuple(item.candidate_id for item in self.candidates)
        if len(ids) != len(set(ids)):
            raise CandidateEnumerationError("enumerated candidate IDs must be unique")
        reasons = tuple(item.reason for item in self.exclusions)
        if reasons != tuple(sorted(reasons)) or len(reasons) != len(set(reasons)):
            raise CandidateEnumerationError("candidate exclusions must be unique and canonical")

    def to_record(self) -> dict[str, object]:
        return {
            "version": self.version,
            "seed": self.seed,
            "graph_node_count": self.graph_node_count,
            "eligible_count": self.eligible_count,
            "candidate_count": len(self.candidates),
            "candidates": [item.to_record() for item in self.candidates],
            "exclusions": [
                {"reason": item.reason, "count": item.count} for item in self.exclusions
            ],
        }


@dataclass(frozen=True, slots=True)
class _PreCandidate:
    scope: CandidateScope
    node: GraphNode
    layer_index: int | None
    request: MutationRequest


@dataclass(order=True, frozen=True, slots=True)
class _RankedCandidate:
    negative_rank: int
    negative_mutation: int
    candidate: _PreCandidate = field(compare=False)


def _scope_for_node(node: GraphNode) -> CandidateScope | None:
    logical = _SCOPE_BY_LOGICAL_KIND.get(node.kind)
    if logical is not None:
        return logical
    if node.kind in _COMPONENT_KINDS:
        return CandidateScope.COMPONENT
    return None


def _layer_index(node: GraphNode) -> int | None:
    value = dict(node.attributes).get("layer_index")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    segments = node.component_id.segments
    for position, segment in enumerate(segments[:-1]):
        if segment.value != "layers":
            continue
        candidate = segments[position + 1].value
        if isinstance(candidate, int) and candidate >= 0:
            return candidate
    return None


def _index_attribute(node: GraphNode, key: str) -> int | None:
    value = dict(node.attributes).get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _is_under(component_id: ComponentId, prefix: ComponentId) -> bool:
    prefix_segments = prefix.segments
    return component_id.segments[: len(prefix_segments)] == prefix_segments


def _filter_reason(
    node: GraphNode,
    scope: CandidateScope,
    layer_index: int | None,
    filters: CandidateFilter,
) -> str | None:
    if scope not in filters.scopes:
        return f"filtered-scope:{scope.value}"
    if filters.include_kinds and node.kind not in filters.include_kinds:
        return f"filtered-kind:{node.kind}"
    if node.kind in filters.exclude_kinds:
        return f"filtered-kind:{node.kind}"
    if filters.include_prefixes and not any(
        _is_under(node.component_id, prefix) for prefix in filters.include_prefixes
    ):
        return "filtered-prefix"
    if any(_is_under(node.component_id, prefix) for prefix in filters.exclude_prefixes):
        return "filtered-prefix"
    if filters.layer_indices and layer_index not in filters.layer_indices:
        return "filtered-layer"
    return None


def _request_for_node(
    node: GraphNode,
    scope: CandidateScope,
    layer_index: int | None,
) -> MutationRequest | None:
    parameters: list[tuple[str, MutationPrimitive]] = [("candidate_scope", scope.value)]
    if scope is CandidateScope.ATTENTION_HEAD:
        index = _index_attribute(node, "head_index")
        if index is None or layer_index is None:
            return None
        parameters.extend((("head_index", index), ("layer_index", layer_index)))
    elif scope is CandidateScope.MLP_CHANNEL:
        index = _index_attribute(node, "channel_index")
        if index is None or layer_index is None:
            return None
        parameters.extend((("channel_index", index), ("layer_index", layer_index)))
    elif scope is CandidateScope.TRANSFORMER_LAYER:
        if layer_index is None:
            return None
        parameters.append(("layer_index", layer_index))
    return MutationRequest(
        MutationKind.MASK,
        (node.component_id,),
        tuple(sorted(parameters)),
    )


def _rank(seed: int, request: MutationRequest) -> tuple[int, int]:
    payload = f"{seed}:{request.mutation_id}".encode("ascii")
    rank = int.from_bytes(hashlib.sha256(payload).digest(), "big")
    mutation = int(request.mutation_id, 16)
    return rank, mutation


def _ranked(seed: int, candidate: _PreCandidate) -> _RankedCandidate:
    rank, mutation = _rank(seed, candidate.request)
    return _RankedCandidate(-rank, -mutation, candidate)


def _select_bounded(
    items: list[_RankedCandidate],
    candidate: _PreCandidate,
    seed: int,
    maximum: int | None,
) -> None:
    entry = _ranked(seed, candidate)
    if maximum is None:
        items.append(entry)
        return
    if len(items) < maximum:
        heapq.heappush(items, entry)
        return
    if entry > items[0]:
        heapq.heapreplace(items, entry)


def _selected_candidates(
    selected: list[_RankedCandidate],
) -> tuple[_PreCandidate, ...]:
    ordered = sorted(
        selected,
        key=lambda entry: (-entry.negative_rank, -entry.negative_mutation),
    )
    return tuple(entry.candidate for entry in ordered)


def enumerate_mutation_candidates(
    graph: ComponentGraph,
    run_id: str,
    config: CandidateEnumeratorConfig,
) -> CandidateEnumerationReport:
    """Enumerate filtered single-target masks with deterministic seeded ordering."""

    if not run_id.startswith("run_"):
        raise CandidateEnumerationError("candidate enumeration requires a canonical run ID")
    canonical = ComponentGraph.build(graph.nodes, graph.edges, graph.constraints)
    resolver = MutationTargetResolver(canonical)
    exclusions: Counter[str] = Counter()
    selected: list[_RankedCandidate] = []
    eligible_count = 0

    for node in canonical.nodes:
        scope = _scope_for_node(node)
        if scope is None:
            exclusions[f"unsupported-kind:{node.kind}"] += 1
            continue
        layer_index = _layer_index(node)
        filtered = _filter_reason(node, scope, layer_index, config.filters)
        if filtered is not None:
            exclusions[filtered] += 1
            continue
        request = _request_for_node(node, scope, layer_index)
        if request is None:
            exclusions[f"invalid-metadata:{scope.value}"] += 1
            continue
        eligible_count += 1
        _select_bounded(
            selected,
            _PreCandidate(scope, node, layer_index, request),
            config.seed,
            config.max_candidates,
        )

    if config.max_candidates is not None and eligible_count > len(selected):
        exclusions["sampled-out"] += eligible_count - len(selected)

    candidates: list[MutationCandidate] = []
    for preliminary in _selected_candidates(selected):
        try:
            resolution = resolver.resolve(preliminary.request)
        except MutationTargetResolutionError:
            exclusions[f"planner-rejected:{preliminary.scope.value}"] += 1
            continue
        candidate_id = derive_candidate_identity(
            run_id,
            preliminary.request.mutation_id,
        ).candidate_id
        candidates.append(
            MutationCandidate(
                candidate_id,
                preliminary.scope,
                preliminary.node.component_id,
                preliminary.node.kind,
                preliminary.layer_index,
                preliminary.request,
                resolution.affected_components,
                resolution.constraint_ids,
            )
        )

    return CandidateEnumerationReport(
        CANDIDATE_ENUMERATOR_VERSION,
        config.seed,
        len(canonical.nodes),
        eligible_count,
        tuple(candidates),
        tuple(
            CandidateExclusionCount(reason, count)
            for reason, count in sorted(exclusions.items())
            if count > 0
        ),
    )
