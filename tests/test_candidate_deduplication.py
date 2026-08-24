from __future__ import annotations

from modelsurgeon.active_learning import (
    AdapterEquivalenceDeclarations,
    candidate_equivalence_key,
    deduplicate_candidates,
)
from modelsurgeon.experiments.candidates import CandidateScope, MutationCandidate
from modelsurgeon.graph import ComponentId
from modelsurgeon.surgery.contracts import MutationKind, MutationRequest


def _candidate(
    identity: str, target: str, *, strength: int, affected: tuple[str, ...] | None = None
) -> MutationCandidate:
    component = ComponentId.parse(target)
    request = MutationRequest(
        MutationKind.MASK,
        (component,),
        (("candidate_scope", "channel"), ("strength", strength)),
    )
    return MutationCandidate(
        f"cand_{identity}",
        CandidateScope.MLP_CHANNEL,
        component,
        "mlp_channel",
        0,
        request,
        tuple(ComponentId.parse(item) for item in (affected or (target,))),
        (),
    )


def test_component_sets_are_order_insensitive_but_false_equivalence_stays_distinct() -> None:
    first = _candidate(
        "a",
        "model.layers.0.mlp.channel.0",
        strength=1,
        affected=("model.layers.0.mlp.channel.0", "model.layers.0.mlp.channel.1"),
    )
    reordered = _candidate(
        "b",
        "model.layers.0.mlp.channel.1",
        strength=1,
        affected=("model.layers.0.mlp.channel.1", "model.layers.0.mlp.channel.0"),
    )
    different_parameter = _candidate(
        "c",
        "model.layers.0.mlp.channel.0",
        strength=2,
        affected=("model.layers.0.mlp.channel.0", "model.layers.0.mlp.channel.1"),
    )

    assert candidate_equivalence_key(first) == candidate_equivalence_key(reordered)
    assert candidate_equivalence_key(first) != candidate_equivalence_key(different_parameter)
    report = deduplicate_candidates((first, reordered, different_parameter))
    assert [item.candidate_id for item in report.candidates] == ["cand_a", "cand_c"]


def test_completed_and_inflight_equivalence_classes_are_excluded() -> None:
    completed = _candidate("completed", "model.layers.0.mlp.channel.0", strength=1)
    inflight = _candidate("inflight", "model.layers.0.mlp.channel.1", strength=1)
    fresh = _candidate("fresh", "model.layers.0.mlp.channel.2", strength=1)

    report = deduplicate_candidates(
        (completed, inflight, fresh), completed=(completed,), in_flight=(inflight,)
    )

    assert report.candidates == (fresh,)
    assert [item.reason for item in report.exclusions] == [
        "already-completed",
        "already-in-flight",
    ]


def test_adapter_declared_equivalence_is_namespaced_and_explicit() -> None:
    first = _candidate("a", "model.layers.0.mlp.channel.0", strength=1)
    second = _candidate("b", "model.layers.1.mlp.channel.0", strength=1)
    declarations = AdapterEquivalenceDeclarations(
        "test-adapter", "1", {first.mutation_id: "shared", second.mutation_id: "shared"}
    )

    report = deduplicate_candidates((first, second), declarations=declarations)

    assert report.candidates == (first,)
    assert report.exclusions[0].reason == "duplicate-equivalence"
