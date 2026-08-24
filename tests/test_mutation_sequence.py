import pytest

from modelsurgeon.graph import ComponentId, ComponentIdentityMapping, ComponentIdentityRemap
from modelsurgeon.search.sequence import (
    MutationSequenceError,
    MutationSequenceState,
    SequenceMutationPlan,
)
from modelsurgeon.surgery.contracts import (
    MutationDelta,
    MutationKind,
    MutationPlan,
    MutationRequest,
)


def _id(name: str) -> ComponentId:
    return ComponentId.parse(f"model.{name}")


def _step(
    state: MutationSequenceState,
    source: ComponentId,
    targets: tuple[ComponentId, ...],
    *,
    proof: str | None = None,
    delta: int = -1,
) -> SequenceMutationPlan:
    request = MutationRequest(MutationKind.REMOVE, (source,))
    plan = MutationPlan(request, (source,), (), MutationDelta(parameters=delta))
    remap = ComponentIdentityRemap.build(
        (ComponentIdentityMapping(source, targets, "test mutation"),)
    )
    return SequenceMutationPlan(state.state_id, plan, remap, proof)


def test_sequence_composes_remaps_costs_and_never_reuses_invalidated_ids() -> None:
    a, b, c, renamed = map(_id, ("a", "b", "c", "renamed"))
    state = MutationSequenceState.initial((a, b, c))
    after_remove = state.extend(_step(state, b, (), delta=-10))
    assert after_remove.active_components == (a, c)
    assert after_remove.invalidated_components == (b,)
    assert after_remove.cumulative_delta.parameters == -10

    after_rename = after_remove.extend(_step(after_remove, c, (renamed,), delta=-2))
    assert after_rename.active_components == (a, renamed)
    assert after_rename.invalidated_components == (b, c)
    assert after_rename.root_to_current.resolve(c) == (renamed,)
    assert after_rename.cumulative_delta.parameters == -12

    with pytest.raises(MutationSequenceError, match="stale"):
        after_rename.extend(_step(after_remove, a, (a,)))
    with pytest.raises(MutationSequenceError, match="cannot be reused"):
        after_rename.extend(_step(after_rename, renamed, (b,)))


def test_only_explicit_disjoint_retained_proofs_deduplicate_equivalent_orders() -> None:
    a, b = map(_id, ("a", "b"))
    root = MutationSequenceState.initial((a, b))
    left_first = root.extend(_step(root, a, (a,), proof="independent-masks"))
    left_first = left_first.extend(_step(left_first, b, (b,), proof="independent-masks"))
    right_first = root.extend(_step(root, b, (b,), proof="independent-masks"))
    right_first = right_first.extend(_step(right_first, a, (a,), proof="independent-masks"))
    assert left_first.state_id != right_first.state_id
    assert left_first.equivalence_id == right_first.equivalence_id

    unproven = root.extend(_step(root, a, (a,)))
    unproven = unproven.extend(_step(unproven, b, (b,)))
    assert unproven.equivalence_id != left_first.equivalence_id


def test_sequence_plan_must_remap_exactly_its_affected_closure() -> None:
    a, b = map(_id, ("a", "b"))
    state = MutationSequenceState.initial((a, b))
    request = MutationRequest(MutationKind.MASK, (a,))
    plan = MutationPlan(request, (a,), (), MutationDelta())
    extra = ComponentIdentityRemap.retained((a, b))
    with pytest.raises(MutationSequenceError, match="exactly"):
        SequenceMutationPlan(state.state_id, plan, extra)
