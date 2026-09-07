import pytest

from modelsurgeon.search.candidate_space import (
    ArchitectureCandidateSpace,
    ArchitectureChoice,
    AxisDomain,
    CandidateSpaceConfig,
    CandidateSpaceError,
    CandidateSpaceRules,
    HardwareRule,
)
from modelsurgeon.search.deployable_state import ArchitectureAxis


def _space(*, reversed_inputs: bool = False) -> ArchitectureCandidateSpace:
    domains = (
        AxisDomain(
            ArchitectureAxis.QUERY_HEADS,
            (
                ArchitectureChoice(8, 8_000, 800, 2),
                ArchitectureChoice(4, 4_000, 400, 1),
            ),
        ),
        AxisDomain(
            ArchitectureAxis.KV_HEADS,
            (
                ArchitectureChoice(2, 2_000, 200, 1),
                ArchitectureChoice(1, 1_000, 100, 0.5),
            ),
        ),
    )
    profiles = (
        HardwareRule("cpu", max_parameter_count=6_000),
        HardwareRule("gpu", max_parameter_count=20_000),
    )
    if reversed_inputs:
        domains = tuple(reversed(domains))
        profiles = tuple(reversed(profiles))
    return ArchitectureCandidateSpace(
        "state_root",
        domains,
        profiles,
        CandidateSpaceRules(
            divisible_axes=((ArchitectureAxis.KV_HEADS, ArchitectureAxis.QUERY_HEADS),)
        ),
        CandidateSpaceConfig(seed=7, max_candidates=3, page_size=2),
    )


def test_candidate_space_is_bounded_lazy_and_resumable() -> None:
    space = _space()
    first = space.generate()
    second = space.generate(cursor=first.next_cursor)
    assert len(first.candidates) == 2
    assert second.complete is True
    assert len(first.candidates) + len(second.candidates) == 3
    assert first.space_id == second.space_id == space.space_id
    assert first.candidates[0].candidate_id != first.candidates[1].candidate_id
    assert all(candidate.parameter_count <= 20_000 for candidate in second.candidates)


def test_input_order_does_not_change_canonical_candidates_or_resume_boundaries() -> None:
    normal = _space()
    reversed_space = _space(reversed_inputs=True)
    normal_records = normal.generate().to_record()
    reversed_records = reversed_space.generate().to_record()
    assert normal.space_id == reversed_space.space_id
    assert normal_records["candidates"] == reversed_records["candidates"]


def test_static_legality_and_hardware_ceilings_are_retained() -> None:
    page = _space().generate()
    rejection_reasons = {reason for reason, _ in page.rejection_counts}
    assert "illegal_divisibility:kv_heads:query_heads" not in rejection_reasons
    assert all(
        candidate.assignments[0][0] is ArchitectureAxis.KV_HEADS
        for candidate in page.candidates
    )


def test_space_rejects_invalid_bounds_and_cursors() -> None:
    with pytest.raises(CandidateSpaceError, match=r"within 1\.\.100000"):
        CandidateSpaceConfig(seed=0, max_candidates=100_001)
    with pytest.raises(CandidateSpaceError, match="exceeds"):
        _space().generate(cursor=4)
