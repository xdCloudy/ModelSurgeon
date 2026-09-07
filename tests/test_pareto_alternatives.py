"""Measured Pareto alternatives and trade-off explanations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelsurgeon.explain import (
    CandidateDisposition,
    CandidateEvidenceStatus,
    CanonicalEvidenceArchive,
    ExplorerCell,
    ExplorerCellStatus,
    FeasibilityCandidateEvidence,
    FeasibilityProvenance,
    ParetoAlternativesError,
    ParetoAlternativeStatus,
    ParetoResourceBounds,
    build_pareto_alternatives,
    generate_explorer,
    render_pareto_explanation,
    render_pareto_html,
)
from modelsurgeon.search.objective_contract import (
    ConstraintDirection,
    ContractMetric,
    HardConstraint,
    MetricObservation,
    MetricUnit,
    ObjectiveContract,
    ObjectiveDirection,
    ObjectiveNormalization,
    SoftObjective,
)

SOURCE = "sha256:" + "b" * 64


def _contract() -> ObjectiveContract:
    return ObjectiveContract(
        constraints=(
            HardConstraint(
                ContractMetric.QUALITY,
                ConstraintDirection.MINIMUM,
                0.95,
                MetricUnit.RATIO,
            ),
        ),
        objectives=(
            SoftObjective(
                ContractMetric.QUALITY,
                ObjectiveDirection.MAXIMIZE,
                MetricUnit.RATIO,
                baseline=1.0,
            ),
            SoftObjective(
                ContractMetric.LATENCY,
                ObjectiveDirection.MINIMIZE,
                MetricUnit.MILLISECONDS,
                normalization=ObjectiveNormalization.IDENTITY,
            ),
        ),
    )


def _candidate(
    name: str,
    *,
    quality: float | None = None,
    latency: float | None = None,
    quality_lower: float | None = None,
    quality_upper: float | None = None,
    status: CandidateEvidenceStatus = CandidateEvidenceStatus.MEASURED,
    disposition: CandidateDisposition = CandidateDisposition.ACCEPTED,
) -> FeasibilityCandidateEvidence:
    observations: list[MetricObservation] = []
    if quality is not None:
        observations.append(
            MetricObservation(
                ContractMetric.QUALITY,
                quality,
                MetricUnit.RATIO,
                lower=quality_lower,
                upper=quality_upper,
            )
        )
    if latency is not None:
        observations.append(
            MetricObservation(ContractMetric.LATENCY, latency, MetricUnit.MILLISECONDS)
        )
    measured = tuple(sorted(observations, key=lambda item: item.metric))
    if status is not CandidateEvidenceStatus.MEASURED:
        measured = ()
    return FeasibilityCandidateEvidence(
        f"candidate_{name}",
        f"evidence_{name}",
        status,
        SOURCE,
        measured,
        (),
        disposition,
        f"evaluation_{name}",
        {"fixture": "pareto-v1", "candidate": name},
    )


def _explanation(candidates: tuple[FeasibilityCandidateEvidence, ...]):
    contract = _contract()
    archive = CanonicalEvidenceArchive.build(contract, candidates)
    return build_pareto_alternatives(
        contract,
        archive,
        provenance=FeasibilityProvenance(
            "spec_pareto_fixture",
            SOURCE,
            archive.archive_id,
            approval_id="approval_pareto_fixture",
            approval_provenance={"operator": "test"},
        ),
    )


def test_frontier_rendering_retains_tradeoffs_and_dominated_evidence() -> None:
    result = _explanation(
        (
            _candidate("a", quality=0.96, latency=100.0),
            _candidate("b", quality=0.95, latency=80.0),
            _candidate("d", quality=0.96, latency=120.0),
        )
    )

    assert result.frontier_candidate_ids == ("candidate_a", "candidate_b")
    by_id = {item.candidate_id: item for item in result.alternatives}
    assert by_id["candidate_a"].status is ParetoAlternativeStatus.FRONTIER
    assert by_id["candidate_b"].status is ParetoAlternativeStatus.FRONTIER
    assert by_id["candidate_d"].status is ParetoAlternativeStatus.DOMINATED
    assert by_id["candidate_d"].dominated_by_candidate_ids == ("candidate_a",)
    assert by_id["candidate_d"].evidence_id == "evidence_d"

    direct = render_pareto_explanation(result, format="direct")
    html = render_pareto_html(result)
    assert isinstance(direct, dict)
    assert direct["frontier_candidate_ids"] == ["candidate_a", "candidate_b"]
    assert "Measured alternatives and trade-offs" in html
    assert "candidate_d" in html
    assert "evidence_d" in html

    golden = json.loads(
        (Path(__file__).parent / "golden" / "pareto_alternatives_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert direct["record_type"] == golden["record_type"]
    assert direct["schema_version"] == golden["schema_version"]
    assert direct["frontier_candidate_ids"] == golden["frontier_candidate_ids"]
    assert {item["candidate_id"]: item["status"] for item in direct["alternatives"]} == golden[
        "statuses"
    ]
    assert {
        item["candidate_id"]: item["dominated_by_candidate_ids"]
        for item in direct["alternatives"]
        if item["dominated_by_candidate_ids"]
    } == golden["dominated_by"]


def test_tied_and_uncertain_candidates_are_explicit_and_not_fabricated() -> None:
    result = _explanation(
        (
            _candidate("a", quality=0.96, latency=100.0),
            _candidate("tie", quality=0.96, latency=100.0),
            _candidate(
                "uncertain",
                quality=0.96,
                quality_lower=0.95,
                quality_upper=0.97,
                latency=90.0,
            ),
        )
    )

    by_id = {item.candidate_id: item for item in result.alternatives}
    assert by_id["candidate_a"].status is ParetoAlternativeStatus.TIED_FRONTIER
    assert by_id["candidate_tie"].status is ParetoAlternativeStatus.TIED_FRONTIER
    assert by_id["candidate_a"].tied_candidate_ids == ("candidate_tie",)
    assert by_id["candidate_uncertain"].uncertain
    assert by_id["candidate_uncertain"].uncertainty_metrics == ("quality",)
    assert "candidate_uncertain" in result.frontier_candidate_ids
    assert all(item.to_record()["measured"] for item in result.alternatives)


def test_hard_constraint_violations_and_terminal_statuses_remain_visible() -> None:
    result = _explanation(
        (
            _candidate(
                "bad", quality=0.90, latency=80.0, disposition=CandidateDisposition.REJECTED
            ),
            _candidate(
                "rejected_frontier",
                quality=0.97,
                latency=70.0,
                disposition=CandidateDisposition.REJECTED,
            ),
            _candidate("unsupported", status=CandidateEvidenceStatus.UNSUPPORTED),
            _candidate("failed", status=CandidateEvidenceStatus.FAILED),
            _candidate("unknown", status=CandidateEvidenceStatus.UNKNOWN),
            _candidate("inconclusive", status=CandidateEvidenceStatus.INCONCLUSIVE),
        )
    )

    bad = result.alternatives[0]
    assert bad.status is ParetoAlternativeStatus.CONSTRAINT_VIOLATION
    assert bad.constraint_violations[0].constraint.metric == ContractMetric.QUALITY
    assert bad.constraint_violations[0].gap == pytest.approx(0.05)
    assert bad.disposition is CandidateDisposition.REJECTED
    rejected = next(
        item for item in result.alternatives if item.candidate_id == "candidate_rejected_frontier"
    )
    assert rejected.status is ParetoAlternativeStatus.DISPOSITION_NOT_ACCEPTED
    assert not rejected.on_frontier
    assert {item.status for item in result.retained_evidence} == {
        CandidateEvidenceStatus.FAILED,
        CandidateEvidenceStatus.INCONCLUSIVE,
        CandidateEvidenceStatus.UNKNOWN,
        CandidateEvidenceStatus.UNSUPPORTED,
        CandidateEvidenceStatus.MEASURED,
    }
    chat = render_pareto_explanation(result, format="chat")
    assert isinstance(chat, str)
    for item in result.retained_evidence:
        assert item.evidence_id in chat
    assert "hard-constraint violations" in chat


def test_direct_and_chat_representations_replay_independently_of_arrival_order() -> None:
    candidates = (
        _candidate("z", quality=0.95, latency=80.0),
        _candidate("a", quality=0.95, latency=80.0),
        _candidate("dominated", quality=0.95, latency=90.0),
    )
    first = _explanation(candidates)
    second = _explanation(tuple(reversed(candidates)))

    assert first.to_record() == second.to_record()
    assert first.chat_text() == second.chat_text()
    assert json.loads(json.dumps(first.to_record(), sort_keys=True)) == first.to_record()
    assert first.frontier_candidate_ids == ("candidate_a", "candidate_z")
    assert "candidate_dominated" in first.chat_text()


def test_existing_explorer_can_embed_the_same_canonical_pareto_projection() -> None:
    result = _explanation(
        (
            _candidate("a", quality=0.96, latency=100.0),
            _candidate("b", quality=0.95, latency=80.0),
        )
    )
    explorer = generate_explorer(
        (
            ExplorerCell(
                "cell-a",
                ExplorerCellStatus.SUPPORTED,
                "evidence_a",
                (("cost", 1.0), ("quality", 0.96)),
            ),
        ),
        pareto_explanation=result,
    )
    assert explorer.data["pareto_explanation"] == result.to_record()
    assert "Measured Pareto alternatives" in explorer.html_text
    assert "candidate_a" in explorer.html_text


def test_bounds_fail_closed_without_truncating_evidence() -> None:
    candidate = _candidate("one", quality=0.96, latency=10.0)
    with pytest.raises(ParetoAlternativesError, match="output bound"):
        contract = _contract()
        archive = CanonicalEvidenceArchive.build(contract, (candidate,))
        build_pareto_alternatives(
            contract,
            archive,
            provenance=FeasibilityProvenance("spec", SOURCE, archive.archive_id),
            resource_bounds=ParetoResourceBounds(max_output_bytes=100),
        )
