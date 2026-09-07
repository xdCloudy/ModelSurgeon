"""Tests for deterministic, evidence-grounded negative experiment explanations."""

from __future__ import annotations

import hashlib
from dataclasses import replace

from modelsurgeon.conversation import (
    CampaignBudget,
    CampaignEvidence,
    CampaignOutcome,
    CampaignSpec,
    EvidenceCursor,
    EvidenceQuery,
    EvidenceQueryOutcome,
    EvidenceSnapshot,
    direct_evidence_report,
    new_campaign_state,
)
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.explain import (
    ExplanationCompleteness,
    NegativeEvidenceReport,
    direct_negative_evidence_report,
    explain_negative_evidence,
)

SOURCE = "sha256:" + "a" * 64
ARTIFACT = "sha256:" + "b" * 64


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_identity_json(value).encode()).hexdigest()


def _state() -> object:
    payload = {"constraints": [{"metric": "quality_retention", "minimum": 0.95}]}
    spec = CampaignSpec(
        "negative_contract_fixture",
        _digest(payload),
        payload,
        (payload["constraints"][0],),  # type: ignore[arg-type]
    )
    return new_campaign_state(
        session_id="chat_session_negative_fixture",
        run_id="run_negative_fixture",
        source_model_digest=SOURCE,
        spec=spec,
        policy_state={"record_type": "negative_fixture_policy"},
        provider_context={"provider_id": "fixture"},
        budget=CampaignBudget(60.0, 4096, 8, 8192),
        campaign_id="campaign_negative_fixture",
    )


def _records(
    index: int,
    *,
    include_evaluation: bool = True,
    include_rollback: bool = False,
) -> dict[str, object]:
    records: dict[str, object] = {
        "mutation_record": {
            "record_type": "canonical_mutation",
            "mutation_id": f"mutation_fixture_{index}",
            "source_digest": SOURCE,
        }
    }
    if include_evaluation:
        records["evaluation_record"] = {
            "record_type": "canonical_evaluation",
            "evaluation_id": f"evaluation_fixture_{index}",
            "measured": True,
            "constraint_evaluation": {
                "passed": False,
                "results": [
                    {
                        "constraint": {
                            "metric": "quality_retention",
                            "comparison": "minimum",
                            "threshold": 0.95,
                            "unit": "ratio",
                            "baseline": "immutable_source",
                        },
                        "observed": 0.91,
                        "passed": False,
                        "reason": "threshold_violation",
                    }
                ],
            },
        }
    if include_rollback:
        records["rollback_record"] = {
            "record_type": "canonical_rollback",
            "rollback_id": f"rollback_fixture_{index}",
            "kind": "rollback",
            "candidate_state_id": f"state_fixture_{index}",
            "parent_checkpoint_id": "checkpoint_source",
            "evaluation_id": f"evaluation_fixture_{index}",
            "artifact_digest": ARTIFACT,
            "source_artifact_digest": SOURCE,
        }
    return records


def _evidence(
    index: int,
    outcome: CampaignOutcome,
    *,
    decision: str,
    records: dict[str, object] | None = None,
    inconclusive: bool = False,
) -> CampaignEvidence:
    provenance: dict[str, object] = {
        "record_type": "canonical_negative_fixture",
        "run_id": "run_negative_fixture",
        "plan_id": "plan_negative_fixture",
        "decision": decision,
        "observed_at": "2026-09-07T12:00:00Z",
        "measurements": {
            "quality_retention": {
                "value": 0.91,
                "unit": "ratio",
                "uncertainty": {
                    "lower_bound": 0.89,
                    "upper_bound": 0.93,
                    "confidence": 0.95,
                    "sample_count": 4,
                },
            }
        },
    }
    if records is not None:
        provenance.update(records)
    return CampaignEvidence(
        f"evidence_fixture_{index}",
        SOURCE,
        outcome,
        f"fixture {decision} outcome",
        provenance,
        artifact_digest=ARTIFACT if decision == "accepted" else None,
        inconclusive=inconclusive,
    )


def _snapshot() -> EvidenceSnapshot:
    evidence = (
        _evidence(0, CampaignOutcome.SUPPORTED, decision="accepted", records=_records(0)),
        _evidence(1, CampaignOutcome.SUPPORTED, decision="rejected", records=_records(1)),
        _evidence(
            2,
            CampaignOutcome.FAILED,
            decision="rolled_back",
            records=_records(2, include_rollback=True),
        ),
        _evidence(
            3,
            CampaignOutcome.FAILED,
            decision="failed",
            records=_records(3, include_evaluation=False),
        ),
        _evidence(
            4,
            CampaignOutcome.UNSUPPORTED,
            decision="unsupported",
            records=_records(4, include_evaluation=False),
        ),
        _evidence(
            5,
            CampaignOutcome.UNKNOWN,
            decision="unknown",
            records=_records(5, include_evaluation=False),
        ),
        _evidence(
            6,
            CampaignOutcome.SUPPORTED,
            decision="inconclusive",
            records=_records(6, include_evaluation=False),
            inconclusive=True,
        ),
    )
    state = replace(
        _state(),
        evidence_cursor=EvidenceCursor(len(evidence), evidence[-1].evidence_id),
        state_version=len(evidence),
    )
    return EvidenceSnapshot(state, evidence)  # type: ignore[arg-type]


def test_all_outcome_classes_preserve_negative_semantics_and_metric_context() -> None:
    snapshot = _snapshot()
    report = explain_negative_evidence(
        snapshot,
        EvidenceQuery(snapshot.campaign.campaign_id),
    )
    outcomes = {item.evidence_id: item for item in report.explanations}

    assert {item.outcome.value for item in report.explanations} == {
        "accepted",
        "rejected",
        "rolled_back",
        "failed",
        "unsupported",
        "unknown",
        "inconclusive",
    }
    rejected = outcomes["evidence_fixture_1"]
    metric = rejected.metrics[0]
    assert metric.measured is True
    assert metric.value == 0.91
    assert metric.unit == "ratio"
    assert metric.direction == "minimum"
    assert metric.threshold == 0.95
    assert metric.uncertainty is not None
    assert metric.uncertainty.lower_bound == 0.89
    assert "threshold_violation" in (metric.reason or "")

    rolled_back = outcomes["evidence_fixture_2"]
    assert "not acceptance" in rolled_back.why_not_qualified
    assert rolled_back.lineage.rollback_id == "rollback_fixture_2"
    assert rolled_back.lineage.parent_ids == ("checkpoint_source",)
    assert rolled_back.canonical_records["rollback_record"] is not None
    assert "not treated as a failure" in outcomes["evidence_fixture_4"].why_not_qualified
    assert "unknown" in outcomes["evidence_fixture_5"].why_not_qualified
    assert outcomes["evidence_fixture_5"].completeness is ExplanationCompleteness.INCOMPLETE
    assert outcomes["evidence_fixture_6"].outcome.value == "inconclusive"


def test_incomplete_evidence_is_explicit_and_unknown_is_not_rejection() -> None:
    snapshot = _snapshot()
    report = explain_negative_evidence(
        snapshot,
        EvidenceQuery(
            snapshot.campaign.campaign_id,
            evidence_ids=("evidence_fixture_5",),
        ),
    )
    explanation = report.explanations[0]
    assert explanation.outcome.value == "unknown"
    assert explanation.completeness is ExplanationCompleteness.INCOMPLETE
    assert "evaluation_record" not in explanation.canonical_records
    assert "qualification" in explanation.unknown_fields
    assert "rejection" in explanation.why_not_qualified


def test_explanation_replay_and_direct_report_parity_are_exact() -> None:
    snapshot = _snapshot()
    query = EvidenceQuery(
        snapshot.campaign.campaign_id,
        outcomes=(
            EvidenceQueryOutcome.REJECTED,
            EvidenceQueryOutcome.ROLLED_BACK,
            EvidenceQueryOutcome.UNSUPPORTED,
        ),
    )
    first = explain_negative_evidence(snapshot, query)
    second = direct_negative_evidence_report(snapshot, query)
    assert first.report_id == second.report_id
    assert first.canonical_json() == second.canonical_json()
    assert first.to_text() == second.to_text()
    restored = NegativeEvidenceReport.from_record(first.to_record())
    assert restored.canonical_json() == first.canonical_json()
    assert direct_evidence_report(snapshot, query).records[0].outcome.value == "rejected"
