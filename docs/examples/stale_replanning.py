"""Small deterministic example for stale-context campaign replanning."""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import replace
from pathlib import Path

from modelsurgeon.conversation import (
    CampaignBudget,
    CampaignSpec,
    CampaignStateStore,
    apply_replan,
    build_replan_approval_request,
    build_replan_proposal,
    context_from_campaign,
    new_campaign_state,
)
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.experiments.optimization_package import ApprovalDecision, ApprovalDecisionKind


def digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_identity_json(value).encode()).hexdigest()


def main() -> None:
    original_payload = {"constraints": [{"metric": "quality", "minimum": 0.95}]}
    next_payload = {"constraints": [{"metric": "quality", "minimum": 0.97}]}
    original_spec = CampaignSpec(
        "contract_quality",
        digest(original_payload),
        original_payload,
        tuple(original_payload["constraints"]),
    )
    next_spec = CampaignSpec(
        "contract_quality_v2",
        digest(next_payload),
        next_payload,
        tuple(next_payload["constraints"]),
    )
    old_plan = {"plan_id": "plan_old", "objective": {"quality": 0.95}}
    new_plan = {"plan_id": "plan_new", "objective": {"quality": 0.97}}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "campaign.sqlite3"
        with CampaignStateStore(path) as store:
            initial = new_campaign_state(
                session_id="chat_session_fixture",
                run_id="run_fixture",
                source_model_digest="sha256:" + "a" * 64,
                spec=original_spec,
                policy_state={"record_type": "example_policy", "plan_version": 1},
                provider_context={"provider_id": "fixture", "revision": "v1"},
                budget=CampaignBudget(60.0, 1024, 4, 4096),
                provenance={"record_type": "example_campaign"},
            )
            campaign = store.create(initial)
            context = context_from_campaign(
                campaign, old_plan, (), transaction_id="tool_transaction_fixture"
            )
            proposal = build_replan_proposal(
                store,
                campaign_id=campaign.campaign_id,
                expected_context=context,
                current_plan=old_plan,
                candidate_plan=new_plan,
                candidate_spec=next_spec,
                candidate_provider_context=campaign.provider_context,
                candidate_budget=campaign.budget,
                transaction_id="tool_transaction_fixture",
            )
            request = build_replan_approval_request(
                proposal,
                requested_at="2026-09-07T10:00:00Z",
                expires_at="2030-01-01T00:00:00Z",
                operator_id="operator_fixture",
            )
            decision = ApprovalDecision(
                request.request_id,
                request.code,
                request.plan_id,
                request.plan_digest,
                request.diff_id,
                ApprovalDecisionKind.APPROVED,
                "2026-09-07T10:01:00Z",
                "operator_fixture",
                "approved after review",
            )
            result = apply_replan(
                store,
                replace(proposal, approval_request=request),
                candidate_spec=next_spec,
                candidate_policy_state={"record_type": "example_policy"},
                candidate_provider_context=campaign.provider_context,
                candidate_budget=campaign.budget,
                approval_decision=decision,
            )
            print(result.to_record()["child_campaign_id"])


if __name__ == "__main__":
    main()
