"""Small deterministic example of a canonical conversational evidence query."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

from modelsurgeon.conversation import (
    CampaignBudget,
    CampaignEvidence,
    CampaignOutcome,
    CampaignSpec,
    EvidenceCursor,
    EvidenceQuery,
    EvidenceQueryEngine,
    EvidenceSnapshot,
    new_campaign_state,
)
from modelsurgeon.experiments.identity import canonical_identity_json


def digest(value: object) -> str:
    return (
        "sha256:" + hashlib.sha256(canonical_identity_json(value).encode()).hexdigest()
    )


def main() -> None:
    spec_payload = {"constraints": [{"metric": "quality", "minimum": 0.95}]}
    spec = CampaignSpec(
        "example_contract",
        digest(spec_payload),
        spec_payload,
        (spec_payload["constraints"][0],),  # type: ignore[arg-type]
    )
    state = new_campaign_state(
        session_id="chat_session_example",
        run_id="run_example",
        source_model_digest="sha256:" + "a" * 64,
        spec=spec,
        policy_state={"record_type": "example_policy"},
        provider_context={"provider_id": "fixture"},
        budget=CampaignBudget(30.0, 1024 * 1024, 1, 4096),
        campaign_id="campaign_example",
    )
    evidence = CampaignEvidence(
        "evidence_example",
        state.source_model_digest,
        CampaignOutcome.SUPPORTED,
        "measured fixture accepted",
        {
            "record_type": "example_measurement",
            "decision": "accepted",
            "measurements": {
                "quality": {
                    "value": 0.97,
                    "unit": "score",
                    "uncertainty": {"confidence": 0.95, "sample_count": 3},
                }
            },
        },
        artifact_digest="sha256:" + "b" * 64,
    )
    state = replace(
        state,
        evidence_cursor=EvidenceCursor(1, evidence.evidence_id),
        state_version=1,
    )
    snapshot = EvidenceSnapshot(state, (evidence,))
    query = EvidenceQuery(
        state.campaign_id,
        fields=("artifact_digest", "measurements", "outcome", "provenance_refs"),
        expected_snapshot_id=snapshot.snapshot_id,
    )
    response = EvidenceQueryEngine(snapshot).query(query)
    print(json.dumps(response.to_record(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
