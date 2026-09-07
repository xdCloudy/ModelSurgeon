"""Small deterministic example for canonical campaign recovery."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from modelsurgeon.conversation import (
    ApprovalStatus,
    CampaignApproval,
    CampaignBudget,
    CampaignSpec,
    CampaignStateStore,
    new_campaign_state,
)
from modelsurgeon.experiments.identity import canonical_identity_json


def digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_identity_json(value).encode()).hexdigest()


def main() -> None:
    payload = {"constraints": [{"metric": "quality", "minimum": 0.95}]}
    spec = CampaignSpec(
        "recovery_example",
        digest(payload),
        payload,
        tuple(payload["constraints"]),  # type: ignore[arg-type]
    )
    approval = CampaignApproval(
        ApprovalStatus.APPROVED,
        spec.spec_digest,
        approval_id="approval_example",
        recorded_by="operator_example",
        expires_at="2099-01-01T00:00:00+00:00",
        provenance={"record_type": "example_approval", "source": "trusted"},
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "campaign.sqlite3"
        with CampaignStateStore(path) as store:
            created = store.create(
                new_campaign_state(
                    session_id="chat_session_example",
                    run_id="run_example",
                    source_model_digest="sha256:" + "a" * 64,
                    spec=spec,
                    policy_state={"record_type": "example_policy"},
                    provider_context={"provider_id": "fixture", "revision": "v1"},
                    budget=CampaignBudget(30.0, 1024, 2, 4096),
                    approval=approval,
                    provenance={"record_type": "example_campaign", "source": "trusted"},
                )
            )
            paused = store.pause(
                created.campaign_id,
                created.session_id,
                operation_id="pause_example",
            )
            recovered = store.restart(
                paused.campaign_id, paused.session_id, operation_id="restart_example"
            )
            print(recovered.canonical_json())


if __name__ == "__main__":
    main()
