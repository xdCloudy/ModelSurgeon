"""Audit the claim-to-evidence renderer contract without external fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from modelsurgeon.conversation import EvidenceQueryResponse, EvidenceSnapshot
from modelsurgeon.explain import render_claim_evidence


def audit_response(response: EvidenceQueryResponse) -> dict[str, object]:
    """Return compact audit evidence for one typed response."""

    explanation = render_claim_evidence(response)
    statuses = {item.status.value for item in explanation.claims}
    return {
        "response_digest": response.response_digest,
        "explanation_digest": explanation.explanation_digest,
        "claim_count": len(explanation.claims),
        "negative_evidence_retained": len(explanation.claims)
        == len(response.records),
        "unavailable_fields_marked": all(
            not item.unavailable or item.unavailable_fields or item.missing_fields
            for item in explanation.claims
        ),
        "statuses": sorted(statuses),
        "resource_usage": explanation.resource_usage.to_record(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("response", type=Path, help="canonical EvidenceQueryResponse JSON")
    parser.add_argument("snapshot", type=Path, help="matching canonical EvidenceSnapshot JSON")
    args = parser.parse_args()
    snapshot = EvidenceSnapshot.from_record(
        json.loads(args.snapshot.read_text(encoding="utf-8"))
    )
    response = EvidenceQueryResponse.from_record(
        json.loads(args.response.read_text(encoding="utf-8")), snapshot=snapshot
    )
    print(json.dumps(audit_response(response), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
