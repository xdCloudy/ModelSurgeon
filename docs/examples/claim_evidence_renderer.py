"""Render every typed canonical evidence-query row without laundering status."""

from modelsurgeon.conversation import EvidenceQueryResponse
from modelsurgeon.explain import render_claim_evidence


def render(canonical_response: EvidenceQueryResponse) -> str:
    """Render a response obtained from EvidenceQueryEngine.

    In an application, ``canonical_response`` should be the exact
    ``EvidenceQueryResponse`` returned by the #475 query boundary. The renderer
    rejects untyped mappings and recalculates canonical report parity.
    """

    return render_claim_evidence(canonical_response).render_text()
