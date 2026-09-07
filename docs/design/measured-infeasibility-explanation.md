# Measured feasibility and infeasibility explanation

The feasibility explanation is a read-only projection over a canonical
candidate-evidence archive. It evaluates the exact hard constraints in the
`ObjectiveContract`; it never lowers a threshold, substitutes a prediction
for a measurement, or removes a rejected or rolled-back result.

## Evidence semantics

Each candidate has a stable candidate ID, evidence ID, source-model digest,
evaluation ID, provenance, optional resource usage, and one explicit evidence
status. Only `measured` candidates can be evaluated against hard constraints.
Predicted, unsupported, failed, unknown, and inconclusive cells remain in the
archive and produce a distinct result instead of an infeasibility claim.

For a measured observation with an interval, minimum constraints use the lower
bound and maximum constraints use the upper bound. Thus uncertainty can make a
nominally passing point a measured near miss. A candidate with an absent hard
metric is incomplete, not infeasible.

The result is `infeasible` only when every complete measured candidate violates
at least one declared hard constraint. It includes the unmet constraints and
deterministically ordered closest measured candidates. If no complete
measured candidate supports that statement, the result is `missing_evidence`,
`predicted_only`, `unsupported`, `failed`, `unknown`, or `inconclusive` as
appropriate.

Rejected and rolled-back measurements are retained in closest-candidate
explanations. Their disposition is evidence context, not permission to treat
the candidate as accepted. An explicit objective amendment is emitted as a
next action only with `requires_approval=true`; the original contract is
unchanged.

## Bounded, replayable API

`build_feasibility_explanation` accepts either a sequence of typed candidate
records or a `CanonicalEvidenceArchive`. Candidates are sorted by ID, near
misses by conservative normalized distance and ID, and actions by action kind.
The result records contract identity, hard constraints, source-model digest,
archive/evidence IDs, approval provenance, resource bounds, usage, and a
content-addressed result ID. Re-running with the same canonical evidence and
provenance produces the same result JSON.

```python
from modelsurgeon.explain import (
    CandidateEvidenceStatus,
    CanonicalEvidenceArchive,
    FeasibilityCandidateEvidence,
    FeasibilityProvenance,
    build_feasibility_explanation,
)

archive = CanonicalEvidenceArchive.build(contract, (candidate_a, candidate_b))
provenance = FeasibilityProvenance(
    "spec_2026_09_07",
    source_model_digest,
    archive.archive_id,
    approval_id="approval_123",
    approval_provenance={"operator": "researcher", "scope": "evaluation"},
)
result = build_feasibility_explanation(
    contract,
    archive,
    provenance=provenance,
)
print(result.outcome.value, result.closest_candidates)
```

The API is intentionally separate from the contract evaluator's lower-level
fail-closed missing-observation behavior: contract evaluation can still reject
an individual candidate, while this projection distinguishes evidence absence
from a measured infeasibility claim across the complete archive.
