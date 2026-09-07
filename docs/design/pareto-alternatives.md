# Measured Pareto alternatives

`modelsurgeon.explain.pareto_alternatives` is a read-only explanation layer on
top of the canonical evidence archive introduced for measured feasibility and
infeasibility. It does not run a search, mutate a model, relax a contract, or
invent a point between two measurements.

## Evidence contract

Every item in `alternatives` is a measured candidate and contains both its
`candidate_id` and `evidence_id`, plus its evaluation ID, candidate provenance,
disposition, measured objective values, and optional resource usage. The
separate `retained_evidence` list preserves unsupported, failed, unknown, and
inconclusive records from the same archive. They remain visible but cannot be
frontier alternatives.

Hard constraints are evaluated conservatively using the same rules as the
measured infeasibility explanation: lower bounds are used for minimums and
upper bounds for maximums. A violation is visible with its observed value,
absolute gap, normalized gap, and uncertainty flag. Missing hard or objective
metrics produce `incomplete`, not a fabricated value.

## Frontier and trade-off semantics

Only measured candidates with complete objective evidence and no conservative
hard-constraint violation and an accepted measured disposition enter the
frontier comparison. Rejected, rolled-back, failed, inconclusive, or unknown
dispositions are retained as `disposition_not_accepted` context and are not
offered as accepted frontier alternatives. For a maximize
objective, one candidate can dominate another only when its lower bound is at
least the other's upper bound. For a minimize objective the comparison uses
the symmetric upper/lower rule. At least one objective must be strictly better
for dominance. Overlapping uncertainty therefore prevents an unsupported
dominance claim. Exact objective equality is retained as `tied_frontier`.

The output records the frontier IDs, dominator IDs, tie IDs, objective metric
deltas, uncertainty metrics, and hard-constraint violations for every measured
candidate. Frontier membership is never inferred from a prediction, a failed
evaluation, or a negative result.

## Determinism, provenance, and bounds

Archives and alternatives are sorted by candidate ID. Frontier IDs, witness
IDs, metric names, violations, and retained evidence are sorted and unique.
The result ID is a SHA-256 digest of the canonical record. The supplied
`FeasibilityProvenance` is retained together with the linked feasibility result
from #458, including source-model digest, archive identity, evidence IDs, and
approval context. Candidate and output limits are hard fail-closed bounds; the
implementation does not truncate evidence to fit a limit.

`render_pareto_explanation` supports `direct`, `chat`, and `html` views. All
three are derived from the same typed result, so a chat summary cannot claim a
different frontier from the direct report. The HTML view is self-contained and
links each measured alternative to its immutable evidence fragment.

```python
from modelsurgeon.explain import (
    CanonicalEvidenceArchive,
    FeasibilityProvenance,
    build_pareto_alternatives,
    render_pareto_explanation,
)

archive = CanonicalEvidenceArchive.build(contract, candidate_evidence)
result = build_pareto_alternatives(
    contract,
    archive,
    provenance=FeasibilityProvenance(
        "spec_2026_09_07",
        source_model_digest,
        archive.archive_id,
        approval_id="approval_123",
    ),
)
print(render_pareto_explanation(result, format="chat"))
```

Limitations: dominance is conservative and archive-local; it is not a global
optimum claim. A candidate with overlapping intervals may be incomparable even
when its point estimate looks better. A terminal negative result can explain
what was tried, but cannot become a frontier point until a new measured
evidence record is produced. Objective baselines are shown only when the
contract declares one; otherwise `delta_from_baseline` is explicitly `null`.
