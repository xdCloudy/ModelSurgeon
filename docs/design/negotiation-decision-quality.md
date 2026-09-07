# v2.5 negotiation decision-quality protocol

This is the bounded v2.5 research protocol for deciding whether an
alternative surface is safe to ship after measured infeasibility. It compares
three control-plane policies over typed fixture campaign traces:

- `refusal_only`: retain the measured explanation and refuse to propose an
  alternative;
- `measured_pareto`: present only the canonical measured Pareto projection,
  then use the immutable #460 amendment APIs when a fixture includes an
  explicit feasible target; and
- `prediction_only`: a deliberately unsafe control. Predicted candidates are
  retained and counted as misleading when presented as alternatives, but they
  are never passed to the measured Pareto or amendment APIs and are never
  shippable.

The protocol record is
[`negotiation_decision_quality_v1.json`](../../tests/fixtures/negotiation_decision_quality_v1.json).
It is bound to `v25-negotiation-decision-quality-v1`, four scenarios, three
seeds per scenario (`0`, `1`, `2`), and exactly four evaluation records per
scenario. Candidate ordering is seed-derived for the trace, while the
canonical evidence, alternatives, IDs, and output ordering remain sorted and
replayable. No live model, provider, credential, or network benchmark is
part of this evidence.

## Measurements and stop rule

Every policy/seed/scenario cell retains the source-model digest, original
contract identity and hard-constraint digest, explanation outcome, all
candidate evidence IDs, presented-alternative status, amendment IDs/diff IDs,
application lineage, interaction cost, and error state. Failed, unsupported,
unknown, predicted-only, and inconclusive cells remain in the denominator.

The metrics are:

- constraint-preservation rate;
- measured-alternative grounding rate;
- amendment traceability and amendment accuracy;
- feasible-target recovery rate;
- misleading-claim rate; and
- bounded interaction cost (mean and total trace steps).

The runner stops and fails closed if an original hard constraint changes,
an effective constraint change lacks an approved immutable amendment, a
non-measured candidate enters the measured Pareto result, or a prediction is
marked measured. A changed preference/objective is allowed only through the
explicit amendment proposal, approval, application, new campaign identity,
and preserved evidence archive.

Replay the summary or complete canonical evidence from the repository root:

```text
uv run python tools/run_negotiation_quality_study.py
uv run python tools/run_negotiation_quality_study.py --json
uv run python tools/audit_v25_negotiation_quality.py
```

## Decision and unsupported cells

The retained run has 36 cells. The measured-Pareto policy preserves all hard
constraints, grounds every alternative claim in measured evidence, traces all
six amendment applications, recovers all six expected target cells, and has a
zero misleading-claim rate. Refusal-only is a safe fallback with no recovery
claims. Prediction-only has a misleading-claim rate of `1.0` in its predicted
cells and is permanently non-shippable.

The decision for this bounded evidence is to permit the measured-Pareto
surface only when its canonical evidence and amendment gates pass, retain
refusal as the fail-closed fallback, and reject prediction-only promotion.
This does not establish live model quality or general user preference
elicitation.

The following cells are explicit non-claims or unsupported boundaries:

| Cell | Status | Reason |
| --- | --- | --- |
| live HF/GGUF quality measurement | not run | Fixture traces use canonical evidence and do not execute models. |
| prediction-only promotion | unsupported / never shippable | A prediction cannot establish feasibility or measured Pareto membership. |
| fluent user preference selection | not claimed | The protocol tests typed amendment traces, not language understanding. |
| hostile-process containment | not claimed | Resource bounds are protocol limits, not sandbox enforcement. |
| silent constraint changes | stop/fail closed | No cell may continue or be promoted after a silent hard-constraint change. |

The exact dependency commits, run ID, metrics, negative/inconclusive counts,
documentation paths, and quality-gate commands are frozen in the v2.5
research record. The milestone-level reconciliation with the v2.0 approval
boundary, direct API compatibility, and duplicate-scope review is frozen in
the [v2.5 constraint-negotiation release boundary](../release/v2.5-constraint-negotiation-boundary.md)
and its [versioned evidence manifest](../research/v2.5-constraint-negotiation-release-v1.json).
