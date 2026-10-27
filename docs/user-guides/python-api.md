# Stable Python API and notebook workflow

This is the smallest supported workflow for a technically competent user. It
resolves a pinned source, a bounded hardware/quality profile, approvals, and a
read-only optimize plan. It does not download a model, execute a mutation, or
overwrite an artifact. The same cells are available in
[`docs/notebooks/optimize_plan_quickstart.ipynb`](../notebooks/optimize_plan_quickstart.ipynb),
and the script form is [`docs/examples/plan_optimize.py`](../examples/plan_optimize.py).

## Clean setup

From a fresh checkout:

```bash
uv sync --extra dev --locked
python docs/examples/plan_optimize.py \
  --model models/tiny-supported \
  --revision sha256:replace-with-an-immutable-revision \
  --output work/quickstart/plan.json
```

The example only needs the core package because planning is deliberately
read-only. Replace the model path and revision with an immutable, licensed
source you control. The example refuses to overwrite its output.

## The public API

```python
from pathlib import Path

from modelsurgeon.config import ModelConfig, Settings
from modelsurgeon.optimization import build_optimize_plan, write_optimize_plan

settings = Settings(
    model=ModelConfig(
        path="models/tiny-supported",
        revision="sha256:replace-with-an-immutable-revision",
    )
)
plan = build_optimize_plan(
    settings,
    preset="balanced",
    hardware_profile="cpu-small",
    quality_profile="balanced",
)
if not plan.executable:
    raise RuntimeError(f"plan is not executable: {plan.outcome.value}: {plan.uncertainties}")
write_optimize_plan(Path("work/quickstart/plan.json"), plan)
print(plan.plan_id, plan.resume_token)
```

The plan retains the resolved configuration, source lineage, profile IDs,
budget, conservative cost estimate, approval points, exact command tuple,
rollback policy, and uncertainties. `plan_id` and `resume_token` are stable
for equivalent inputs. Treat a plan with `unknown`, `unsupported`, or `failed`
outcome as evidence, not as permission to execute.

## Execute a first-party campaign

For a real Hugging Face campaign, install the optional model dependencies and
use `execute_first_party_optimize`. Planning stays read-only; execution loads
the pinned model, runs the approval-bound orchestrator, and writes run state
and new artifacts to the configured output locations. It never overwrites the
source model. For native GGUF, follow the separate [native GGUF guide](native-gguf.md).

```bash
uv sync --extra dev --extra hf --locked
```

Review the exact executable plan first, then pass only the approval codes you
have explicitly accepted. If approvals are missing, execution returns a
durable paused run without loading or mutating the source.

```python
from modelsurgeon.config import ModelConfig, Settings
from modelsurgeon.optimization import build_optimize_plan

settings = Settings(
    model=ModelConfig(
        path="models/tiny-supported",
        revision="sha256:replace-with-an-immutable-revision",
    )
)
options = {
    "preset": "balanced",
    "hardware_profile": "cpu-small",
    "quality_profile": "balanced",
}
plan = build_optimize_plan(settings, **options, dry_run=False)
if not plan.executable:
    raise RuntimeError(f"plan is not executable: {plan.outcome.value}: {plan.uncertainties}")
print(plan.canonical_json())  # Review inputs, budgets, approvals, and uncertainties first.
```

Only after reviewing that output and explicitly accepting each approval, run
the campaign in a separate notebook cell. If using a separate script, recreate
the same `settings` and `options` values there:

```python
from modelsurgeon.optimization import execute_first_party_optimize

run = execute_first_party_optimize(
    settings,
    "work/quickstart/optimize-run.json",
    **options,
    approvals=("plan_review", "source_model", "resource_budget", "artifact_write"),
)
print(run.outcome.value)
```

To resume an interrupted run, use the same settings, options, state path, and
approved codes, and set `resume=True`. Keep the state file and its
content-addressed artifacts together as the audit record. See the
[`unified optimize plan` contract](../design/unified-optimize-plan.md) for
the persisted fields, approval semantics, and native GGUF requirements.

No conversational provider is needed for this API. `Settings()` selects the
explicit `provider.kind=none` mode; callers that need to inspect provider
availability can use `modelsurgeon.providers.provider_diagnostics(settings)`.
Provider configuration is metadata and a bounded request budget only: it does
not override hard optimization constraints or provide execution authority.

## Explain measured infeasibility

Use the read-only feasibility API when a bounded search has no accepted
candidate. It distinguishes missing, predicted-only, unsupported, failed, and
inconclusive evidence from a measured hard-constraint violation. The complete
example is [`docs/examples/infeasibility_explanation.py`](../examples/infeasibility_explanation.py).

```python
archive = CanonicalEvidenceArchive.build(contract, candidate_evidence)
result = build_feasibility_explanation(
    contract,
    archive,
    provenance=FeasibilityProvenance(
        "spec_2026_09_07",
        source_model_digest,
        archive.archive_id,
        approval_id="approval_123",
    ),
)
if result.outcome is FeasibilityOutcome.INFEASIBLE:
    for constraint in result.unmet_constraints:
        print(constraint.constraint.metric, constraint.candidate_ids)
else:
    print(result.outcome.value, result.next_actions)
```

Near misses are measured candidates only and are ordered by conservative
constraint distance, then candidate ID. Predictions can nominate a follow-up
measurement but never establish feasibility. Rejected and rolled-back
evidence remains visible, and changing a hard constraint requires a separate
approved objective amendment.

## Explain measured Pareto alternatives

Use `build_pareto_alternatives` to explain the trade-offs among measured,
hard-constraint-feasible candidates in the same canonical archive. Each
alternative includes its candidate/evidence IDs, measured objective values,
metric deltas, uncertainty, constraint violations, disposition, provenance,
and frontier or dominance context. Unsupported, failed, unknown, and
inconclusive evidence remains in `retained_evidence` and is never promoted to
the frontier. The complete example is
[`docs/examples/pareto_alternatives.py`](../examples/pareto_alternatives.py).

```python
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
print(render_pareto_explanation(result, format="direct"))
print(render_pareto_explanation(result, format="chat"))
```

Dominance is conservative over objective intervals and only compares complete
measured candidates with an accepted disposition that pass hard constraints.
Ties, overlapping uncertainty, and rejected or rolled-back dispositions are
explicit. Candidate and output resource bounds fail closed;
the result is never truncated to manufacture a smaller frontier.

## Explain final Pareto selection

Use `build_pareto_selection_explanation` after the measured alternatives
projection to explain a final choice. It renders hard constraints before soft
trade-offs and recomputes the selected candidate from the canonical objective,
measured feasible frontier, and v2.0 decision replay. Ties, dominance,
uncertainty, negative evidence, approval context, and objective-amendment
identity remain visible. See
[`docs/examples/pareto_selection.py`](../examples/pareto_selection.py) and
[`docs/design/pareto-selection-explanations.md`](../design/pareto-selection-explanations.md).

## HF and GGUF boundaries

For Hugging Face/safetensors, inspect a revision-pinned source with the
[`huggingface` guide](huggingface.md), then run a bounded proof before any
physical write. Use a new destination and retain the inspection/proof records.
For native GGUF, use the [native GGUF guide](native-gguf.md); generic optimize
planning intentionally reports GGUF mutation as `unsupported` rather than
silently switching formats or materializing a full model.

Do not enable remote model code, pass mutable branches, embed credentials, or
use the source directory as an output path. Those controls are part of the
workflow contract, not optional prose.
