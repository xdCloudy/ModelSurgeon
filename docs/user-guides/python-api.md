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
