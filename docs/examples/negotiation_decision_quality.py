"""Minimal offline replay example for the bounded v2.5 study."""

from pathlib import Path

from modelsurgeon.conversation import (
    NegotiationPolicy,
    load_and_run_negotiation_study,
)

study = Path("tests/fixtures/negotiation_decision_quality_v1.json")
run = load_and_run_negotiation_study(study)
measured = next(item for item in run.metrics if item.policy is NegotiationPolicy.MEASURED_PARETO)

print(run.run_id)
print(measured.to_record())
