"""Audit the v2.8 Pareto-selection explanation evidence manifest."""

from pathlib import Path

from tools.audit_pareto_selection_explanations import audit_release


def test_pareto_selection_audit_passes() -> None:
    audit_release(Path(__file__).resolve().parents[1])
