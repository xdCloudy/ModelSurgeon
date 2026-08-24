"""Prediction explanations and reports."""

from modelsurgeon.explain.attribution import (
    ATTRIBUTION_SCHEMA_VERSION,
    AttributionError,
    AttributionReport,
    AttributionResult,
    AttributionUnavailable,
    FeatureContribution,
    FeatureProvenance,
    PredictionAttribution,
    attribute_predictions,
)
from modelsurgeon.explain.decision_summary import (
    DECISION_SUMMARY_SCHEMA_VERSION,
    DecisionEvidence,
    DecisionSummaryError,
    ExpectedDeltaSummary,
    MutationDecisionSummary,
    QuantizationContext,
    generate_mutation_decision_summary,
)

__all__ = [
    "ATTRIBUTION_SCHEMA_VERSION",
    "DECISION_SUMMARY_SCHEMA_VERSION",
    "AttributionError",
    "AttributionReport",
    "AttributionResult",
    "AttributionUnavailable",
    "DecisionEvidence",
    "DecisionSummaryError",
    "ExpectedDeltaSummary",
    "FeatureContribution",
    "FeatureProvenance",
    "MutationDecisionSummary",
    "PredictionAttribution",
    "QuantizationContext",
    "attribute_predictions",
    "generate_mutation_decision_summary",
]
