"""Deterministic bounded-batch scoring for active-learning candidate pools."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from modelsurgeon.surgeon.calibration import ProbabilityCalibrator

CANDIDATE_SCORING_SCHEMA_VERSION: Final[int] = 1


class CandidateScoringError(ValueError):
    """Raised when scoring configuration or model output is invalid."""


class BatchPredictor(Protocol):
    def predict(self, rows: Sequence[Sequence[float]]) -> Sequence[float]: ...


@dataclass(frozen=True, slots=True)
class ScoringCandidate:
    candidate_id: str
    feature_schema_version: int
    feature_names: tuple[str, ...]
    feature_values: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.candidate_id.startswith("cand_"):
            raise CandidateScoringError("scoring candidates require canonical candidate IDs")
        if self.feature_schema_version <= 0:
            raise CandidateScoringError("candidate feature schema version must be positive")


@dataclass(frozen=True, slots=True)
class CandidateScore:
    candidate_id: str
    utility: float
    outcomes: tuple[tuple[str, float], ...]
    raw_safe_probability: float
    safe_probability: float
    uncertainty: float
    schema_version: int = CANDIDATE_SCORING_SCHEMA_VERSION

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "utility": self.utility,
            "outcomes": dict(self.outcomes),
            "raw_safe_probability": self.raw_safe_probability,
            "safe_probability": self.safe_probability,
            "uncertainty": self.uncertainty,
        }


@dataclass(frozen=True, slots=True)
class QuarantinedCandidate:
    candidate_id: str
    reason: str
    observed_schema_version: int
    observed_feature_names: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "reason": self.reason,
            "observed_schema_version": self.observed_schema_version,
            "observed_feature_names": list(self.observed_feature_names),
        }


@dataclass(frozen=True, slots=True)
class CandidateScoringReport:
    scores: tuple[CandidateScore, ...]
    quarantined: tuple[QuarantinedCandidate, ...]
    batch_size: int
    expected_feature_schema_version: int
    expected_feature_names: tuple[str, ...]
    model_revision: str
    tool_revision: str
    schema_version: int = CANDIDATE_SCORING_SCHEMA_VERSION

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "batch_size": self.batch_size,
            "expected_feature_schema_version": self.expected_feature_schema_version,
            "expected_feature_names": list(self.expected_feature_names),
            "model_revision": self.model_revision,
            "tool_revision": self.tool_revision,
            "score_count": len(self.scores),
            "quarantine_count": len(self.quarantined),
            "scores": [item.to_record() for item in self.scores],
            "quarantined": [item.to_record() for item in self.quarantined],
        }


def score_candidate_pool(
    candidates: Sequence[ScoringCandidate],
    *,
    utility_predictor: BatchPredictor,
    outcome_predictors: Mapping[str, BatchPredictor],
    safe_predictor: BatchPredictor,
    uncertainty_predictor: BatchPredictor,
    calibrator: ProbabilityCalibrator,
    expected_feature_schema_version: int,
    expected_feature_names: Sequence[str],
    model_revision: str,
    tool_revision: str,
    batch_size: int = 256,
) -> CandidateScoringReport:
    """Score compatible candidates in stable input order and quarantine schema drift."""

    if not 1 <= batch_size <= 4096:
        raise CandidateScoringError("candidate scoring batch size must be within 1..4096")
    names = tuple(expected_feature_names)
    if expected_feature_schema_version <= 0 or not names:
        raise CandidateScoringError("candidate scoring expected schema is invalid")
    if not model_revision or not tool_revision:
        raise CandidateScoringError("candidate scoring revisions are required")
    if len(names) != len(set(names)) or any(not name for name in names):
        raise CandidateScoringError("candidate scoring feature names must be unique")
    outcomes = tuple(sorted(outcome_predictors.items()))
    if any(not name for name, _ in outcomes):
        raise CandidateScoringError("candidate outcome names cannot be blank")
    compatible: list[ScoringCandidate] = []
    quarantined: list[QuarantinedCandidate] = []
    for candidate in candidates:
        reason: str | None = None
        if candidate.feature_schema_version != expected_feature_schema_version:
            reason = "feature-schema-version-incompatible"
        elif candidate.feature_names != names:
            reason = "feature-schema-names-incompatible"
        elif len(candidate.feature_values) != len(names):
            reason = "feature-width-incompatible"
        elif any(not math.isfinite(value) for value in candidate.feature_values):
            reason = "feature-values-non-finite"
        if reason is None:
            compatible.append(candidate)
        else:
            quarantined.append(
                QuarantinedCandidate(
                    candidate.candidate_id,
                    reason,
                    candidate.feature_schema_version,
                    candidate.feature_names,
                )
            )
    scores: list[CandidateScore] = []
    for start in range(0, len(compatible), batch_size):
        batch = compatible[start : start + batch_size]
        rows = tuple(item.feature_values for item in batch)
        utility = _predictions(utility_predictor, rows, "utility")
        safe_raw = _predictions(safe_predictor, rows, "safe probability")
        uncertainty = _predictions(uncertainty_predictor, rows, "uncertainty")
        calibrated = calibrator.calibrate(safe_raw)
        outcome_values = tuple(
            (name, _predictions(predictor, rows, f"outcome {name!r}"))
            for name, predictor in outcomes
        )
        if any(not 0.0 <= value <= 1.0 for value in (*safe_raw, *calibrated)):
            raise CandidateScoringError("safe probability outputs must be within [0, 1]")
        if any(value < 0.0 for value in uncertainty):
            raise CandidateScoringError("uncertainty outputs cannot be negative")
        for index, candidate in enumerate(batch):
            scores.append(
                CandidateScore(
                    candidate.candidate_id,
                    utility[index],
                    tuple((name, values[index]) for name, values in outcome_values),
                    safe_raw[index],
                    calibrated[index],
                    uncertainty[index],
                )
            )
    return CandidateScoringReport(
        tuple(scores),
        tuple(quarantined),
        batch_size,
        expected_feature_schema_version,
        names,
        model_revision,
        tool_revision,
    )


def _predictions(
    predictor: BatchPredictor, rows: Sequence[Sequence[float]], label: str
) -> tuple[float, ...]:
    values = tuple(float(value) for value in predictor.predict(rows))
    if len(values) != len(rows) or any(not math.isfinite(value) for value in values):
        raise CandidateScoringError(f"{label} predictor returned invalid batch output")
    return values
