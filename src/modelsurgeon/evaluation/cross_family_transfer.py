"""Cross-family transfer protocols and explicit schema-failure reporting."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from modelsurgeon.surgeon.matrix import TrainingMatrixError

from .cross_model_transfer import (
    CrossModelTransferResult,
    TransferDataset,
    _identity,
    run_cross_model_transfer,
)
from .static_feature_study import StaticFeatureStudyConfig, StaticFeatureStudyError


class TransferProtocol(StrEnum):
    SINGLE_FAMILY = "single_family"
    MULTI_FAMILY = "multi_family"
    HELD_OUT_FAMILY = "held_out_family"


@dataclass(frozen=True, slots=True)
class TransferExperiment:
    protocol: TransferProtocol
    sources: tuple[TransferDataset, ...]
    target: TransferDataset


@dataclass(frozen=True, slots=True)
class TransferProtocolOutcome:
    protocol: TransferProtocol
    source_families: tuple[str, ...]
    target_family: str
    status: str
    result: CrossModelTransferResult | None = None
    failure_kind: str | None = None
    failure_detail: str | None = None

    def to_record(self) -> dict[str, object]:
        return {
            "protocol": self.protocol.value,
            "source_families": list(self.source_families),
            "target_family": self.target_family,
            "status": self.status,
            "result": None if self.result is None else self.result.to_record(),
            "failure": (
                None
                if self.failure_kind is None
                else {"kind": self.failure_kind, "detail": self.failure_detail}
            ),
        }


@dataclass(frozen=True, slots=True)
class CrossFamilyTransferStudyResult:
    outcomes: tuple[TransferProtocolOutcome, ...]
    multi_family_improvement: Mapping[str, float] | None

    def to_record(self) -> dict[str, object]:
        return {
            "outcomes": [outcome.to_record() for outcome in self.outcomes],
            "multi_family_improvement": (
                None
                if self.multi_family_improvement is None
                else dict(self.multi_family_improvement)
            ),
        }


def _metric(result: CrossModelTransferResult, task: str, name: str) -> float:
    report = result.cross_classifier if task == "classifier" else result.cross_regressor
    for estimate in report.metrics:
        if estimate.name == name and estimate.defined and estimate.value is not None:
            return estimate.value
    raise StaticFeatureStudyError(f"Q6 comparison metric is undefined: {task}.{name}")


def _validate_experiment(experiment: TransferExperiment) -> tuple[tuple[str, ...], str]:
    if not experiment.sources:
        raise StaticFeatureStudyError("Q6 protocol requires source datasets")
    source_families = tuple(sorted({_identity(item.records)[2] for item in experiment.sources}))
    target_family = _identity(experiment.target.records)[2]
    if experiment.protocol is TransferProtocol.SINGLE_FAMILY:
        if len(source_families) != 1 or target_family not in source_families:
            raise StaticFeatureStudyError("single-family protocol topology is invalid")
    elif experiment.protocol is TransferProtocol.MULTI_FAMILY:
        if len(source_families) < 2 or target_family not in source_families:
            raise StaticFeatureStudyError("multi-family protocol topology is invalid")
    elif target_family in source_families:
        raise StaticFeatureStudyError("held-out-family protocol includes the target family")
    return source_families, target_family


def run_cross_family_transfer_study(
    experiments: Sequence[TransferExperiment],
    config: StaticFeatureStudyConfig | None = None,
) -> CrossFamilyTransferStudyResult:
    """Run Q6 protocols while retaining expected schema failures as outcomes."""

    resolved = config or StaticFeatureStudyConfig()
    protocols = tuple(experiment.protocol for experiment in experiments)
    if len(set(protocols)) != len(protocols):
        raise StaticFeatureStudyError("Q6 protocols must be unique")
    outcomes: list[TransferProtocolOutcome] = []
    for experiment in experiments:
        source_families, target_family = _validate_experiment(experiment)
        try:
            result = run_cross_model_transfer(
                experiment.sources,
                experiment.target,
                resolved,
                require_represented_family=(
                    experiment.protocol is not TransferProtocol.HELD_OUT_FAMILY
                ),
            )
        except (StaticFeatureStudyError, TrainingMatrixError) as error:
            outcomes.append(
                TransferProtocolOutcome(
                    experiment.protocol,
                    source_families,
                    target_family,
                    "failed",
                    failure_kind=type(error).__name__,
                    failure_detail=str(error),
                )
            )
        else:
            outcomes.append(
                TransferProtocolOutcome(
                    experiment.protocol,
                    source_families,
                    target_family,
                    "succeeded",
                    result=result,
                )
            )

    successes = {
        outcome.protocol: outcome.result for outcome in outcomes if outcome.result is not None
    }
    single = successes.get(TransferProtocol.SINGLE_FAMILY)
    multi = successes.get(TransferProtocol.MULTI_FAMILY)
    improvement: dict[str, float] | None = None
    if single is not None and multi is not None and single.target_model == multi.target_model:
        improvement = {
            "auc_gain": _metric(multi, "classifier", "auc") - _metric(single, "classifier", "auc"),
            "calibration_error_reduction": _metric(single, "classifier", "calibration_error")
            - _metric(multi, "classifier", "calibration_error"),
            "precision_at_top_n_gain": _metric(
                multi, "classifier", f"precision_at_{resolved.top_n}"
            )
            - _metric(single, "classifier", f"precision_at_{resolved.top_n}"),
            "mae_reduction": _metric(single, "regressor", "mae")
            - _metric(multi, "regressor", "mae"),
            "rmse_reduction": _metric(single, "regressor", "rmse")
            - _metric(multi, "regressor", "rmse"),
        }
    return CrossFamilyTransferStudyResult(tuple(outcomes), improvement)
