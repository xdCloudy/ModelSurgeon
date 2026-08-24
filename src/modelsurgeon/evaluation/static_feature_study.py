"""Static-only Surgeon evaluation for the v0.8 cross-model Q1 study."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast

from modelsurgeon.datasets.grouped_splits import GroupedSplitManifest
from modelsurgeon.surgeon.matrix import TrainingMatrices, build_training_matrices
from modelsurgeon.surgeon.metrics import MetricReport, evaluate_classification, evaluate_regression
from modelsurgeon.surgeon.models import LightGBMConfig, ModelTask, train_lightgbm
from modelsurgeon.surgeon.targets import DEFAULT_TARGET_SCHEMA, schema_with_thresholds

STATIC_FEATURE_STUDY_VERSION: Final[str] = "1"


class StaticFeatureStudyError(ValueError):
    """Raised when records cannot support a leakage-safe static-only study."""


@dataclass(frozen=True, slots=True)
class StaticFeatureStudyConfig:
    safe_perplexity_delta: float = 0.25
    top_n: int = 10
    threads: int = 4
    seed: int = 42
    bootstrap_repetitions: int = 1000
    bootstrap_confidence: float = 0.95

    def __post_init__(self) -> None:
        if not math.isfinite(self.safe_perplexity_delta) or self.safe_perplexity_delta < 0:
            raise StaticFeatureStudyError("safe_perplexity_delta must be finite and non-negative")
        if self.top_n <= 0 or self.threads <= 0 or self.threads > 32:
            raise StaticFeatureStudyError("top_n and threads must be positive")
        if self.bootstrap_repetitions <= 0:
            raise StaticFeatureStudyError("bootstrap_repetitions must be positive")
        if not 0.0 < self.bootstrap_confidence < 1.0:
            raise StaticFeatureStudyError("bootstrap_confidence must be within (0, 1)")
        if isinstance(self.seed, bool) or self.seed < 0 or self.seed >= 1 << 31:
            raise StaticFeatureStudyError("seed must fit LightGBM's 31-bit seed contract")


@dataclass(frozen=True, slots=True)
class StaticFeatureStudyResult:
    model_identifier: str
    model_revision: str
    family: str
    example_count: int
    train_count: int
    validation_count: int
    test_count: int
    static_feature_names: tuple[str, ...]
    classifier: MetricReport
    regressor: MetricReport
    version: str = STATIC_FEATURE_STUDY_VERSION

    def to_record(self) -> dict[str, object]:
        return {
            "version": self.version,
            "feature_profile": "static_only",
            "model": {
                "identifier": self.model_identifier,
                "revision": self.model_revision,
                "family": self.family,
            },
            "counts": {
                "examples": self.example_count,
                "train": self.train_count,
                "validation": self.validation_count,
                "test": self.test_count,
            },
            "static_feature_names": list(self.static_feature_names),
            "classifier": self.classifier.to_record(),
            "regressor": self.regressor.to_record(),
        }


def _static_record(record: Mapping[str, object]) -> Mapping[str, object]:
    raw_features = record.get("pre_mutation_features")
    if not isinstance(raw_features, list):
        raise StaticFeatureStudyError("pre_mutation_features must be a list")
    static: list[object] = []
    for raw in raw_features:
        if not isinstance(raw, Mapping):
            raise StaticFeatureStudyError("feature entries must be objects")
        sample_context = raw.get("sample_context")
        if sample_context is None:
            static.append(raw)
    if not static:
        raise StaticFeatureStudyError("record contains no static features")
    filtered = dict(record)
    filtered["pre_mutation_features"] = static
    return filtered


def select_static_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """Remove every calibration-dependent feature before preprocessing or fitting."""

    if not records:
        raise StaticFeatureStudyError("static study requires records")
    return tuple(_static_record(record) for record in records)


def _identity(records: Sequence[Mapping[str, object]]) -> tuple[str, str, str]:
    identities: set[tuple[str, str, str]] = set()
    for record in records:
        model = record.get("model")
        if not isinstance(model, Mapping):
            raise StaticFeatureStudyError("records require model provenance")
        identifier = model.get("identifier")
        revision = model.get("revision")
        family = model.get("family")
        if not all(isinstance(value, str) and value for value in (identifier, revision, family)):
            raise StaticFeatureStudyError("model identifier, revision, and family are required")
        identities.add(cast(tuple[str, str, str], (identifier, revision, family)))
    if len(identities) != 1:
        raise StaticFeatureStudyError("each static study run must contain exactly one model")
    return next(iter(identities))


def _require_labels(matrices: TrainingMatrices, *, classification: bool) -> None:
    for matrix in (matrices.train, matrices.validation, matrices.test):
        if not all(matrix.target_mask):
            raise StaticFeatureStudyError(
                f"{matrix.partition.value} contains missing {matrix.target_name} labels"
            )
    if classification:
        train_classes = {int(value) for value in matrices.train.target_values}
        if train_classes != {0, 1}:
            raise StaticFeatureStudyError("classification training split requires both classes")


def run_static_feature_study(
    records: Sequence[Mapping[str, object]],
    split: GroupedSplitManifest,
    config: StaticFeatureStudyConfig | None = None,
) -> StaticFeatureStudyResult:
    """Fit static-only LightGBMs and evaluate one held-out target model."""

    if config is None:
        config = StaticFeatureStudyConfig()
    identity = _identity(records)
    static_records = select_static_records(records)
    target_schema = schema_with_thresholds(
        {"perplexity": config.safe_perplexity_delta},
        base=DEFAULT_TARGET_SCHEMA,
    )
    classifier_matrices = build_training_matrices(
        static_records,
        split,
        target_schema=target_schema,
        target_name="safe_mutation",
    )
    regressor_matrices = build_training_matrices(
        static_records,
        split,
        target_schema=target_schema,
        target_name="perplexity",
    )
    _require_labels(classifier_matrices, classification=True)
    _require_labels(regressor_matrices, classification=False)

    classifier = train_lightgbm(
        classifier_matrices.train,
        classifier_matrices.validation,
        config=LightGBMConfig(
            ModelTask.CLASSIFICATION,
            num_threads=config.threads,
            seed=config.seed,
        ),
    )
    regressor = train_lightgbm(
        regressor_matrices.train,
        regressor_matrices.validation,
        config=LightGBMConfig(
            ModelTask.REGRESSION,
            num_threads=config.threads,
            seed=config.seed,
        ),
    )
    classifier_predictions = tuple(
        float(value) for value in classifier.predict(classifier_matrices.test.values)
    )
    regressor_predictions = tuple(
        float(value) for value in regressor.predict(regressor_matrices.test.values)
    )
    labels = tuple(int(value) for value in classifier_matrices.test.target_values)
    classifier_report = evaluate_classification(
        labels,
        classifier_predictions,
        top_n=config.top_n,
        group_ids=classifier_matrices.test.group_ids,
        bootstrap_repetitions=config.bootstrap_repetitions,
        confidence=config.bootstrap_confidence,
        seed=config.seed,
    )
    regressor_report = evaluate_regression(
        regressor_matrices.test.target_values,
        regressor_predictions,
        group_ids=regressor_matrices.test.group_ids,
        bootstrap_repetitions=config.bootstrap_repetitions,
        confidence=config.bootstrap_confidence,
        seed=config.seed,
    )
    return StaticFeatureStudyResult(
        identity[0],
        identity[1],
        identity[2],
        len(static_records),
        len(classifier_matrices.train.example_ids),
        len(classifier_matrices.validation.example_ids),
        len(classifier_matrices.test.example_ids),
        tuple(item.name for item in classifier_matrices.preprocessor.numeric),
        classifier_report,
        regressor_report,
    )
