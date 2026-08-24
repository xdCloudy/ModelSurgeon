"""Static-only Surgeon evaluation for the v0.8 cross-model Q1 study."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, cast

from modelsurgeon.datasets.grouped_splits import GroupedSplitManifest
from modelsurgeon.surgeon.matrix import TrainingMatrices, build_training_matrices
from modelsurgeon.surgeon.metrics import MetricReport, evaluate_classification, evaluate_regression
from modelsurgeon.surgeon.models import LightGBMConfig, ModelTask, train_lightgbm
from modelsurgeon.surgeon.targets import DEFAULT_TARGET_SCHEMA, schema_with_thresholds

STATIC_FEATURE_STUDY_VERSION: Final[str] = "1"


class StaticFeatureStudyError(ValueError):
    """Raised when records cannot support a leakage-safe static-only study."""


class FeatureProfile(StrEnum):
    STATIC_ONLY = "static_only"
    STATIC_ACTIVATION = "static_activation"
    STATIC_ACTIVATION_GRADIENT = "static_activation_gradient"


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
    feature_names: tuple[str, ...]
    classifier: MetricReport
    regressor: MetricReport
    feature_profile: FeatureProfile = FeatureProfile.STATIC_ONLY
    test_labels: tuple[int, ...] = ()
    test_targets: tuple[float, ...] = ()
    test_classifier_predictions: tuple[float, ...] = ()
    test_regressor_predictions: tuple[float, ...] = ()
    test_group_ids: tuple[str, ...] = ()
    test_example_ids: tuple[str, ...] = ()
    version: str = STATIC_FEATURE_STUDY_VERSION

    def to_record(self) -> dict[str, object]:
        return {
            "version": self.version,
            "feature_profile": self.feature_profile.value,
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
            "feature_names": list(self.feature_names),
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


def select_feature_profile_records(
    records: Sequence[Mapping[str, object]],
    profile: FeatureProfile,
) -> tuple[Mapping[str, object], ...]:
    """Select a persisted feature profile without changing examples or labels."""

    if profile is FeatureProfile.STATIC_ONLY:
        return select_static_records(records)
    if not records:
        raise StaticFeatureStudyError("feature study requires records")
    output: list[Mapping[str, object]] = []
    for record in records:
        raw_features = record.get("pre_mutation_features")
        if not isinstance(raw_features, list) or not raw_features:
            raise StaticFeatureStudyError("pre_mutation_features must be a non-empty list")
        activation_present = any(
            isinstance(raw, Mapping)
            and raw.get("sample_context") is not None
            and raw.get("extractor") != "gradient_features"
            for raw in raw_features
        )
        if not activation_present:
            raise StaticFeatureStudyError("static-plus-activation profile requires activations")
        gradient_present = any(
            isinstance(raw, Mapping) and raw.get("extractor") == "gradient_features"
            for raw in raw_features
        )
        if profile is FeatureProfile.STATIC_ACTIVATION_GRADIENT and not gradient_present:
            raise StaticFeatureStudyError("gradient feature profile requires gradients")
        selected_features = [
            raw
            for raw in raw_features
            if profile is FeatureProfile.STATIC_ACTIVATION_GRADIENT
            or not isinstance(raw, Mapping)
            or raw.get("extractor") != "gradient_features"
        ]
        filtered = dict(record)
        filtered["pre_mutation_features"] = selected_features
        output.append(filtered)
    return tuple(output)


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


def run_feature_profile_study(
    records: Sequence[Mapping[str, object]],
    split: GroupedSplitManifest,
    profile: FeatureProfile,
    config: StaticFeatureStudyConfig | None = None,
    *,
    include_context: bool = True,
) -> StaticFeatureStudyResult:
    """Fit one feature-profile pair of LightGBMs on a held-out target model."""

    if config is None:
        config = StaticFeatureStudyConfig()
    identity = _identity(records)
    selected_records = select_feature_profile_records(records, profile)
    target_schema = schema_with_thresholds(
        {"perplexity": config.safe_perplexity_delta},
        base=DEFAULT_TARGET_SCHEMA,
    )
    classifier_matrices = build_training_matrices(
        selected_records,
        split,
        target_schema=target_schema,
        target_name="safe_mutation",
        include_context=include_context,
    )
    regressor_matrices = build_training_matrices(
        selected_records,
        split,
        target_schema=target_schema,
        target_name="perplexity",
        include_context=include_context,
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
        len(selected_records),
        len(classifier_matrices.train.example_ids),
        len(classifier_matrices.validation.example_ids),
        len(classifier_matrices.test.example_ids),
        tuple(item.name for item in classifier_matrices.preprocessor.numeric),
        classifier_report,
        regressor_report,
        profile,
        labels,
        tuple(regressor_matrices.test.target_values),
        classifier_predictions,
        regressor_predictions,
        tuple(classifier_matrices.test.group_ids),
        tuple(classifier_matrices.test.example_ids),
    )


def run_static_feature_study(
    records: Sequence[Mapping[str, object]],
    split: GroupedSplitManifest,
    config: StaticFeatureStudyConfig | None = None,
) -> StaticFeatureStudyResult:
    """Fit static-only LightGBMs and evaluate one held-out target model."""

    return run_feature_profile_study(records, split, FeatureProfile.STATIC_ONLY, config)
