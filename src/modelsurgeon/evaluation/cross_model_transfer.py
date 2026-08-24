"""Zero-target-example cross-model transfer evaluation for v0.8 Q5."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from modelsurgeon.datasets.grouped_splits import GroupedSplitManifest, SplitPartition
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.surgeon.matrix import build_training_matrices, transform_inference_record
from modelsurgeon.surgeon.metrics import MetricReport, evaluate_classification, evaluate_regression
from modelsurgeon.surgeon.models import LightGBMConfig, ModelTask, train_lightgbm
from modelsurgeon.surgeon.targets import (
    DEFAULT_TARGET_SCHEMA,
    derive_supervised_targets,
    schema_with_thresholds,
)

from .activation_feature_study import PairedGainEstimate, paired_feature_gains
from .static_feature_study import (
    FeatureProfile,
    StaticFeatureStudyConfig,
    StaticFeatureStudyError,
    StaticFeatureStudyResult,
    run_feature_profile_study,
    select_feature_profile_records,
)


@dataclass(frozen=True, slots=True)
class TransferDataset:
    records: tuple[Mapping[str, object], ...]
    split: GroupedSplitManifest


@dataclass(frozen=True, slots=True)
class CrossModelTransferResult:
    source_models: tuple[tuple[str, str, str], ...]
    target_model: tuple[str, str, str]
    source_train_count: int
    source_validation_count: int
    target_test_count: int
    source_preprocessor: Mapping[str, object]
    source_preprocessor_sha256: str
    within_classifier: MetricReport
    cross_classifier: MetricReport
    within_regressor: MetricReport
    cross_regressor: MetricReport
    degradations: tuple[PairedGainEstimate, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "source_models": [
                {"identifier": item[0], "revision": item[1], "family": item[2]}
                for item in self.source_models
            ],
            "target_model": {
                "identifier": self.target_model[0],
                "revision": self.target_model[1],
                "family": self.target_model[2],
            },
            "counts": {
                "source_train": self.source_train_count,
                "source_validation": self.source_validation_count,
                "target_test": self.target_test_count,
            },
            "source_preprocessor": dict(self.source_preprocessor),
            "source_preprocessor_sha256": self.source_preprocessor_sha256,
            "within": {
                "classifier": self.within_classifier.to_record(),
                "regressor": self.within_regressor.to_record(),
            },
            "cross": {
                "classifier": self.cross_classifier.to_record(),
                "regressor": self.cross_regressor.to_record(),
            },
            "degradations": [item.to_record() for item in self.degradations],
        }


def _identity(records: Sequence[Mapping[str, object]]) -> tuple[str, str, str]:
    identities: set[tuple[str, str, str]] = set()
    for record in records:
        model = record.get("model")
        if not isinstance(model, Mapping):
            raise StaticFeatureStudyError("transfer records require model provenance")
        values = (model.get("identifier"), model.get("revision"), model.get("family"))
        if not all(isinstance(value, str) and value for value in values):
            raise StaticFeatureStudyError("transfer model identity is incomplete")
        identities.add((str(values[0]), str(values[1]), str(values[2])))
    if len(identities) != 1:
        raise StaticFeatureStudyError("each transfer dataset must contain one model")
    return next(iter(identities))


def _partition_maps(
    split: GroupedSplitManifest,
) -> tuple[dict[str, SplitPartition], dict[str, str]]:
    partitions: dict[str, SplitPartition] = {}
    groups: dict[str, str] = {}
    for group in split.groups:
        for example_id in group.example_ids:
            partitions[example_id] = group.partition
            groups[example_id] = group.group_id
    return partitions, groups


def _rename_degradations(
    gains: Sequence[PairedGainEstimate],
) -> tuple[PairedGainEstimate, ...]:
    names = {
        "auc_gain": "auc_degradation",
        "mae_reduction": "mae_increase",
        "rmse_reduction": "rmse_increase",
    }
    output: list[PairedGainEstimate] = []
    for gain in gains:
        name = names.get(gain.name)
        if name is None and gain.name.startswith("precision_at_"):
            name = gain.name.replace("_gain", "_degradation")
        if name is None:
            raise StaticFeatureStudyError(f"unknown transfer degradation {gain.name}")
        output.append(
            PairedGainEstimate(
                name,
                gain.value,
                gain.confidence_low,
                gain.confidence_high,
                gain.bootstrap_repetitions,
            )
        )
    return tuple(sorted(output, key=lambda item: item.name))


def run_cross_model_transfer(
    sources: Sequence[TransferDataset],
    target: TransferDataset,
    config: StaticFeatureStudyConfig | None = None,
) -> CrossModelTransferResult:
    """Train only on source models and evaluate an unseen represented-family model."""

    if not sources:
        raise StaticFeatureStudyError("cross-model transfer requires source datasets")
    resolved = config or StaticFeatureStudyConfig()
    source_identities = tuple(sorted(_identity(source.records) for source in sources))
    target_identity = _identity(target.records)
    if target_identity in source_identities:
        raise StaticFeatureStudyError("target model must be completely unseen in sources")
    if target_identity[2] not in {identity[2] for identity in source_identities}:
        raise StaticFeatureStudyError("Q5 target family must be represented in sources")

    selected_source: list[Mapping[str, object]] = []
    source_split: dict[str, SplitPartition] = {}
    for source in sources:
        partitions, _ = _partition_maps(source.split)
        for record in select_feature_profile_records(
            source.records, FeatureProfile.STATIC_ACTIVATION_GRADIENT
        ):
            example_id = str(record.get("example_id"))
            partition = partitions.get(example_id)
            if partition in (SplitPartition.TRAIN, SplitPartition.VALIDATION):
                if example_id in source_split:
                    raise StaticFeatureStudyError("source example IDs overlap across models")
                selected_source.append(record)
                source_split[example_id] = partition

    target_schema = schema_with_thresholds(
        {"perplexity": resolved.safe_perplexity_delta}, base=DEFAULT_TARGET_SCHEMA
    )
    classifier_matrices = build_training_matrices(
        selected_source,
        source_split,
        target_schema=target_schema,
        target_name="safe_mutation",
    )
    regressor_matrices = build_training_matrices(
        selected_source,
        source_split,
        target_schema=target_schema,
        target_name="perplexity",
    )
    if {int(value) for value in classifier_matrices.train.target_values} != {0, 1}:
        raise StaticFeatureStudyError("cross-model source training requires both classes")
    classifier = train_lightgbm(
        classifier_matrices.train,
        classifier_matrices.validation,
        config=LightGBMConfig(
            ModelTask.CLASSIFICATION,
            num_threads=resolved.threads,
            seed=resolved.seed,
        ),
    )
    regressor = train_lightgbm(
        regressor_matrices.train,
        regressor_matrices.validation,
        config=LightGBMConfig(
            ModelTask.REGRESSION,
            num_threads=resolved.threads,
            seed=resolved.seed,
        ),
    )

    target_partitions, target_groups = _partition_maps(target.split)
    target_test = tuple(
        record
        for record in select_feature_profile_records(
            target.records, FeatureProfile.STATIC_ACTIVATION_GRADIENT
        )
        if target_partitions.get(str(record.get("example_id"))) is SplitPartition.TEST
    )
    if not target_test:
        raise StaticFeatureStudyError("cross-model target test partition is empty")
    rows = tuple(
        transform_inference_record(record, classifier_matrices.preprocessor)
        for record in target_test
    )
    regression_rows = tuple(
        transform_inference_record(record, regressor_matrices.preprocessor)
        for record in target_test
    )
    classifier_predictions = classifier.predict(rows)
    regressor_predictions = regressor.predict(regression_rows)
    derived = tuple(derive_supervised_targets(record, target_schema) for record in target_test)
    labels = tuple(1 if item.safe_mutation else 0 for item in derived)
    target_values: list[float] = []
    for item in derived:
        value = item.value("perplexity").value
        if value is None:
            raise StaticFeatureStudyError("cross-model target labels are incomplete")
        target_values.append(value)
    targets = tuple(target_values)
    example_ids = tuple(str(record.get("example_id")) for record in target_test)
    group_ids = tuple(target_groups[example_id] for example_id in example_ids)
    cross_classifier = evaluate_classification(
        labels,
        classifier_predictions,
        top_n=resolved.top_n,
        group_ids=group_ids,
        bootstrap_repetitions=resolved.bootstrap_repetitions,
        confidence=resolved.bootstrap_confidence,
        seed=resolved.seed,
    )
    cross_regressor = evaluate_regression(
        targets,
        regressor_predictions,
        group_ids=group_ids,
        bootstrap_repetitions=resolved.bootstrap_repetitions,
        confidence=resolved.bootstrap_confidence,
        seed=resolved.seed,
    )
    within = run_feature_profile_study(
        target.records,
        target.split,
        FeatureProfile.STATIC_ACTIVATION_GRADIENT,
        resolved,
    )
    cross_result = StaticFeatureStudyResult(
        target_identity[0],
        target_identity[1],
        target_identity[2],
        len(selected_source),
        len(classifier_matrices.train.example_ids),
        len(classifier_matrices.validation.example_ids),
        len(target_test),
        tuple(item.name for item in classifier_matrices.preprocessor.numeric),
        cross_classifier,
        cross_regressor,
        FeatureProfile.STATIC_ACTIVATION_GRADIENT,
        labels,
        targets,
        tuple(float(value) for value in classifier_predictions),
        tuple(float(value) for value in regressor_predictions),
        group_ids,
        example_ids,
    )
    preprocessor_record = classifier_matrices.preprocessor.to_record()
    preprocessor_sha = hashlib.sha256(
        canonical_identity_json(preprocessor_record).encode("utf-8")
    ).hexdigest()
    return CrossModelTransferResult(
        source_identities,
        target_identity,
        len(classifier_matrices.train.example_ids),
        len(classifier_matrices.validation.example_ids),
        len(target_test),
        preprocessor_record,
        preprocessor_sha,
        within.classifier,
        cross_classifier,
        within.regressor,
        cross_regressor,
        _rename_degradations(paired_feature_gains(cross_result, within, resolved)),
    )
