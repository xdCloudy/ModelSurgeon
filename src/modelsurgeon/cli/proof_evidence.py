"""Train and evaluate the complete empirical First Surgeon LightGBM proof."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, cast

import typer

from modelsurgeon.cli.surgeon import (
    SurgeonCommandError,
    _predict_values,
    _training_models,
    load_surgeon_records,
)
from modelsurgeon.datasets.grouped_splits import (
    GROUPED_SPLIT_ALGORITHM,
    GROUPED_SPLIT_VERSION,
    GroupedSplitManifest,
    GroupedSplitMode,
    SplitGroup,
    SplitPartition,
    SplitRatios,
)
from modelsurgeon.experiments.candidates import MutationCandidate
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.features.schema import FeatureKind, FeatureRecord
from modelsurgeon.graph import ComponentId
from modelsurgeon.instrumentation.memory_telemetry import (
    MemoryTelemetryConfig,
    MemoryTelemetryReport,
    collect_memory_telemetry,
)
from modelsurgeon.surgeon.matrix import (
    ExampleRecord,
    TrainingMatrices,
    build_training_matrices,
    transform_inference_record,
)
from modelsurgeon.surgeon.metrics import (
    MetricEstimate,
    MetricReport,
    evaluate_classification,
    evaluate_regression,
)
from modelsurgeon.surgeon.models import (
    LightGBMConfig,
    ModelTask,
    SerializableModel,
    train_lightgbm,
)
from modelsurgeon.surgeon.ranking import (
    ComponentAggregation,
    MagnitudeNormalization,
    MagnitudeRankingConfig,
    RankingResult,
    rank_magnitude,
    rank_random,
)
from modelsurgeon.surgeon.registry import SurgeonModelRegistry
from modelsurgeon.surgeon.targets import DEFAULT_TARGET_SCHEMA, schema_with_thresholds

FIRST_SURGEON_EVIDENCE_VERSION = "1"


class FirstSurgeonEvidenceError(ValueError):
    """Raised when proof artifacts cannot support a complete comparable evidence report."""


@dataclass(frozen=True, slots=True)
class FirstSurgeonEvidenceConfig:
    safe_perplexity_delta: float = 0.25
    threads: int = 4
    seed: int = 42
    top_n: int = 50
    bootstrap_repetitions: int = 1000
    bootstrap_confidence: float = 0.95
    magnitude_normalization: MagnitudeNormalization = MagnitudeNormalization.MEAN_ABSOLUTE
    magnitude_component_aggregation: ComponentAggregation = ComponentAggregation.MEAN

    def __post_init__(self) -> None:
        if not math.isfinite(self.safe_perplexity_delta) or self.safe_perplexity_delta < 0:
            raise FirstSurgeonEvidenceError(
                "safe_perplexity_delta must be finite and non-negative"
            )
        if self.threads <= 0 or self.top_n <= 0 or self.bootstrap_repetitions <= 0:
            raise FirstSurgeonEvidenceError(
                "threads, top_n, and bootstrap_repetitions must be positive"
            )
        if isinstance(self.seed, bool) or self.seed < 0 or self.seed >= 1 << 64:
            raise FirstSurgeonEvidenceError("seed must be an unsigned 64-bit integer")
        if not 0.0 < self.bootstrap_confidence < 1.0:
            raise FirstSurgeonEvidenceError("bootstrap_confidence must be within (0, 1)")


@dataclass(frozen=True, slots=True)
class TrainingResourceSummary:
    wall_seconds: float
    peak_rss_bytes: int | None
    peak_cuda_allocated_bytes: int | None
    peak_cuda_reserved_bytes: int | None
    telemetry_version: str

    def to_record(self) -> dict[str, object]:
        return {
            "wall_seconds": self.wall_seconds,
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_cuda_allocated_bytes": self.peak_cuda_allocated_bytes,
            "peak_cuda_reserved_bytes": self.peak_cuda_reserved_bytes,
            "telemetry_version": self.telemetry_version,
        }


@dataclass(frozen=True, slots=True)
class TrainedProofModel:
    model: SerializableModel
    report: MetricReport
    digest: str
    resources: TrainingResourceSummary

    def to_record(self) -> dict[str, object]:
        return {
            "artifact_digest": self.digest,
            "model_kind": self.model.to_record().get("kind"),
            "target": self.model.target_name,
            "metrics": self.report.to_record(),
            "resources": self.resources.to_record(),
        }


@dataclass(frozen=True, slots=True)
class FirstSurgeonEvidenceResult:
    classifier: TrainedProofModel
    regressor: TrainedProofModel
    random_ranking: RankingResult
    random_metrics: MetricReport
    magnitude_ranking: RankingResult
    magnitude_metrics: MetricReport
    test_example_ids: tuple[str, ...]
    test_group_ids: tuple[str, ...]
    dataset_sha256: str
    split_sha256: str
    model_identities: tuple[tuple[str, str, str | None], ...]
    tool_revisions: tuple[str, ...]
    config: FirstSurgeonEvidenceConfig
    classifier_smoke_prediction: float
    regressor_smoke_prediction: float
    version: str = FIRST_SURGEON_EVIDENCE_VERSION

    def to_record(self) -> dict[str, object]:
        return {
            "record_type": "first_surgeon_evidence",
            "version": self.version,
            "dataset_sha256": self.dataset_sha256,
            "split_sha256": self.split_sha256,
            "test_example_count": len(self.test_example_ids),
            "test_group_count": len(set(self.test_group_ids)),
            "test_example_ids": list(self.test_example_ids),
            "model_identities": [
                {
                    "identifier": identifier,
                    "revision": revision,
                    "quantization": quantization,
                }
                for identifier, revision, quantization in self.model_identities
            ],
            "tool_revisions": list(self.tool_revisions),
            "config": {
                "safe_perplexity_delta": self.config.safe_perplexity_delta,
                "threads": self.config.threads,
                "seed": self.config.seed,
                "top_n": self.config.top_n,
                "bootstrap_repetitions": self.config.bootstrap_repetitions,
                "bootstrap_confidence": self.config.bootstrap_confidence,
                "magnitude_normalization": self.config.magnitude_normalization.value,
                "magnitude_component_aggregation": (
                    self.config.magnitude_component_aggregation.value
                ),
            },
            "learned": {
                "classifier": self.classifier.to_record(),
                "regressor": self.regressor.to_record(),
            },
            "baselines": {
                "random": {
                    "ranking": self.random_ranking.to_record(),
                    "metrics": self.random_metrics.to_record(),
                },
                "magnitude": {
                    "ranking": self.magnitude_ranking.to_record(),
                    "metrics": self.magnitude_metrics.to_record(),
                },
            },
            "inference_smoke": {
                "classifier_prediction": self.classifier_smoke_prediction,
                "regressor_prediction": self.regressor_smoke_prediction,
            },
        }


@dataclass(frozen=True, slots=True)
class _RankingCandidate:
    candidate_id: str
    affected_components: tuple[ComponentId, ...]


@dataclass(frozen=True, slots=True)
class _RankingFeature:
    component_id: ComponentId
    name: str
    kind: FeatureKind
    value: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise FirstSurgeonEvidenceError(f"cannot hash proof artifact {path}: {error}") from error
    return digest.hexdigest()


def _integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise FirstSurgeonEvidenceError(f"{label} must be an integer")
    return value


def _string_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise FirstSurgeonEvidenceError(f"{label} must be a non-empty-string list")
    return tuple(cast(list[str], value))


def load_grouped_proof_split(path: Path) -> GroupedSplitManifest:
    """Load the full grouped manifest rather than flattening away leakage-group identity."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FirstSurgeonEvidenceError("proof split is unreadable or invalid JSON") from error
    if not isinstance(raw, Mapping):
        raise FirstSurgeonEvidenceError("proof split must be a JSON object")
    if raw.get("version") != GROUPED_SPLIT_VERSION:
        raise FirstSurgeonEvidenceError("proof split version is unsupported")
    if raw.get("algorithm") != GROUPED_SPLIT_ALGORITHM:
        raise FirstSurgeonEvidenceError("proof split algorithm is unsupported")
    try:
        mode = GroupedSplitMode(str(raw["mode"]))
    except (KeyError, ValueError) as error:
        raise FirstSurgeonEvidenceError("proof split mode is invalid") from error
    seed = _integer(raw.get("seed"), "proof split seed")
    ratios_raw = raw.get("ratios")
    groups_raw = raw.get("groups")
    if not isinstance(ratios_raw, Mapping) or not isinstance(groups_raw, list):
        raise FirstSurgeonEvidenceError("proof split ratios/groups are malformed")
    try:
        ratios = SplitRatios(
            float(ratios_raw["train"]),
            float(ratios_raw["validation"]),
            float(ratios_raw["test"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FirstSurgeonEvidenceError("proof split ratios are invalid") from error
    groups: list[SplitGroup] = []
    for item in groups_raw:
        if not isinstance(item, Mapping):
            raise FirstSurgeonEvidenceError("proof split group must be an object")
        group_id = item.get("group_id")
        partition = item.get("partition")
        if not isinstance(group_id, str) or not isinstance(partition, str):
            raise FirstSurgeonEvidenceError("proof split group identity/partition is invalid")
        try:
            resolved_partition = SplitPartition(partition)
        except ValueError as error:
            raise FirstSurgeonEvidenceError("proof split contains unknown partition") from error
        groups.append(
            SplitGroup(
                group_id,
                resolved_partition,
                tuple(sorted(_string_list(item.get("keys"), "proof split group keys"))),
                tuple(
                    sorted(
                        _string_list(
                            item.get("example_ids"),
                            "proof split group example_ids",
                        )
                    )
                ),
            )
        )
    return GroupedSplitManifest(
        mode,
        seed,
        ratios,
        tuple(sorted(groups, key=lambda item: item.group_id)),
    )


def _test_group_map(split: GroupedSplitManifest) -> dict[str, str]:
    output: dict[str, str] = {}
    for group in split.groups:
        if group.partition is not SplitPartition.TEST:
            continue
        for example_id in group.example_ids:
            output[example_id] = group.group_id
    if not output:
        raise FirstSurgeonEvidenceError("proof split has no held-out test examples")
    return output


def _test_records(
    records: Sequence[Mapping[str, object]],
    groups: Mapping[str, str],
) -> tuple[Mapping[str, object], ...]:
    by_id: dict[str, Mapping[str, object]] = {}
    for record in records:
        example_id = record.get("example_id")
        if not isinstance(example_id, str) or not example_id:
            raise FirstSurgeonEvidenceError("proof dataset record is missing example_id")
        if example_id in by_id:
            raise FirstSurgeonEvidenceError(f"duplicate proof example {example_id!r}")
        by_id[example_id] = record
    missing = set(groups) - set(by_id)
    if missing:
        raise FirstSurgeonEvidenceError(
            f"proof split references absent test examples: {sorted(missing)[:5]}"
        )
    return tuple(by_id[example_id] for example_id in sorted(groups))


def _measured_matrix(
    matrices: TrainingMatrices,
    *,
    require_both_classes: bool = False,
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...], tuple[str, ...]]:
    rows: list[tuple[float, ...]] = []
    targets: list[float] = []
    groups: list[str] = []
    for row, target, mask, group in zip(
        matrices.test.values,
        matrices.test.target_values,
        matrices.test.target_mask,
        matrices.test.group_ids,
        strict=True,
    ):
        if not mask:
            raise FirstSurgeonEvidenceError(
                f"held-out example is missing {matrices.test.target_name!r}; identical-set comparison requires complete labels"
            )
        rows.append(tuple(row))
        targets.append(target)
        groups.append(group)
    if not rows:
        raise FirstSurgeonEvidenceError("held-out test matrix is empty")
    if require_both_classes and {int(value) for value in targets} != {0, 1}:
        raise FirstSurgeonEvidenceError(
            "held-out safe-mutation labels contain only one class; AUC cannot satisfy #104"
        )
    return tuple(rows), tuple(targets), tuple(groups)


def _validate_training_classes(matrices: TrainingMatrices) -> None:
    labels = {
        int(target)
        for target, mask in zip(
            matrices.train.target_values,
            matrices.train.target_mask,
            strict=True,
        )
        if mask
    }
    if labels != {0, 1}:
        raise FirstSurgeonEvidenceError(
            "training safe-mutation labels must contain both safe and unsafe examples"
        )


def _resource_summary(report: MemoryTelemetryReport) -> TrainingResourceSummary:
    wall = report.samples[-1].elapsed_seconds
    return TrainingResourceSummary(
        wall,
        report.peak_rss_bytes,
        report.peak_cuda_allocated_bytes,
        report.peak_cuda_reserved_bytes,
        report.version,
    )


def _train_lightgbm_with_resources(
    matrices: TrainingMatrices,
    *,
    task: ModelTask,
    threads: int,
    seed: int,
) -> tuple[SerializableModel, TrainingResourceSummary]:
    holder: list[SerializableModel] = []

    def operation() -> None:
        holder.append(
            train_lightgbm(
                matrices.train,
                matrices.validation,
                config=LightGBMConfig(task, num_threads=threads, seed=seed),
            )
        )

    telemetry = collect_memory_telemetry(
        f"first-surgeon-lightgbm-{task.value}",
        operation,
        MemoryTelemetryConfig(
            sampling_enabled=True,
            sample_interval_seconds=0.25,
            max_samples=4096,
        ),
    )
    if len(holder) != 1:
        raise FirstSurgeonEvidenceError("LightGBM training did not produce exactly one model")
    return holder[0], _resource_summary(telemetry)


def _metric_mapping(report: MetricReport) -> dict[str, float | None]:
    return {item.name: item.value for item in report.metrics}


def _training_identity(records: Sequence[Mapping[str, object]]) -> tuple[
    tuple[tuple[str, str, str | None], ...],
    tuple[str, ...],
]:
    models: set[tuple[str, str, str | None]] = set()
    tools: set[str] = set()
    for record in records:
        model = record.get("model")
        versions = record.get("versions")
        if not isinstance(model, Mapping) or not isinstance(versions, Mapping):
            raise FirstSurgeonEvidenceError("proof records require model/version provenance")
        identifier = model.get("identifier")
        revision = model.get("revision")
        quantization = model.get("quantization")
        tool_revision = versions.get("tool_revision")
        if (
            not isinstance(identifier, str)
            or not isinstance(revision, str)
            or (quantization is not None and not isinstance(quantization, str))
            or not isinstance(tool_revision, str)
        ):
            raise FirstSurgeonEvidenceError("proof model/tool provenance is malformed")
        models.add((identifier, revision, quantization))
        tools.add(tool_revision)
    return (
        tuple(sorted(models, key=lambda item: (item[0], item[1], item[2] or ""))),
        tuple(sorted(tools)),
    )


def _ranking_inputs(
    records: Sequence[Mapping[str, object]],
) -> tuple[
    tuple[_RankingCandidate, ...],
    tuple[_RankingFeature, ...],
]:
    candidates: list[_RankingCandidate] = []
    features: list[_RankingFeature] = []
    for record in records:
        example_id = record.get("example_id")
        components_raw = record.get("components")
        features_raw = record.get("pre_mutation_features")
        if (
            not isinstance(example_id, str)
            or not isinstance(components_raw, list)
            or not isinstance(features_raw, list)
        ):
            raise FirstSurgeonEvidenceError("test example ranking fields are malformed")
        try:
            components = tuple(sorted(ComponentId.parse(item) for item in components_raw))
        except (TypeError, ValueError) as error:
            raise FirstSurgeonEvidenceError("test example contains invalid components") from error
        candidates.append(_RankingCandidate(example_id, components))
        for raw in features_raw:
            if not isinstance(raw, Mapping):
                raise FirstSurgeonEvidenceError("pre-mutation feature must be an object")
            if raw.get("kind") != FeatureKind.SCALAR.value:
                continue
            component_raw = raw.get("component_id")
            name = raw.get("name")
            value = raw.get("value")
            if (
                not isinstance(component_raw, str)
                or not isinstance(name, str)
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
            ):
                raise FirstSurgeonEvidenceError("scalar ranking feature is malformed")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise FirstSurgeonEvidenceError("scalar ranking feature is non-finite")
            features.append(
                _RankingFeature(
                    ComponentId.parse(component_raw),
                    name,
                    FeatureKind.SCALAR,
                    numeric,
                )
            )
    return tuple(candidates), tuple(features)


def _rank_scores(result: RankingResult, expected_ids: Sequence[str]) -> tuple[float, ...]:
    if result.exclusions:
        first = result.exclusions[0]
        raise FirstSurgeonEvidenceError(
            f"baseline {result.method} excluded held-out candidate {first.candidate_id}: {first.reason}"
        )
    rank_by_id = {entry.candidate_id: entry.rank for entry in result.entries}
    if set(rank_by_id) != set(expected_ids):
        raise FirstSurgeonEvidenceError(
            f"baseline {result.method} did not rank the identical held-out candidate set"
        )
    denominator = max(1, len(expected_ids) - 1)
    return tuple(
        1.0 - (rank_by_id[example_id] - 1) / denominator
        for example_id in expected_ids
    )


def _ranking_metric_subset(report: MetricReport, top_n: int) -> MetricReport:
    wanted = {"auc", "pr_auc", f"precision_at_{top_n}", f"recall_at_{top_n}"}
    selected = tuple(item for item in report.metrics if item.name in wanted)
    if {item.name for item in selected} != wanted:
        raise FirstSurgeonEvidenceError("ranking metric report is missing required entries")
    return MetricReport(tuple(sorted(selected, key=lambda item: item.name)))


def _require_defined(report: MetricReport, names: Sequence[str], label: str) -> None:
    for name in names:
        metric = report.metric(name)
        if metric.value is None:
            raise FirstSurgeonEvidenceError(
                f"{label} metric {name} is undefined: {metric.reason}"
            )


def _publish_model(
    registry: SurgeonModelRegistry,
    model: SerializableModel,
    matrices: TrainingMatrices,
    report: MetricReport,
    resources: TrainingResourceSummary,
    records: Sequence[Mapping[str, object]],
    *,
    baseline: str,
    config: FirstSurgeonEvidenceConfig,
    dataset: Path,
    split: Path,
) -> str:
    artifact = registry.publish(
        model,
        matrices.preprocessor,
        schema_with_thresholds(
            {"perplexity": config.safe_perplexity_delta},
            base=DEFAULT_TARGET_SCHEMA,
        ),
        training_models=_training_models(records),
        metrics=_metric_mapping(report),
        split_manifest=matrices.split_manifest,
        provenance={
            "baseline": baseline,
            "seed": config.seed,
            "threads": config.threads,
            "target": model.target_name,
            "dataset": str(dataset),
            "split": str(split),
            "resource_use": resources.to_record(),
            "proof_evidence_version": FIRST_SURGEON_EVIDENCE_VERSION,
        },
    )
    return str(artifact.metadata.digest)


def _smoke_prediction(
    registry: SurgeonModelRegistry,
    digest: str,
    record: Mapping[str, object],
) -> float:
    loaded = registry.load(digest)
    row = transform_inference_record(
        cast(ExampleRecord, record),
        loaded.preprocessor,
        refuse_missing=True,
    )
    value = _predict_values(loaded.model, (row,))[0]
    if not math.isfinite(value):
        raise FirstSurgeonEvidenceError("immutable-bundle inference smoke produced non-finite output")
    return value


def run_first_surgeon_evidence(
    dataset: Path,
    split: Path,
    registry_root: Path,
    config: FirstSurgeonEvidenceConfig,
) -> FirstSurgeonEvidenceResult:
    """Train both LightGBM surgeons and compare them against identical held-out baselines."""

    records = load_surgeon_records(dataset)
    if not records:
        raise FirstSurgeonEvidenceError("proof dataset contains no examples")
    manifest = load_grouped_proof_split(split)
    target_schema = schema_with_thresholds(
        {"perplexity": config.safe_perplexity_delta},
        base=DEFAULT_TARGET_SCHEMA,
    )
    example_records = cast(Sequence[ExampleRecord], records)
    regression_matrices = build_training_matrices(
        example_records,
        manifest,
        target_schema=target_schema,
        target_name="perplexity",
    )
    classification_matrices = build_training_matrices(
        example_records,
        manifest,
        target_schema=target_schema,
        target_name="safe_mutation",
    )
    _validate_training_classes(classification_matrices)
    if config.top_n > len(classification_matrices.test.example_ids):
        raise FirstSurgeonEvidenceError(
            f"top_n={config.top_n} exceeds held-out candidate count {len(classification_matrices.test.example_ids)}"
        )

    regression_model, regression_resources = _train_lightgbm_with_resources(
        regression_matrices,
        task=ModelTask.REGRESSION,
        threads=config.threads,
        seed=config.seed,
    )
    classifier_model, classifier_resources = _train_lightgbm_with_resources(
        classification_matrices,
        task=ModelTask.CLASSIFICATION,
        threads=config.threads,
        seed=config.seed,
    )

    regression_rows, regression_targets, regression_groups = _measured_matrix(
        regression_matrices
    )
    classifier_rows, classifier_targets, classifier_groups = _measured_matrix(
        classification_matrices,
        require_both_classes=True,
    )
    if regression_matrices.test.example_ids != classification_matrices.test.example_ids:
        raise FirstSurgeonEvidenceError("regression/classification held-out example sets differ")
    if regression_groups != classifier_groups:
        raise FirstSurgeonEvidenceError("regression/classification held-out group identities differ")

    regression_predictions = _predict_values(regression_model, regression_rows)
    classifier_probabilities = _predict_values(classifier_model, classifier_rows)
    regression_report = evaluate_regression(
        regression_targets,
        regression_predictions,
        group_ids=regression_groups,
        bootstrap_repetitions=config.bootstrap_repetitions,
        confidence=config.bootstrap_confidence,
        seed=config.seed,
    )
    labels = tuple(int(value) for value in classifier_targets)
    classifier_report = evaluate_classification(
        labels,
        classifier_probabilities,
        top_n=config.top_n,
        group_ids=classifier_groups,
        bootstrap_repetitions=config.bootstrap_repetitions,
        confidence=config.bootstrap_confidence,
        seed=config.seed,
    )
    _require_defined(regression_report, ("mae", "rmse"), "regression")
    _require_defined(
        classifier_report,
        ("auc", "pr_auc", f"precision_at_{config.top_n}"),
        "classification",
    )

    test_groups = _test_group_map(manifest)
    test_records = _test_records(records, test_groups)
    test_ids = tuple(cast(str, record["example_id"]) for record in test_records)
    if test_ids != regression_matrices.test.example_ids:
        raise FirstSurgeonEvidenceError(
            "held-out record ordering disagrees with the matrix; comparison cannot be certified identical"
        )
    candidates, features = _ranking_inputs(test_records)
    random_ranking = rank_random(
        tuple(candidate.candidate_id for candidate in candidates),
        seed=config.seed,
        select_count=config.top_n,
    )
    magnitude_ranking = rank_magnitude(
        cast(Sequence[MutationCandidate], candidates),
        cast(Sequence[FeatureRecord], features),
        config=MagnitudeRankingConfig(
            config.magnitude_normalization,
            config.magnitude_component_aggregation,
            config.top_n,
        ),
    )
    random_scores = _rank_scores(random_ranking, test_ids)
    magnitude_scores = _rank_scores(magnitude_ranking, test_ids)
    random_report = _ranking_metric_subset(
        evaluate_classification(
            labels,
            random_scores,
            top_n=config.top_n,
            group_ids=classifier_groups,
            bootstrap_repetitions=config.bootstrap_repetitions,
            confidence=config.bootstrap_confidence,
            seed=config.seed + 101,
        ),
        config.top_n,
    )
    magnitude_report = _ranking_metric_subset(
        evaluate_classification(
            labels,
            magnitude_scores,
            top_n=config.top_n,
            group_ids=classifier_groups,
            bootstrap_repetitions=config.bootstrap_repetitions,
            confidence=config.bootstrap_confidence,
            seed=config.seed + 202,
        ),
        config.top_n,
    )

    registry = SurgeonModelRegistry(str(registry_root))
    regression_digest = _publish_model(
        registry,
        regression_model,
        regression_matrices,
        regression_report,
        regression_resources,
        records,
        baseline="lightgbm-regressor",
        config=config,
        dataset=dataset,
        split=split,
    )
    classifier_digest = _publish_model(
        registry,
        classifier_model,
        classification_matrices,
        classifier_report,
        classifier_resources,
        records,
        baseline="lightgbm-classifier",
        config=config,
        dataset=dataset,
        split=split,
    )
    classifier_smoke = _smoke_prediction(registry, classifier_digest, test_records[0])
    regression_smoke = _smoke_prediction(registry, regression_digest, test_records[0])
    model_identities, tool_revisions = _training_identity(records)
    return FirstSurgeonEvidenceResult(
        TrainedProofModel(
            classifier_model,
            classifier_report,
            classifier_digest,
            classifier_resources,
        ),
        TrainedProofModel(
            regression_model,
            regression_report,
            regression_digest,
            regression_resources,
        ),
        random_ranking,
        random_report,
        magnitude_ranking,
        magnitude_report,
        test_ids,
        classifier_groups,
        _sha256(dataset),
        _sha256(split),
        model_identities,
        tool_revisions,
        config,
        classifier_smoke,
        regression_smoke,
    )


def write_first_surgeon_evidence(path: Path, result: FirstSurgeonEvidenceResult) -> None:
    payload = canonical_identity_json(result.to_record()) + "\n"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
    except FileExistsError as error:
        raise FirstSurgeonEvidenceError(f"evidence output already exists: {path}") from error
    except OSError as error:
        raise FirstSurgeonEvidenceError(f"cannot write evidence report: {error}") from error


def first_surgeon_evidence_command(
    dataset: Annotated[
        Path,
        typer.Argument(help="First Surgeon proof examples.jsonl"),
    ],
    split: Annotated[
        Path,
        typer.Option("--split", help="Grouped proof split.json"),
    ],
    registry: Annotated[
        Path,
        typer.Option("--registry", help="Immutable registry for both LightGBM bundles"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="New JSON evidence report path"),
    ] = Path("first-surgeon-evidence.json"),
    safe_perplexity_delta: Annotated[
        float,
        typer.Option("--safe-perplexity-delta", min=0.0),
    ] = 0.25,
    threads: Annotated[int, typer.Option("--threads", min=1)] = 4,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 42,
    top_n: Annotated[int, typer.Option("--top-n", min=1)] = 50,
    bootstrap_repetitions: Annotated[
        int,
        typer.Option("--bootstrap-repetitions", min=1),
    ] = 1000,
    bootstrap_confidence: Annotated[
        float,
        typer.Option("--bootstrap-confidence", min=0.5, max=0.999),
    ] = 0.95,
    magnitude_normalization: Annotated[
        MagnitudeNormalization,
        typer.Option("--magnitude-normalization"),
    ] = MagnitudeNormalization.MEAN_ABSOLUTE,
    magnitude_component_aggregation: Annotated[
        ComponentAggregation,
        typer.Option("--magnitude-component-aggregation"),
    ] = ComponentAggregation.MEAN,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Also emit the full evidence record to stdout"),
    ] = False,
) -> None:
    """Train both proof LightGBMs and publish the complete held-out comparison evidence."""

    try:
        result = run_first_surgeon_evidence(
            dataset,
            split,
            registry,
            FirstSurgeonEvidenceConfig(
                safe_perplexity_delta=safe_perplexity_delta,
                threads=threads,
                seed=seed,
                top_n=top_n,
                bootstrap_repetitions=bootstrap_repetitions,
                bootstrap_confidence=bootstrap_confidence,
                magnitude_normalization=magnitude_normalization,
                magnitude_component_aggregation=magnitude_component_aggregation,
            ),
        )
        write_first_surgeon_evidence(output, result)
        record = result.to_record()
        if output_json:
            typer.echo(canonical_identity_json(record))
            return
        classifier_metrics = result.classifier.report
        regressor_metrics = result.regressor.report
        typer.echo(
            "First Surgeon evidence complete: "
            f"AUC={classifier_metrics.metric('auc').value:.6g} "
            f"P@{top_n}={classifier_metrics.metric(f'precision_at_{top_n}').value:.6g} "
            f"MAE={regressor_metrics.metric('mae').value:.6g} "
            f"RMSE={regressor_metrics.metric('rmse').value:.6g}"
        )
        typer.echo(f"classifier: {result.classifier.digest}")
        typer.echo(f"regressor: {result.regressor.digest}")
        typer.echo(f"evidence: {output}")
    except (OSError, RuntimeError, ValueError, SurgeonCommandError) as error:
        typer.echo(f"first-surgeon-evidence error: {error}", err=True)
        raise typer.Exit(2) from error
