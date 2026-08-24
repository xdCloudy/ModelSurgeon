"""CLI helpers for training and scoring baseline surgeon models."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, cast

import typer

from modelsurgeon.datasets.grouped_splits import (
    GROUPED_SPLIT_ALGORITHM,
    GROUPED_SPLIT_VERSION,
    GroupedSplitManifest,
    GroupedSplitMode,
    SplitGroup,
    SplitPartition,
    SplitRatios,
)
from modelsurgeon.datasets.parquet_store import (
    PartitionedParquetStore,
    PartitionKind,
    PartitionPredicate,
)
from modelsurgeon.experiments.identity import canonical_identity_json
from modelsurgeon.surgeon.matrix import (
    ExampleRecord,
    TrainingMatrices,
    build_training_matrices,
    transform_inference_record,
)
from modelsurgeon.surgeon.metrics import (
    MetricReport,
    evaluate_classification,
    evaluate_regression,
)
from modelsurgeon.surgeon.models import (
    LightGBMConfig,
    LogisticConfig,
    LogisticSurgeonModel,
    MLPConfig,
    ModelTask,
    SerializableModel,
    train_lightgbm,
    train_linear,
    train_logistic,
    train_mlp,
)
from modelsurgeon.surgeon.registry import (
    SurgeonModelRegistry,
    TrainingModelIdentity,
)
from modelsurgeon.surgeon.targets import (
    DEFAULT_TARGET_SCHEMA,
    schema_with_thresholds,
)


class SurgeonCommandError(ValueError):
    """Raised for actionable train/predict CLI contract failures."""


def _predict_values(
    model: SerializableModel, rows: Sequence[Sequence[float]]
) -> tuple[float, ...]:
    if isinstance(model, LogisticSurgeonModel):
        return model.predict_proba(rows)
    return tuple(float(value) for value in model.predict(rows))


def _payload_record(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SurgeonCommandError("dataset entries must be JSON objects")
    payload_json = value.get("payload_json")
    if payload_json is None:
        return cast(Mapping[str, object], value)
    if not isinstance(payload_json, str):
        raise SurgeonCommandError("dataset payload_json must be a string")
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError as error:
        raise SurgeonCommandError("dataset payload_json is invalid JSON") from error
    if not isinstance(payload, Mapping):
        raise SurgeonCommandError("dataset payload_json must decode to an object")
    return cast(Mapping[str, object], payload)


def load_surgeon_records(path: Path) -> tuple[Mapping[str, object], ...]:
    """Load canonical example payloads from Parquet store, JSON, or JSONL."""

    if path.is_dir():
        rows = PartitionedParquetStore(path).read_rows(
            PartitionPredicate(kind=PartitionKind.EXAMPLES)
        )
        return tuple(_payload_record(row) for row in rows)
    if not path.is_file():
        raise SurgeonCommandError(f"dataset path does not exist: {path}")
    if path.suffix.lower() == ".jsonl":
        records: list[Mapping[str, object]] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise SurgeonCommandError(
                    f"invalid JSONL dataset line {line_number}"
                ) from error
            records.append(_payload_record(value))
        return tuple(records)
    if path.suffix.lower() == ".json":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise SurgeonCommandError("dataset JSON is invalid") from error
        if isinstance(value, list):
            return tuple(_payload_record(item) for item in value)
        if isinstance(value, Mapping) and isinstance(value.get("examples"), list):
            return tuple(
                _payload_record(item)
                for item in cast(list[object], value["examples"])
            )
        return (_payload_record(value),)
    raise SurgeonCommandError(
        "dataset input must be a Parquet store directory, .json, or .jsonl"
    )


def _split_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SurgeonCommandError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SurgeonCommandError(f"{label} must be finite")
    return result


def _split_strings(value: object, label: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise SurgeonCommandError(f"{label} must be a non-empty-string list")
    return tuple(cast(list[str], value))


def _grouped_split(value: Mapping[str, object]) -> GroupedSplitManifest:
    if value.get("version") != GROUPED_SPLIT_VERSION:
        raise SurgeonCommandError("grouped split version is unsupported")
    if value.get("algorithm") != GROUPED_SPLIT_ALGORITHM:
        raise SurgeonCommandError("grouped split algorithm is unsupported")
    mode_raw = value.get("mode")
    seed = value.get("seed")
    ratios_raw = value.get("ratios")
    groups_raw = value.get("groups")
    if not isinstance(mode_raw, str):
        raise SurgeonCommandError("grouped split mode is invalid")
    if (
        not isinstance(seed, int)
        or isinstance(seed, bool)
        or seed < 0
        or seed >= 1 << 64
    ):
        raise SurgeonCommandError("grouped split seed must be unsigned 64-bit")
    if not isinstance(ratios_raw, Mapping) or not isinstance(groups_raw, list):
        raise SurgeonCommandError("grouped split ratios/groups are malformed")
    try:
        mode = GroupedSplitMode(mode_raw)
        ratios = SplitRatios(
            _split_number(ratios_raw["train"], "train split ratio"),
            _split_number(ratios_raw["validation"], "validation split ratio"),
            _split_number(ratios_raw["test"], "test split ratio"),
        )
    except (KeyError, ValueError) as error:
        raise SurgeonCommandError("grouped split mode/ratios are invalid") from error

    groups: list[SplitGroup] = []
    for raw in groups_raw:
        if not isinstance(raw, Mapping):
            raise SurgeonCommandError("grouped split entries must be objects")
        group_id = raw.get("group_id")
        partition = raw.get("partition")
        if (
            not isinstance(group_id, str)
            or not group_id
            or not isinstance(partition, str)
        ):
            raise SurgeonCommandError("grouped split identity/partition is malformed")
        try:
            resolved_partition = SplitPartition(partition)
        except ValueError as error:
            raise SurgeonCommandError("grouped split partition is unknown") from error
        groups.append(
            SplitGroup(
                group_id,
                resolved_partition,
                tuple(sorted(_split_strings(raw.get("keys"), "grouped split keys"))),
                tuple(
                    sorted(
                        _split_strings(
                            raw.get("example_ids"),
                            "grouped split example_ids",
                        )
                    )
                ),
            )
        )
    return GroupedSplitManifest(
        mode,
        seed,
        ratios,
        tuple(sorted(groups, key=lambda group: group.group_id)),
    )


def load_split_assignments(
    path: Path,
) -> GroupedSplitManifest | dict[str, SplitPartition]:
    """Load inline assignments or preserve a full serialized grouped split manifest."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SurgeonCommandError(
            "split manifest is unreadable or invalid JSON"
        ) from error
    if not isinstance(value, Mapping):
        raise SurgeonCommandError("split manifest must be a JSON object")
    assignments_raw = value.get("assignments")
    assignments: dict[str, SplitPartition] = {}
    if isinstance(assignments_raw, Mapping):
        for example_id, partition in assignments_raw.items():
            if not isinstance(example_id, str) or not isinstance(partition, str):
                raise SurgeonCommandError(
                    "split assignments must map strings to partition strings"
                )
            if example_id in assignments:
                raise SurgeonCommandError("split assignment IDs must be unique")
            try:
                assignments[example_id] = SplitPartition(partition)
            except ValueError as error:
                raise SurgeonCommandError(
                    f"unknown split partition {partition!r}"
                ) from error
        return assignments
    if "groups" in value:
        return _grouped_split(cast(Mapping[str, object], value))
    raise SurgeonCommandError(
        "split JSON must contain either assignments or GroupedSplitManifest groups"
    )


def _thresholds(values: Sequence[str]) -> dict[str, float]:
    output: dict[str, float] = {}
    for item in values:
        try:
            name, raw = item.split("=", 1)
            threshold = float(raw)
        except (ValueError, TypeError) as error:
            raise SurgeonCommandError(
                f"safe threshold {item!r} must use name=value form"
            ) from error
        if not name or name in output:
            raise SurgeonCommandError(
                "safe threshold names must be non-empty and unique"
            )
        output[name] = threshold
    return output


def _training_models(
    records: Sequence[Mapping[str, object]],
) -> tuple[TrainingModelIdentity, ...]:
    identities: set[tuple[str, str, str | None]] = set()
    for record in records:
        model = record.get("model")
        if not isinstance(model, Mapping):
            raise SurgeonCommandError("training example is missing model identity")
        identifier = model.get("identifier")
        revision = model.get("revision")
        quantization = model.get("quantization")
        if (
            not isinstance(identifier, str)
            or not isinstance(revision, str)
            or (quantization is not None and not isinstance(quantization, str))
        ):
            raise SurgeonCommandError(
                "training model identity fields are malformed"
            )
        identities.add((identifier, revision, quantization))
    return tuple(
        TrainingModelIdentity(identifier, revision, quantization)
        for identifier, revision, quantization in sorted(
            identities,
            key=lambda item: (item[0], item[1], item[2] or ""),
        )
    )


def _measured_test(
    matrix_values: Sequence[Sequence[float]],
    targets: Sequence[float],
    masks: Sequence[bool],
    groups: Sequence[str],
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...], tuple[str, ...]]:
    rows: list[tuple[float, ...]] = []
    labels: list[float] = []
    selected_groups: list[str] = []
    for row, target, mask, group in zip(
        matrix_values, targets, masks, groups, strict=True
    ):
        if mask:
            rows.append(tuple(row))
            labels.append(target)
            selected_groups.append(group)
    if not rows:
        raise SurgeonCommandError(
            "test split has no measured labels for the selected target"
        )
    return tuple(rows), tuple(labels), tuple(selected_groups)


def _metric_mapping(report: MetricReport) -> dict[str, float | None]:
    return {metric.name: metric.value for metric in report.metrics}


def _split_group_counts(matrices: TrainingMatrices) -> dict[str, int]:
    return {
        "train": len(set(matrices.train.group_ids)),
        "validation": len(set(matrices.validation.group_ids)),
        "test": len(set(matrices.test.group_ids)),
    }


def train_surgeon_command(
    dataset: Annotated[
        Path, typer.Argument(help="Example Parquet store, JSON, or JSONL")
    ],
    split: Annotated[
        Path, typer.Option("--split", help="Leakage-safe split manifest JSON")
    ],
    registry: Annotated[
        Path, typer.Option("--registry", help="Immutable surgeon registry root")
    ],
    target: Annotated[
        str, typer.Option("--target", help="Target metric or safe_mutation")
    ] = "perplexity",
    baseline: Annotated[
        str,
        typer.Option(
            "--baseline",
            help=(
                "linear, logistic, lightgbm-regressor, lightgbm-classifier, "
                "mlp-regressor, mlp-classifier"
            ),
        ),
    ] = "linear",
    safe_threshold: Annotated[
        list[str] | None,
        typer.Option(
            "--safe-threshold",
            help=(
                "Absolute degradation threshold as metric=value; repeat for multiple metrics"
            ),
        ),
    ] = None,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    threads: Annotated[int, typer.Option("--threads", min=1)] = 4,
    top_n: Annotated[int, typer.Option("--top-n", min=1)] = 50,
    output_json: Annotated[
        bool, typer.Option("--json", help="Emit machine-readable result")
    ] = False,
) -> None:
    """Train a baseline surgeon from leakage-safe examples and publish one immutable version."""

    try:
        records = load_surgeon_records(dataset)
        if not records:
            raise SurgeonCommandError("training dataset contains no examples")
        assignments = load_split_assignments(split)
        thresholds = _thresholds(safe_threshold or ())
        if target == "safe_mutation" and not thresholds:
            raise SurgeonCommandError(
                "safe_mutation training requires at least one --safe-threshold metric=value"
            )
        target_schema = schema_with_thresholds(
            thresholds, base=DEFAULT_TARGET_SCHEMA
        )
        matrices = build_training_matrices(
            cast(Sequence[ExampleRecord], records),
            assignments,
            target_schema=target_schema,
            target_name=target,
        )

        model: SerializableModel
        task: ModelTask
        if baseline == "linear":
            if target == "safe_mutation":
                raise SurgeonCommandError(
                    "linear baseline requires a continuous target"
                )
            task = ModelTask.REGRESSION
            model = train_linear(
                matrices.train, validation=matrices.validation
            )
        elif baseline == "logistic":
            if target != "safe_mutation":
                raise SurgeonCommandError(
                    "logistic baseline requires target=safe_mutation"
                )
            task = ModelTask.CLASSIFICATION
            model = train_logistic(
                matrices.train,
                config=LogisticConfig(seed=seed),
                validation=matrices.validation,
            )
        elif baseline in {"lightgbm-regressor", "lightgbm-classifier"}:
            task = (
                ModelTask.REGRESSION
                if baseline.endswith("regressor")
                else ModelTask.CLASSIFICATION
            )
            if task is ModelTask.CLASSIFICATION and target != "safe_mutation":
                raise SurgeonCommandError(
                    "LightGBM classifier requires target=safe_mutation"
                )
            if task is ModelTask.REGRESSION and target == "safe_mutation":
                raise SurgeonCommandError(
                    "LightGBM regressor requires a continuous target"
                )
            model = train_lightgbm(
                matrices.train,
                matrices.validation,
                config=LightGBMConfig(task, num_threads=threads, seed=seed),
            )
        elif baseline in {"mlp-regressor", "mlp-classifier"}:
            task = (
                ModelTask.REGRESSION
                if baseline.endswith("regressor")
                else ModelTask.CLASSIFICATION
            )
            if task is ModelTask.CLASSIFICATION and target != "safe_mutation":
                raise SurgeonCommandError(
                    "MLP classifier requires target=safe_mutation"
                )
            if task is ModelTask.REGRESSION and target == "safe_mutation":
                raise SurgeonCommandError(
                    "MLP regressor requires a continuous target"
                )
            model = train_mlp(
                matrices.train,
                matrices.validation,
                config=MLPConfig(task=task, seed=seed),
            )
        else:
            raise SurgeonCommandError(
                f"unknown surgeon baseline {baseline!r}"
            )

        test_rows, test_targets, test_groups = _measured_test(
            matrices.test.values,
            matrices.test.target_values,
            matrices.test.target_mask,
            matrices.test.group_ids,
        )
        predictions = _predict_values(model, test_rows)
        if task is ModelTask.CLASSIFICATION:
            report = evaluate_classification(
                tuple(int(value) for value in test_targets),
                predictions,
                top_n=top_n,
                group_ids=test_groups,
                seed=seed,
            )
        else:
            report = evaluate_regression(
                test_targets,
                predictions,
                group_ids=test_groups,
                seed=seed,
            )

        training_models = _training_models(records)
        published = SurgeonModelRegistry(str(registry)).publish(
            model,
            matrices.preprocessor,
            target_schema,
            training_models=training_models,
            metrics=_metric_mapping(report),
            split_manifest=matrices.split_manifest,
            provenance={
                "baseline": baseline,
                "seed": seed,
                "threads": threads,
                "target": target,
                "dataset": str(dataset),
                "split": str(split),
            },
        )
        result = {
            "record_type": "surgeon_training_result",
            "artifact_digest": str(published.metadata.digest),
            "artifact_size_bytes": published.metadata.size_bytes,
            "baseline": baseline,
            "target": target,
            "training_models": [item.to_record() for item in training_models],
            "source_feature_schema_version": (
                matrices.preprocessor.source_feature_schema_version
            ),
            "target_schema_version": matrices.preprocessor.target_schema_version,
            "feature_count": len(matrices.preprocessor.output_feature_names),
            "split_manifest_version": matrices.split_manifest.get("version"),
            "split_algorithm": matrices.split_manifest.get("algorithm"),
            "split_mode": matrices.split_manifest.get("mode"),
            "split_counts": {
                "train": len(matrices.train.example_ids),
                "validation": len(matrices.validation.example_ids),
                "test": len(matrices.test.example_ids),
            },
            "split_group_counts": _split_group_counts(matrices),
            "metrics": report.to_record(),
        }
        if output_json:
            typer.echo(canonical_identity_json(result))
        else:
            typer.echo(
                f"surgeon {baseline} target={target} artifact={result['artifact_digest']} "
                f"features={result['feature_count']}"
            )
            for identity in training_models:
                typer.echo(
                    f"model={identity.identifier}@{identity.revision} "
                    f"quantization={identity.quantization or 'none'}"
                )
            for metric in report.metrics:
                value = (
                    "undefined"
                    if metric.value is None
                    else f"{metric.value:.6g}"
                )
                typer.echo(f"{metric.name}={value}")
    except (OSError, ValueError) as error:
        typer.echo(f"train-surgeon error: {error}", err=True)
        raise typer.Exit(2) from error


def predict_surgeon_command(
    dataset: Annotated[
        Path, typer.Argument(help="Pre-mutation candidate JSON/JSONL/Parquet")
    ],
    registry: Annotated[
        Path, typer.Option("--registry", help="Immutable surgeon registry root")
    ],
    bundle: Annotated[
        str, typer.Option("--bundle", help="sha256:... surgeon artifact digest")
    ],
    output_json: Annotated[
        bool, typer.Option("--json", help="Emit newline-delimited predictions")
    ] = False,
) -> None:
    """Score compatible pre-mutation candidate records with a persisted surgeon."""

    try:
        records = load_surgeon_records(dataset)
        loaded = SurgeonModelRegistry(str(registry)).load(bundle)
        rows = tuple(
            transform_inference_record(
                cast(ExampleRecord, record),
                loaded.preprocessor,
                refuse_missing=True,
            )
            for record in records
        )
        predictions = _predict_values(loaded.model, rows)
        training_models = [
            item.to_record() for item in loaded.card.training_models
        ]
        for record, prediction in zip(records, predictions, strict=True):
            example_id = record.get("example_id")
            mutation_id = record.get("mutation_id")
            payload = {
                "record_type": "surgeon_prediction",
                "example_id": example_id,
                "mutation_id": mutation_id,
                "target": loaded.card.target_name,
                "prediction": prediction,
                "artifact_digest": str(loaded.artifact.metadata.digest),
                "model_kind": loaded.card.model_kind,
                "training_models": training_models,
                "source_feature_schema_version": (
                    loaded.preprocessor.source_feature_schema_version
                ),
                "target_schema_version": loaded.preprocessor.target_schema_version,
            }
            if output_json:
                typer.echo(canonical_identity_json(payload))
            else:
                typer.echo(
                    f"{example_id or mutation_id}: "
                    f"{loaded.card.target_name}={prediction:.8g} "
                    f"artifact={loaded.artifact.metadata.digest}"
                )
    except (OSError, ValueError) as error:
        typer.echo(f"predict-surgeon error: {error}", err=True)
        raise typer.Exit(2) from error
