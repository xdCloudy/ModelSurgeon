"""Leakage-safe typed training matrices and train-only preprocessing."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, cast

from modelsurgeon.adapters.gguf.quantization import QUANT_LAYOUTS, GGMLQuantizationType
from modelsurgeon.datasets.grouped_splits import GroupedSplitManifest, SplitPartition
from modelsurgeon.experiments.schema import MutationExampleRecord
from modelsurgeon.features.schema import FeatureKind, FeatureRecord

from .targets import SupervisedTargets, TargetSchema, derive_supervised_targets

TRAINING_MATRIX_SCHEMA_VERSION: Final[int] = 1
_UNKNOWN_CATEGORY: Final[str] = "<unknown>"


class TrainingMatrixError(ValueError):
    """Raised when examples/splits cannot form leakage-safe matrices."""


type ExampleRecord = MutationExampleRecord | Mapping[str, object]


@dataclass(frozen=True, slots=True)
class RawSurgeonRow:
    example_id: str
    partition: SplitPartition
    numeric: tuple[tuple[str, float], ...]
    categorical: tuple[tuple[str, str], ...]
    targets: SupervisedTargets
    sample_weight: float
    feature_schema_version: int

    def __post_init__(self) -> None:
        if not self.example_id:
            raise TrainingMatrixError("matrix rows require an example ID")
        if not math.isfinite(self.sample_weight) or self.sample_weight <= 0:
            raise TrainingMatrixError("sample weights must be finite and positive")
        if self.feature_schema_version <= 0:
            raise TrainingMatrixError("feature schema version must be positive")
        numeric_names = tuple(name for name, _ in self.numeric)
        categorical_names = tuple(name for name, _ in self.categorical)
        if numeric_names != tuple(sorted(set(numeric_names))):
            raise TrainingMatrixError("numeric row features must be unique and canonical")
        if categorical_names != tuple(sorted(set(categorical_names))):
            raise TrainingMatrixError("categorical row features must be unique and canonical")


@dataclass(frozen=True, slots=True)
class NumericPreprocessor:
    name: str
    mean: float
    scale: float

    def __post_init__(self) -> None:
        if not self.name or not math.isfinite(self.mean):
            raise TrainingMatrixError("numeric preprocessing requires finite train statistics")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise TrainingMatrixError("numeric preprocessing scale must be finite and positive")


@dataclass(frozen=True, slots=True)
class CategoricalPreprocessor:
    name: str
    categories: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.categories:
            raise TrainingMatrixError("categorical preprocessing requires train categories")
        if self.categories != tuple(sorted(set(self.categories))):
            raise TrainingMatrixError("categorical categories must be unique and canonical")
        if _UNKNOWN_CATEGORY not in self.categories:
            raise TrainingMatrixError("categorical preprocessing must include unknown bucket")


@dataclass(frozen=True, slots=True)
class SurgeonPreprocessor:
    numeric: tuple[NumericPreprocessor, ...]
    categorical: tuple[CategoricalPreprocessor, ...]
    source_feature_schema_version: int
    target_schema_version: int
    schema_version: int = TRAINING_MATRIX_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TRAINING_MATRIX_SCHEMA_VERSION:
            raise TrainingMatrixError("unsupported matrix schema version")
        if self.source_feature_schema_version <= 0 or self.target_schema_version <= 0:
            raise TrainingMatrixError("source schema versions must be positive")
        numeric_names = tuple(item.name for item in self.numeric)
        categorical_names = tuple(item.name for item in self.categorical)
        if numeric_names != tuple(sorted(set(numeric_names))):
            raise TrainingMatrixError("numeric preprocessors must be unique and canonical")
        if categorical_names != tuple(sorted(set(categorical_names))):
            raise TrainingMatrixError("categorical preprocessors must be unique and canonical")

    @property
    def output_feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for numeric_item in self.numeric:
            names.extend((f"num:{numeric_item.name}", f"missing:{numeric_item.name}"))
        for categorical_item in self.categorical:
            names.extend(
                f"cat:{categorical_item.name}={category}"
                for category in categorical_item.categories
            )
        return tuple(names)

    def to_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_feature_schema_version": self.source_feature_schema_version,
            "target_schema_version": self.target_schema_version,
            "numeric": [
                {"name": item.name, "mean": item.mean, "scale": item.scale} for item in self.numeric
            ],
            "categorical": [
                {"name": item.name, "categories": list(item.categories)}
                for item in self.categorical
            ],
            "output_feature_names": list(self.output_feature_names),
        }

    @classmethod
    def from_record(cls, value: Mapping[str, object]) -> SurgeonPreprocessor:
        schema_version = value.get("schema_version")
        source = value.get("source_feature_schema_version")
        target = value.get("target_schema_version")
        numeric_raw = value.get("numeric")
        categorical_raw = value.get("categorical")
        if schema_version != TRAINING_MATRIX_SCHEMA_VERSION:
            raise TrainingMatrixError("unsupported persisted matrix schema version")
        if not isinstance(source, int) or isinstance(source, bool) or source <= 0:
            raise TrainingMatrixError("persisted source feature schema version is invalid")
        if not isinstance(target, int) or isinstance(target, bool) or target <= 0:
            raise TrainingMatrixError("persisted target schema version is invalid")
        if not isinstance(numeric_raw, list) or not isinstance(categorical_raw, list):
            raise TrainingMatrixError("persisted preprocessing lists are invalid")
        numeric: list[NumericPreprocessor] = []
        for raw in numeric_raw:
            if not isinstance(raw, Mapping):
                raise TrainingMatrixError("persisted numeric preprocessor is invalid")
            name = raw.get("name")
            mean = raw.get("mean")
            scale = raw.get("scale")
            if (
                not isinstance(name, str)
                or isinstance(mean, bool)
                or not isinstance(mean, (int, float))
                or isinstance(scale, bool)
                or not isinstance(scale, (int, float))
            ):
                raise TrainingMatrixError("persisted numeric preprocessor fields are invalid")
            numeric.append(NumericPreprocessor(name, float(mean), float(scale)))
        categorical: list[CategoricalPreprocessor] = []
        for raw in categorical_raw:
            if not isinstance(raw, Mapping):
                raise TrainingMatrixError("persisted categorical preprocessor is invalid")
            name = raw.get("name")
            categories = raw.get("categories")
            if (
                not isinstance(name, str)
                or not isinstance(categories, list)
                or not all(isinstance(item, str) for item in categories)
            ):
                raise TrainingMatrixError("persisted categorical preprocessor fields are invalid")
            categorical.append(CategoricalPreprocessor(name, tuple(cast(list[str], categories))))
        return cls(tuple(numeric), tuple(categorical), source, target)


@dataclass(frozen=True, slots=True)
class SurgeonMatrix:
    partition: SplitPartition
    example_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    target_name: str
    target_values: tuple[float, ...]
    target_mask: tuple[bool, ...]
    sample_weights: tuple[float, ...]
    group_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        count = len(self.example_ids)
        arrays = (
            len(self.values),
            len(self.target_values),
            len(self.target_mask),
            len(self.sample_weights),
            len(self.group_ids),
        )
        if any(length != count for length in arrays):
            raise TrainingMatrixError("matrix row arrays must align")
        if any(len(row) != len(self.feature_names) for row in self.values):
            raise TrainingMatrixError("matrix feature rows must match feature names")
        if any(not math.isfinite(value) for row in self.values for value in row):
            raise TrainingMatrixError("matrix features must be finite")
        if any(
            mask and not math.isfinite(value)
            for value, mask in zip(self.target_values, self.target_mask, strict=True)
        ):
            raise TrainingMatrixError("present matrix targets must be finite")


@dataclass(frozen=True, slots=True)
class TrainingMatrices:
    preprocessor: SurgeonPreprocessor
    train: SurgeonMatrix
    validation: SurgeonMatrix
    test: SurgeonMatrix
    split_manifest: dict[str, object]


def _partition_map(
    split: GroupedSplitManifest | Mapping[str, str | SplitPartition],
) -> tuple[dict[str, SplitPartition], dict[str, str], dict[str, object]]:
    if isinstance(split, GroupedSplitManifest):
        partitions: dict[str, SplitPartition] = {}
        groups: dict[str, str] = {}
        for group in split.groups:
            for example_id in group.example_ids:
                partitions[example_id] = group.partition
                groups[example_id] = group.group_id
        return partitions, groups, split.to_record()

    partitions = {}
    groups = {}
    for example_id, raw in split.items():
        if not example_id:
            raise TrainingMatrixError("split mappings cannot contain blank example IDs")
        try:
            partition = raw if isinstance(raw, SplitPartition) else SplitPartition(raw)
        except ValueError as error:
            raise TrainingMatrixError(f"unknown split partition {raw!r}") from error
        partitions[example_id] = partition
        groups[example_id] = example_id
    return (
        partitions,
        groups,
        {
            "version": "inline-1",
            "assignments": {
                example_id: partition.value for example_id, partition in sorted(partitions.items())
            },
        },
    )


def _record(example: ExampleRecord) -> Mapping[str, object]:
    if isinstance(example, MutationExampleRecord):
        return example.to_record()
    return example


def _example_id(record: Mapping[str, object]) -> str:
    value = record.get("example_id")
    if not isinstance(value, str) or not value:
        raise TrainingMatrixError("examples require non-empty example_id")
    return value


def _feature_schema_version(example: ExampleRecord, record: Mapping[str, object]) -> int:
    if isinstance(example, MutationExampleRecord):
        return example.versions.feature_schema_version
    versions = record.get("versions")
    if not isinstance(versions, Mapping):
        raise TrainingMatrixError("example versions record is required")
    value = versions.get("feature_schema_version")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TrainingMatrixError("example feature schema version is invalid")
    return value


def _numeric_from_feature_records(features: Sequence[FeatureRecord]) -> dict[str, float]:
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for feature in features:
        if feature.kind is FeatureKind.SCALAR:
            if isinstance(feature.value, tuple):
                raise TrainingMatrixError("scalar feature unexpectedly contains a vector")
            grouped[feature.name].append(feature.value)
        else:
            if not isinstance(feature.value, tuple):
                raise TrainingMatrixError("vector feature unexpectedly contains a scalar")
            for index, value in enumerate(feature.value):
                grouped[f"{feature.name}[{index}]"].append(value)
    return {name: math.fsum(values) / len(values) for name, values in sorted(grouped.items())}


def _numeric_from_feature_dicts(raw: object) -> dict[str, float]:
    if not isinstance(raw, list):
        raise TrainingMatrixError("pre_mutation_features must be a list")
    grouped: defaultdict[str, list[float]] = defaultdict(list)
    for item in raw:
        if not isinstance(item, Mapping):
            raise TrainingMatrixError("feature records must be objects")
        name = item.get("name")
        kind = item.get("kind")
        value = item.get("value")
        if not isinstance(name, str) or not name:
            raise TrainingMatrixError("feature records require names")
        if kind == FeatureKind.SCALAR.value:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TrainingMatrixError(f"scalar feature {name!r} is not numeric")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise TrainingMatrixError(f"scalar feature {name!r} is non-finite")
            grouped[name].append(numeric)
        elif kind == FeatureKind.VECTOR.value:
            if not isinstance(value, list):
                raise TrainingMatrixError(f"vector feature {name!r} is not a list")
            for index, raw_value in enumerate(value):
                if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                    raise TrainingMatrixError(f"vector feature {name!r} is non-numeric")
                numeric = float(raw_value)
                if not math.isfinite(numeric):
                    raise TrainingMatrixError(f"vector feature {name!r} is non-finite")
                grouped[f"{name}[{index}]"].append(numeric)
        else:
            raise TrainingMatrixError(f"feature {name!r} has unknown kind {kind!r}")
    return {name: math.fsum(values) / len(values) for name, values in sorted(grouped.items())}


def _categorical(record: Mapping[str, object], *, include_context: bool = True) -> dict[str, str]:
    model = record.get("model")
    mutation = record.get("mutation")
    if not isinstance(model, Mapping) or not isinstance(mutation, Mapping):
        raise TrainingMatrixError("example model and mutation records are required")
    plan = mutation.get("plan")
    if not isinstance(plan, Mapping):
        raise TrainingMatrixError("mutation plan is required")
    request = plan.get("request")
    if not isinstance(request, Mapping):
        raise TrainingMatrixError("mutation request is required")
    kind = request.get("kind")
    family = model.get("family")
    format_name = model.get("format")
    quantization = model.get("quantization")
    if not isinstance(kind, str) or not isinstance(family, str) or not isinstance(format_name, str):
        raise TrainingMatrixError("mutation kind, model family, and format must be strings")
    if quantization is not None and not isinstance(quantization, str):
        raise TrainingMatrixError("model quantization must be string or null")
    parameters = request.get("parameters")
    scope = ""
    if isinstance(parameters, Mapping):
        raw_scope = parameters.get("candidate_scope")
        if isinstance(raw_scope, str):
            scope = raw_scope
    output = {
        "model_family": family,
        "model_format": format_name,
        "mutation_kind": kind,
        "candidate_scope": scope or "unspecified",
    }
    if include_context:
        context = _context(record)
        output.update(
            {
                "model_quantization": quantization or "none",
                "feature_source_precision": context[1],
                "hardware_accelerator": context[2],
                "hardware_os": context[3],
            }
        )
    return output


def _quantization_type(raw: object) -> GGMLQuantizationType | None:
    if not isinstance(raw, str):
        return None
    canonical = raw.upper()
    aliases = {"Q4_K_M": "Q4_K", "Q4_K_S": "Q4_K", "Q5_K_M": "Q5_K", "Q5_K_S": "Q5_K"}
    canonical = aliases.get(canonical, canonical)
    try:
        return GGMLQuantizationType(canonical)
    except ValueError:
        return None


def _context(
    record: Mapping[str, object],
) -> tuple[dict[str, float], str, str, str]:
    """Extract source-only model, precision, and hardware context without outcomes."""

    numeric: dict[str, float] = {}
    model = record.get("model")
    if isinstance(model, Mapping):
        parameters = model.get("parameter_count")
        if isinstance(parameters, int) and not isinstance(parameters, bool) and parameters > 0:
            numeric["context_model_parameter_count"] = float(parameters)
        quant_type = _quantization_type(model.get("quantization"))
        if quant_type is not None:
            layout = QUANT_LAYOUTS[quant_type]
            numeric["context_bits_per_weight"] = 8.0 * layout.type_size / layout.block_size

    precision_sources: set[str] = set()
    storage_dtypes: set[str] = set()
    errors: list[float] = []
    raw_features = record.get("pre_mutation_features")
    if isinstance(raw_features, list):
        for feature in raw_features:
            if not isinstance(feature, Mapping) or not isinstance(
                feature.get("precision"), Mapping
            ):
                continue
            precision = cast(Mapping[str, object], feature["precision"])
            source = precision.get("source")
            storage = precision.get("storage_dtype")
            if isinstance(source, str):
                precision_sources.add(source)
            if isinstance(storage, str):
                storage_dtypes.add(storage.lower())
            error = precision.get("error")
            if isinstance(error, Mapping):
                absolute = error.get("absolute_error")
                if isinstance(absolute, (int, float)) and not isinstance(absolute, bool):
                    value = float(absolute)
                    if math.isfinite(value) and value >= 0.0:
                        errors.append(value)
    if "context_bits_per_weight" not in numeric and len(storage_dtypes) == 1:
        bits = {
            "float64": 64.0,
            "float32": 32.0,
            "bfloat16": 16.0,
            "float16": 16.0,
        }.get(next(iter(storage_dtypes)))
        if bits is not None:
            numeric["context_bits_per_weight"] = bits
    if errors:
        numeric["context_feature_error_mean"] = math.fsum(errors) / len(errors)
        numeric["context_feature_error_max"] = max(errors)
    precision_source = (
        "unknown"
        if not precision_sources
        else next(iter(precision_sources))
        if len(precision_sources) == 1
        else "mixed"
    )

    accelerator = "unknown"
    os_name = "unknown"
    hardware = record.get("hardware")
    if isinstance(hardware, Mapping):
        os_record = hardware.get("os")
        if isinstance(os_record, Mapping) and isinstance(os_record.get("name"), str):
            os_name = str(os_record["name"])
        cpu = hardware.get("cpu")
        if isinstance(cpu, Mapping):
            cores = cpu.get("logical_cores")
            if isinstance(cores, int) and not isinstance(cores, bool) and cores > 0:
                numeric["context_cpu_logical_cores"] = float(cores)
        memory = hardware.get("memory")
        if isinstance(memory, Mapping):
            total_memory = memory.get("total_bytes")
            if (
                isinstance(total_memory, int)
                and not isinstance(total_memory, bool)
                and total_memory > 0
            ):
                numeric["context_system_memory_bytes"] = float(total_memory)
        cuda = hardware.get("cuda")
        if isinstance(cuda, Mapping) and cuda.get("available") is True:
            devices = cuda.get("devices")
            if isinstance(devices, list) and devices and isinstance(devices[0], Mapping):
                device = cast(Mapping[str, object], devices[0])
                name = device.get("name")
                accelerator = str(name) if isinstance(name, str) else "cuda"
                total_vram = device.get("total_memory_bytes")
                if (
                    isinstance(total_vram, int)
                    and not isinstance(total_vram, bool)
                    and total_vram > 0
                ):
                    numeric["context_accelerator_memory_bytes"] = float(total_vram)
        elif isinstance(cuda, Mapping):
            accelerator = "cpu"
    return numeric, precision_source, accelerator, os_name


def materialize_raw_rows(
    examples: Sequence[ExampleRecord],
    split: GroupedSplitManifest | Mapping[str, str | SplitPartition],
    *,
    target_schema: TargetSchema,
    sample_weights: Mapping[str, float] | None = None,
    include_context: bool = True,
) -> tuple[RawSurgeonRow, ...]:
    """Materialize typed raw rows without learning any preprocessing state."""

    partitions, _, _ = _partition_map(split)
    rows: list[RawSurgeonRow] = []
    seen: set[str] = set()
    for example in examples:
        record = _record(example)
        example_id = _example_id(record)
        if example_id in seen:
            raise TrainingMatrixError(f"duplicate training example {example_id!r}")
        seen.add(example_id)
        partition = partitions.get(example_id)
        if partition is None:
            raise TrainingMatrixError(f"example {example_id!r} is absent from split manifest")
        if isinstance(example, MutationExampleRecord):
            numeric = _numeric_from_feature_records(example.pre_mutation_features)
        else:
            numeric = _numeric_from_feature_dicts(record.get("pre_mutation_features"))
        if include_context:
            numeric.update(_context(record)[0])
        weight = 1.0 if sample_weights is None else sample_weights.get(example_id, 1.0)
        rows.append(
            RawSurgeonRow(
                example_id,
                partition,
                tuple(sorted(numeric.items())),
                tuple(sorted(_categorical(record, include_context=include_context).items())),
                derive_supervised_targets(example, target_schema),
                weight,
                _feature_schema_version(example, record),
            )
        )
    extra_assignments = set(partitions) - seen
    if extra_assignments:
        raise TrainingMatrixError(
            f"split manifest references examples not supplied: {sorted(extra_assignments)[:5]}"
        )
    return tuple(rows)


def fit_preprocessor(
    rows: Sequence[RawSurgeonRow],
    *,
    target_schema_version: int,
) -> SurgeonPreprocessor:
    """Fit imputation/scaling/vocabularies exclusively from the training partition."""

    train = tuple(row for row in rows if row.partition is SplitPartition.TRAIN)
    if not train:
        raise TrainingMatrixError("preprocessing requires at least one training row")
    feature_versions = {row.feature_schema_version for row in rows}
    if len(feature_versions) != 1:
        raise TrainingMatrixError(
            f"mixed feature schema versions are unsupported: {sorted(feature_versions)}"
        )
    numeric_names = sorted({name for row in train for name, _ in row.numeric})
    numeric: list[NumericPreprocessor] = []
    for name in numeric_names:
        measured = [dict(row.numeric)[name] for row in train if name in dict(row.numeric)]
        mean = math.fsum(measured) / len(measured)
        variance = math.fsum((value - mean) ** 2 for value in measured) / len(measured)
        scale = math.sqrt(variance)
        if scale == 0.0:
            scale = 1.0
        numeric.append(NumericPreprocessor(name, mean, scale))

    categorical_names = sorted({name for row in train for name, _ in row.categorical})
    categorical: list[CategoricalPreprocessor] = []
    for name in categorical_names:
        categories = {dict(row.categorical)[name] for row in train if name in dict(row.categorical)}
        categories.add(_UNKNOWN_CATEGORY)
        categorical.append(CategoricalPreprocessor(name, tuple(sorted(categories))))

    return SurgeonPreprocessor(
        tuple(numeric),
        tuple(categorical),
        next(iter(feature_versions)),
        target_schema_version,
    )


def transform_row(
    row: RawSurgeonRow,
    preprocessor: SurgeonPreprocessor,
) -> tuple[float, ...]:
    numeric = dict(row.numeric)
    categorical = dict(row.categorical)
    output: list[float] = []
    for numeric_item in preprocessor.numeric:
        missing = numeric_item.name not in numeric
        numeric_value = numeric_item.mean if missing else numeric[numeric_item.name]
        output.extend(
            (
                (numeric_value - numeric_item.mean) / numeric_item.scale,
                1.0 if missing else 0.0,
            )
        )
    for categorical_item in preprocessor.categorical:
        categorical_value = categorical.get(categorical_item.name, _UNKNOWN_CATEGORY)
        if categorical_value not in categorical_item.categories:
            categorical_value = _UNKNOWN_CATEGORY
        output.extend(
            1.0 if categorical_value == category else 0.0
            for category in categorical_item.categories
        )
    return tuple(output)


def _target_for_row(
    row: RawSurgeonRow,
    target_name: str,
) -> tuple[float, bool]:
    if target_name == "safe_mutation":
        return (
            (1.0 if row.targets.safe_mutation else 0.0) if row.targets.safe_mutation_mask else 0.0,
            row.targets.safe_mutation_mask,
        )
    target = row.targets.value(target_name)
    return (0.0 if target.value is None else target.value, target.mask)


def _matrix(
    rows: Sequence[RawSurgeonRow],
    partition: SplitPartition,
    preprocessor: SurgeonPreprocessor,
    *,
    target_name: str,
    groups: Mapping[str, str],
) -> SurgeonMatrix:
    selected = tuple(row for row in rows if row.partition is partition)
    target_pairs = tuple(_target_for_row(row, target_name) for row in selected)
    return SurgeonMatrix(
        partition,
        tuple(row.example_id for row in selected),
        preprocessor.output_feature_names,
        tuple(transform_row(row, preprocessor) for row in selected),
        target_name,
        tuple(value for value, _ in target_pairs),
        tuple(mask for _, mask in target_pairs),
        tuple(row.sample_weight for row in selected),
        tuple(groups[row.example_id] for row in selected),
    )


def build_training_matrices(
    examples: Sequence[ExampleRecord],
    split: GroupedSplitManifest | Mapping[str, str | SplitPartition],
    *,
    target_schema: TargetSchema,
    target_name: str,
    sample_weights: Mapping[str, float] | None = None,
    include_context: bool = True,
) -> TrainingMatrices:
    """Build train/validation/test matrices while fitting state on train only."""

    partitions, groups, split_record = _partition_map(split)
    del partitions
    rows = materialize_raw_rows(
        examples,
        split,
        target_schema=target_schema,
        sample_weights=sample_weights,
        include_context=include_context,
    )
    preprocessor = fit_preprocessor(rows, target_schema_version=target_schema.version)
    matrices = TrainingMatrices(
        preprocessor,
        _matrix(rows, SplitPartition.TRAIN, preprocessor, target_name=target_name, groups=groups),
        _matrix(
            rows,
            SplitPartition.VALIDATION,
            preprocessor,
            target_name=target_name,
            groups=groups,
        ),
        _matrix(rows, SplitPartition.TEST, preprocessor, target_name=target_name, groups=groups),
        split_record,
    )
    if not matrices.train.example_ids:
        raise TrainingMatrixError("training split is empty")
    return matrices


def transform_inference_record(
    example: ExampleRecord,
    preprocessor: SurgeonPreprocessor,
    *,
    refuse_missing: bool = True,
) -> tuple[float, ...]:
    """Transform one pre-mutation candidate record without requiring post-mutation labels."""

    record = _record(example)
    schema_version = _feature_schema_version(example, record)
    if schema_version != preprocessor.source_feature_schema_version:
        raise TrainingMatrixError(
            "inference feature schema version does not match trained preprocessing"
        )
    if isinstance(example, MutationExampleRecord):
        numeric_values = _numeric_from_feature_records(example.pre_mutation_features)
    else:
        numeric_values = _numeric_from_feature_dicts(record.get("pre_mutation_features"))
    numeric_values.update(_context(record)[0])
    categorical_values = _categorical(record)
    if refuse_missing:
        missing_numeric = [
            item.name for item in preprocessor.numeric if item.name not in numeric_values
        ]
        missing_categorical = [
            item.name for item in preprocessor.categorical if item.name not in categorical_values
        ]
        if missing_numeric or missing_categorical:
            raise TrainingMatrixError(
                "inference record is missing required trained features: "
                + ", ".join((*missing_numeric, *missing_categorical))
            )
    dummy_targets = SupervisedTargets(
        _example_id(record),
        (),
        None,
        False,
        "inference row has no supervised labels",
    )
    row = RawSurgeonRow(
        _example_id(record),
        SplitPartition.TEST,
        tuple(sorted(numeric_values.items())),
        tuple(sorted(categorical_values.items())),
        dummy_targets,
        1.0,
        schema_version,
    )
    return transform_row(row, preprocessor)
