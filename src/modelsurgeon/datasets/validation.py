"""Machine-readable validation for serialized supervised mutation datasets."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from modelsurgeon.experiments.schema import (
    MUTATION_EXAMPLE_SCHEMA_VERSION,
    ExperimentOutcomeKind,
    MetricState,
    MutationExampleRecord,
)
from modelsurgeon.graph import ComponentId
from modelsurgeon.surgery.serialization import MutationRecordError, MutationRunRecord

DATASET_VALIDATOR_VERSION = "1"


class DatasetValidationRule(StrEnum):
    SCHEMA = "schema"
    FINITE_RANGE = "finite_range"
    COMPONENT_REFERENCE = "component_reference"
    REVISION_PROVENANCE = "revision_provenance"
    TARGET_CALCULATION = "target_calculation"
    DUPLICATE_ID = "duplicate_id"


@dataclass(frozen=True, slots=True)
class DatasetValidationIssue:
    rule: DatasetValidationRule
    record_index: int
    path: str
    code: str
    detail: str
    example_id: str | None = None

    def __post_init__(self) -> None:
        if self.record_index < 0 or not all((self.path, self.code, self.detail)):
            raise ValueError("dataset validation issues require complete context")

    def to_record(self) -> dict[str, object]:
        return {
            "rule": self.rule.value,
            "record_index": self.record_index,
            "example_id": self.example_id,
            "path": self.path,
            "code": self.code,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class DatasetValidationConfig:
    target_absolute_tolerance: float = 1e-9
    target_relative_tolerance: float = 1e-9

    def __post_init__(self) -> None:
        tolerances = (self.target_absolute_tolerance, self.target_relative_tolerance)
        if any(not math.isfinite(value) or value < 0 for value in tolerances):
            raise ValueError("dataset target tolerances must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class DatasetValidationReport:
    version: str
    record_count: int
    issues: tuple[DatasetValidationIssue, ...]

    def __post_init__(self) -> None:
        if self.version != DATASET_VALIDATOR_VERSION or self.record_count < 0:
            raise ValueError("dataset validation report metadata is invalid")

    @property
    def valid(self) -> bool:
        return not self.issues

    def to_record(self) -> dict[str, object]:
        return {
            "version": self.version,
            "record_count": self.record_count,
            "valid": self.valid,
            "issues": [issue.to_record() for issue in self.issues],
        }


_ROOT_FIELDS = {
    "schema_version",
    "example_id",
    "experiment_id",
    "mutation_id",
    "model",
    "dataset",
    "components",
    "mutation",
    "pre_mutation_features",
    "baseline_metrics",
    "post_metrics",
    "delta_metrics",
    "outcome",
    "hardware",
    "versions",
    "seeds",
    "timings",
    "quantization_control",
}


def _mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        return None
    return dict(cast(Mapping[str, object], value))


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and bool(value) else None


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _add(
    issues: list[DatasetValidationIssue],
    rule: DatasetValidationRule,
    index: int,
    example_id: str | None,
    path: str,
    code: str,
    detail: str,
) -> None:
    issues.append(DatasetValidationIssue(rule, index, path, code, detail, example_id))


def _model_and_dataset(
    record: dict[str, object],
    issues: list[DatasetValidationIssue],
    index: int,
    example_id: str | None,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    model = _mapping(record.get("model"))
    dataset = _mapping(record.get("dataset"))
    if model is None:
        _add(
            issues,
            DatasetValidationRule.SCHEMA,
            index,
            example_id,
            "model",
            "expected_object",
            "model must be an object",
        )
    else:
        for field in ("identifier", "revision", "family", "format"):
            if _string(model.get(field)) is None:
                _add(
                    issues,
                    DatasetValidationRule.SCHEMA,
                    index,
                    example_id,
                    f"model.{field}",
                    "required_string",
                    "model identity fields must be non-empty strings",
                )
        raw_count = model.get("parameter_count")
        if raw_count is not None:
            count = _integer(raw_count)
            if count is None or count <= 0:
                _add(
                    issues,
                    DatasetValidationRule.FINITE_RANGE,
                    index,
                    example_id,
                    "model.parameter_count",
                    "positive_integer",
                    "model parameter count must be positive when present",
                )
    if dataset is None:
        _add(
            issues,
            DatasetValidationRule.SCHEMA,
            index,
            example_id,
            "dataset",
            "expected_object",
            "dataset must be an object",
        )
    else:
        for field in (
            "identifier",
            "revision",
            "split",
            "manifest_id",
            "tokenizer",
            "tokenizer_revision",
        ):
            if _string(dataset.get(field)) is None:
                _add(
                    issues,
                    DatasetValidationRule.SCHEMA,
                    index,
                    example_id,
                    f"dataset.{field}",
                    "required_string",
                    "dataset identity fields must be non-empty strings",
                )
    return model, dataset


def _versions_and_seeds(
    record: dict[str, object],
    issues: list[DatasetValidationIssue],
    index: int,
    example_id: str | None,
) -> dict[str, object] | None:
    versions = _mapping(record.get("versions"))
    if versions is None:
        _add(
            issues,
            DatasetValidationRule.SCHEMA,
            index,
            example_id,
            "versions",
            "expected_object",
            "versions must be an object",
        )
    else:
        for field in ("tool_revision", "config_digest", "evaluator_version"):
            if _string(versions.get(field)) is None:
                _add(
                    issues,
                    DatasetValidationRule.SCHEMA,
                    index,
                    example_id,
                    f"versions.{field}",
                    "required_string",
                    "version identity fields must be non-empty strings",
                )
        for field in ("feature_schema_version", "mutation_record_schema_version"):
            version = _integer(versions.get(field))
            if version is None or version <= 0:
                _add(
                    issues,
                    DatasetValidationRule.FINITE_RANGE,
                    index,
                    example_id,
                    f"versions.{field}",
                    "positive_integer",
                    "referenced schema versions must be positive integers",
                )
    seeds = _mapping(record.get("seeds"))
    if seeds is None:
        _add(
            issues,
            DatasetValidationRule.SCHEMA,
            index,
            example_id,
            "seeds",
            "expected_object",
            "seeds must be an object",
        )
    else:
        for field in ("experiment_seed", "data_seed", "mutation_seed"):
            seed = _integer(seeds.get(field))
            if seed is None or seed < 0 or seed >= 1 << 64:
                _add(
                    issues,
                    DatasetValidationRule.FINITE_RANGE,
                    index,
                    example_id,
                    f"seeds.{field}",
                    "unsigned_64_bit_integer",
                    "experiment seeds must be unsigned 64-bit integers",
                )
    return versions


def _components(
    value: object,
    issues: list[DatasetValidationIssue],
    index: int,
    example_id: str | None,
) -> tuple[str, ...] | None:
    if not isinstance(value, list) or not value:
        _add(
            issues,
            DatasetValidationRule.SCHEMA,
            index,
            example_id,
            "components",
            "nonempty_array",
            "components must be a non-empty array",
        )
        return None
    parsed: list[str] = []
    for offset, raw in enumerate(value):
        if not isinstance(raw, str):
            valid = False
        else:
            try:
                parsed.append(str(ComponentId.parse(raw)))
                valid = True
            except ValueError:
                valid = False
        if not valid:
            _add(
                issues,
                DatasetValidationRule.COMPONENT_REFERENCE,
                index,
                example_id,
                f"components[{offset}]",
                "invalid_component_id",
                "component reference is not a valid canonical component ID",
            )
    result = tuple(parsed)
    if result and result != tuple(sorted(set(result))):
        _add(
            issues,
            DatasetValidationRule.COMPONENT_REFERENCE,
            index,
            example_id,
            "components",
            "noncanonical_component_set",
            "components must be sorted and unique",
        )
    return result


def _mutation(
    raw: object,
    mutation_id: str | None,
    components: tuple[str, ...] | None,
    model: dict[str, object] | None,
    versions: dict[str, object] | None,
    issues: list[DatasetValidationIssue],
    index: int,
    example_id: str | None,
) -> None:
    value = _mapping(raw)
    if value is None:
        _add(
            issues,
            DatasetValidationRule.SCHEMA,
            index,
            example_id,
            "mutation",
            "expected_object",
            "mutation must be an object",
        )
        return
    try:
        parsed = MutationRunRecord.from_json(
            json.dumps(value, ensure_ascii=False, allow_nan=False)
        )
    except (MutationRecordError, TypeError, ValueError) as error:
        _add(
            issues,
            DatasetValidationRule.COMPONENT_REFERENCE,
            index,
            example_id,
            "mutation",
            "invalid_mutation_record",
            str(error),
        )
        return
    if mutation_id is not None and parsed.mutation_id != mutation_id:
        _add(
            issues,
            DatasetValidationRule.COMPONENT_REFERENCE,
            index,
            example_id,
            "mutation_id",
            "mutation_id_mismatch",
            "top-level mutation ID does not match the canonical request",
        )
    affected = tuple(str(item) for item in parsed.plan.affected_components)
    if components is not None and affected != components:
        _add(
            issues,
            DatasetValidationRule.COMPONENT_REFERENCE,
            index,
            example_id,
            "mutation.plan.affected_components",
            "affected_component_mismatch",
            "mutation affected components do not match example components",
        )
    model_revision = None if model is None else _string(model.get("revision"))
    if model_revision is not None and parsed.provenance.input_revision != model_revision:
        _add(
            issues,
            DatasetValidationRule.REVISION_PROVENANCE,
            index,
            example_id,
            "mutation.provenance.input_revision",
            "model_revision_mismatch",
            "mutation input revision does not match model revision",
        )
    tool_revision = None if versions is None else _string(versions.get("tool_revision"))
    if tool_revision is not None and parsed.provenance.tool_revision != tool_revision:
        _add(
            issues,
            DatasetValidationRule.REVISION_PROVENANCE,
            index,
            example_id,
            "mutation.provenance.tool_revision",
            "tool_revision_mismatch",
            "mutation tool revision does not match example version context",
        )


def _features(
    raw: object,
    dataset: dict[str, object] | None,
    issues: list[DatasetValidationIssue],
    index: int,
    example_id: str | None,
) -> None:
    if not isinstance(raw, list):
        _add(
            issues,
            DatasetValidationRule.SCHEMA,
            index,
            example_id,
            "pre_mutation_features",
            "expected_array",
            "pre-mutation features must be an array",
        )
        return
    identities: list[tuple[str, str, str, str]] = []
    for offset, item in enumerate(raw):
        feature = _mapping(item)
        path = f"pre_mutation_features[{offset}]"
        if feature is None:
            _add(
                issues,
                DatasetValidationRule.SCHEMA,
                index,
                example_id,
                path,
                "expected_object",
                "feature record must be an object",
            )
            continue
        component = _string(feature.get("component_id"))
        if component is not None:
            try:
                component = str(ComponentId.parse(component))
            except ValueError:
                component = None
        if component is None:
            _add(
                issues,
                DatasetValidationRule.COMPONENT_REFERENCE,
                index,
                example_id,
                f"{path}.component_id",
                "invalid_component_id",
                "feature component ID is invalid",
            )
        name = _string(feature.get("name"))
        extractor = _string(feature.get("extractor"))
        extractor_version = _string(feature.get("extractor_version"))
        if name is None or extractor is None or extractor_version is None:
            _add(
                issues,
                DatasetValidationRule.SCHEMA,
                index,
                example_id,
                path,
                "incomplete_feature_identity",
                "feature name and extractor identity are required",
            )
        elif component is not None:
            identities.append((component, name, extractor, extractor_version))
        raw_value = feature.get("value")
        values = raw_value if isinstance(raw_value, list) else [raw_value]
        if not values or any(_finite(value) is None for value in values):
            _add(
                issues,
                DatasetValidationRule.FINITE_RANGE,
                index,
                example_id,
                f"{path}.value",
                "finite_feature_value",
                "feature values must be finite numeric values",
            )
        context_raw = feature.get("sample_context")
        if context_raw is None:
            continue
        context = _mapping(context_raw)
        if context is None:
            _add(
                issues,
                DatasetValidationRule.SCHEMA,
                index,
                example_id,
                f"{path}.sample_context",
                "expected_object",
                "feature sample context must be an object",
            )
            continue
        if dataset is None:
            continue
        provenance_pairs = (
            ("dataset", "identifier"),
            ("revision", "revision"),
            ("split", "split"),
            ("tokenizer", "tokenizer"),
            ("tokenizer_revision", "tokenizer_revision"),
        )
        for sample_field, dataset_field in provenance_pairs:
            if context.get(sample_field) != dataset.get(dataset_field):
                _add(
                    issues,
                    DatasetValidationRule.REVISION_PROVENANCE,
                    index,
                    example_id,
                    f"{path}.sample_context.{sample_field}",
                    "sample_context_mismatch",
                    "feature sample context does not match dataset provenance",
                )
    if identities != sorted(set(identities)):
        _add(
            issues,
            DatasetValidationRule.COMPONENT_REFERENCE,
            index,
            example_id,
            "pre_mutation_features",
            "duplicate_feature_identity",
            "feature identities must be sorted and unique",
        )


def _metrics(
    raw: object,
    phase: str,
    issues: list[DatasetValidationIssue],
    index: int,
    example_id: str | None,
) -> dict[str, tuple[float, str | None]]:
    measured: dict[str, tuple[float, str | None]] = {}
    if not isinstance(raw, list):
        _add(
            issues,
            DatasetValidationRule.SCHEMA,
            index,
            example_id,
            f"{phase}_metrics",
            "expected_array",
            "metric collections must be arrays",
        )
        return measured
    names: list[str] = []
    for offset, item in enumerate(raw):
        metric = _mapping(item)
        path = f"{phase}_metrics[{offset}]"
        if metric is None:
            _add(
                issues,
                DatasetValidationRule.SCHEMA,
                index,
                example_id,
                path,
                "expected_object",
                "metric observation must be an object",
            )
            continue
        name = _string(metric.get("name"))
        state_name = _string(metric.get("state"))
        if name is None or state_name is None:
            _add(
                issues,
                DatasetValidationRule.SCHEMA,
                index,
                example_id,
                path,
                "incomplete_metric_identity",
                "metric name and state are required",
            )
            continue
        names.append(name)
        try:
            state = MetricState(state_name)
        except ValueError:
            _add(
                issues,
                DatasetValidationRule.SCHEMA,
                index,
                example_id,
                f"{path}.state",
                "unknown_metric_state",
                "metric state is unknown",
            )
            continue
        if state is MetricState.MEASURED:
            value = _finite(metric.get("value"))
            if value is None:
                _add(
                    issues,
                    DatasetValidationRule.FINITE_RANGE,
                    index,
                    example_id,
                    f"{path}.value",
                    "finite_measured_metric",
                    "measured metrics require a finite numeric value",
                )
            else:
                unit_raw = metric.get("unit")
                measured[name] = (value, unit_raw if isinstance(unit_raw, str) else None)
            if metric.get("reason") is not None:
                _add(
                    issues,
                    DatasetValidationRule.SCHEMA,
                    index,
                    example_id,
                    f"{path}.reason",
                    "measured_metric_reason",
                    "measured metrics cannot carry missingness reasons",
                )
        else:
            if metric.get("value") is not None:
                _add(
                    issues,
                    DatasetValidationRule.SCHEMA,
                    index,
                    example_id,
                    f"{path}.value",
                    "missing_metric_has_value",
                    "non-measured metrics cannot carry numeric values",
                )
            if _string(metric.get("reason")) is None:
                _add(
                    issues,
                    DatasetValidationRule.SCHEMA,
                    index,
                    example_id,
                    f"{path}.reason",
                    "missing_metric_reason",
                    "non-measured metrics require an explicit reason",
                )
    if names != sorted(set(names)):
        _add(
            issues,
            DatasetValidationRule.SCHEMA,
            index,
            example_id,
            f"{phase}_metrics",
            "noncanonical_metric_names",
            "metric names must be sorted and unique",
        )
    return measured


def _delta_base_name(name: str, baseline: set[str], post: set[str]) -> str | None:
    candidates = [name]
    if name.startswith("delta_"):
        candidates.append(name[6:])
    if name.endswith("_delta"):
        candidates.append(name[:-6])
    return next(
        (
            candidate
            for candidate in candidates
            if candidate in baseline and candidate in post
        ),
        None,
    )


def _target_calculations(
    baseline: dict[str, tuple[float, str | None]],
    post: dict[str, tuple[float, str | None]],
    delta: dict[str, tuple[float, str | None]],
    config: DatasetValidationConfig,
    issues: list[DatasetValidationIssue],
    index: int,
    example_id: str | None,
) -> None:
    for name, (observed, unit) in delta.items():
        base_name = _delta_base_name(name, set(baseline), set(post))
        if base_name is None:
            continue
        before, before_unit = baseline[base_name]
        after, after_unit = post[base_name]
        if unit is not None and before_unit is not None and unit != before_unit:
            _add(
                issues,
                DatasetValidationRule.TARGET_CALCULATION,
                index,
                example_id,
                f"delta_metrics.{name}",
                "delta_unit_mismatch",
                "delta metric unit does not match its baseline metric",
            )
        if before_unit is not None and after_unit is not None and before_unit != after_unit:
            _add(
                issues,
                DatasetValidationRule.TARGET_CALCULATION,
                index,
                example_id,
                f"delta_metrics.{name}",
                "source_unit_mismatch",
                "baseline and post metric units differ for a delta target",
            )
        expected = after - before
        if not math.isclose(
            observed,
            expected,
            rel_tol=config.target_relative_tolerance,
            abs_tol=config.target_absolute_tolerance,
        ):
            _add(
                issues,
                DatasetValidationRule.TARGET_CALCULATION,
                index,
                example_id,
                f"delta_metrics.{name}",
                "incorrect_delta",
                f"delta target {observed} does not equal post-baseline {expected}",
            )


def _timings(
    raw: object,
    issues: list[DatasetValidationIssue],
    index: int,
    example_id: str | None,
) -> None:
    if not isinstance(raw, list):
        _add(
            issues,
            DatasetValidationRule.SCHEMA,
            index,
            example_id,
            "timings",
            "expected_array",
            "timings must be an array",
        )
        return
    stages: list[str] = []
    for offset, item in enumerate(raw):
        timing = _mapping(item)
        path = f"timings[{offset}]"
        if timing is None:
            _add(
                issues,
                DatasetValidationRule.SCHEMA,
                index,
                example_id,
                path,
                "expected_object",
                "stage timing must be an object",
            )
            continue
        stage = _string(timing.get("stage"))
        if stage is None:
            _add(
                issues,
                DatasetValidationRule.SCHEMA,
                index,
                example_id,
                f"{path}.stage",
                "required_string",
                "stage timing identity is required",
            )
        else:
            stages.append(stage)
        for field in ("wall_seconds", "cpu_seconds"):
            raw_value = timing.get(field)
            if raw_value is None and field == "cpu_seconds":
                continue
            value = _finite(raw_value)
            if value is None or value < 0:
                _add(
                    issues,
                    DatasetValidationRule.FINITE_RANGE,
                    index,
                    example_id,
                    f"{path}.{field}",
                    "nonnegative_finite_timing",
                    "stage timings must be finite and non-negative",
                )
        for field in ("tokens", "candidates"):
            raw_count = timing.get(field)
            if raw_count is None:
                continue
            count = _integer(raw_count)
            if count is None or count < 0:
                _add(
                    issues,
                    DatasetValidationRule.FINITE_RANGE,
                    index,
                    example_id,
                    f"{path}.{field}",
                    "nonnegative_integer",
                    "stage timing counts must be non-negative integers",
                )
    if stages != sorted(set(stages)):
        _add(
            issues,
            DatasetValidationRule.SCHEMA,
            index,
            example_id,
            "timings",
            "noncanonical_timing_stages",
            "stage timings must be sorted and unique",
        )


def _outcome(
    raw: object,
    issues: list[DatasetValidationIssue],
    index: int,
    example_id: str | None,
) -> None:
    outcome = _mapping(raw)
    if outcome is None:
        _add(
            issues,
            DatasetValidationRule.SCHEMA,
            index,
            example_id,
            "outcome",
            "expected_object",
            "outcome must be an object",
        )
        return
    kind_name = _string(outcome.get("kind"))
    try:
        kind = None if kind_name is None else ExperimentOutcomeKind(kind_name)
    except ValueError:
        kind = None
    if kind is None:
        _add(
            issues,
            DatasetValidationRule.SCHEMA,
            index,
            example_id,
            "outcome.kind",
            "unknown_outcome",
            "experiment outcome kind is unknown",
        )
        return
    reason = outcome.get("reason")
    if kind is ExperimentOutcomeKind.SUCCEEDED and reason is not None:
        _add(
            issues,
            DatasetValidationRule.SCHEMA,
            index,
            example_id,
            "outcome.reason",
            "successful_outcome_reason",
            "successful outcomes cannot carry a failure reason",
        )
    if kind is not ExperimentOutcomeKind.SUCCEEDED and _string(reason) is None:
        _add(
            issues,
            DatasetValidationRule.SCHEMA,
            index,
            example_id,
            "outcome.reason",
            "missing_outcome_reason",
            "rejected and failed outcomes require a reason",
        )


def _record(
    value: MutationExampleRecord | Mapping[str, object],
    index: int,
    config: DatasetValidationConfig,
    issues: list[DatasetValidationIssue],
) -> str | None:
    record = value.to_record() if isinstance(value, MutationExampleRecord) else _mapping(value)
    if record is None:
        _add(
            issues,
            DatasetValidationRule.SCHEMA,
            index,
            None,
            "$",
            "expected_object",
            "dataset record must be an object",
        )
        return None
    example_id = _string(record.get("example_id"))
    if set(record) != _ROOT_FIELDS:
        _add(
            issues,
            DatasetValidationRule.SCHEMA,
            index,
            example_id,
            "$",
            "root_fields",
            "mutation example has missing or unknown top-level fields",
        )
    if record.get("schema_version") != MUTATION_EXAMPLE_SCHEMA_VERSION:
        _add(
            issues,
            DatasetValidationRule.SCHEMA,
            index,
            example_id,
            "schema_version",
            "unsupported_schema_version",
            "mutation example schema version is unsupported",
        )
    if example_id is None:
        _add(
            issues,
            DatasetValidationRule.SCHEMA,
            index,
            None,
            "example_id",
            "required_string",
            "example ID must be a non-empty string",
        )
    for field in ("experiment_id", "mutation_id"):
        if _string(record.get(field)) is None:
            _add(
                issues,
                DatasetValidationRule.SCHEMA,
                index,
                example_id,
                field,
                "required_string",
                f"{field} must be a non-empty string",
            )
    model, dataset = _model_and_dataset(record, issues, index, example_id)
    versions = _versions_and_seeds(record, issues, index, example_id)
    component_ids = _components(record.get("components"), issues, index, example_id)
    _mutation(
        record.get("mutation"),
        _string(record.get("mutation_id")),
        component_ids,
        model,
        versions,
        issues,
        index,
        example_id,
    )
    _features(record.get("pre_mutation_features"), dataset, issues, index, example_id)
    baseline = _metrics(record.get("baseline_metrics"), "baseline", issues, index, example_id)
    post = _metrics(record.get("post_metrics"), "post", issues, index, example_id)
    delta = _metrics(record.get("delta_metrics"), "delta", issues, index, example_id)
    _target_calculations(baseline, post, delta, config, issues, index, example_id)
    _timings(record.get("timings"), issues, index, example_id)
    _outcome(record.get("outcome"), issues, index, example_id)
    if _mapping(record.get("hardware")) is None:
        _add(
            issues,
            DatasetValidationRule.SCHEMA,
            index,
            example_id,
            "hardware",
            "expected_object",
            "hardware provenance must be an object",
        )
    quantization = record.get("quantization_control")
    if quantization is not None and _mapping(quantization) is None:
        _add(
            issues,
            DatasetValidationRule.SCHEMA,
            index,
            example_id,
            "quantization_control",
            "expected_object",
            "quantization control must be an object when present",
        )
    return example_id


def validate_mutation_dataset(
    records: Iterable[MutationExampleRecord | Mapping[str, object]],
    config: DatasetValidationConfig | None = None,
) -> DatasetValidationReport:
    """Validate every row and return all deterministic machine-readable issues."""

    resolved = config or DatasetValidationConfig()
    issues: list[DatasetValidationIssue] = []
    seen_ids: dict[str, int] = {}
    count = 0
    for index, value in enumerate(records):
        count += 1
        example_id = _record(value, index, resolved, issues)
        if example_id is None:
            continue
        first_index = seen_ids.get(example_id)
        if first_index is None:
            seen_ids[example_id] = index
            continue
        _add(
            issues,
            DatasetValidationRule.DUPLICATE_ID,
            index,
            example_id,
            "example_id",
            "duplicate_example_id",
            f"example ID duplicates record index {first_index}",
        )
    ordered = tuple(
        sorted(
            issues,
            key=lambda item: (
                item.record_index,
                item.rule.value,
                item.path,
                item.code,
                item.detail,
            ),
        )
    )
    return DatasetValidationReport(DATASET_VALIDATOR_VERSION, count, ordered)
