"""Conservative, artifact-bound Pareto study evidence for v1.1."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from itertools import product

PARETO_STUDY_SCHEMA_VERSION = 1


class ParetoStudyError(ValueError):
    """Raised when study evidence is incomplete, ambiguous, or unsafe."""


class StudyOutcome(StrEnum):
    MEASURED = "measured"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


class MetricDirection(StrEnum):
    HIGHER = "higher"
    LOWER = "lower"


def _text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ParetoStudyError(f"{label} is required")


def _finite(value: float, label: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ParetoStudyError(f"{label} must be numeric")
    if not math.isfinite(float(value)):
        raise ParetoStudyError(f"{label} must be finite")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class StudyModel:
    rung: str
    identifier: str
    revision: str
    family: str
    size_parameters: int
    license: str

    def __post_init__(self) -> None:
        for label, value in (
            ("model rung", self.rung),
            ("model identifier", self.identifier),
            ("model revision", self.revision),
            ("model family", self.family),
            ("model license", self.license),
        ):
            _text(value, label)
        if self.size_parameters <= 0:
            raise ParetoStudyError("model size must be positive")

    def to_record(self) -> dict[str, object]:
        return {
            "rung": self.rung,
            "identifier": self.identifier,
            "revision": self.revision,
            "family": self.family,
            "size_parameters": self.size_parameters,
            "license": self.license,
        }


@dataclass(frozen=True, slots=True)
class StudyCorpus:
    identifier: str
    revision: str
    split: str
    license: str
    token_count: int

    def __post_init__(self) -> None:
        for label, value in (
            ("corpus identifier", self.identifier),
            ("corpus revision", self.revision),
            ("corpus split", self.split),
            ("corpus license", self.license),
        ):
            _text(value, label)
        if self.token_count <= 0:
            raise ParetoStudyError("corpus token count must be positive")

    def to_record(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "revision": self.revision,
            "split": self.split,
            "license": self.license,
            "token_count": self.token_count,
        }


@dataclass(frozen=True, slots=True)
class StudyBudget:
    optimization_tokens: int
    evaluation_tokens: int
    max_wall_seconds: float
    max_gpu_memory_gib: float

    def __post_init__(self) -> None:
        if self.optimization_tokens <= 0 or self.evaluation_tokens <= 0:
            raise ParetoStudyError("study token budgets must be positive")
        if self.max_wall_seconds <= 0 or self.max_gpu_memory_gib <= 0:
            raise ParetoStudyError("study resource budgets must be positive")

    def to_record(self) -> dict[str, int | float]:
        return {
            "optimization_tokens": self.optimization_tokens,
            "evaluation_tokens": self.evaluation_tokens,
            "max_wall_seconds": self.max_wall_seconds,
            "max_gpu_memory_gib": self.max_gpu_memory_gib,
        }


@dataclass(frozen=True, slots=True)
class StudyMetric:
    name: str
    unit: str
    direction: MetricDirection
    value: float | None = None
    confidence_low: float | None = None
    confidence_high: float | None = None
    bootstrap_repetitions: int = 0
    reason: str | None = None

    def __post_init__(self) -> None:
        _text(self.name, "metric name")
        _text(self.unit, "metric unit")
        if self.value is None:
            _text(self.reason or "", "unavailable metric reason")
            if any(bound is not None for bound in (self.confidence_low, self.confidence_high)):
                raise ParetoStudyError("unavailable metrics cannot carry intervals")
            return
        _finite(self.value, "metric value")
        if self.confidence_low is None or self.confidence_high is None:
            raise ParetoStudyError("measured metrics require confidence bounds")
        _finite(self.confidence_low, "metric confidence_low")
        _finite(self.confidence_high, "metric confidence_high")
        if not self.confidence_low <= self.value <= self.confidence_high:
            raise ParetoStudyError("confidence bounds must contain the metric value")
        if self.bootstrap_repetitions < 100:
            raise ParetoStudyError("measured metrics require at least 100 bootstrap repetitions")
        if self.reason is not None:
            raise ParetoStudyError("measured metrics cannot carry an unavailable reason")

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "unit": self.unit,
            "direction": self.direction.value,
            "value": self.value,
            "confidence_low": self.confidence_low,
            "confidence_high": self.confidence_high,
            "bootstrap_repetitions": self.bootstrap_repetitions,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class StudyArtifact:
    digest: str | None
    size_bytes: int | None
    reloadable: bool | None
    generation_smoke: bool | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.digest is not None and (
            len(self.digest) != 64
            or any(character not in "0123456789abcdef" for character in self.digest)
        ):
            raise ParetoStudyError("artifact digest must be a lowercase SHA-256")
        if self.size_bytes is not None and self.size_bytes <= 0:
            raise ParetoStudyError("artifact size must be positive")
        if self.reloadable is True and self.generation_smoke is not True:
            raise ParetoStudyError("reloadable artifacts require generation smoke evidence")
        if self.digest is None or self.size_bytes is None:
            _text(self.reason or "", "unavailable artifact reason")

    def to_record(self) -> dict[str, object]:
        return {
            "digest": self.digest,
            "size_bytes": self.size_bytes,
            "reloadable": self.reloadable,
            "generation_smoke": self.generation_smoke,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class StudyEvidencePoint:
    model: StudyModel
    method: str
    seed: int
    corpus: StudyCorpus
    budget: StudyBudget
    artifact: StudyArtifact
    outcome: StudyOutcome
    metrics: tuple[StudyMetric, ...]
    reason: str | None = None

    def __post_init__(self) -> None:
        _text(self.method, "study method")
        if self.seed < 0:
            raise ParetoStudyError("study seed must be unsigned")
        names = tuple(metric.name for metric in self.metrics)
        if len(names) != len(set(names)):
            raise ParetoStudyError("study metric names must be unique")
        if self.outcome is StudyOutcome.MEASURED:
            if self.artifact.digest is None or self.artifact.reloadable is not True:
                raise ParetoStudyError("measured points require a reloadable artifact digest")
            if any(metric.value is None for metric in self.metrics):
                raise ParetoStudyError("measured points require complete metrics")
            if self.reason is not None:
                raise ParetoStudyError("measured points cannot carry a reason")
        else:
            _text(self.reason or "", "study outcome reason")

    @property
    def point_id(self) -> str:
        identity = {
            "schema_version": PARETO_STUDY_SCHEMA_VERSION,
            "model": self.model.to_record(),
            "method": self.method,
            "seed": self.seed,
            "corpus": self.corpus.to_record(),
            "budget": self.budget.to_record(),
            "artifact": self.artifact.to_record(),
        }
        return "pareto_point_" + hashlib.sha256(_canonical(identity).encode()).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {
            "point_id": self.point_id,
            "model": self.model.to_record(),
            "method": self.method,
            "seed": self.seed,
            "corpus": self.corpus.to_record(),
            "budget": self.budget.to_record(),
            "artifact": self.artifact.to_record(),
            "outcome": self.outcome.value,
            "metrics": [metric.to_record() for metric in self.metrics],
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ParetoStudy:
    models: tuple[StudyModel, ...]
    methods: tuple[str, ...]
    seeds: tuple[int, ...]
    corpus: StudyCorpus
    budget: StudyBudget
    points: tuple[StudyEvidencePoint, ...]
    frontier_point_ids: tuple[str, ...]
    superiority_claim: str | None
    schema_version: int = PARETO_STUDY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PARETO_STUDY_SCHEMA_VERSION:
            raise ParetoStudyError("unsupported Pareto study schema version")
        if len(self.models) < 2 or len({model.family for model in self.models}) < 2:
            raise ParetoStudyError("study requires two model families")
        if len({model.size_parameters for model in self.models}) < 2:
            raise ParetoStudyError("study requires two model sizes")
        if len(self.seeds) < 3 or self.seeds != tuple(sorted(set(self.seeds))):
            raise ParetoStudyError("study requires three sorted unique seeds")
        if not self.methods or len(set(self.methods)) != len(self.methods):
            raise ParetoStudyError("study methods must be unique")
        expected = {
            (model.rung, method, seed)
            for model, method, seed in product(self.models, self.methods, self.seeds)
        }
        actual = {(point.model.rung, point.method, point.seed) for point in self.points}
        if actual != expected or len(actual) != len(self.points):
            raise ParetoStudyError("study must retain every model/method/seed cell")
        point_ids = {point.point_id for point in self.points}
        if any(point_id not in point_ids for point_id in self.frontier_point_ids):
            raise ParetoStudyError("frontier references an unknown point")
        if self.superiority_claim is not None:
            _text(self.superiority_claim, "superiority claim")
            if not self.frontier_point_ids:
                raise ParetoStudyError("a superiority claim requires a measured frontier")

    @property
    def study_id(self) -> str:
        identity = self._identity_record()
        return "pareto_study_" + hashlib.sha256(_canonical(identity).encode()).hexdigest()

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "models": [model.to_record() for model in self.models],
            "methods": list(self.methods),
            "seeds": list(self.seeds),
            "corpus": self.corpus.to_record(),
            "budget": self.budget.to_record(),
            "points": [point.to_record() for point in self.points],
            "frontier_point_ids": list(self.frontier_point_ids),
            "superiority_claim": self.superiority_claim,
        }

    def to_record(self) -> dict[str, object]:
        record = self._identity_record()
        record["study_id"] = self.study_id
        return record


def _default_metric_set(reason: str) -> tuple[StudyMetric, ...]:
    return tuple(
        StudyMetric(name, unit, direction, reason=reason)
        for name, unit, direction in (
            ("quality", "nats/token", MetricDirection.LOWER),
            ("artifact_size", "bytes", MetricDirection.LOWER),
            ("ram", "bytes", MetricDirection.LOWER),
            ("vram", "bytes", MetricDirection.LOWER),
            ("prompt_speed", "tokens/second", MetricDirection.HIGHER),
            ("decode_speed", "tokens/second", MetricDirection.HIGHER),
            ("optimization_cost", "gpu-seconds", MetricDirection.LOWER),
        )
    )


def build_default_pareto_study() -> ParetoStudy:
    models = (
        StudyModel(
            "smollm2_135m",
            "HuggingFaceTB/SmolLM2-135M",
            "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
            "llama",
            135_000_000,
            "Apache-2.0",
        ),
        StudyModel(
            "qwen2.5_500m",
            "Qwen/Qwen2.5-0.5B",
            "060db6499f32faf8b98477b0a26969ef7d8b9987",
            "qwen",
            500_000_000,
            "Apache-2.0",
        ),
    )
    methods = (
        "reference_dense",
        "modelsurgeon_equal_budget",
        "wanda",
        "sparsegpt",
        "structured_baselines",
        "quantization_only",
        "pruning_plus_quantization",
        "magnitude_random_controls",
    )
    corpus = StudyCorpus(
        "wikitext-2-raw-v1",
        "modelsurgeon-benchmark-datasets-v1.1",
        "test",
        "CC-BY-SA-4.0",
        100_000,
    )
    budget = StudyBudget(100_000, 100_000, 3600, 24)
    reason = "no complete reloadable artifact bundle is available for this study cell"
    points = tuple(
        StudyEvidencePoint(
            model,
            method,
            seed,
            corpus,
            budget,
            StudyArtifact(None, None, None, None, reason),
            StudyOutcome.UNSUPPORTED,
            _default_metric_set(reason),
            reason,
        )
        for model, method, seed in product(models, methods, (11, 23, 47))
    )
    return ParetoStudy(
        models,
        methods,
        (11, 23, 47),
        corpus,
        budget,
        points,
        (),
        None,
    )


DEFAULT_PARETO_STUDY = build_default_pareto_study()


def render_pareto_study(
    study: ParetoStudy = DEFAULT_PARETO_STUDY,
    *,
    format: str = "json",
) -> str:
    if format == "json":
        return _canonical(study.to_record()) + "\n"
    if format == "markdown":
        measured = sum(point.outcome is StudyOutcome.MEASURED for point in study.points)
        return "\n".join(
            (
                f"# Pareto study {study.study_id}",
                "",
                f"- Points: {measured}/{len(study.points)} measured",
                f"- Frontier points: {len(study.frontier_point_ids)}",
                "- Superiority claim: none; deployable confidence-bounded evidence is incomplete.",
            )
        ) + "\n"
    raise ParetoStudyError(f"unsupported study format: {format}")
