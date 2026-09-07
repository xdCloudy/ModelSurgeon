"""Frozen, content-addressed protocol for equal-budget competitive benchmarks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from itertools import product

from modelsurgeon.evaluation.benchmark_schema import BenchmarkBudget, BenchmarkBudgetLimit
from modelsurgeon.evaluation.model_ladder import (
    PERMISSIVE_MODEL_LADDER,
    ModelEvaluationLadder,
)
from modelsurgeon.experiments.identity import canonical_identity_json

BENCHMARK_PROTOCOL_SCHEMA_VERSION = 1


class BenchmarkProtocolError(ValueError):
    """Raised when a benchmark protocol is ambiguous, incomplete, or mutable."""


class ProtocolApplicability(StrEnum):
    """Execution eligibility retained for every selected protocol cell."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    INFEASIBLE = "infeasible"
    UNKNOWN = "unknown"


class LicenseDecision(StrEnum):
    """License review result retained independently from execution eligibility."""

    ALLOWED = "allowed"
    EXCLUDED = "excluded"
    PENDING = "pending"


class ProtocolMetricCategory(StrEnum):
    QUALITY = "quality"
    ARTIFACT = "artifact"
    RUNTIME = "runtime"
    COST = "cost"
    RELIABILITY = "reliability"


def _require_text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkProtocolError(f"{label} is required")


def _hash_record(record: object) -> str:
    return hashlib.sha256(canonical_identity_json(record).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ProtocolMetricDefinition:
    """A metric with an exact unit, aggregation, and decision direction."""

    name: str
    category: ProtocolMetricCategory
    unit: str
    aggregation: str
    direction: str
    decision_rule: str

    def __post_init__(self) -> None:
        for label, value in (
            ("metric name", self.name),
            ("metric unit", self.unit),
            ("metric aggregation", self.aggregation),
            ("metric direction", self.direction),
            ("metric decision rule", self.decision_rule),
        ):
            _require_text(value, label)

    def to_record(self) -> dict[str, str]:
        return {
            "name": self.name,
            "category": self.category.value,
            "unit": self.unit,
            "aggregation": self.aggregation,
            "direction": self.direction,
            "decision_rule": self.decision_rule,
        }


@dataclass(frozen=True, slots=True)
class ProtocolTask:
    """Pinned task, split, license, and contamination-gate identity."""

    task_id: str
    dataset: str
    revision: str
    split: str
    license: str
    metric_names: tuple[str, ...]
    contamination_gate: str
    source_url: str

    def __post_init__(self) -> None:
        for label, value in (
            ("task id", self.task_id),
            ("task dataset", self.dataset),
            ("task revision", self.revision),
            ("task split", self.split),
            ("task license", self.license),
            ("task contamination gate", self.contamination_gate),
            ("task source URL", self.source_url),
        ):
            _require_text(value, label)
        if not self.metric_names or len(set(self.metric_names)) != len(self.metric_names):
            raise BenchmarkProtocolError("tasks require unique metric names")

    def to_record(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "dataset": self.dataset,
            "revision": self.revision,
            "split": self.split,
            "license": self.license,
            "metric_names": list(self.metric_names),
            "contamination_gate": self.contamination_gate,
            "source_url": self.source_url,
        }


@dataclass(frozen=True, slots=True)
class ProtocolMethod:
    """Competitor/reference implementation and its source provenance."""

    method_id: str
    name: str
    revision: str
    variant: str
    license: str
    source_url: str

    def __post_init__(self) -> None:
        for label, value in (
            ("method id", self.method_id),
            ("method name", self.name),
            ("method revision", self.revision),
            ("method variant", self.variant),
            ("method license", self.license),
            ("method source URL", self.source_url),
        ):
            _require_text(value, label)

    def to_record(self) -> dict[str, str]:
        return {
            "method_id": self.method_id,
            "name": self.name,
            "revision": self.revision,
            "variant": self.variant,
            "license": self.license,
            "source_url": self.source_url,
        }


@dataclass(frozen=True, slots=True)
class ProtocolCompressionTarget:
    """A matched output target with a measurable target unit."""

    target_id: str
    format: str
    target: float
    unit: str
    applicability: ProtocolApplicability
    reason: str

    def __post_init__(self) -> None:
        for label, value in (
            ("compression target id", self.target_id),
            ("compression format", self.format),
            ("compression unit", self.unit),
            ("compression decision reason", self.reason),
        ):
            _require_text(value, label)
        if self.target < 0:
            raise BenchmarkProtocolError("compression target must be non-negative")

    def to_record(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "format": self.format,
            "target": self.target,
            "unit": self.unit,
            "applicability": self.applicability.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ProtocolCell:
    """One model/method/task decision, including negative and unknown outcomes."""

    model_rung: str
    method_id: str
    task_id: str
    applicability: ProtocolApplicability
    license_decision: LicenseDecision
    reason: str

    def __post_init__(self) -> None:
        for label, value in (
            ("cell model rung", self.model_rung),
            ("cell method id", self.method_id),
            ("cell task id", self.task_id),
            ("cell decision reason", self.reason),
        ):
            _require_text(value, label)
        if self.applicability is ProtocolApplicability.SUPPORTED and (
            self.license_decision is not LicenseDecision.ALLOWED
        ):
            raise BenchmarkProtocolError("supported cells require an allowed license decision")
        if self.applicability is not ProtocolApplicability.SUPPORTED and len(self.reason) < 12:
            raise BenchmarkProtocolError("non-supported cells require an explanatory reason")

    @property
    def cell_key(self) -> tuple[str, str, str]:
        return self.model_rung, self.method_id, self.task_id

    def to_record(self) -> dict[str, str]:
        return {
            "model_rung": self.model_rung,
            "method_id": self.method_id,
            "task_id": self.task_id,
            "applicability": self.applicability.value,
            "license_decision": self.license_decision.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class StatisticalPlan:
    """Pre-registered repeated-run comparison and declared evidence limits."""

    seeds: tuple[int, ...]
    bootstrap_unit: str
    bootstrap_resamples: int
    confidence_level: float
    hierarchical_levels: tuple[str, ...]
    decision_thresholds: tuple[str, ...]
    power_limitation: str
    precision_limitation: str

    def __post_init__(self) -> None:
        if len(self.seeds) < 3 or self.seeds != tuple(sorted(set(self.seeds))):
            raise BenchmarkProtocolError(
                "statistical plan requires at least three sorted unique seeds"
            )
        if any(isinstance(seed, bool) or seed < 0 for seed in self.seeds):
            raise BenchmarkProtocolError("statistical plan seeds must be unsigned integers")
        if self.bootstrap_resamples < 100:
            raise BenchmarkProtocolError("bootstrap resamples must be at least 100")
        if not 0 < self.confidence_level < 1:
            raise BenchmarkProtocolError("confidence level must be between zero and one")
        for label, value in (
            ("bootstrap unit", self.bootstrap_unit),
            ("power limitation", self.power_limitation),
            ("precision limitation", self.precision_limitation),
        ):
            _require_text(value, label)
        if not self.hierarchical_levels or not self.decision_thresholds:
            raise BenchmarkProtocolError("statistical plan requires hierarchy and thresholds")

    def to_record(self) -> dict[str, object]:
        return {
            "seeds": list(self.seeds),
            "bootstrap_unit": self.bootstrap_unit,
            "bootstrap_resamples": self.bootstrap_resamples,
            "confidence_level": self.confidence_level,
            "hierarchical_levels": list(self.hierarchical_levels),
            "decision_thresholds": list(self.decision_thresholds),
            "power_limitation": self.power_limitation,
            "precision_limitation": self.precision_limitation,
        }


@dataclass(frozen=True, slots=True)
class ProtocolExclusion:
    """An explicitly audited exclusion, such as a non-permissive model license."""

    item: str
    revision: str
    license: str
    decision: LicenseDecision
    reason: str
    source_url: str

    def __post_init__(self) -> None:
        for label, value in (
            ("excluded item", self.item),
            ("excluded revision", self.revision),
            ("excluded license", self.license),
            ("exclusion reason", self.reason),
            ("exclusion source URL", self.source_url),
        ):
            _require_text(value, label)
        if self.decision is not LicenseDecision.EXCLUDED:
            raise BenchmarkProtocolError("audit exclusions must have an excluded decision")

    def to_record(self) -> dict[str, str]:
        return {
            "item": self.item,
            "revision": self.revision,
            "license": self.license,
            "decision": self.decision.value,
            "reason": self.reason,
            "source_url": self.source_url,
        }


@dataclass(frozen=True, slots=True)
class ContaminationLicenseAudit:
    """Content-addressed audit of source revisions, contamination gates, and exclusions."""

    checked_on: date
    auditor: str
    policy: str
    source_revisions: tuple[str, ...]
    exclusions: tuple[ProtocolExclusion, ...]

    def __post_init__(self) -> None:
        for label, value in (("audit auditor", self.auditor), ("audit policy", self.policy)):
            _require_text(value, label)
        if not self.source_revisions or len(set(self.source_revisions)) != len(
            self.source_revisions
        ):
            raise BenchmarkProtocolError("audit source revisions must be unique and non-empty")
        if not self.exclusions:
            raise BenchmarkProtocolError("audit must retain explicit exclusions")

    def _identity_record(self) -> dict[str, object]:
        return {
            "checked_on": self.checked_on.isoformat(),
            "auditor": self.auditor,
            "policy": self.policy,
            "source_revisions": list(self.source_revisions),
            "exclusions": [item.to_record() for item in self.exclusions],
        }

    @property
    def audit_id(self) -> str:
        return f"audit_{_hash_record(self._identity_record())}"

    def to_record(self) -> dict[str, object]:
        record = self._identity_record()
        record["audit_id"] = self.audit_id
        return record


@dataclass(frozen=True, slots=True)
class BenchmarkProtocolManifest:
    """Complete preregistration consumed by competitive benchmark runners."""

    ladder: ModelEvaluationLadder
    tasks: tuple[ProtocolTask, ...]
    methods: tuple[ProtocolMethod, ...]
    compression_targets: tuple[ProtocolCompressionTarget, ...]
    budget: BenchmarkBudget
    metrics: tuple[ProtocolMetricDefinition, ...]
    cells: tuple[ProtocolCell, ...]
    statistical_plan: StatisticalPlan
    audit: ContaminationLicenseAudit
    hardware_profiles: tuple[str, ...]
    amendment_policy: str
    schema_version: int = BENCHMARK_PROTOCOL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != BENCHMARK_PROTOCOL_SCHEMA_VERSION:
            raise BenchmarkProtocolError("unsupported benchmark protocol schema version")
        if not self.hardware_profiles or not self.amendment_policy.strip():
            raise BenchmarkProtocolError("protocol requires hardware profiles and amendment policy")
        if self.statistical_plan.seeds != tuple(self.statistical_plan.seeds):
            raise BenchmarkProtocolError("statistical seeds must be deterministic")
        task_ids = tuple(task.task_id for task in self.tasks)
        method_ids = tuple(method.method_id for method in self.methods)
        if not task_ids or len(set(task_ids)) != len(task_ids):
            raise BenchmarkProtocolError("protocol tasks must be uniquely identified")
        if not method_ids or len(set(method_ids)) != len(method_ids):
            raise BenchmarkProtocolError("protocol methods must be uniquely identified")
        metric_names = tuple(metric.name for metric in self.metrics)
        if len(metric_names) != len(set(metric_names)):
            raise BenchmarkProtocolError("protocol metrics must be uniquely identified")
        metric_name_set = set(metric_names)
        if any(not set(task.metric_names).issubset(metric_name_set) for task in self.tasks):
            raise BenchmarkProtocolError("every task metric must have a protocol definition")
        if {metric.category for metric in self.metrics} != set(ProtocolMetricCategory):
            raise BenchmarkProtocolError(
                "protocol must define quality, artifact, runtime, cost, and reliability metrics"
            )
        if not self.compression_targets:
            raise BenchmarkProtocolError("protocol requires at least one compression target")
        expected = {
            (target.rung, method.method_id, task.task_id)
            for target, method, task in product(self.ladder.targets, self.methods, self.tasks)
        }
        actual = tuple(cell.cell_key for cell in self.cells)
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise BenchmarkProtocolError(
                "protocol must record exactly one decision for every model/method/task cell"
            )
        source_revisions = set(self.audit.source_revisions)
        required_revisions = {
            *(target.revision for target in self.ladder.targets),
            *(task.revision for task in self.tasks),
            *(method.revision for method in self.methods),
        }
        if not required_revisions.issubset(source_revisions):
            raise BenchmarkProtocolError(
                "audit must include every model, task, and method revision"
            )

    @property
    def protocol_id(self) -> str:
        return f"protocol_{_hash_record(self._identity_record())}"

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "ladder_id": self.ladder.ladder_id,
            "ladder": self.ladder.to_record(),
            "tasks": [task.to_record() for task in self.tasks],
            "methods": [method.to_record() for method in self.methods],
            "compression_targets": [target.to_record() for target in self.compression_targets],
            "budget": self.budget.to_record(),
            "metrics": [metric.to_record() for metric in self.metrics],
            "cells": [cell.to_record() for cell in self.cells],
            "statistical_plan": self.statistical_plan.to_record(),
            "audit": self.audit.to_record(),
            "hardware_profiles": list(self.hardware_profiles),
            "amendment_policy": self.amendment_policy,
        }

    def to_record(self) -> dict[str, object]:
        record = self._identity_record()
        record["protocol_id"] = self.protocol_id
        return record


def _default_tasks() -> tuple[ProtocolTask, ...]:
    revision = "modelsurgeon-benchmark-datasets-v1.1"
    return (
        ProtocolTask(
            "heldout_perplexity",
            "wikitext-2-raw-v1",
            revision,
            "test",
            "CC-BY-SA-4.0",
            ("perplexity",),
            "exclude any training-document overlap recorded by the dataset manifest",
            "https://huggingface.co/datasets/Salesforce/wikitext",
        ),
        ProtocolTask(
            "arc",
            "allenai/ai2_arc",
            revision,
            "ARC-Challenge:test",
            "CC-BY-SA-4.0",
            ("multiple_choice_accuracy",),
            "reject prompts whose source document appears in model training disclosures",
            "https://huggingface.co/datasets/allenai/ai2_arc",
        ),
        ProtocolTask(
            "hellaswag",
            "Rowan/hellaswag",
            revision,
            "validation",
            "MIT",
            ("multiple_choice_accuracy",),
            "retain the supplied validation contamination metadata before scoring",
            "https://huggingface.co/datasets/Rowan/hellaswag",
        ),
        ProtocolTask(
            "winogrande",
            "allenai/winogrande",
            revision,
            "validation",
            "MIT",
            ("multiple_choice_accuracy",),
            "retain source identifiers and exclude any disclosed training overlap",
            "https://huggingface.co/datasets/allenai/winogrande",
        ),
        ProtocolTask(
            "mmlu_successor",
            "cais/mmlu",
            revision,
            "test",
            "Apache-2.0",
            ("multiple_choice_accuracy",),
            "use the pinned successor mapping only after subject-level contamination review",
            "https://huggingface.co/datasets/cais/mmlu",
        ),
        ProtocolTask(
            "gsm8k",
            "openai/gsm8k",
            revision,
            "test",
            "MIT",
            ("exact_match_accuracy",),
            "exclude leaked solutions and retain exclusions in the evidence manifest",
            "https://huggingface.co/datasets/openai/gsm8k",
        ),
        ProtocolTask(
            "code",
            "bigcode/humanevalpack",
            revision,
            "python",
            "Apache-2.0",
            ("pass_at_1",),
            "run only languages and licenses admitted by the evaluator capability gate",
            "https://huggingface.co/datasets/bigcode/humanevalpack",
        ),
    )


def build_default_benchmark_protocol(
    ladder: ModelEvaluationLadder = PERMISSIVE_MODEL_LADDER,
) -> BenchmarkProtocolManifest:
    """Build the checked-in v1.1 protocol without running any benchmark."""

    tasks = _default_tasks()
    methods = (
        ProtocolMethod(
            "reference_dense",
            "unmodified reference",
            "reference-implementation-v1.1",
            "dense checkpoint, no surgery",
            "Apache-2.0",
            "https://github.com/xdCloudy/ModelSurgeon",
        ),
        ProtocolMethod(
            "modelsurgeon_equal_budget",
            "ModelSurgeon",
            "77e3b4f8bc2f2aedd78202a01b4bee2c7942a298",
            "equal optimization-cost budget",
            "Apache-2.0",
            "https://github.com/xdCloudy/ModelSurgeon/tree/77e3b4f8bc2f2aedd78202a01b4bee2c7942a298",
        ),
    )
    metrics = (
        ProtocolMetricDefinition(
            "perplexity",
            ProtocolMetricCategory.QUALITY,
            "nats/token",
            "token-weighted mean",
            "lower",
            "report delta and 95% CI; no success threshold before execution",
        ),
        ProtocolMetricDefinition(
            "multiple_choice_accuracy",
            ProtocolMetricCategory.QUALITY,
            "proportion",
            "paired item mean",
            "higher",
            "report paired bootstrap delta and 95% CI",
        ),
        ProtocolMetricDefinition(
            "exact_match_accuracy",
            ProtocolMetricCategory.QUALITY,
            "proportion",
            "paired item mean",
            "higher",
            "report paired bootstrap delta and 95% CI",
        ),
        ProtocolMetricDefinition(
            "pass_at_1",
            ProtocolMetricCategory.QUALITY,
            "proportion",
            "task mean",
            "higher",
            "report task bootstrap delta and 95% CI",
        ),
        ProtocolMetricDefinition(
            "artifact_size",
            ProtocolMetricCategory.ARTIFACT,
            "bytes",
            "final artifact byte count",
            "lower",
            "compare only artifacts passing integrity validation",
        ),
        ProtocolMetricDefinition(
            "runtime",
            ProtocolMetricCategory.RUNTIME,
            "seconds",
            "median per completed run",
            "lower",
            "report median and 95% bootstrap CI",
        ),
        ProtocolMetricDefinition(
            "optimization_cost",
            ProtocolMetricCategory.COST,
            "gpu-seconds",
            "sum of optimization phases",
            "lower",
            "reject runs over the equal-budget ceiling",
        ),
        ProtocolMetricDefinition(
            "failure_rate",
            ProtocolMetricCategory.RELIABILITY,
            "proportion",
            "failed repetitions / attempted repetitions",
            "lower",
            "retain every failure and report Wilson interval",
        ),
    )
    cells = tuple(
        ProtocolCell(
            target.rung,
            method.method_id,
            task.task_id,
            ProtocolApplicability.SUPPORTED
            if task.task_id == "heldout_perplexity"
            else ProtocolApplicability.UNKNOWN,
            LicenseDecision.ALLOWED,
            "pinned held-out perplexity is supported by the current evaluator"
            if task.task_id == "heldout_perplexity"
            else (
                "task is preregistered but evaluator capability and contamination gate "
                "remain to be resolved before execution"
            ),
        )
        for target, method, task in product(ladder.targets, methods, tasks)
    )
    audit = ContaminationLicenseAudit(
        date(2026, 8, 24),
        "ModelSurgeon evaluation maintainers",
        "pin source revisions and licenses; exclude disclosed overlap; retain every exclusion",
        tuple(
            dict.fromkeys(
                target.revision for target in ladder.targets
            )
        )
        + tuple(
            revision
            for revision in dict.fromkeys(task.revision for task in tasks)
            if revision not in {target.revision for target in ladder.targets}
        )
        + tuple(
            revision
            for revision in dict.fromkeys(method.revision for method in methods)
            if revision not in {target.revision for target in ladder.targets}
            and revision not in {task.revision for task in tasks}
        ),
        (
            ProtocolExclusion(
                "google/gemma-2-2b",
                "license-review-v1.1",
                "Gemma Terms of Use",
                LicenseDecision.EXCLUDED,
                "not admitted to the permissive ladder until its license is explicitly "
                "approved for this study",
                "https://ai.google.dev/gemma/terms",
            ),
        ),
    )
    return BenchmarkProtocolManifest(
        ladder=ladder,
        tasks=tasks,
        methods=methods,
        compression_targets=(
            ProtocolCompressionTarget(
                "dense_reference",
                "safetensors",
                1.0,
                "fraction_of_source_bytes",
                ProtocolApplicability.SUPPORTED,
                "uncompressed reference artifact",
            ),
            ProtocolCompressionTarget(
                "gguf_q4_k_m",
                "GGUF",
                4.5,
                "bits_per_weight",
                ProtocolApplicability.UNKNOWN,
                "requires the pinned GGUF conversion and integrity gate before execution",
            ),
        ),
        budget=BenchmarkBudget(
            limits=(
                # Names are sorted because BenchmarkBudget is itself content-addressed.
                BenchmarkBudgetLimit("evaluation_tokens", 100000, "tokens"),
                BenchmarkBudgetLimit("max_gpu_memory", 24, "GiB"),
                BenchmarkBudgetLimit("optimization_wall_time", 3600, "seconds"),
            ),
            objective=(
                "equal optimization and evaluation resource ceiling per "
                "model/method/task cell"
            ),
        ),
        metrics=metrics,
        cells=cells,
        statistical_plan=StatisticalPlan(
            (11, 23, 47),
            "model x task x seed paired bootstrap unit with method pairing",
            2000,
            0.95,
            ("model", "task", "seed"),
            (
                "95% CI excludes zero for a directional claim",
                "otherwise report negative or inconclusive",
            ),
            "three seeds cannot establish small effects with high power; no claim is made "
            "below the preregistered detectable-effect review threshold",
            "finite test sets and three repeated seeds may produce wide intervals; report "
            "the interval and do not round away uncertainty",
        ),
        audit=audit,
        hardware_profiles=("cpu_streaming", "gpu_12_gb", "gpu_12_gb_quantized", "gpu_24_gb"),
        amendment_policy=(
            "Any change to a model, task, method, budget, seed, metric, or decision rule "
            "creates a new protocol_id and preserves the superseded manifest; amendments "
            "are prohibited after the first scored cell."
        ),
    )


DEFAULT_BENCHMARK_PROTOCOL = build_default_benchmark_protocol()


def render_benchmark_protocol(
    protocol: BenchmarkProtocolManifest = DEFAULT_BENCHMARK_PROTOCOL,
    *,
    format: str = "json",
) -> str:
    """Render a deterministic protocol manifest as JSON or reviewable Markdown."""

    if format == "json":
        return canonical_identity_json(protocol.to_record()) + "\n"
    if format != "markdown":
        raise BenchmarkProtocolError(f"unsupported protocol format: {format!r}")
    record = protocol.to_record()
    return "\n".join(
        (
            f"# Benchmark protocol {protocol.protocol_id}",
            "",
            f"- Ladder: `{record['ladder_id']}`",
            f"- Audit: `{protocol.audit.audit_id}`",
            f"- Cells: {len(protocol.cells)} model/method/task decisions",
            f"- Seeds: {', '.join(str(seed) for seed in protocol.statistical_plan.seeds)}",
            "- Negative and unknown cells are retained in the manifest; absence is not success.",
        )
    )


def protocol_from_record(raw: dict[str, object]) -> BenchmarkProtocolManifest:
    """Reject accidental use of this writer-only schema as an unvalidated object."""

    if not isinstance(raw, dict) or raw.get("schema_version") != BENCHMARK_PROTOCOL_SCHEMA_VERSION:
        raise BenchmarkProtocolError("protocol records must carry the supported schema version")
    raise BenchmarkProtocolError(
        "protocol_from_record is intentionally not a lossy parser; construct typed records "
        "or use the checked-in manifest"
    )
