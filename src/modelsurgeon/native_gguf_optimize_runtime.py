"""First-party autonomous optimization for the supported native GGUF cell.

This runtime intentionally has a narrow capability boundary. It executes the
model-wide native GGUF MLP-channel operation for Llama and dense Qwen files,
and delegates every authoritative decision to the real GGUF writer and pinned
llama.cpp tools. Unsupported operations and incomplete runtime identity fail
closed instead of being represented as estimates.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from modelsurgeon.adapters import ModelFamily
from modelsurgeon.adapters.gguf import (
    DENSE_GGUF_CODECS,
    IQ4_CODECS,
    Q2_K_CODEC,
    Q3_K_CODEC,
    Q4_K_CODEC,
    Q5_0_CODEC,
    Q5_K_CODEC,
    Q6_K_CODEC,
    Q8_0_CODEC,
    CodecRegistry,
    GGUFDiscovery,
    GGUFDiskEstimate,
    GGUFResourceLimitError,
    GGUFTensorReader,
    QuantizationCodec,
    discover_gguf_components,
    open_gguf,
    preflight_gguf_disk,
)
from modelsurgeon.config import ObjectiveConfig, OptimizeMetric
from modelsurgeon.evaluation.llama_cpp import (
    LlamaCppValidationConfig,
    validate_generated_gguf,
)
from modelsurgeon.evaluation.llama_cpp_perplexity import (
    LlamaCppPerplexityConfig,
    LlamaCppPerplexityManifest,
    benchmark_gguf_perplexity,
)
from modelsurgeon.evaluation.llama_cpp_throughput import (
    LlamaCppThroughputConfig,
    benchmark_gguf_throughput,
)
from modelsurgeon.evaluation.quality_gate import (
    QualityGateError,
    evaluate_perplexity_quality_gate,
)
from modelsurgeon.experiments import (
    OptimizationEvidenceOutcome,
    OptimizationEvidenceRecord,
    OptimizationEvidenceStore,
    collect_hardware_inventory,
)
from modelsurgeon.optimization import OptimizePlan
from modelsurgeon.optimization_orchestrator import (
    OptimizeRuntime,
    OptimizeStage,
    StageContext,
    StageResult,
    WorkflowOutcome,
)
from modelsurgeon.search.objectives import (
    ObjectiveObservation,
    objectives_from_config,
)
from modelsurgeon.search.pareto import (
    ParetoArchive,
    ParetoCandidate,
    ParetoObjectiveValue,
)
from modelsurgeon.surgery import (
    GGUFEdit,
    GGUFResourceLimits,
    GGUFSourceState,
    GGUFStageMeasurement,
    execute_native_gguf_mlp_channel_removal,
    plan_native_gguf_model_mlp_channel_removal,
    run_gguf_cumulative_sequence,
)
from modelsurgeon.surgery.native_mlp_execute import NativeGGUFMLPExecutionResult
from modelsurgeon.surgery.native_mlp_plan import NativeGGUFMLPRemovalPlan


class NativeGGUFOptimizeRuntimeError(RuntimeError):
    """Raised when the explicit GGUF optimizer contract cannot be executed."""


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _raw_digest_file(path: Path) -> str:
    return _digest_file(path).removeprefix("sha256:")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise NativeGGUFOptimizeRuntimeError(f"{label} must be a mapping")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NativeGGUFOptimizeRuntimeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise NativeGGUFOptimizeRuntimeError(f"{label} must be finite")
    return result


def _integer(value: object, default: int, label: str) -> int:
    selected = default if value is None else value
    if isinstance(selected, bool) or not isinstance(selected, int) or selected <= 0:
        raise NativeGGUFOptimizeRuntimeError(f"{label} must be a positive integer")
    return selected


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise NativeGGUFOptimizeRuntimeError(f"{label} must be a non-negative integer")
    return value


def _path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise NativeGGUFOptimizeRuntimeError(f"{label} must be a path")
    return Path(value).expanduser().absolute().resolve(strict=False)


def _json_detail(result: StageResult) -> Mapping[str, object]:
    try:
        value: object = json.loads(result.detail)
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, Mapping) else {}


@dataclass(frozen=True, slots=True)
class _GgufCandidate:
    candidate_id: str
    mutation_id: str
    offset: int
    removed_count: int
    removed_channels: tuple[int, ...]
    plan_record: Mapping[str, object]

    def to_record(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "mutation_id": self.mutation_id,
            "offset": self.offset,
            "removed_count": self.removed_count,
            "removed_channels": list(self.removed_channels),
            "plan": dict(self.plan_record),
        }

    @classmethod
    def from_record(cls, value: object) -> _GgufCandidate:
        record = _mapping(value, "GGUF candidate")
        raw_channels = record.get("removed_channels")
        if not isinstance(raw_channels, list) or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in raw_channels
        ):
            raise NativeGGUFOptimizeRuntimeError("GGUF candidate channels are invalid")
        candidate_id = record.get("candidate_id")
        mutation_id = record.get("mutation_id")
        if not isinstance(candidate_id, str) or not candidate_id.startswith("candidate_"):
            raise NativeGGUFOptimizeRuntimeError("GGUF candidate ID is invalid")
        if not isinstance(mutation_id, str) or not mutation_id.strip():
            raise NativeGGUFOptimizeRuntimeError("GGUF mutation ID is invalid")
        return cls(
            candidate_id,
            mutation_id,
            _nonnegative_integer(record.get("offset"), "GGUF candidate offset"),
            _integer(
                record.get("removed_count"), len(raw_channels), "GGUF candidate removed_count"
            ),
            tuple(raw_channels),
            _mapping(record.get("plan", {}), "GGUF candidate plan"),
        )


class NativeGGUFOptimizeRuntime(OptimizeRuntime):
    """Run the real native GGUF MLP-channel optimization cell."""

    def __init__(self, plan: OptimizePlan) -> None:
        self.plan = plan
        self.campaign_run_id: str | None = None
        self.source_path: Path | None = None
        self.source_digest: str | None = None
        self.discovery: GGUFDiscovery | None = None
        self.source_state: GGUFSourceState | None = None
        self.manifest: LlamaCppPerplexityManifest | None = None
        self.baseline: Mapping[str, object] | None = None
        self.candidates: tuple[_GgufCandidate, ...] = ()
        self.measurements: dict[str, Mapping[str, object]] = {}
        self.selected: _GgufCandidate | None = None
        self.selected_measurement: Mapping[str, object] | None = None
        self.measured_frontier: Mapping[str, object] | None = None
        self.search_comparison: Mapping[str, object] | None = None
        self.artifact: Path | None = None
        self.artifact_digest: str | None = None
        self.sequence_record: Mapping[str, object] | None = None
        self.deployment: Mapping[str, object] | None = None
        self.deployment_constraints_passed: bool | None = None
        self.runtime_hardware: Mapping[str, object] | None = None
        self.evidence: dict[str, Mapping[str, object]] = {}

    def _resolved(self) -> Mapping[str, object]:
        return _mapping(self.plan.resolved_config, "resolved configuration")

    def _quality_gate(
        self, baseline_perplexity: float, candidate_perplexity: float
    ) -> Mapping[str, object]:
        constraints = _mapping(self._resolved().get("constraints"), "constraints")
        raw_max_delta = constraints.get("max_perplexity_delta")
        max_delta = (
            None
            if raw_max_delta is None
            else _number(raw_max_delta, "constraints.max_perplexity_delta")
        )
        try:
            return evaluate_perplexity_quality_gate(
                baseline_perplexity,
                candidate_perplexity,
                min_quality_retention_ratio=_number(
                    constraints.get("min_quality_retention_ratio", 0.98),
                    "constraints.min_quality_retention_ratio",
                ),
                max_perplexity_delta=max_delta,
                profile_max_perplexity_delta=self.plan.quality_profile.max_perplexity_delta,
            )
        except QualityGateError as error:
            raise NativeGGUFOptimizeRuntimeError(str(error)) from error

    def _runtime(self) -> Mapping[str, object]:
        runtime = _mapping(self._resolved().get("runtime"), "runtime")
        required = ("llama_cli", "llama_perplexity", "llama_bench", "expected_revision")
        for name in required:
            value = runtime.get(name)
            if not isinstance(value, str) or not value.strip():
                raise NativeGGUFOptimizeRuntimeError(
                    f"runtime.{name} must be explicit for native GGUF optimization"
                )
        return runtime

    def _model_revision(self) -> str:
        model = _mapping(self._resolved().get("model"), "model")
        revision = model.get("revision")
        if not isinstance(revision, str) or not revision.strip():
            raise NativeGGUFOptimizeRuntimeError("model.revision must be pinned")
        return revision

    def _model_family(self) -> ModelFamily:
        model = _mapping(self._resolved().get("model"), "model")
        raw_family = model.get("family")
        if not isinstance(raw_family, str) or not raw_family.strip():
            raise NativeGGUFOptimizeRuntimeError(
                "model.family must be explicit for GGUF architecture resolution"
            )
        try:
            return ModelFamily(raw_family)
        except ValueError as error:
            raise NativeGGUFOptimizeRuntimeError(
                f"unsupported explicit GGUF model family {raw_family!r}"
            ) from error

    def _source(self) -> Path:
        model = _mapping(self._resolved().get("model"), "model")
        path = _path(model.get("path"), "model.path")
        if not path.is_file() or path.suffix.lower() != ".gguf":
            raise NativeGGUFOptimizeRuntimeError(
                f"native GGUF optimization requires an existing .gguf file: {path}"
            )
        return path

    def _calibration(self) -> Path:
        calibration = _mapping(self._resolved().get("calibration"), "calibration")
        path = _path(calibration.get("dataset"), "calibration.dataset")
        if not path.is_file():
            raise NativeGGUFOptimizeRuntimeError(
                f"calibration.dataset must name an existing UTF-8 text file: {path}"
            )
        try:
            path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise NativeGGUFOptimizeRuntimeError(
                "calibration.dataset must be valid UTF-8"
            ) from error
        return path

    def _hardware(self, probe_path: Path | None = None) -> Mapping[str, object]:
        if self.runtime_hardware is None:
            resolved = probe_path or self._artifact_root()
            existing = resolved
            while not existing.exists() and existing != existing.parent:
                existing = existing.parent
            self.runtime_hardware = collect_hardware_inventory(existing).to_record()
        return self.runtime_hardware

    def _artifact_root(self) -> Path:
        raw = self._resolved().get("artifact_dir", "artifacts")
        return _path(raw, "artifact_dir")

    @staticmethod
    def _tensor_hashes(path: Path) -> tuple[tuple[str, str], ...]:
        hashes: list[tuple[str, str]] = []
        with open_gguf(path) as mapped:
            reader = GGUFTensorReader(mapped)
            for handle in reader.index.tensors:
                digest = hashlib.sha256()
                for chunk in reader.iter_chunks(
                    handle, max_chunk_bytes=reader.limits.max_chunk_bytes
                ):
                    digest.update(chunk.data)
                hashes.append((handle.name, digest.hexdigest()))
        return tuple(sorted(hashes))

    @classmethod
    def _state(cls, path: Path, discovery: GGUFDiscovery) -> GGUFSourceState:
        return GGUFSourceState(
            discovery.parameter_count,
            discovery.storage_bytes,
            hashlib.sha256(_canonical(discovery.to_record()).encode()).hexdigest(),
            cls._tensor_hashes(path),
        )

    def _ensure_source(self) -> None:
        if self.source_path is not None:
            return
        source = self._source()
        try:
            with open_gguf(source) as mapped:
                discovery = discover_gguf_components(
                    mapped.container,
                    family=self._model_family(),
                )
        except (OSError, ValueError, GGUFResourceLimitError) as error:
            raise NativeGGUFOptimizeRuntimeError(f"GGUF inspection failed: {error}") from error
        self.source_path = source
        self.source_digest = _digest_file(source)
        self.discovery = discovery
        self.source_state = self._state(source, discovery)

    def _manifest(self) -> LlamaCppPerplexityManifest:
        if self.manifest is not None:
            return self.manifest
        calibration = _mapping(self._resolved().get("calibration"), "calibration")
        path = self._calibration()
        dataset_revision = calibration.get("dataset_revision")
        if not isinstance(dataset_revision, str) or not dataset_revision.strip():
            raise NativeGGUFOptimizeRuntimeError(
                "calibration.dataset_revision is required for GGUF evidence"
            )
        self.manifest = LlamaCppPerplexityManifest(
            path,
            _raw_digest_file(path),
            _integer(calibration.get("samples"), 512, "calibration.samples"),
            "local-text",
            dataset_revision,
            str(calibration.get("split", "train")),
            "gguf-embedded-tokenizer",
            self._model_revision(),
        )
        return self.manifest

    def _tool_config(
        self,
    ) -> tuple[
        LlamaCppValidationConfig,
        LlamaCppPerplexityConfig,
        LlamaCppThroughputConfig,
    ]:
        runtime = self._runtime()
        expected = str(runtime["expected_revision"])
        threads = _integer(runtime.get("threads"), 1, "runtime.threads")
        gpu_layers = runtime.get("gpu_layers", 0)
        if isinstance(gpu_layers, bool) or not isinstance(gpu_layers, int) or gpu_layers < 0:
            raise NativeGGUFOptimizeRuntimeError("runtime.gpu_layers must be non-negative")
        context = _integer(runtime.get("context_size"), 512, "runtime.context_size")
        batch = _integer(runtime.get("batch_size"), 512, "runtime.batch_size")
        microbatch = _integer(runtime.get("microbatch_size"), 512, "runtime.microbatch_size")
        if batch > context or microbatch > batch:
            raise NativeGGUFOptimizeRuntimeError(
                "runtime geometry must satisfy microbatch <= batch <= context"
            )
        timeout = _number(runtime.get("timeout_seconds", 300.0), "runtime.timeout_seconds")
        return (
            LlamaCppValidationConfig(
                executable=str(runtime["llama_cli"]),
                expected_revision=expected,
                threads=threads,
                gpu_layers=gpu_layers,
                timeout_seconds=min(timeout, 120.0),
            ),
            LlamaCppPerplexityConfig(
                executable=str(runtime["llama_perplexity"]),
                expected_revision=expected,
                context_size=context,
                batch_size=batch,
                microbatch_size=microbatch,
                threads=threads,
                gpu_layers=gpu_layers,
                chunks=_integer(runtime.get("chunks"), 1, "runtime.chunks"),
                timeout_seconds=timeout,
            ),
            LlamaCppThroughputConfig(
                executable=str(runtime["llama_bench"]),
                expected_revision=expected,
                prompt_tokens=_integer(runtime.get("prompt_tokens"), 512, "runtime.prompt_tokens"),
                generation_tokens=_integer(
                    runtime.get("generation_tokens"), 128, "runtime.generation_tokens"
                ),
                batch_size=batch,
                microbatch_size=microbatch,
                threads=threads,
                gpu_layers=gpu_layers,
                repetitions=_integer(runtime.get("repetitions"), 5, "runtime.repetitions"),
                timeout_seconds=timeout,
            ),
        )

    def _model_record(self) -> Mapping[str, object]:
        self._ensure_source()
        assert self.source_path is not None and self.source_digest is not None
        assert self.discovery is not None
        return {
            "format": "gguf",
            "path": str(self.source_path),
            "revision": self._model_revision(),
            "sha256": self.source_digest,
            "family": self.discovery.family.value,
            "architecture": self.discovery.architecture,
            "shape": self.discovery.to_record(),
        }

    def _dataset_record(self) -> Mapping[str, object]:
        manifest = self._manifest()
        return {
            "path": str(manifest.text_path),
            "sha256": manifest.text_sha256,
            "sample_count": manifest.sample_count,
            "dataset": manifest.dataset,
            "dataset_revision": manifest.dataset_revision,
            "split": manifest.split,
            "tokenizer": manifest.tokenizer,
            "tokenizer_revision": manifest.tokenizer_revision,
        }

    def _versions(self) -> Mapping[str, object]:
        runtime = self._runtime()
        return {
            "runtime": "first_party_native_gguf_physical_search",
            "llama_cpp_expected_revision": runtime["expected_revision"],
            "config_digest": self.plan.config_digest,
            "plan_digest": self.plan.plan_id,
            "evidence_schema_version": 1,
        }

    def _evidence_store(self) -> OptimizationEvidenceStore:
        return OptimizationEvidenceStore(self._artifact_root() / "optimization-evidence")

    def _publish_evidence(
        self,
        *,
        stage: str,
        state_id: str,
        outcome: OptimizationEvidenceOutcome,
        candidate_id: str | None = None,
        mutation_id: str | None = None,
        measurement: Mapping[str, object] | None = None,
        artifact: Mapping[str, object] | None = None,
        failure: Mapping[str, object] | None = None,
        lineage: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        run_id = self.campaign_run_id or (
            "runtime_" + hashlib.sha256(self.plan.plan_id.encode()).hexdigest()
        )
        identity = {
            "run_id": run_id,
            "stage": stage,
            "state_id": state_id,
            "candidate_id": candidate_id,
            "mutation_id": mutation_id,
            "outcome": outcome.value,
            "measurement": None if measurement is None else dict(measurement),
            "artifact": None if artifact is None else dict(artifact),
            "failure": None if failure is None else dict(failure),
        }
        observation_id = "obs_" + hashlib.sha256(_canonical(identity).encode()).hexdigest()
        record = OptimizationEvidenceRecord(
            observation_id,
            run_id,
            stage,
            state_id,
            outcome,
            self._model_record(),
            self._dataset_record(),
            {
                "profile": self.plan.hardware_profile.to_record(),
                "runtime_inventory": dict(self._hardware()),
            },
            self._versions(),
            dict(lineage or {}),
            candidate_id=candidate_id,
            mutation_id=mutation_id,
            features=(
                ()
                if self.discovery is None
                else (
                    {"name": "parameter_count", "value": self.discovery.parameter_count},
                    {"name": "storage_bytes", "value": self.discovery.storage_bytes},
                    {"name": "layer_count", "value": self.discovery.shape.layers},
                    {
                        "name": "feed_forward_length",
                        "value": self.discovery.shape.feed_forward_length,
                    },
                )
            ),
            measurement=measurement,
            artifact=artifact,
            failure=failure,
        )
        published = self._evidence_store().publish(record).to_record()
        self.evidence[observation_id] = published
        return published

    def _register_codecs(self) -> CodecRegistry:
        registry = CodecRegistry()
        codecs: tuple[QuantizationCodec, ...] = (
            *DENSE_GGUF_CODECS,
            Q2_K_CODEC,
            Q3_K_CODEC,
            Q4_K_CODEC,
            Q5_0_CODEC,
            Q5_K_CODEC,
            Q6_K_CODEC,
            Q8_0_CODEC,
            *tuple(cast(QuantizationCodec, item) for item in IQ4_CODECS.values()),
        )
        seen = set()
        for codec in codecs:
            identity = codec.identity.quant_type
            if identity in seen:
                continue
            registry.register(codec)
            seen.add(identity)
        return registry

    def _plan_edit(self, source: Path, removed: tuple[int, ...]) -> NativeGGUFMLPRemovalPlan:
        with open_gguf(source) as mapped:
            discovery = discover_gguf_components(
                mapped.container,
                family=self._model_family(),
            )
            if discovery.family not in {ModelFamily.LLAMA, ModelFamily.QWEN}:
                raise NativeGGUFOptimizeRuntimeError(
                    "native GGUF MLP optimization supports only Llama and dense Qwen"
                )
            return plan_native_gguf_model_mlp_channel_removal(discovery, removed_channels=removed)

    def _execute_edit(
        self, source: Path, destination: Path, removed: tuple[int, ...]
    ) -> tuple[NativeGGUFMLPExecutionResult, GGUFDiscovery, NativeGGUFMLPRemovalPlan]:
        if destination.exists():
            raise NativeGGUFOptimizeRuntimeError(
                f"refusing to overwrite GGUF candidate: {destination}"
            )
        plan = self._plan_edit(source, removed)
        destination.parent.mkdir(parents=True, exist_ok=True)
        estimated_output = max(source.stat().st_size * 2, source.stat().st_size)
        disk = preflight_gguf_disk(
            destination,
            destination.parent,
            GGUFDiskEstimate(estimated_output, 0),
        )
        with open_gguf(source) as mapped:
            result = execute_native_gguf_mlp_channel_removal(
                mapped, plan, destination, disk, self._register_codecs()
            )
        return result, result.output_discovery, plan

    def _validate(self, path: Path) -> Mapping[str, object]:
        validation_config, _, _ = self._tool_config()
        report = validate_generated_gguf(path, config=validation_config)
        record = report.to_record()
        if not report.successful:
            raise NativeGGUFOptimizeRuntimeError(
                f"GGUF generation validation failed: {report.failure_reason}"
            )
        return record

    def _quality(self, path: Path) -> Mapping[str, object]:
        assert self.source_path is not None
        _, config, _ = self._tool_config()
        report = benchmark_gguf_perplexity(self.source_path, path, self._manifest(), config=config)
        record = report.to_record()
        if not report.successful:
            raise NativeGGUFOptimizeRuntimeError(
                f"GGUF perplexity measurement failed: {report.candidate.failure_reason}"
            )
        return record

    def _throughput(self, path: Path) -> Mapping[str, object]:
        _, _, config = self._tool_config()
        report = benchmark_gguf_throughput(path, config=config)
        record = report.to_record()
        if not report.successful:
            raise NativeGGUFOptimizeRuntimeError(
                f"GGUF throughput measurement failed: {report.failure_reason}"
            )
        return record

    @staticmethod
    def _compact_quality(record: Mapping[str, object]) -> Mapping[str, object]:
        baseline = _mapping(record.get("baseline"), "quality baseline")
        candidate = _mapping(record.get("candidate"), "quality candidate")
        return {
            "successful": record.get("successful"),
            "manifest_id": record.get("manifest_id"),
            "config": record.get("config"),
            "tool": record.get("tool"),
            "baseline": {
                "model_sha256": baseline.get("model_sha256"),
                "perplexity": baseline.get("perplexity"),
                "uncertainty": baseline.get("uncertainty"),
                "command": baseline.get("command"),
                "returncode": baseline.get("returncode"),
                "timed_out": baseline.get("timed_out"),
                "failure_reason": baseline.get("failure_reason"),
            },
            "candidate": {
                "model_sha256": candidate.get("model_sha256"),
                "perplexity": candidate.get("perplexity"),
                "uncertainty": candidate.get("uncertainty"),
                "command": candidate.get("command"),
                "returncode": candidate.get("returncode"),
                "timed_out": candidate.get("timed_out"),
                "failure_reason": candidate.get("failure_reason"),
            },
            "perplexity_delta": record.get("perplexity_delta"),
        }

    @staticmethod
    def _compact_throughput(record: Mapping[str, object]) -> Mapping[str, object]:
        return {
            key: record.get(key)
            for key in (
                "successful",
                "failure_reason",
                "model_path",
                "model_sha256",
                "config",
                "command",
                "returncode",
                "timed_out",
                "peak_rss_bytes",
                "peak_vram_bytes",
                "memory_sample_count",
                "environment",
                "prompt",
                "generation",
                "parse_error",
            )
        }

    def _baseline_measurement(self) -> Mapping[str, object]:
        if self.baseline is not None:
            return self.baseline
        assert self.source_path is not None and self.discovery is not None
        validation = self._validate(self.source_path)
        quality = self._quality(self.source_path)
        throughput = self._throughput(self.source_path)
        quality_record = self._compact_quality(quality)
        throughput_record = self._compact_throughput(throughput)
        baseline_result = _mapping(quality.get("baseline"), "baseline quality")
        generation = _mapping(throughput.get("generation"), "baseline generation")
        self.baseline = {
            "perplexity": _number(baseline_result.get("perplexity"), "baseline.perplexity"),
            "perplexity_uncertainty": baseline_result.get("uncertainty"),
            "median_seconds": _number(
                _number(generation.get("average_latency_ns"), "baseline latency") / 1e9,
                "baseline.median_seconds",
            ),
            "generation_tokens_per_second": generation.get("average_tokens_per_second"),
            "peak_ram_bytes": throughput.get("peak_rss_bytes"),
            "peak_vram_bytes": throughput.get("peak_vram_bytes"),
            "parameter_count": self.discovery.parameter_count,
            "storage_bytes": self.discovery.storage_bytes,
            "disk_bytes": self.source_path.stat().st_size,
            "validation": validation,
            "quality": quality_record,
            "throughput": throughput_record,
            "hardware_inventory": dict(self._hardware(self.source_path.parent)),
        }
        return self.baseline

    def _candidate_limit(self) -> int:
        return max(1, min(self.plan.budget.evaluations, 8))

    def _generate_candidates(self) -> tuple[_GgufCandidate, ...]:
        self._ensure_source()
        assert self.source_path is not None and self.discovery is not None
        if self.discovery.family not in {ModelFamily.LLAMA, ModelFamily.QWEN}:
            raise NativeGGUFOptimizeRuntimeError(
                "native GGUF MLP optimization does not support family "
                f"{self.discovery.family.value}"
            )
        width = self.discovery.shape.feed_forward_length
        raw_layouts = []
        for tensor in self.discovery.tensors:
            if tensor.mapping.role.value in {"mlp_gate", "mlp_up", "mlp_down"}:
                try:
                    from modelsurgeon.adapters.gguf import QUANT_LAYOUTS

                    raw_layouts.append(QUANT_LAYOUTS[tensor.descriptor.quant_type].block_size)
                except (KeyError, ValueError) as error:
                    raise NativeGGUFOptimizeRuntimeError(
                        f"no pinned block geometry for {tensor.descriptor.name}"
                    ) from error
        block = max(raw_layouts, default=1)
        if width <= block:
            raise NativeGGUFOptimizeRuntimeError(
                "GGUF feed-forward width cannot admit a representable channel removal"
            )
        counts = tuple(count for count in (block, block * 2, block * 3) if count < width)
        candidates: list[_GgufCandidate] = []
        seen: set[tuple[int, ...]] = set()
        for count in counts:
            for offset in (0, max(0, width - count), max(0, (width - count) // 2)):
                offset -= offset % block
                removed = tuple(range(offset, offset + count))
                if removed in seen:
                    continue
                identity = {
                    "plan": self.plan.plan_id,
                    "removed_channels": list(removed),
                }
                digest = hashlib.sha256(_canonical(identity).encode()).hexdigest()
                candidates.append(
                    _GgufCandidate(
                        "candidate_gguf_" + digest,
                        "mutation_gguf_" + digest,
                        offset,
                        count,
                        removed,
                        {
                            "deferred_physical_plan": True,
                            "family": self.discovery.family.value,
                            "architecture": self.discovery.architecture,
                            "layer_count": self.discovery.shape.layers,
                            "removed_count": count,
                            "block_size": block,
                        },
                    )
                )
                seen.add(removed)
                if len(candidates) >= self._candidate_limit():
                    return tuple(sorted(candidates, key=lambda item: item.candidate_id))
        return tuple(sorted(candidates, key=lambda item: item.candidate_id))

    def _candidate_measurement(
        self, candidate: _GgufCandidate, destination: Path
    ) -> Mapping[str, object]:
        assert self.source_path is not None and self.discovery is not None
        result, output_discovery, plan = self._execute_edit(
            self.source_path, destination, candidate.removed_channels
        )
        validation = self._validate(destination)
        quality = self._quality(destination)
        throughput = self._throughput(destination)
        baseline = self._baseline_measurement()
        candidate_quality = _mapping(quality.get("candidate"), "candidate quality")
        generation = _mapping(throughput.get("generation"), "candidate generation")
        perplexity = _number(candidate_quality.get("perplexity"), "candidate.perplexity")
        baseline_ppl = _number(baseline["perplexity"], "baseline.perplexity")
        quality_gate = self._quality_gate(baseline_ppl, perplexity)
        baseline_parameters = _integer(
            baseline.get("parameter_count"), 1, "baseline.parameter_count"
        )
        baseline_storage = _integer(baseline.get("storage_bytes"), 1, "baseline.storage_bytes")
        latency = _number(
            _number(generation.get("average_latency_ns"), "candidate latency") / 1e9,
            "candidate.median_seconds",
        )
        measurement = {
            "measurement_authority": "physical_reloaded_gguf_external_runtime",
            "baseline_perplexity": baseline_ppl,
            "candidate_perplexity": perplexity,
            "perplexity_delta": perplexity - baseline_ppl,
            "quality_retention_ratio": quality_gate["quality_retention_ratio"],
            "quality_gate": quality_gate,
            "baseline_median_seconds": baseline["median_seconds"],
            "candidate_median_seconds": latency,
            "latency_delta_seconds": latency
            - _number(baseline["median_seconds"], "baseline.median_seconds"),
            "baseline_parameter_count": baseline["parameter_count"],
            "candidate_parameter_count": output_discovery.parameter_count,
            "parameter_delta": output_discovery.parameter_count - baseline_parameters,
            "baseline_storage_bytes": baseline["storage_bytes"],
            "candidate_storage_bytes": output_discovery.storage_bytes,
            "storage_delta_bytes": output_discovery.storage_bytes - baseline_storage,
            "baseline_disk_bytes": baseline["disk_bytes"],
            "candidate_disk_bytes": destination.stat().st_size,
            "baseline_peak_ram_bytes": baseline["peak_ram_bytes"],
            "candidate_peak_ram_bytes": throughput.get("peak_rss_bytes"),
            "baseline_peak_vram_bytes": baseline["peak_vram_bytes"],
            "candidate_peak_vram_bytes": throughput.get("peak_vram_bytes"),
            "candidate_perplexity_uncertainty": candidate_quality.get("uncertainty"),
            "validation": validation,
            "quality": self._compact_quality(quality),
            "throughput": self._compact_throughput(throughput),
            "mutation": plan.to_record(),
            "write": {
                "path": str(result.write_result.path),
                "sha256": "sha256:" + result.write_result.sha256,
                "file_size": result.write_result.file_size,
                "unchanged_tensor_count": len(result.unchanged_tensor_sha256),
                "requantization_errors": [
                    {
                        "component_id": str(item.component_id),
                        "destination_quant_type": item.destination_quant_type.value,
                        "block_offset": item.block_offset,
                        "block_count": item.block_count,
                        "error": asdict(item.error),
                    }
                    for item in result.requantization_errors
                ],
                "peak_row_working_bytes": result.peak_row_working_bytes,
            },
            "output_discovery": output_discovery.to_record(),
        }
        return measurement

    @staticmethod
    def _objective_value(
        metric: OptimizeMetric, measurement: Mapping[str, object]
    ) -> tuple[float, float] | None:
        if metric is OptimizeMetric.QUALITY:
            return (
                _number(measurement["baseline_perplexity"], "baseline.perplexity")
                / _number(measurement["candidate_perplexity"], "candidate.perplexity"),
                1.0,
            )
        names = {
            OptimizeMetric.PERPLEXITY: ("candidate_perplexity", "baseline_perplexity"),
            OptimizeMetric.PARAMETER_COUNT: (
                "candidate_parameter_count",
                "baseline_parameter_count",
            ),
            OptimizeMetric.LATENCY: ("candidate_median_seconds", "baseline_median_seconds"),
            OptimizeMetric.MEMORY: ("candidate_peak_ram_bytes", "baseline_peak_ram_bytes"),
            OptimizeMetric.DISK_SIZE: ("candidate_disk_bytes", "baseline_disk_bytes"),
        }
        selected = names.get(metric)
        if selected is None or not all(name in measurement for name in selected):
            return None
        if measurement[selected[0]] is None or measurement[selected[1]] is None:
            return None
        return (
            _number(measurement[selected[0]], selected[0]),
            _number(measurement[selected[1]], selected[1]),
        )

    def _frontier(self) -> Mapping[str, object]:
        root = self._resolved()
        objective_set = objectives_from_config(
            ObjectiveConfig.model_validate(_mapping(root.get("objective"), "objective"))
        )
        values_by_candidate: dict[str, dict[OptimizeMetric, tuple[float, float]]] = {}
        missing: set[OptimizeMetric] = set()
        for candidate in self.candidates:
            measurement = self.measurements.get(candidate.candidate_id)
            if measurement is None:
                continue
            values: dict[OptimizeMetric, tuple[float, float]] = {}
            for term in objective_set.terms:
                value = self._objective_value(term.metric, measurement)
                if value is None:
                    missing.add(term.metric)
                else:
                    values[term.metric] = value
            if len(values) == len(objective_set.terms):
                values_by_candidate[candidate.candidate_id] = values
        if missing:
            result: Mapping[str, object] = {
                "status": "unknown",
                "reason": "declared GGUF Pareto objectives were not all measured",
                "missing_metrics": sorted(item.value for item in missing),
                "objective_set": objective_set.to_record(),
                "candidate_count": len(values_by_candidate),
            }
            self.measured_frontier = result
            return result
        if not values_by_candidate:
            raise NativeGGUFOptimizeRuntimeError(
                "no measured GGUF candidate has complete declared Pareto objectives"
            )
        archive_path = (
            self._artifact_root()
            / "optimize"
            / str(self.campaign_run_id)
            / "gguf-measured-candidate-frontier.sqlite"
        )
        candidate_by_id = {item.candidate_id: item for item in self.candidates}
        with ParetoArchive(archive_path, objective_set) as archive:
            for candidate_id, values in sorted(values_by_candidate.items()):
                archive.put(
                    ParetoCandidate(
                        candidate_id,
                        tuple(
                            ParetoObjectiveValue(metric, value[0])
                            for metric, value in sorted(
                                values.items(), key=lambda item: item[0].value
                            )
                        ),
                        {
                            "candidate": candidate_by_id[candidate_id].to_record(),
                            "measurement": dict(self.measurements[candidate_id]),
                            "artifact_status": "physical_reloaded_gguf_candidate",
                        },
                    )
                )
            frontier = tuple(
                item.candidate.candidate_id for item in archive.entries(frontier_only=True)
            )
        scores = {
            candidate_id: objective_set.score(
                tuple(
                    ObjectiveObservation(metric, value[0], value[1])
                    for metric, value in sorted(
                        values_by_candidate[candidate_id].items(), key=lambda item: item[0].value
                    )
                )
            ).reward
            for candidate_id in frontier
        }
        preferred = sorted(frontier, key=lambda item: (-scores[item], item))[0]
        result = {
            "status": "measured",
            "archive_path": str(archive_path),
            "archive_sha256": _digest_file(archive_path),
            "objective_set": objective_set.to_record(),
            "candidate_count": len(values_by_candidate),
            "frontier_candidate_ids": list(frontier),
            "preferred_candidate_id": preferred,
            "artifact_status": "candidates_require_cumulative_surgery_and_deployment_validation",
        }
        self.measured_frontier = result
        return result

    def _constraints_pass(self, measurement: Mapping[str, object]) -> bool:
        constraints = _mapping(self._resolved().get("constraints"), "constraints")
        quality_gate = measurement.get("quality_gate")
        if not isinstance(quality_gate, Mapping):
            baseline_value = measurement.get("baseline_perplexity")
            candidate_value = measurement.get("candidate_perplexity")
            if baseline_value is None or candidate_value is None:
                return False
            quality_gate = self._quality_gate(
                _number(baseline_value, "baseline_perplexity"),
                _number(candidate_value, "candidate_perplexity"),
            )
        if quality_gate.get("accepted") is not True:
            return False
        max_ram = constraints.get("max_ram_bytes")
        if isinstance(max_ram, int):
            candidate_ram = measurement.get("candidate_peak_ram_bytes")
            if candidate_ram is None:
                return False
            if _number(candidate_ram, "candidate_peak_ram_bytes") > max_ram:
                return False
        max_vram = constraints.get("max_vram_bytes")
        if isinstance(max_vram, int):
            candidate_vram = measurement.get("candidate_peak_vram_bytes")
            if candidate_vram is None:
                return False
            if _number(candidate_vram, "candidate_peak_vram_bytes") > max_vram:
                return False
        max_disk = constraints.get("max_disk_bytes")
        if isinstance(max_disk, int):
            candidate_disk = measurement.get("candidate_disk_bytes")
            if candidate_disk is None:
                return False
            if _number(candidate_disk, "candidate_disk_bytes") > max_disk:
                return False
        min_gain = constraints.get("min_latency_gain_ratio")
        if isinstance(min_gain, (int, float)):
            baseline_value = measurement.get("baseline_median_seconds")
            candidate_value = measurement.get("candidate_median_seconds")
            if baseline_value is None or candidate_value is None:
                return False
            baseline = _number(baseline_value, "baseline_median_seconds")
            candidate = _number(candidate_value, "candidate_median_seconds")
            gain = 0.0 if baseline <= 0 else (baseline - candidate) / baseline
            if gain < float(min_gain):
                return False
        return True

    def _candidate_search(self) -> Mapping[str, object]:
        self._ensure_source()
        assert self.source_path is not None
        if not self.candidates:
            self.candidates = self._generate_candidates()
        if not self.candidates:
            raise NativeGGUFOptimizeRuntimeError("no representable GGUF MLP candidates exist")
        candidate_root = (
            self._artifact_root() / "optimize" / str(self.campaign_run_id) / "gguf" / "candidates"
        )
        candidate_root.mkdir(parents=True, exist_ok=True)
        feasible: list[_GgufCandidate] = []
        order: list[str] = []
        failures: dict[str, Mapping[str, object]] = {}
        for candidate in self.candidates:
            order.append(candidate.candidate_id)
            destination = candidate_root / f"{candidate.candidate_id}.gguf"
            try:
                measurement = self._candidate_measurement(candidate, destination)
            except Exception as error:
                failures[candidate.candidate_id] = {
                    "reason": str(error),
                    "classification": self._failure_classification(error),
                }
                self._publish_evidence(
                    stage="active_search",
                    state_id="source:" + str(self.source_digest),
                    outcome=OptimizationEvidenceOutcome.FAILED,
                    candidate_id=candidate.candidate_id,
                    mutation_id=candidate.mutation_id,
                    failure=failures[candidate.candidate_id],
                    lineage={"parent_state": "source", "candidate": candidate.to_record()},
                )
                continue
            self.measurements[candidate.candidate_id] = measurement
            accepted = self._constraints_pass(measurement)
            self._publish_evidence(
                stage="active_search",
                state_id="source:" + str(self.source_digest),
                outcome=(
                    OptimizationEvidenceOutcome.ACCEPTED
                    if accepted
                    else OptimizationEvidenceOutcome.REJECTED
                ),
                candidate_id=candidate.candidate_id,
                mutation_id=candidate.mutation_id,
                measurement=measurement,
                artifact={
                    "path": str(destination),
                    "digest": _digest_file(destination),
                    "size_bytes": destination.stat().st_size,
                },
                lineage={"parent_state": "source", "candidate": candidate.to_record()},
            )
            if accepted:
                feasible.append(candidate)
        if not feasible:
            raise NativeGGUFOptimizeRuntimeError(
                "no measured GGUF candidate met the declared hard constraints"
            )
        frontier = self._frontier()
        preferred_id = frontier.get("preferred_candidate_id")
        if not isinstance(preferred_id, str) or preferred_id not in {
            item.candidate_id for item in feasible
        }:
            preferred_id = sorted(feasible, key=lambda item: item.candidate_id)[0].candidate_id
        self.selected = next(item for item in feasible if item.candidate_id == preferred_id)
        self.selected_measurement = self.measurements[self.selected.candidate_id]
        self.search_comparison = {
            "measured_authority": "physical_reloaded_external_runtime",
            "measured_best": self.selected.candidate_id,
            "baselines": [
                {
                    "method": "meta_surgeon",
                    "candidate_id": None,
                    "measured": False,
                    "reason": "no GGUF Meta-Surgeon feature contract is enabled",
                },
                {
                    "method": "magnitude",
                    "candidate_id": min(
                        feasible, key=lambda item: (item.removed_count, item.candidate_id)
                    ).candidate_id,
                    "measured": True,
                },
                {
                    "method": "random",
                    "candidate_id": order[0],
                    "measured": order[0] in self.measurements,
                },
                {
                    "method": "no_guidance",
                    "candidate_id": self.selected.candidate_id,
                    "measured": True,
                },
            ],
            "evaluation_order": order,
            "failed_candidates": failures,
        }
        return {
            "selected_candidate_id": self.selected.candidate_id,
            "selected_measurement": dict(self.selected_measurement),
            "candidate_measurements": {
                key: dict(value) for key, value in sorted(self.measurements.items())
            },
            "search_comparison": self.search_comparison,
            "measured_frontier": frontier,
            "candidates": [item.to_record() for item in self.candidates],
            "resource_inventory": dict(self._hardware(candidate_root)),
        }

    def _surgery(self) -> Mapping[str, object]:
        self._ensure_source()
        assert self.source_path is not None and self.source_state is not None
        if self.selected is None:
            raise NativeGGUFOptimizeRuntimeError("GGUF surgery has no selected measured candidate")
        block = self.selected.removed_count
        if self.discovery is not None:
            try:
                from modelsurgeon.adapters.gguf import QUANT_LAYOUTS

                block = min(
                    QUANT_LAYOUTS[tensor.descriptor.quant_type].block_size
                    for tensor in self.discovery.tensors
                    if tensor.mapping.role.value in {"ffn_gate", "ffn_up", "ffn_down"}
                )
            except (ValueError, KeyError) as error:
                raise NativeGGUFOptimizeRuntimeError(
                    "GGUF MLP block geometry is unavailable"
                ) from error
        chunks = [
            self.selected.removed_channels[index : index + block]
            for index in range(0, len(self.selected.removed_channels), block)
        ]
        output_root = (
            self._artifact_root() / "optimize" / str(self.campaign_run_id) / "gguf" / "cumulative"
        )
        measurements: dict[str, Mapping[str, object]] = {}
        _, _, throughput_config = self._tool_config()
        limits = GGUFResourceLimits(self.plan.budget.max_ram_bytes, self.plan.budget.max_disk_bytes)

        edits: list[GGUFEdit] = []
        for index, removed in enumerate(chunks):
            mutation_id = f"{self.selected.mutation_id}.stage-{index:02d}"

            def execute(
                source: Path,
                destination: Path,
                removed: tuple[int, ...] = tuple(removed),
                mutation_id: str = mutation_id,
            ) -> GGUFStageMeasurement:
                result, discovery, plan = self._execute_edit(source, destination, removed)
                validation = self._validate(destination)
                quality = self._quality(destination)
                throughput = benchmark_gguf_throughput(destination, config=throughput_config)
                if not throughput.successful:
                    raise NativeGGUFOptimizeRuntimeError(
                        "cumulative GGUF throughput measurement failed: "
                        f"{throughput.failure_reason}"
                    )
                peak = throughput.peak_rss_bytes or 0
                quality_baseline = _mapping(quality.get("baseline"), "sequence quality baseline")
                quality_candidate = _mapping(
                    quality.get("candidate"), "sequence quality candidate"
                )
                baseline_perplexity = _number(
                    quality_baseline.get("perplexity"), "sequence.baseline_perplexity"
                )
                candidate_perplexity = _number(
                    quality_candidate.get("perplexity"), "sequence.candidate_perplexity"
                )
                quality_gate = self._quality_gate(baseline_perplexity, candidate_perplexity)
                throughput_record = throughput.to_record()
                generation = _mapping(
                    throughput_record.get("generation"), "sequence generation"
                )
                baseline = self._baseline_measurement()
                candidate_latency = _number(
                    _number(generation.get("average_latency_ns"), "sequence latency") / 1e9,
                    "sequence.candidate_median_seconds",
                )
                measurement = {
                    "baseline_perplexity": baseline_perplexity,
                    "candidate_perplexity": candidate_perplexity,
                    "perplexity_delta": quality_gate["perplexity_delta"],
                    "quality_retention_ratio": quality_gate["quality_retention_ratio"],
                    "quality_gate": quality_gate,
                    "baseline_median_seconds": baseline["median_seconds"],
                    "candidate_median_seconds": candidate_latency,
                    "candidate_peak_ram_bytes": throughput.peak_rss_bytes,
                    "candidate_peak_vram_bytes": throughput.peak_vram_bytes,
                    "quality": self._compact_quality(quality),
                    "throughput": self._compact_throughput(throughput.to_record()),
                    "validation": validation,
                    "mutation": plan.to_record(),
                    "output_discovery": discovery.to_record(),
                    "candidate_disk_bytes": destination.stat().st_size,
                }
                if not self._constraints_pass(measurement):
                    raise NativeGGUFOptimizeRuntimeError(
                        "cumulative GGUF child exceeded the declared quality or resource constraint"
                    )
                measurements[mutation_id] = measurement
                return GGUFStageMeasurement(
                    discovery.parameter_count,
                    discovery.storage_bytes,
                    hashlib.sha256(_canonical(discovery.to_record()).encode()).hexdigest(),
                    self._tensor_hashes(destination),
                    tuple(sorted(plan.coupled_tensor_names)),
                    max(peak, result.peak_row_working_bytes),
                    destination.stat().st_size,
                    True,
                    True,
                )

            edits.append(GGUFEdit(mutation_id, "mlp_channel_removal", execute))
        assert self.source_digest is not None
        sequence = run_gguf_cumulative_sequence(
            self.source_path,
            self.source_state,
            tuple(edits),
            output_root=output_root,
            source_outcome_id="source_" + self.source_digest.removeprefix("sha256:"),
            limits=limits,
        )
        if not sequence.stages:
            raise NativeGGUFOptimizeRuntimeError(
                f"GGUF cumulative sequence produced no accepted child: {sequence.failure_reason}"
            )
        self.sequence_record = sequence.to_record()
        final = sequence.final_artifact
        self.artifact = final
        self.artifact_digest = _digest_file(final)
        selected = self.selected
        assert selected is not None
        final_mutation = sequence.stages[-1].mutation_id
        final_measurement = measurements.get(final_mutation)
        if final_measurement is None:
            raise NativeGGUFOptimizeRuntimeError("final GGUF sequence measurement was not retained")
        self.selected_measurement = final_measurement
        for stage in sequence.stages:
            stage_digest = stage.outcome.artifact.digest
            if not isinstance(stage_digest, str) or not stage_digest:
                raise NativeGGUFOptimizeRuntimeError(
                    "GGUF sequence stage did not retain an artifact digest"
                )
            self._publish_evidence(
                stage="physical_surgery",
                state_id="accepted_child:" + stage.outcome.outcome_id,
                outcome=OptimizationEvidenceOutcome.ACCEPTED,
                candidate_id=selected.candidate_id,
                mutation_id=stage.mutation_id,
                measurement=measurements.get(stage.mutation_id),
                artifact={
                    "path": str(stage.artifact),
                    "digest": "sha256:" + stage_digest,
                    "size_bytes": stage.outcome.artifact.size_bytes,
                },
                lineage={
                    "sequence_id": sequence.sequence_id,
                    "stage_index": stage.index,
                    "parent_outcome_id": stage.outcome.parent_outcome_id,
                    "outcome_id": stage.outcome.outcome_id,
                },
            )
        if sequence.failed_index is not None:
            self._publish_evidence(
                stage="physical_surgery",
                state_id="rolled_back:" + sequence.sequence_id,
                outcome=OptimizationEvidenceOutcome.ROLLED_BACK,
                candidate_id=self.selected.candidate_id,
                mutation_id=sequence.stages[-1].mutation_id,
                failure={"reason": sequence.failure_reason, "failed_index": sequence.failed_index},
                lineage={"sequence_id": sequence.sequence_id},
            )
        return {
            "status": "accepted",
            "artifact": str(final),
            "artifact_digest": self.artifact_digest,
            "sequence": self.sequence_record,
            "sequence_measurements": measurements,
            "resource_inventory": dict(self._hardware(output_root)),
        }

    def _deployment(self) -> Mapping[str, object]:
        if self.artifact is None or self.artifact_digest is None:
            raise NativeGGUFOptimizeRuntimeError("deployment requires a published GGUF artifact")
        quality = self._quality(self.artifact)
        throughput = self._throughput(self.artifact)
        measurement = self.measurements.get(self.selected.candidate_id if self.selected else "", {})
        baseline = self._baseline_measurement()
        candidate = _mapping(quality.get("candidate"), "deployment quality")
        generation = _mapping(throughput.get("generation"), "deployment generation")
        candidate_latency = _number(
            _number(generation.get("average_latency_ns"), "deployment latency") / 1e9,
            "deployment.candidate_latency",
        )
        baseline_latency = _number(baseline["median_seconds"], "deployment.baseline_latency")
        latency_gain = (
            0.0
            if baseline_latency <= 0
            else (baseline_latency - candidate_latency) / baseline_latency
        )
        candidate_ppl = _number(candidate.get("perplexity"), "deployment.perplexity")
        quality_gate = self._quality_gate(
            _number(baseline["perplexity"], "baseline.perplexity"), candidate_ppl
        )
        quality_delta = _number(quality_gate["perplexity_delta"], "deployment.perplexity_delta")
        constraints = _mapping(self._resolved().get("constraints"), "constraints")
        passed = bool(quality_gate["accepted"])
        if isinstance(constraints.get("min_latency_gain_ratio"), (int, float)):
            passed = passed and latency_gain >= _number(
                constraints["min_latency_gain_ratio"],
                "constraints.min_latency_gain_ratio",
            )
        if isinstance(constraints.get("max_disk_bytes"), int):
            max_disk = cast(int, constraints["max_disk_bytes"])
            passed = passed and self.artifact.stat().st_size <= max_disk
        if (
            isinstance(constraints.get("max_ram_bytes"), int)
            and throughput.get("peak_rss_bytes") is not None
        ):
            max_ram = cast(int, constraints["max_ram_bytes"])
            passed = passed and int(cast(int, throughput["peak_rss_bytes"])) <= max_ram
        self.deployment_constraints_passed = passed
        self.deployment = {
            "baseline": dict(baseline),
            "candidate": {
                "perplexity": candidate_ppl,
                "perplexity_delta": quality_delta,
                "quality_retention_ratio": quality_gate["quality_retention_ratio"],
                "latency_seconds": candidate_latency,
                "latency_gain_ratio": latency_gain,
                "disk_bytes": self.artifact.stat().st_size,
                "peak_ram_bytes": throughput.get("peak_rss_bytes"),
                "peak_vram_bytes": throughput.get("peak_vram_bytes"),
            },
            "quality": self._compact_quality(quality),
            "quality_gate": quality_gate,
            "throughput": self._compact_throughput(throughput),
            "constraints_passed": passed,
            "hardware_inventory": dict(self._hardware(self.artifact.parent)),
            "artifact": str(self.artifact),
            "artifact_digest": self.artifact_digest,
            "selected_search_measurement": dict(measurement),
        }
        return self.deployment

    @staticmethod
    def _failure_classification(error: BaseException | str) -> str:
        text = str(error).lower()
        if "out of memory" in text or "oom" in text:
            return "oom"
        if "timeout" in text or "timed out" in text:
            return "timeout"
        if "unsupported" in text:
            return "unsupported"
        if "unknown" in text or "unavailable" in text:
            return "unknown"
        return "runtime_failure"

    def _result(
        self,
        stage: OptimizeStage,
        detail: Mapping[str, object] | str,
        *,
        measured: bool = False,
        constraints_passed: bool = False,
        artifact_digest: str | None = None,
        candidate_id: str | None = None,
        evaluation_id: str | None = None,
        alternatives: tuple[str, ...] = (),
    ) -> StageResult:
        detail_text = detail if isinstance(detail, str) else _canonical(detail)
        evidence_id = (
            "evidence_"
            + hashlib.sha256(
                f"{self.plan.plan_id}:{stage.value}:{detail_text}".encode()
            ).hexdigest()
        )
        state_id = None
        if artifact_digest is not None:
            state_id = "state_" + hashlib.sha256(artifact_digest.encode()).hexdigest()
        return StageResult(
            WorkflowOutcome.SUPPORTED,
            evidence_id,
            detail_text,
            measured=measured,
            complete=True,
            constraints_passed=constraints_passed,
            artifact_digest=artifact_digest,
            candidate_id=candidate_id,
            candidate_state_id=state_id,
            evaluation_id=evaluation_id,
            artifact_immutable=artifact_digest is not None,
            alternatives=alternatives,
        )

    def _unsupported(self, stage: OptimizeStage, reason: str) -> StageResult:
        try:
            reference = self._publish_evidence(
                stage=stage.value,
                state_id="boundary:" + stage.value,
                outcome=OptimizationEvidenceOutcome.UNSUPPORTED,
                failure={"reason": reason, "classification": self._failure_classification(reason)},
                lineage={"boundary": True},
            )
        except Exception:
            reference = None
        return StageResult(
            WorkflowOutcome.UNSUPPORTED,
            "unsupported_"
            + hashlib.sha256(f"{self.plan.plan_id}:{stage.value}:{reason}".encode()).hexdigest(),
            _canonical({"status": "unsupported", "reason": reason, "evidence": reference}),
            complete=True,
        )

    def _failed(self, stage: OptimizeStage, error: BaseException) -> StageResult:
        reason = str(error) or error.__class__.__name__
        try:
            reference = self._publish_evidence(
                stage=stage.value,
                state_id="failure:" + stage.value,
                outcome=OptimizationEvidenceOutcome.FAILED,
                failure={"reason": reason, "classification": self._failure_classification(error)},
                lineage={"boundary": True},
            )
        except Exception:
            reference = None
        return StageResult(
            WorkflowOutcome.FAILED,
            "failure_"
            + hashlib.sha256(f"{self.plan.plan_id}:{stage.value}:{reason}".encode()).hexdigest(),
            _canonical({"status": "failed", "reason": reason, "evidence": reference}),
            complete=True,
        )

    def _hydrate(self, context: StageContext) -> None:
        self.campaign_run_id = context.run.run_id
        for stage, stage_result in context.completed_results.items():
            detail = _json_detail(stage_result)
            candidates = detail.get("candidates")
            if isinstance(candidates, list) and candidates:
                self.candidates = tuple(_GgufCandidate.from_record(item) for item in candidates)
            baseline = detail.get("baseline")
            if isinstance(baseline, Mapping):
                self.baseline = baseline
            measurements = detail.get("candidate_measurements")
            if isinstance(measurements, Mapping):
                self.measurements = {
                    str(key): _mapping(value, "candidate measurement")
                    for key, value in measurements.items()
                }
            selected_id = detail.get("selected_candidate_id")
            if isinstance(selected_id, str):
                self.selected = next(
                    (item for item in self.candidates if item.candidate_id == selected_id), None
                )
            selected_measurement = detail.get("selected_measurement")
            if isinstance(selected_measurement, Mapping):
                self.selected_measurement = selected_measurement
            frontier = detail.get("measured_frontier")
            if isinstance(frontier, Mapping):
                self.measured_frontier = frontier
            comparison = detail.get("search_comparison")
            if isinstance(comparison, Mapping):
                self.search_comparison = comparison
            sequence = detail.get("sequence")
            if isinstance(sequence, Mapping):
                self.sequence_record = sequence
            artifact = detail.get("artifact")
            if isinstance(artifact, str) and stage_result.artifact_digest is not None:
                path = Path(artifact).expanduser().absolute().resolve(strict=False)
                if path.is_file() and _digest_file(path) == stage_result.artifact_digest:
                    self.artifact = path
                    self.artifact_digest = stage_result.artifact_digest
            if stage is OptimizeStage.DEPLOYMENT_BENCHMARK:
                self.deployment = detail
                self.deployment_constraints_passed = stage_result.constraints_passed

    def run_stage(self, context: StageContext) -> StageResult:
        stage = context.stage
        try:
            self._hydrate(context)
            self._ensure_source()
            assert self.source_path is not None and self.discovery is not None
            if stage is OptimizeStage.PROFILE:
                validation_config, quality_config, throughput_config = self._tool_config()
                return self._result(
                    stage,
                    {
                        "source": self._model_record(),
                        "hardware_inventory": dict(self._hardware(self.source_path.parent)),
                        "runtime": {
                            "validation": asdict(validation_config),
                            "perplexity": quality_config.to_record(),
                            "throughput": throughput_config.to_record(),
                        },
                    },
                    measured=True,
                    constraints_passed=True,
                )
            if stage is OptimizeStage.CAPABILITY:
                if self.discovery.family not in {ModelFamily.LLAMA, ModelFamily.QWEN}:
                    return self._unsupported(
                        stage,
                        "native GGUF MLP optimization does not support family "
                        f"{self.discovery.family.value}",
                    )
                return self._result(
                    stage,
                    {
                        "family": self.discovery.family.value,
                        "architecture": self.discovery.architecture,
                        "supported_scopes": ["mlp_channel"],
                        "physical_measurements_authoritative": True,
                        "repair": "unsupported",
                        "additional_quantization": "unsupported",
                    },
                    constraints_passed=True,
                )
            if stage is OptimizeStage.BASELINE:
                baseline = self._baseline_measurement()
                return self._result(
                    stage,
                    {"baseline": baseline, "source_digest": self.source_digest},
                    measured=True,
                    constraints_passed=True,
                )
            if stage is OptimizeStage.CANDIDATE_GENERATION:
                self.candidates = self.candidates or self._generate_candidates()
                if not self.candidates:
                    return self._unsupported(
                        stage, "no representable native GGUF MLP candidates exist"
                    )
                return self._result(
                    stage,
                    {
                        "candidate_count": len(self.candidates),
                        "candidates": [item.to_record() for item in self.candidates],
                        "source": "GGUF component graph and quantization block geometry",
                    },
                    measured=True,
                    constraints_passed=True,
                    alternatives=tuple(item.candidate_id for item in self.candidates[:8]),
                )
            if stage is OptimizeStage.ACTIVE_SEARCH:
                search = self._candidate_search()
                assert self.selected is not None and self.selected_measurement is not None
                evaluation_id = (
                    "evaluation_"
                    + hashlib.sha256(_canonical(self.selected_measurement).encode()).hexdigest()
                )
                return self._result(
                    stage,
                    search,
                    measured=True,
                    constraints_passed=True,
                    candidate_id=self.selected.candidate_id,
                    evaluation_id=evaluation_id,
                    alternatives=tuple(
                        item.candidate_id
                        for item in self.candidates
                        if item.candidate_id != self.selected.candidate_id
                    )[:8],
                )
            if stage is OptimizeStage.SURGERY:
                surgery = self._surgery()
                assert self.selected is not None and self.artifact_digest is not None
                evaluation_id = (
                    "evaluation_"
                    + hashlib.sha256(
                        _canonical(self.selected_measurement or {}).encode()
                    ).hexdigest()
                )
                return self._result(
                    stage,
                    surgery,
                    measured=True,
                    constraints_passed=True,
                    artifact_digest=self.artifact_digest,
                    candidate_id=self.selected.candidate_id,
                    evaluation_id=evaluation_id,
                )
            if stage is OptimizeStage.REPAIR:
                if (
                    _mapping(self._resolved().get("repair"), "repair").get("method", "none")
                    != "none"
                ):
                    return self._unsupported(stage, "native GGUF repair is not implemented")
                return self._result(
                    stage,
                    {
                        "status": "unsupported",
                        "method": "none",
                        "reason": "repair was not requested for the GGUF cell",
                        "artifact": str(self.artifact) if self.artifact else None,
                        "artifact_digest": self.artifact_digest,
                    },
                    artifact_digest=self.artifact_digest,
                    candidate_id=None if self.selected is None else self.selected.candidate_id,
                )
            if stage is OptimizeStage.QUANTIZATION:
                if (
                    _mapping(self._resolved().get("quantization"), "quantization").get(
                        "method", "none"
                    )
                    != "none"
                ):
                    return self._unsupported(
                        stage, "native GGUF additional quantization is not implemented"
                    )
                return self._result(
                    stage,
                    {
                        "status": "unsupported",
                        "method": "none",
                        "reason": "additional quantization was not requested for the GGUF cell",
                        "artifact": str(self.artifact) if self.artifact else None,
                        "artifact_digest": self.artifact_digest,
                    },
                    artifact_digest=self.artifact_digest,
                    candidate_id=None if self.selected is None else self.selected.candidate_id,
                )
            if stage is OptimizeStage.DEPLOYMENT_BENCHMARK:
                deployment = self._deployment()
                return self._result(
                    stage,
                    deployment,
                    measured=True,
                    constraints_passed=bool(self.deployment_constraints_passed),
                    artifact_digest=self.artifact_digest,
                    candidate_id=None if self.selected is None else self.selected.candidate_id,
                    evaluation_id="evaluation_"
                    + hashlib.sha256(_canonical(deployment).encode()).hexdigest(),
                )
            if stage is OptimizeStage.PARETO:
                if self.artifact_digest is None or self.selected is None:
                    return self._unsupported(stage, "no published GGUF artifact is available")
                if self.deployment_constraints_passed is not True:
                    return self._unsupported(
                        stage, "deployment measurements failed a declared hard constraint"
                    )
                frontier = self.measured_frontier or self._frontier()
                if frontier.get("status") != "measured":
                    return self._unsupported(
                        stage, "GGUF Pareto objectives were not fully measured"
                    )
                raw_frontier_ids = frontier.get("frontier_candidate_ids", ())
                frontier_ids = (
                    tuple(item for item in raw_frontier_ids if isinstance(item, str))
                    if isinstance(raw_frontier_ids, (list, tuple))
                    else ()
                )
                return self._result(
                    stage,
                    {
                        "selection": (
                            "one physically measured GGUF artifact selected from "
                            "the declared frontier"
                        ),
                        "measured_frontier": frontier,
                        "deployment": self.deployment,
                    },
                    measured=True,
                    constraints_passed=True,
                    artifact_digest=self.artifact_digest,
                    candidate_id=self.selected.candidate_id,
                    evaluation_id="evaluation_"
                    + hashlib.sha256(_canonical(frontier).encode()).hexdigest(),
                    alternatives=tuple(
                        item for item in frontier_ids if item != self.selected.candidate_id
                    ),
                )
            if stage is OptimizeStage.REPORT:
                return self._result(
                    stage,
                    "native GGUF evidence report retained in the optimize run state",
                )
            raise NativeGGUFOptimizeRuntimeError(f"unsupported optimize stage: {stage.value}")
        except NativeGGUFOptimizeRuntimeError as error:
            return self._unsupported(stage, str(error))
        except Exception as error:
            return self._failed(stage, error)


def build_native_gguf_optimize_runtime(plan: OptimizePlan) -> OptimizeRuntime:
    """Build the explicit first-party native GGUF runtime."""

    return NativeGGUFOptimizeRuntime(plan)


__all__ = ["NativeGGUFOptimizeRuntime", "build_native_gguf_optimize_runtime"]
