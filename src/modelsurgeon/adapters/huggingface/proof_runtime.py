"""Concrete Hugging Face/PyTorch runtime for the First Surgeon MLP-channel proof."""

from __future__ import annotations

import gc
import hashlib
import math
import os
import statistics
import time
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, ExitStack
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from types import TracebackType
from typing import Any, Self, cast

from modelsurgeon.adapters import ArchitectureEvidence, detect_model_family
from modelsurgeon.adapters.huggingface.discovery import discover_huggingface_components
from modelsurgeon.adapters.huggingface.loader import (
    HuggingFaceDependencyError,
    HuggingFaceDType,
    HuggingFaceLoadRequest,
    load_causal_lm,
)
from modelsurgeon.cli.experiment import ResolvedExperiment, SingleMutationExperimentResult
from modelsurgeon.evaluation.tiered import (
    EscalationAction,
    EvaluationTier,
    MetricDecision,
    ThresholdComparator,
    TierDecision,
    TieredEvaluationReport,
    TierThreshold,
)
from modelsurgeon.experiments.candidates import CandidateScope, MutationCandidate
from modelsurgeon.experiments.hardware import collect_hardware_inventory
from modelsurgeon.experiments.identity import (
    ExperimentIdentitySpec,
    canonical_identity_json,
    derive_experiment_identity,
    derive_run_identity,
)
from modelsurgeon.experiments.schema import (
    DatasetTarget,
    ExperimentOutcome,
    ExperimentOutcomeKind,
    ExperimentRecord,
    MetricObservation,
    MetricState,
    ModelTarget,
    SeedContext,
    StageTiming,
    VersionContext,
)
from modelsurgeon.features.cache import FeaturePartition, FeaturePartitionKey
from modelsurgeon.features.gradient_features import GradientFeatures
from modelsurgeon.features.schema import (
    FEATURE_SCHEMA_VERSION,
    FeatureKind,
    FeatureRecord,
    FeatureSampleContext,
    PrecisionProvenance,
    PrecisionSource,
)
from modelsurgeon.graph import (
    ComponentGraph,
    ComponentId,
    ComponentRecordLike,
    build_component_graph,
)
from modelsurgeon.instrumentation.memory_telemetry import (
    MemoryTelemetryConfig,
    MemoryTelemetryError,
    MemoryTelemetryReport,
    TorchCudaMemoryProvider,
    collect_memory_telemetry,
)
from modelsurgeon.surgery.contracts import (
    MutationDelta,
    MutationKind,
    MutationPlan,
    MutationRequest,
    MutationTransaction,
    require_safe_transaction,
)
from modelsurgeon.surgery.serialization import (
    MUTATION_RECORD_SCHEMA_VERSION,
    MutationOutcome,
    MutationOutcomeStatus,
    MutationProvenance,
)
from modelsurgeon.surgery.target_resolution import MutationTargetResolver
from modelsurgeon.surgery.transaction import InMemoryMutationTransaction

HF_MLP_PROOF_RUNTIME_VERSION = "1"
HF_MLP_FEATURE_EXTRACTOR = "hf_mlp_channel"
HF_MLP_FEATURE_EXTRACTOR_VERSION = "1"
HF_MLP_EVALUATOR_VERSION = "hf_mlp_perplexity_v1"


class HuggingFaceMLPProofError(ValueError):
    """Raised when the concrete HF proof runtime cannot preserve its evidence contract."""


@dataclass(frozen=True, slots=True)
class HuggingFaceMLPProofConfig:
    model: str
    calibration_text: Path
    revision: str | None = None
    tokenizer: str | None = None
    tokenizer_revision: str | None = None
    device_map: str | None = "auto"
    dtype: HuggingFaceDType = HuggingFaceDType.AUTO
    trust_remote_code: bool = False
    local_files_only: bool = False
    sequence_length: int = 256
    max_tokens: int = 4096
    safe_perplexity_delta: float = 0.25
    seed: int = 42
    tool_revision: str | None = None

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise HuggingFaceMLPProofError("model cannot be blank")
        if self.revision is not None and not self.revision.strip():
            raise HuggingFaceMLPProofError("revision cannot be blank")
        if self.tokenizer is not None and not self.tokenizer.strip():
            raise HuggingFaceMLPProofError("tokenizer cannot be blank")
        if self.tokenizer_revision is not None and not self.tokenizer_revision.strip():
            raise HuggingFaceMLPProofError("tokenizer revision cannot be blank")
        if self.sequence_length < 2:
            raise HuggingFaceMLPProofError("sequence_length must be at least 2")
        if self.max_tokens < 2:
            raise HuggingFaceMLPProofError("max_tokens must be at least 2")
        if not math.isfinite(self.safe_perplexity_delta) or self.safe_perplexity_delta < 0:
            raise HuggingFaceMLPProofError("safe_perplexity_delta must be finite and non-negative")
        if isinstance(self.seed, bool) or self.seed < 0 or self.seed >= 1 << 64:
            raise HuggingFaceMLPProofError("seed must be an unsigned 64-bit integer")


@dataclass(frozen=True, slots=True)
class _PerplexityMeasurement:
    mean_loss: float
    perplexity: float
    token_count: int

    def __post_init__(self) -> None:
        if self.token_count <= 0:
            raise HuggingFaceMLPProofError("perplexity measurement requires target tokens")
        if (
            not math.isfinite(self.mean_loss)
            or self.mean_loss < 0
            or not math.isfinite(self.perplexity)
            or self.perplexity < 1
        ):
            raise HuggingFaceMLPProofError("perplexity measurement is not finite and valid")


@dataclass(frozen=True, slots=True)
class _CandidateMeasurement:
    mutation_id: str
    post: _PerplexityMeasurement
    loss_delta: float
    perplexity_delta: float
    accepted: bool
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class MLPChannelSetMeasurement:
    coordinates: tuple[tuple[int, int], ...]
    baseline_perplexity: float
    masked_perplexity: float
    perplexity_delta: float
    baseline_median_seconds: float
    masked_median_seconds: float
    latency_delta_seconds: float
    measurement_wall_seconds: float
    repetitions: int
    token_count: int
    quantization_error: float = 0.0

    def to_record(self) -> dict[str, object]:
        return {
            "coordinates": [
                {"layer_index": layer, "channel_index": channel}
                for layer, channel in self.coordinates
            ],
            "baseline_perplexity": self.baseline_perplexity,
            "masked_perplexity": self.masked_perplexity,
            "perplexity_delta": self.perplexity_delta,
            "baseline_median_seconds": self.baseline_median_seconds,
            "masked_median_seconds": self.masked_median_seconds,
            "latency_delta_seconds": self.latency_delta_seconds,
            "measurement_wall_seconds": self.measurement_wall_seconds,
            "repetitions": self.repetitions,
            "token_count": self.token_count,
            "quantization_error": self.quantization_error,
            "quantization_error_reason": "high-precision mask path performs no requantization",
        }


@dataclass(slots=True)
class _ActivationAccumulator:
    count: int = 0
    sums: Any = None
    abs_sums: Any = None
    square_sums: Any = None
    zero_counts: Any = None
    max_abs: Any = None
    storage_dtype: str | None = None


@dataclass(frozen=True, slots=True)
class _LayerWeightStatistics:
    count: int
    combined_dtype: str
    part_dtypes: tuple[str, str, str]
    values: Any


@dataclass(frozen=True, slots=True)
class FeatureCollectionCost:
    wall_seconds: float
    initial_rss_bytes: int | None
    peak_rss_bytes: int | None
    incremental_peak_rss_bytes: int | None
    initial_cuda_allocated_bytes: int | None
    peak_cuda_allocated_bytes: int | None
    incremental_peak_cuda_allocated_bytes: int | None
    peak_cuda_reserved_bytes: int | None

    def to_record(self) -> dict[str, object]:
        return {
            "wall_seconds": self.wall_seconds,
            "initial_rss_bytes": self.initial_rss_bytes,
            "peak_rss_bytes": self.peak_rss_bytes,
            "incremental_peak_rss_bytes": self.incremental_peak_rss_bytes,
            "initial_cuda_allocated_bytes": self.initial_cuda_allocated_bytes,
            "peak_cuda_allocated_bytes": self.peak_cuda_allocated_bytes,
            "incremental_peak_cuda_allocated_bytes": (self.incremental_peak_cuda_allocated_bytes),
            "peak_cuda_reserved_bytes": self.peak_cuda_reserved_bytes,
        }


@dataclass(frozen=True, slots=True)
class FeatureCollectionCosts:
    static: FeatureCollectionCost
    forward_baseline: FeatureCollectionCost
    activation: FeatureCollectionCost

    def to_record(self) -> dict[str, object]:
        return {
            "static": self.static.to_record(),
            "forward_baseline": self.forward_baseline.to_record(),
            "activation": self.activation.to_record(),
            "activation_overhead": {
                "wall_seconds": self.activation.wall_seconds - self.forward_baseline.wall_seconds,
                "peak_rss_bytes": _difference(
                    self.activation.peak_rss_bytes,
                    self.forward_baseline.peak_rss_bytes,
                ),
                "peak_cuda_allocated_bytes": _difference(
                    self.activation.peak_cuda_allocated_bytes,
                    self.forward_baseline.peak_cuda_allocated_bytes,
                ),
                "peak_cuda_reserved_bytes": _difference(
                    self.activation.peak_cuda_reserved_bytes,
                    self.forward_baseline.peak_cuda_reserved_bytes,
                ),
            },
        }


@dataclass(frozen=True, slots=True)
class MLPChannelGradientCollection:
    records: tuple[tuple[ComponentId, tuple[FeatureRecord, ...]], ...]
    cost: FeatureCollectionCost
    token_count: int

    def __post_init__(self) -> None:
        if self.token_count <= 0 or not self.records:
            raise HuggingFaceMLPProofError("gradient collection requires records and tokens")

    def to_record(self) -> dict[str, object]:
        return {
            "token_count": self.token_count,
            "component_count": len(self.records),
            "cost": self.cost.to_record(),
        }


def _incremental(peak: int | None, initial: int | None) -> int | None:
    if peak is None or initial is None:
        return None
    return max(0, peak - initial)


def _difference(right: int | None, left: int | None) -> int | None:
    if right is None or left is None:
        return None
    return right - left


def _feature_collection_cost(report: MemoryTelemetryReport) -> FeatureCollectionCost:
    initial = report.samples[0]
    return FeatureCollectionCost(
        report.samples[-1].elapsed_seconds,
        initial.rss_bytes,
        report.peak_rss_bytes,
        _incremental(report.peak_rss_bytes, initial.rss_bytes),
        initial.cuda_allocated_bytes,
        report.peak_cuda_allocated_bytes,
        _incremental(report.peak_cuda_allocated_bytes, initial.cuda_allocated_bytes),
        report.peak_cuda_reserved_bytes,
    )


class _NoopSnapshotTarget:
    def snapshot(self) -> object:
        return None

    def restore(self, snapshot: object) -> None:
        del snapshot


class _DownProjectionChannelMask(AbstractContextManager["_DownProjectionChannelMask"]):
    """Temporarily zero one intermediate channel at the down-projection input."""

    def __init__(self, module: Any, channel_index: int, torch: Any) -> None:
        self._module = module
        self._channel_index = channel_index
        self._torch = torch
        self._handle: Any = None

    def _hook(self, module: object, inputs: tuple[object, ...]) -> tuple[object, ...]:
        del module
        if not inputs:
            raise HuggingFaceMLPProofError("down projection received no positional input")
        tensor: Any = inputs[0]
        if not self._torch.is_tensor(tensor):
            raise HuggingFaceMLPProofError("down projection input is not a tensor")
        if tensor.ndim < 1 or self._channel_index >= int(tensor.shape[-1]):
            raise HuggingFaceMLPProofError("MLP channel index exceeds down-projection input width")
        masked = tensor.clone()
        masked[..., self._channel_index] = 0
        return (masked, *inputs[1:])

    def __enter__(self) -> Self:
        if self._handle is not None:
            raise HuggingFaceMLPProofError("MLP channel mask is already active")
        self._handle = self._module.register_forward_pre_hook(self._hook)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        if self._handle is None:
            return
        handle = self._handle
        self._handle = None
        handle.remove()


class _DownProjectionChannelMaskSet(AbstractContextManager["_DownProjectionChannelMaskSet"]):
    """Temporarily zero a canonical set of channels with one hook and one clone."""

    def __init__(self, module: Any, channels: tuple[int, ...], torch: Any) -> None:
        self._module = module
        self._channels = channels
        self._torch = torch
        self._handle: Any = None

    def _hook(self, module: object, inputs: tuple[object, ...]) -> tuple[object, ...]:
        del module
        if not inputs:
            raise HuggingFaceMLPProofError("down projection received no positional input")
        tensor: Any = inputs[0]
        if not self._torch.is_tensor(tensor):
            raise HuggingFaceMLPProofError("down projection input is not a tensor")
        if tensor.ndim < 1 or self._channels[-1] >= int(tensor.shape[-1]):
            raise HuggingFaceMLPProofError("MLP channel index exceeds down-projection input width")
        masked = tensor.clone()
        for channel in self._channels:
            masked[..., channel] = 0
        return (masked, *inputs[1:])

    def __enter__(self) -> Self:
        if self._handle is not None:
            raise HuggingFaceMLPProofError("MLP channel mask set is already active")
        self._handle = self._module.register_forward_pre_hook(self._hook)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def _architecture_evidence(config: object) -> ArchitectureEvidence:
    model_type = getattr(config, "model_type", None)
    if not isinstance(model_type, str):
        model_type = None
    architectures = getattr(config, "architectures", ())
    if not isinstance(architectures, (list, tuple)):
        architectures = ()
    return ArchitectureEvidence(
        model_type=model_type,
        architecture_names=tuple(item for item in architectures if isinstance(item, str)),
    )


def _tool_revision(explicit: str | None) -> str:
    if explicit is not None:
        if not explicit.strip():
            raise HuggingFaceMLPProofError("tool_revision cannot be blank")
        return explicit
    environment = os.environ.get("MODELSURGEON_TOOL_REVISION")
    if environment:
        return environment
    try:
        return f"modelsurgeon-{package_version('modelsurgeon')}"
    except PackageNotFoundError:
        return "modelsurgeon-source"


def _read_calibration_text(path: Path) -> tuple[str, str]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise HuggingFaceMLPProofError(f"cannot read calibration text: {error}") from error
    if not payload:
        raise HuggingFaceMLPProofError("calibration text cannot be empty")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise HuggingFaceMLPProofError("calibration text must be UTF-8") from error
    return text, hashlib.sha256(payload).hexdigest()


def _token_chunks(
    token_ids: object,
    *,
    sequence_length: int,
    max_tokens: int,
) -> tuple[tuple[int, ...], ...]:
    if not isinstance(token_ids, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in token_ids
    ):
        raise HuggingFaceMLPProofError("tokenizer returned unsupported input_ids")
    bounded = token_ids[:max_tokens]
    return _nontrivial_chunks(bounded, sequence_length)


def _nontrivial_chunks(
    token_ids: list[int],
    sequence_length: int,
) -> tuple[tuple[int, ...], ...]:
    chunks = tuple(
        tuple(token_ids[start : start + sequence_length])
        for start in range(0, len(token_ids), sequence_length)
        if len(token_ids[start : start + sequence_length]) >= 2
    )
    if not chunks:
        raise HuggingFaceMLPProofError("calibration text produced fewer than two usable tokens")
    return chunks


def _sample_ids(chunks: tuple[tuple[int, ...], ...]) -> tuple[str, ...]:
    result: list[str] = []
    for index, chunk in enumerate(chunks):
        payload = canonical_identity_json({"index": index, "token_ids": chunk}).encode("utf-8")
        result.append(f"sample_{hashlib.sha256(payload).hexdigest()}")
    return tuple(result)


def _tokenizer_revision(
    tokenizer: Any,
    *,
    source: str,
    requested: str | None,
    model_source: str,
    model_revision: str,
) -> str:
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, Mapping):
        commit = init_kwargs.get("_commit_hash")
        if isinstance(commit, str) and commit.strip():
            return commit
    if requested is not None:
        return requested
    if source == model_source:
        return model_revision
    local = Path(source)
    if local.exists():
        return str(local.resolve())
    raise HuggingFaceMLPProofError(
        "tokenizer revision could not be resolved; pin --tokenizer-revision explicitly"
    )


def _find_module(modules: Mapping[str, Any], suffix: str) -> Any:
    exact = modules.get(suffix)
    if exact is not None:
        return exact
    matches = [module for name, module in modules.items() if name.endswith(f".{suffix}")]
    if len(matches) != 1:
        raise HuggingFaceMLPProofError(
            f"expected exactly one Hugging Face module matching {suffix!r}, found {len(matches)}"
        )
    return matches[0]


def _channel_coordinates(request: MutationRequest) -> tuple[int, int]:
    if request.kind is not MutationKind.MASK or len(request.targets) != 1:
        raise HuggingFaceMLPProofError("HF proof runtime supports single-target masks only")
    parameters = dict(request.parameters)
    if parameters.get("candidate_scope") != CandidateScope.MLP_CHANNEL.value:
        raise HuggingFaceMLPProofError("HF proof runtime supports MLP-channel candidates only")
    layer = parameters.get("layer_index")
    channel = parameters.get("channel_index")
    if (
        not isinstance(layer, int)
        or isinstance(layer, bool)
        or layer < 0
        or not isinstance(channel, int)
        or isinstance(channel, bool)
        or channel < 0
    ):
        raise HuggingFaceMLPProofError("MLP-channel request is missing valid layer/channel indices")
    expected = ComponentId.parse(f"model.layers.{layer}.mlp.channel.{channel}")
    if request.targets != (expected,):
        raise HuggingFaceMLPProofError(
            "MLP-channel request target disagrees with its layer/channel metadata"
        )
    return layer, channel


def _tensor_dtype(tensor: Any) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def _scalar(tensor: Any) -> float:
    value = float(tensor.detach().cpu().item())
    if not math.isfinite(value):
        raise HuggingFaceMLPProofError("feature/evaluation computation produced a non-finite value")
    return value


def _high_precision(storage_dtype: str) -> PrecisionProvenance:
    return PrecisionProvenance(
        PrecisionSource.HIGH_PRECISION,
        storage_dtype=storage_dtype,
        compute_dtype="float32",
    )


def _metric(name: str, value: float, unit: str) -> MetricObservation:
    return MetricObservation(name, MetricState.MEASURED, value, unit)


class HuggingFaceMLPProofRuntime:
    """Real causal-LM MLP-channel masking, activation capture, and perplexity evaluation."""

    def __init__(self, config: HuggingFaceMLPProofConfig) -> None:
        self.config = config
        self._torch = self._load_torch()
        loaded = load_causal_lm(
            HuggingFaceLoadRequest(
                model=config.model,
                revision=config.revision,
                trust_remote_code=config.trust_remote_code,
                device_map=config.device_map,
                dtype=config.dtype,
                local_files_only=config.local_files_only,
            )
        )
        self.model = loaded.model
        self.model.eval()
        family = detect_model_family(_architecture_evidence(self.model.config))
        self._discovery = discover_huggingface_components(self.model, family.family)
        graph_records = cast(
            tuple[ComponentRecordLike, ...],
            tuple(self._discovery.components()),
        )
        self._component_graph = build_component_graph(graph_records)
        self._target_resolver = MutationTargetResolver(self._component_graph)
        self._modules = dict(self.model.named_modules())
        self._validate_mlp_layout()

        tokenizer_source = config.tokenizer or config.model
        requested_tokenizer_revision = config.tokenizer_revision
        if requested_tokenizer_revision is None and tokenizer_source == config.model:
            requested_tokenizer_revision = loaded.provenance.resolved_revision
        self._tokenizer = self._load_tokenizer(tokenizer_source, requested_tokenizer_revision)
        resolved_tokenizer_revision = _tokenizer_revision(
            self._tokenizer,
            source=tokenizer_source,
            requested=requested_tokenizer_revision,
            model_source=config.model,
            model_revision=loaded.provenance.resolved_revision,
        )
        text, text_revision = _read_calibration_text(config.calibration_text)
        encoded = self._tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise HuggingFaceMLPProofError("tokenizer did not return an input_ids mapping")
        self._chunks = _token_chunks(
            encoded["input_ids"],
            sequence_length=config.sequence_length,
            max_tokens=config.max_tokens,
        )
        self._sample_ids = _sample_ids(self._chunks)
        self._dataset = self._dataset_target(
            text_revision,
            tokenizer_source,
            resolved_tokenizer_revision,
        )
        self._model_target = ModelTarget(
            identifier=config.model,
            revision=loaded.provenance.resolved_revision,
            family=family.family.value,
            format="huggingface",
            parameter_count=self._discovery.parameter_count,
            quantization=None,
        )
        self._tool_revision = _tool_revision(config.tool_revision)
        identity = derive_experiment_identity(
            ExperimentIdentitySpec(
                model=self._model_target,
                dataset=self._dataset,
                resolved_config=self._identity_config(),
                seeds=SeedContext(config.seed, config.seed, config.seed),
                tool_revision=self._tool_revision,
                evaluator_version=HF_MLP_EVALUATOR_VERSION,
                feature_schema_version=FEATURE_SCHEMA_VERSION,
                mutation_record_schema_version=MUTATION_RECORD_SCHEMA_VERSION,
            )
        )
        self._experiment_id = identity.experiment_id
        self._config_digest = identity.config_digest
        self._run_id = derive_run_identity(identity.experiment_id, "hf-mlp-channel-proof").run_id
        self._hardware = collect_hardware_inventory(config.calibration_text.parent)
        self._input_device = self._resolve_input_device()
        self._baseline: _PerplexityMeasurement | None = None
        self._activation_stats: dict[int, _ActivationAccumulator] = {}
        self._weight_stats: dict[int, _LayerWeightStatistics] = {}
        self._last_measurement: _CandidateMeasurement | None = None

    @staticmethod
    def _load_torch() -> Any:
        try:
            return import_module("torch")
        except ImportError as error:
            raise HuggingFaceDependencyError(
                "Hugging Face proof runtime requires `uv sync --extra hf`"
            ) from error

    def _load_tokenizer(self, source: str, revision: str | None) -> Any:
        try:
            auto_tokenizer = import_module("transformers").AutoTokenizer
        except (AttributeError, ImportError) as error:
            raise HuggingFaceDependencyError(
                "Hugging Face proof runtime requires `uv sync --extra hf`"
            ) from error
        try:
            return auto_tokenizer.from_pretrained(
                source,
                revision=revision,
                trust_remote_code=self.config.trust_remote_code,
                local_files_only=self.config.local_files_only,
            )
        except Exception as error:
            raise HuggingFaceMLPProofError(
                f"failed to load tokenizer {source!r}: {error}"
            ) from error

    def _identity_config(self) -> dict[str, object]:
        return {
            "runtime_version": HF_MLP_PROOF_RUNTIME_VERSION,
            "sequence_length": self.config.sequence_length,
            "max_tokens": self.config.max_tokens,
            "safe_perplexity_delta": self.config.safe_perplexity_delta,
            "device_map": self.config.device_map,
            "dtype": self.config.dtype.value,
            "trust_remote_code": self.config.trust_remote_code,
        }

    def _dataset_target(
        self,
        text_revision: str,
        tokenizer_source: str,
        tokenizer_revision: str,
    ) -> DatasetTarget:
        manifest_payload = {
            "calibration_revision": text_revision,
            "tokenizer": tokenizer_source,
            "tokenizer_revision": tokenizer_revision,
            "sequence_length": self.config.sequence_length,
            "max_tokens": self.config.max_tokens,
            "samples": self._sample_ids,
        }
        digest = hashlib.sha256(
            canonical_identity_json(manifest_payload).encode("utf-8")
        ).hexdigest()
        return DatasetTarget(
            identifier=f"local-text:{self.config.calibration_text.name}",
            revision=text_revision,
            split="calibration",
            manifest_id=f"manifest_{digest}",
            tokenizer=tokenizer_source,
            tokenizer_revision=tokenizer_revision,
        )

    @property
    def component_graph(self) -> ComponentGraph:
        return self._component_graph

    @property
    def run_id(self) -> str:
        return self._run_id

    def _validate_mlp_layout(self) -> None:
        width = self._discovery.shape.intermediate_size
        for layer in range(self._discovery.shape.layers):
            gate, up, down = self._projection_weights(layer)
            for label, weight in (("gate_proj", gate), ("up_proj", up), ("down_proj", down)):
                if not self._torch.is_tensor(weight) or weight.ndim != 2:
                    raise HuggingFaceMLPProofError(
                        f"layer {layer} {label} must expose a rank-2 tensor weight"
                    )
            if int(gate.shape[0]) != width or int(up.shape[0]) != width:
                raise HuggingFaceMLPProofError(
                    f"layer {layer} gate/up output width disagrees with intermediate_size"
                )
            if int(down.shape[1]) != width:
                raise HuggingFaceMLPProofError(
                    f"layer {layer} down-projection input width disagrees with intermediate_size"
                )

    def _resolve_input_device(self) -> Any:
        embeddings = self.model.get_input_embeddings()
        weight = getattr(embeddings, "weight", None)
        if weight is not None and hasattr(weight, "device"):
            return weight.device
        try:
            return next(iter(self.model.parameters())).device
        except StopIteration as error:
            raise HuggingFaceMLPProofError("loaded model exposes no parameter device") from error

    def _down_projection(self, layer: int) -> Any:
        return _find_module(self._modules, f"model.layers.{layer}.mlp.down_proj")

    def _projection_weights(self, layer: int) -> tuple[Any, Any, Any]:
        gate = _find_module(self._modules, f"model.layers.{layer}.mlp.gate_proj")
        up = _find_module(self._modules, f"model.layers.{layer}.mlp.up_proj")
        down = self._down_projection(layer)
        for label, module in (("gate_proj", gate), ("up_proj", up), ("down_proj", down)):
            weight = getattr(module, "weight", None)
            if weight is None:
                raise HuggingFaceMLPProofError(f"{label} does not expose a weight tensor")
        return gate.weight, up.weight, down.weight

    def _activation_hook(self, layer: int) -> Any:
        def capture(module: object, inputs: tuple[object, ...]) -> None:
            del module
            if not inputs or not self._torch.is_tensor(inputs[0]):
                raise HuggingFaceMLPProofError("down projection activation input is not a tensor")
            tensor: Any = inputs[0]
            if tensor.ndim < 1:
                raise HuggingFaceMLPProofError("down projection activation has no channel axis")
            flat = tensor.detach().float().reshape(-1, int(tensor.shape[-1]))
            if int(flat.shape[-1]) != self._discovery.shape.intermediate_size:
                raise HuggingFaceMLPProofError(
                    "captured activation width disagrees with model config"
                )
            state = self._activation_stats.setdefault(layer, _ActivationAccumulator())
            sums = flat.sum(dim=0).cpu()
            abs_sums = flat.abs().sum(dim=0).cpu()
            square_sums = flat.square().sum(dim=0).cpu()
            zero_counts = (flat == 0).sum(dim=0).cpu()
            maximum = flat.abs().amax(dim=0).cpu()
            if state.sums is None:
                state.sums = sums
                state.abs_sums = abs_sums
                state.square_sums = square_sums
                state.zero_counts = zero_counts
                state.max_abs = maximum
                state.storage_dtype = _tensor_dtype(tensor)
            else:
                state.sums += sums
                state.abs_sums += abs_sums
                state.square_sums += square_sums
                state.zero_counts += zero_counts
                state.max_abs = self._torch.maximum(state.max_abs, maximum)
            state.count += int(flat.shape[0])

        return capture

    def _forward_measurement(self, *, capture_activations: bool = False) -> _PerplexityMeasurement:
        handles: list[Any] = []
        if capture_activations:
            self._activation_stats.clear()
            for layer in range(self._discovery.shape.layers):
                handles.append(
                    self._down_projection(layer).register_forward_pre_hook(
                        self._activation_hook(layer)
                    )
                )
        total_nll = 0.0
        token_count = 0
        try:
            with self._torch.inference_mode():
                for chunk in self._chunks:
                    input_ids = self._torch.tensor(
                        [chunk],
                        dtype=self._torch.long,
                        device=self._input_device,
                    )
                    output = self.model(input_ids=input_ids, use_cache=False)
                    logits = getattr(output, "logits", None)
                    if logits is None or not self._torch.is_tensor(logits) or logits.ndim != 3:
                        raise HuggingFaceMLPProofError(
                            "causal LM forward did not return rank-3 logits"
                        )
                    shifted_logits = logits[:, :-1, :].float().contiguous()
                    shifted_targets = input_ids[:, 1:].to(shifted_logits.device).contiguous()
                    nll = self._torch.nn.functional.cross_entropy(
                        shifted_logits.view(-1, int(shifted_logits.shape[-1])),
                        shifted_targets.view(-1),
                        reduction="sum",
                    )
                    total_nll += _scalar(nll)
                    token_count += int(shifted_targets.numel())
        finally:
            while handles:
                handles.pop().remove()
        if token_count <= 0:
            raise HuggingFaceMLPProofError("calibration forward produced no shifted target tokens")
        mean_loss = total_nll / token_count
        try:
            perplexity = math.exp(mean_loss)
        except OverflowError as error:
            raise HuggingFaceMLPProofError("perplexity overflowed finite range") from error
        return _PerplexityMeasurement(mean_loss, perplexity, token_count)

    def _ensure_baseline(self) -> _PerplexityMeasurement:
        if self._baseline is None:
            self._baseline = self._forward_measurement(capture_activations=True)
            if len(self._activation_stats) != self._discovery.shape.layers:
                raise HuggingFaceMLPProofError(
                    "baseline activation capture missed one or more layers"
                )
        return self._baseline

    def _feature_sample_context(self) -> FeatureSampleContext:
        return FeatureSampleContext(
            dataset=self._dataset.identifier,
            revision=self._dataset.revision,
            split=self._dataset.split,
            sample_ids=self._sample_ids,
            preprocessing_version="hf_text_chunk_v1",
            tokenizer=self._dataset.tokenizer,
            tokenizer_revision=self._dataset.tokenizer_revision,
        )

    def _feature_records(self, layer: int, channel: int) -> tuple[FeatureRecord, ...]:
        self._ensure_baseline()
        component = ComponentId.parse(f"model.layers.{layer}.mlp.channel.{channel}")
        metadata: tuple[tuple[str, str | int], ...] = (
            ("channel_index", channel),
            ("layer_index", layer),
            ("source_phase", "pre_mutation"),
        )
        records = self._weight_features(component, layer, channel, metadata)
        records.extend(self._activation_features(component, layer, channel, metadata))
        return tuple(sorted(records, key=lambda item: item.name))

    def measure_feature_collection_costs(self) -> FeatureCollectionCosts:
        """Measure static summaries and activation capture after model load."""

        cuda = None
        if str(self._input_device).startswith("cuda"):
            try:
                cuda = TorchCudaMemoryProvider(self._input_device)
            except MemoryTelemetryError:
                cuda = None
        telemetry_config = MemoryTelemetryConfig(
            sampling_enabled=True,
            sample_interval_seconds=0.005,
            max_samples=4096,
        )

        self._weight_stats.clear()

        def collect_static() -> None:
            for layer in range(self._discovery.shape.layers):
                self._layer_weight_statistics(layer)
            if cuda is not None:
                self._torch.cuda.synchronize(self._input_device)

        static_report = collect_memory_telemetry(
            "hf-static-feature-collection",
            collect_static,
            telemetry_config,
            cuda=cuda,
        )
        self._weight_stats.clear()
        gc.collect()

        # Warm lazy kernels/allocators before the paired no-hook/hook forwards.
        self._forward_measurement(capture_activations=False)
        gc.collect()

        def collect_forward_baseline() -> None:
            self._forward_measurement(capture_activations=False)
            if cuda is not None:
                self._torch.cuda.synchronize(self._input_device)

        baseline_report = collect_memory_telemetry(
            "hf-forward-feature-baseline",
            collect_forward_baseline,
            telemetry_config,
            cuda=cuda,
        )
        gc.collect()

        def collect_activation() -> None:
            self._forward_measurement(capture_activations=True)
            if cuda is not None:
                self._torch.cuda.synchronize(self._input_device)

        activation_report = collect_memory_telemetry(
            "hf-activation-feature-collection",
            collect_activation,
            telemetry_config,
            cuda=cuda,
        )
        return FeatureCollectionCosts(
            _feature_collection_cost(static_report),
            _feature_collection_cost(baseline_report),
            _feature_collection_cost(activation_report),
        )

    def collect_mlp_channel_gradient_features(
        self,
        coordinates: Sequence[tuple[ComponentId, int, int]],
    ) -> MLPChannelGradientCollection:
        """Collect one calibration-loss backward and summarize requested channels."""

        if not coordinates or len(coordinates) != len({item[0] for item in coordinates}):
            raise HuggingFaceMLPProofError(
                "gradient coordinates must be non-empty with unique components"
            )
        by_layer: dict[int, list[tuple[ComponentId, int]]] = {}
        for component, layer, channel in coordinates:
            expected = ComponentId.parse(f"model.layers.{layer}.mlp.channel.{channel}")
            if component != expected or not 0 <= layer < self._discovery.shape.layers:
                raise HuggingFaceMLPProofError("gradient coordinate identity is invalid")
            if not 0 <= channel < self._discovery.shape.intermediate_size:
                raise HuggingFaceMLPProofError("gradient channel exceeds intermediate width")
            by_layer.setdefault(layer, []).append((component, channel))

        requested_parameters: list[Any] = []
        for layer in sorted(by_layer):
            requested_parameters.extend(self._projection_weights(layer))
        original_requires_grad = {
            id(parameter): bool(parameter.requires_grad) for parameter in self.model.parameters()
        }
        requested_parameter_ids = {id(item) for item in requested_parameters}
        for parameter in self.model.parameters():
            parameter.requires_grad_(id(parameter) in requested_parameter_ids)

        holder: list[tuple[tuple[ComponentId, tuple[FeatureRecord, ...]], ...]] = []
        token_holder: list[int] = []
        cuda = None
        if str(self._input_device).startswith("cuda"):
            try:
                cuda = TorchCudaMemoryProvider(self._input_device)
            except MemoryTelemetryError:
                cuda = None

        def operation() -> None:
            self.model.zero_grad(set_to_none=True)
            total_tokens = 0
            total_loss: Any = None
            for chunk in self._chunks:
                input_ids = self._torch.tensor(
                    [chunk], dtype=self._torch.long, device=self._input_device
                )
                output = self.model(input_ids=input_ids, use_cache=False)
                logits = output.logits[:, :-1, :].float().contiguous()
                targets = input_ids[:, 1:].to(logits.device).contiguous()
                loss = self._torch.nn.functional.cross_entropy(
                    logits.view(-1, int(logits.shape[-1])),
                    targets.view(-1),
                    reduction="mean",
                )
                total_loss = loss if total_loss is None else total_loss + loss
                total_tokens += int(targets.numel())
            if total_loss is None or total_tokens <= 0:
                raise HuggingFaceMLPProofError("gradient calibration produced no loss")
            total_loss.backward()

            output_records: list[tuple[ComponentId, tuple[FeatureRecord, ...]]] = []
            context = self._feature_sample_context()
            for layer, layer_coordinates in sorted(by_layer.items()):
                weights = self._projection_weights(layer)
                oriented = (
                    (weights[0], weights[0].grad),
                    (weights[1], weights[1].grad),
                    (
                        weights[2].transpose(0, 1),
                        None if weights[2].grad is None else weights[2].grad.transpose(0, 1),
                    ),
                )
                gradient_l1 = None
                gradient_square = None
                gradient_maximum = None
                product_sum = None
                product_abs_sum = None
                product_square = None
                storage_dtypes: set[str] = set()
                source_devices: set[str] = set()
                element_count = 0
                for weight, gradient in oriented:
                    if gradient is None:
                        raise HuggingFaceMLPProofError(
                            f"gradient missing for MLP projection in layer {layer}"
                        )
                    values = weight.detach().float()
                    gradients = gradient.detach().float()
                    products = values * gradients
                    part_l1 = gradients.abs().sum(dim=1)
                    part_square = gradients.square().sum(dim=1)
                    part_maximum = gradients.abs().amax(dim=1)
                    part_product_sum = products.sum(dim=1)
                    part_product_abs_sum = products.abs().sum(dim=1)
                    part_product_square = products.square().sum(dim=1)
                    gradient_l1 = part_l1 if gradient_l1 is None else gradient_l1 + part_l1
                    gradient_square = (
                        part_square if gradient_square is None else gradient_square + part_square
                    )
                    gradient_maximum = (
                        part_maximum
                        if gradient_maximum is None
                        else self._torch.maximum(gradient_maximum, part_maximum)
                    )
                    product_sum = (
                        part_product_sum if product_sum is None else product_sum + part_product_sum
                    )
                    product_abs_sum = (
                        part_product_abs_sum
                        if product_abs_sum is None
                        else product_abs_sum + part_product_abs_sum
                    )
                    product_square = (
                        part_product_square
                        if product_square is None
                        else product_square + part_product_square
                    )
                    storage_dtypes.add(_tensor_dtype(gradient))
                    source_devices.add(str(gradient.device))
                    element_count += int(values.shape[1])
                columns = self._torch.stack(
                    (
                        gradient_l1,
                        self._torch.sqrt(self._torch.clamp(gradient_square, min=0.0)),
                        gradient_maximum,
                        product_sum,
                        product_abs_sum,
                        self._torch.sqrt(self._torch.clamp(product_square, min=0.0)),
                    ),
                    dim=1,
                ).cpu()
                for component, channel in layer_coordinates:
                    values = columns[channel]
                    removal = -_scalar(values[3])
                    features = GradientFeatures(
                        component,
                        1,
                        element_count,
                        _tensor_dtype(weights[0]),
                        str(weights[0].device),
                        tuple(sorted(storage_dtypes)),
                        tuple(sorted(source_devices)),
                        _scalar(values[0]),
                        _scalar(values[1]),
                        _scalar(values[2]),
                        _scalar(values[3]),
                        _scalar(values[4]),
                        _scalar(values[5]),
                        removal,
                        abs(removal),
                    ).feature_records(sample_context=context)
                    output_records.append((component, features))
            holder.append(tuple(sorted(output_records, key=lambda item: item[0])))
            token_holder.append(total_tokens)
            if cuda is not None:
                self._torch.cuda.synchronize(self._input_device)

        try:
            report = collect_memory_telemetry(
                "hf-mlp-channel-gradient-collection",
                operation,
                MemoryTelemetryConfig(
                    sampling_enabled=True,
                    sample_interval_seconds=0.005,
                    max_samples=4096,
                ),
                cuda=cuda,
            )
        finally:
            self.model.zero_grad(set_to_none=True)
            for parameter in self.model.parameters():
                parameter.requires_grad_(original_requires_grad[id(parameter)])
        if len(holder) != 1 or len(token_holder) != 1:
            raise HuggingFaceMLPProofError("gradient collection did not complete")
        return MLPChannelGradientCollection(
            holder[0], _feature_collection_cost(report), token_holder[0]
        )

    def _layer_weight_statistics(self, layer: int) -> _LayerWeightStatistics:
        cached = self._weight_stats.get(layer)
        if cached is not None:
            return cached

        gate, up, down = self._projection_weights(layer)
        channel_parts = (
            gate.detach().float(),
            up.detach().float(),
            down.detach().float().transpose(0, 1),
        )
        dtypes = (_tensor_dtype(gate), _tensor_dtype(up), _tensor_dtype(down))
        part_statistics: list[tuple[Any, Any, Any, Any, Any]] = []
        l1_parts: list[Any] = []
        l2_parts: list[Any] = []
        max_parts: list[Any] = []
        for values in channel_parts:
            abs_values = values.abs()
            square_values = values.square()
            mean = values.mean(dim=1)
            square_mean = square_values.mean(dim=1)
            part_statistics.append(
                (
                    mean,
                    abs_values.mean(dim=1),
                    self._torch.sqrt(self._torch.clamp(square_mean, min=0.0)),
                    values.std(dim=1, unbiased=False),
                    abs_values.amax(dim=1),
                )
            )
            l1_parts.append(abs_values.sum(dim=1))
            l2_parts.append(square_values.sum(dim=1))
            max_parts.append(abs_values.amax(dim=1))

        l1 = self._torch.stack(l1_parts).sum(dim=0)
        l2 = self._torch.sqrt(self._torch.clamp(self._torch.stack(l2_parts).sum(dim=0), min=0.0))
        maximum = self._torch.stack(max_parts).amax(dim=0)
        columns = [l1, l2, maximum]
        columns.extend(value for part in part_statistics for value in part)
        resolved = _LayerWeightStatistics(
            sum(int(values.shape[1]) for values in channel_parts),
            dtypes[0] if len(set(dtypes)) == 1 else "mixed",
            dtypes,
            self._torch.stack(columns, dim=1).cpu(),
        )
        self._weight_stats[layer] = resolved
        return resolved

    def _weight_features(
        self,
        component: ComponentId,
        layer: int,
        channel: int,
        metadata: tuple[tuple[str, str | int], ...],
    ) -> list[FeatureRecord]:
        statistics = self._layer_weight_statistics(layer)
        values = statistics.values[channel]
        records = [
            self._feature(
                component,
                name,
                value,
                statistics.combined_dtype,
                None,
                metadata,
            )
            for name, value in (
                ("weight_count", float(statistics.count)),
                ("weight_l1_norm", _scalar(values[0])),
                ("weight_l2_norm", _scalar(values[1])),
                ("weight_max_magnitude", _scalar(values[2])),
            )
        ]
        for part_index, (prefix, storage_dtype) in enumerate(
            zip(
                ("gate_weight", "up_weight", "down_weight"),
                statistics.part_dtypes,
                strict=True,
            )
        ):
            offset = 3 + part_index * 5
            part_values = (
                ("mean", _scalar(values[offset])),
                ("abs_mean", _scalar(values[offset + 1])),
                ("rms", _scalar(values[offset + 2])),
                ("std", _scalar(values[offset + 3])),
                ("max_abs", _scalar(values[offset + 4])),
            )
            records.extend(
                self._feature(
                    component,
                    f"{prefix}_{suffix}",
                    value,
                    storage_dtype,
                    None,
                    metadata,
                )
                for suffix, value in part_values
            )
        return records

    def _activation_features(
        self,
        component: ComponentId,
        layer: int,
        channel: int,
        metadata: tuple[tuple[str, str | int], ...],
    ) -> list[FeatureRecord]:
        state = self._activation_stats.get(layer)
        if state is None or state.count <= 0 or state.sums is None:
            raise HuggingFaceMLPProofError(f"activation statistics are missing for layer {layer}")
        mean = _scalar(state.sums[channel]) / state.count
        square_mean = _scalar(state.square_sums[channel]) / state.count
        values = (
            ("activation_mean", mean),
            ("activation_abs_mean", _scalar(state.abs_sums[channel]) / state.count),
            ("activation_rms", math.sqrt(max(0.0, square_mean))),
            ("activation_std", math.sqrt(max(0.0, square_mean - mean * mean))),
            ("activation_zero_fraction", _scalar(state.zero_counts[channel]) / state.count),
            ("activation_max_abs", _scalar(state.max_abs[channel])),
        )
        context = self._feature_sample_context()
        return [
            self._feature(
                component,
                name,
                value,
                state.storage_dtype or "unknown",
                context,
                metadata,
            )
            for name, value in values
        ]

    @staticmethod
    def _feature(
        component: ComponentId,
        name: str,
        value: float,
        storage_dtype: str,
        context: FeatureSampleContext | None,
        metadata: tuple[tuple[str, str | int], ...],
    ) -> FeatureRecord:
        return FeatureRecord(
            component,
            name,
            FeatureKind.SCALAR,
            float(value),
            "float64",
            HF_MLP_FEATURE_EXTRACTOR,
            HF_MLP_FEATURE_EXTRACTOR_VERSION,
            _high_precision(storage_dtype),
            context,
            metadata,
        )

    def pre_mutation_feature_partitions(
        self,
        candidate: MutationCandidate,
    ) -> tuple[FeaturePartition, ...]:
        if candidate.scope is not CandidateScope.MLP_CHANNEL:
            raise HuggingFaceMLPProofError("HF proof runtime only accepts MLP-channel candidates")
        layer, channel = _channel_coordinates(candidate.request)
        records = self._feature_records(layer, channel)
        encoded = canonical_identity_json([item.to_record() for item in records]).encode("utf-8")
        checksum = hashlib.sha256(encoded).hexdigest()
        key = FeaturePartitionKey(
            model_revision=self._model_target.revision,
            input_revision=self._dataset.manifest_id,
            component_id=candidate.component_id,
            extractor=HF_MLP_FEATURE_EXTRACTOR,
            extractor_version=HF_MLP_FEATURE_EXTRACTOR_VERSION,
        )
        return (FeaturePartition(key, records, checksum),)

    def resolve(self, request: MutationRequest) -> ResolvedExperiment:
        _channel_coordinates(request)
        resolution = self._target_resolver.resolve(request)
        plan = resolution.to_plan(preconditions=(), expected_delta=MutationDelta())
        return ResolvedExperiment(
            plan,
            MutationProvenance(
                self._model_target.revision,
                self._tool_revision,
                self.config.model if Path(self.config.model).exists() else None,
            ),
        )

    def transaction(
        self,
        plan: MutationPlan,
    ) -> AbstractContextManager[MutationTransaction]:
        _channel_coordinates(plan.request)
        return InMemoryMutationTransaction(
            self.model,
            {"hook": _NoopSnapshotTarget()},
            ("hook",),
        )

    def mutation_scope(
        self,
        plan: MutationPlan,
        transaction: MutationTransaction,
    ) -> AbstractContextManager[object]:
        require_safe_transaction(transaction)
        layer, channel = _channel_coordinates(plan.request)
        return _DownProjectionChannelMask(self._down_projection(layer), channel, self._torch)

    def evaluate(self, plan: MutationPlan) -> TieredEvaluationReport:
        baseline = self._ensure_baseline()
        started = time.perf_counter()
        post = self._forward_measurement()
        elapsed = time.perf_counter() - started
        loss_delta = post.mean_loss - baseline.mean_loss
        perplexity_delta = post.perplexity - baseline.perplexity
        accepted = perplexity_delta <= self.config.safe_perplexity_delta
        self._last_measurement = _CandidateMeasurement(
            plan.request.mutation_id,
            post,
            loss_delta,
            perplexity_delta,
            accepted,
            elapsed,
        )
        return self._evaluation_report(post, loss_delta, perplexity_delta, accepted)

    def measure_channel_set(
        self,
        coordinates: Sequence[tuple[int, int]],
        *,
        warmup: int = 1,
        repetitions: int = 7,
    ) -> MLPChannelSetMeasurement:
        """Measure an actual cumulative mask against the unchanged model baseline."""

        canonical = tuple(sorted(set(coordinates)))
        if not canonical or len(canonical) != len(coordinates) or warmup < 0 or repetitions <= 0:
            raise HuggingFaceMLPProofError("cumulative mask measurement inputs are invalid")
        for layer, channel in canonical:
            if not 0 <= layer < self._discovery.shape.layers or not (
                0 <= channel < self._discovery.shape.intermediate_size
            ):
                raise HuggingFaceMLPProofError("cumulative mask coordinate is out of range")
        baseline = self._ensure_baseline()
        started = time.perf_counter()

        def measured_forwards() -> tuple[_PerplexityMeasurement, tuple[float, ...]]:
            for _ in range(warmup):
                self._forward_measurement()
            measurements: list[_PerplexityMeasurement] = []
            timings: list[float] = []
            for _ in range(repetitions):
                forward_started = time.perf_counter()
                measurements.append(self._forward_measurement())
                timings.append(time.perf_counter() - forward_started)
            return measurements[-1], tuple(timings)

        _, baseline_timings = measured_forwards()
        by_layer: dict[int, list[int]] = {}
        for layer, channel in canonical:
            by_layer.setdefault(layer, []).append(channel)
        with ExitStack() as stack:
            for layer, channels in sorted(by_layer.items()):
                stack.enter_context(
                    _DownProjectionChannelMaskSet(
                        self._down_projection(layer), tuple(sorted(channels)), self._torch
                    )
                )
            masked, masked_timings = measured_forwards()
        baseline_latency = statistics.median(baseline_timings)
        masked_latency = statistics.median(masked_timings)
        return MLPChannelSetMeasurement(
            canonical,
            baseline.perplexity,
            masked.perplexity,
            masked.perplexity - baseline.perplexity,
            baseline_latency,
            masked_latency,
            masked_latency - baseline_latency,
            time.perf_counter() - started,
            repetitions,
            masked.token_count,
        )

    def _evaluation_report(
        self,
        post: _PerplexityMeasurement,
        loss_delta: float,
        perplexity_delta: float,
        accepted: bool,
    ) -> TieredEvaluationReport:
        threshold = TierThreshold(
            EvaluationTier.TIER1,
            "perplexity_delta",
            ThresholdComparator.MAXIMUM,
            self.config.safe_perplexity_delta,
        )
        tier0_metrics = (
            MetricDecision(EvaluationTier.TIER0, "load_shape_forward_pass", 1.0, None, None),
            MetricDecision(EvaluationTier.TIER0, "numerics_pass", 1.0, None, None),
        )
        tier1_metrics = (
            MetricDecision(EvaluationTier.TIER1, "mean_loss", post.mean_loss, None, None),
            MetricDecision(EvaluationTier.TIER1, "perplexity", post.perplexity, None, None),
            MetricDecision(EvaluationTier.TIER1, "loss_delta", loss_delta, None, None),
            MetricDecision(
                EvaluationTier.TIER1,
                "perplexity_delta",
                perplexity_delta,
                threshold,
                accepted,
            ),
        )
        reason = None
        if not accepted:
            reason = (
                f"perplexity_delta={perplexity_delta:.12g} exceeds maximum "
                f"{self.config.safe_perplexity_delta:.12g}"
            )
        decisions = [
            TierDecision(
                EvaluationTier.TIER0,
                True,
                True,
                EscalationAction.ESCALATE,
                None,
                tier0_metrics,
            ),
            TierDecision(
                EvaluationTier.TIER1,
                True,
                accepted,
                EscalationAction.COMPLETE if accepted else EscalationAction.REJECT,
                reason,
                tier1_metrics,
            ),
        ]
        tail_reason = "tier not configured" if accepted else "candidate rejected by Tier 1"
        decisions.extend(
            TierDecision(tier, False, None, EscalationAction.SKIP, tail_reason, ())
            for tier in (EvaluationTier.TIER2, EvaluationTier.TIER3)
        )
        return TieredEvaluationReport(tuple(decisions), accepted, EvaluationTier.TIER1)

    def rolled_back_outcome(
        self,
        plan: MutationPlan,
        evaluation: TieredEvaluationReport,
    ) -> MutationOutcome:
        del evaluation
        _channel_coordinates(plan.request)
        return MutationOutcome(
            MutationOutcomeStatus.ROLLED_BACK,
            MutationDelta(),
            (),
            "temporary down-projection input mask removed after evaluation",
        )

    def experiment_record(
        self,
        candidate: MutationCandidate,
        result: SingleMutationExperimentResult,
    ) -> ExperimentRecord:
        measurement = self._last_measurement
        if measurement is None or measurement.mutation_id != candidate.mutation_id:
            raise HuggingFaceMLPProofError(
                "candidate experiment record requested without matching evaluation evidence"
            )
        if result.run_record.mutation_id != candidate.mutation_id:
            raise HuggingFaceMLPProofError(
                "experiment result mutation identity does not match candidate"
            )
        baseline = self._ensure_baseline()
        outcome = self._experiment_outcome(measurement)
        mutation_seed = int(candidate.mutation_id[:16], 16)
        return ExperimentRecord(
            run_id=self._run_id,
            experiment_id=self._experiment_id,
            attempt_id=candidate.candidate_id,
            model=self._model_target,
            dataset=self._dataset,
            components=result.run_record.plan.affected_components,
            mutation=result.run_record,
            baseline_metrics=(
                _metric("loss", baseline.mean_loss, "loss"),
                _metric("perplexity", baseline.perplexity, "perplexity"),
            ),
            post_metrics=(
                _metric("loss", measurement.post.mean_loss, "loss"),
                _metric("perplexity", measurement.post.perplexity, "perplexity"),
            ),
            delta_metrics=(
                _metric("loss", measurement.loss_delta, "loss"),
                _metric("perplexity", measurement.perplexity_delta, "perplexity"),
            ),
            outcome=outcome,
            hardware=self._hardware,
            versions=VersionContext(
                self._tool_revision,
                self._config_digest,
                HF_MLP_EVALUATOR_VERSION,
                FEATURE_SCHEMA_VERSION,
                MUTATION_RECORD_SCHEMA_VERSION,
            ),
            seeds=SeedContext(self.config.seed, self.config.seed, mutation_seed),
            timings=(
                StageTiming(
                    "evaluate",
                    measurement.wall_seconds,
                    tokens=measurement.post.token_count,
                    candidates=1,
                ),
            ),
        )

    def _experiment_outcome(self, measurement: _CandidateMeasurement) -> ExperimentOutcome:
        if measurement.accepted:
            return ExperimentOutcome(ExperimentOutcomeKind.SUCCEEDED)
        return ExperimentOutcome(
            ExperimentOutcomeKind.REJECTED,
            (
                f"perplexity delta {measurement.perplexity_delta:.12g} exceeds "
                f"{self.config.safe_perplexity_delta:.12g}"
            ),
        )
