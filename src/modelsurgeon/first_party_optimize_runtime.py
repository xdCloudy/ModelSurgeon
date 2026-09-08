"""First-party execution boundary for the verified Hugging Face optimize cell.

This module intentionally supports a bounded, real path first: local or pinned
Hugging Face causal LMs exposing adapter-defined gated-MLP, attention, layer, or
Linear projection layouts. It uses the existing proof runtime for measurement
and the existing physical surgery and reload contracts for publication. Other
formats and operations return an explicit unsupported outcome instead of
placeholder data.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import statistics
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

from modelsurgeon.adapters import ModelFormat
from modelsurgeon.adapters.huggingface.loader import (
    HuggingFaceDType,
    HuggingFaceLoadRequest,
    load_causal_lm,
)
from modelsurgeon.adapters.huggingface.physical_attention import (
    remove_huggingface_attention_heads,
)
from modelsurgeon.adapters.huggingface.physical_layers import (
    remove_huggingface_transformer_layers,
)
from modelsurgeon.adapters.huggingface.physical_mlp import (
    remove_huggingface_mlp_channels,
)
from modelsurgeon.adapters.huggingface.proof_runtime import (
    HuggingFaceMLPProofConfig,
    HuggingFaceMLPProofRuntime,
)
from modelsurgeon.config import ObjectiveConfig, OptimizeMetric
from modelsurgeon.experiments.candidates import (
    CandidateEnumeratorConfig,
    CandidateFilter,
    CandidateScope,
    MutationCandidate,
    enumerate_mutation_candidates,
)
from modelsurgeon.experiments.hardware import HardwareInventory, collect_hardware_inventory
from modelsurgeon.experiments.identity import derive_candidate_identity
from modelsurgeon.experiments.optimization_evidence import (
    OptimizationEvidenceOutcome,
    OptimizationEvidenceRecord,
    OptimizationEvidenceStore,
)
from modelsurgeon.features.cache import FeaturePartition, FeaturePartitionCache, FeaturePartitionKey
from modelsurgeon.features.schema import (
    FeatureKind,
    FeatureRecord,
    PrecisionProvenance,
    PrecisionSource,
)
from modelsurgeon.instrumentation.memory_telemetry import (
    MemoryTelemetryConfig,
    MemoryTelemetryError,
    TorchCudaMemoryProvider,
    collect_memory_telemetry,
)
from modelsurgeon.optimization import OptimizePlan
from modelsurgeon.optimization_orchestrator import (
    OptimizeRuntime,
    OptimizeStage,
    StageContext,
    StageResult,
    WorkflowOutcome,
)
from modelsurgeon.surgery.contracts import MutationKind, MutationRequest, TransactionState
from modelsurgeon.surgery.distillation_repair import (
    DistillationRepairConfig,
    TokenizerSignature,
    run_distillation_repair,
)
from modelsurgeon.surgery.huggingface_low_rank import (
    HuggingFaceLowRankError,
    load_huggingface_low_rank,
    publish_huggingface_low_rank,
)
from modelsurgeon.surgery.huggingface_quantization import (
    HuggingFaceQuantizationError,
    load_huggingface_dynamic_int8,
    publish_huggingface_dynamic_int8,
    quantize_huggingface_dynamic_int8,
)
from modelsurgeon.surgery.huggingface_sequence import (
    HuggingFaceCumulativeRun,
    HuggingFaceEdit,
    HuggingFaceStageEvidence,
    run_huggingface_cumulative_sequence,
)
from modelsurgeon.surgery.lora_repair import (
    LoRAOutputMode,
    LoRARepairConfig,
    run_bounded_lora_repair,
)


class FirstPartyOptimizeRuntimeError(RuntimeError):
    """Raised when the verified first-party optimize cell cannot run safely."""


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise FirstPartyOptimizeRuntimeError(f"{label} must be a mapping")
    return value


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_tree(path: Path) -> str:
    entries = _tree_entries(path)
    if not entries:
        raise FirstPartyOptimizeRuntimeError(f"model source does not exist: {path}")
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _tree_entries(path: Path) -> list[dict[str, object]]:
    if path.is_file():
        return [{"path": path.name, "sha256": _digest_file(path), "size": path.stat().st_size}]
    if not path.is_dir():
        raise FirstPartyOptimizeRuntimeError(f"model source does not exist: {path}")
    entries = [
        {
            "path": child.relative_to(path).as_posix(),
            "sha256": _digest_file(child),
            "size": child.stat().st_size,
        }
        for child in sorted(item for item in path.rglob("*") if item.is_file())
    ]
    if not entries:
        raise FirstPartyOptimizeRuntimeError("model source directory is empty")
    return entries


def _artifact_digest(path: Path) -> str:
    return "sha256:" + _digest_tree(path)


def _config_value(plan: OptimizePlan, section: str, key: str) -> object:
    root = _mapping(plan.resolved_config, "resolved configuration")
    return _mapping(root.get(section), section).get(key)


def _hf_dtype(value: object) -> HuggingFaceDType:
    return {
        "auto": HuggingFaceDType.AUTO,
        "fp32": HuggingFaceDType.FLOAT32,
        "fp16": HuggingFaceDType.FLOAT16,
        "bf16": HuggingFaceDType.BFLOAT16,
    }.get(str(value), HuggingFaceDType.AUTO)


def _integer(value: object, default: int, label: str) -> int:
    selected = default if value is None else value
    if not isinstance(selected, int) or isinstance(selected, bool):
        raise FirstPartyOptimizeRuntimeError(f"{label} must be an integer")
    return selected


def _number(value: object, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise FirstPartyOptimizeRuntimeError(f"{label} must be numeric")
    return float(value)


def _directory_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        raise FirstPartyOptimizeRuntimeError(f"artifact path does not exist: {path}")
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


class HuggingFaceOptimizeRuntime(OptimizeRuntime):
    """Execute one real, bounded HF structural optimization workflow."""

    def __init__(self, plan: OptimizePlan) -> None:
        self.plan = plan
        self.proof: HuggingFaceMLPProofRuntime | None = None
        self.source_digest: str | None = None
        self.candidates: tuple[MutationCandidate, ...] = ()
        self.selected: MutationCandidate | None = None
        self.selected_channel: int | None = None
        self.selected_measurement: Mapping[str, object] | None = None
        self.baseline_runtime_measurement: Mapping[str, object] | None = None
        self.selected_channels: tuple[int, ...] = ()
        self.selected_candidates: tuple[MutationCandidate, ...] = ()
        self.candidate_features: dict[str, tuple[dict[str, object], ...]] = {}
        self.feature_cache_entries: dict[str, Mapping[str, object]] = {}
        self.actual_measurements: dict[str, Mapping[str, object]] = {}
        self.meta_guidance: Mapping[str, object] | None = None
        self.meta_predictions: dict[str, float] = {}
        self.meta_order: tuple[str, ...] = ()
        self.search_comparison: Mapping[str, object] | None = None
        self.measured_frontier: Mapping[str, object] | None = None
        self.campaign_run_id: str | None = None
        self.state_updates: tuple[Mapping[str, object], ...] = ()
        self.evidence_publications: dict[str, Mapping[str, object]] = {}
        self.search_evidence_publications: dict[str, Mapping[str, object]] = {}
        self.boundary_evidence: dict[str, Mapping[str, object]] = {}
        self.runtime_hardware: Mapping[str, object] | None = None
        self.resource_preflight: Mapping[str, object] | None = None
        self.surgery_sequence: HuggingFaceCumulativeRun | None = None
        self.artifact: Path | None = None
        self.artifact_digest: str | None = None
        self.reloaded_model: Any | None = None
        self.baseline: Mapping[str, object] | None = None
        self.deployment_baseline: Mapping[str, object] | None = None
        self.deployment: Mapping[str, object] | None = None
        self.deployment_constraints_passed: bool | None = None

    def _hydrate(self, context: StageContext) -> None:
        """Rebuild runtime state from durable stage evidence after a restart."""

        self.campaign_run_id = context.run.run_id
        for stage in context.completed_results.values():
            if stage.artifact_digest is not None and self.artifact_digest is None:
                self.artifact_digest = stage.artifact_digest
            if stage.candidate_id is not None and self.selected is None:
                candidates = self.candidates or self._enumerate()
                candidate_id = "cand_" + stage.candidate_id.removeprefix("candidate_")
                self.selected = next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate.candidate_id == candidate_id
                    ),
                    None,
                )
                if self.selected is not None:
                    if self.selected.scope is CandidateScope.MLP_CHANNEL:
                        self.selected_channel = self._channel(self.selected)
                        self.selected_channels = (self.selected_channel,)
                    else:
                        self.selected_channel = None
                        self.selected_channels = ()
            try:
                detail = json.loads(stage.detail)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(detail, Mapping):
                continue
            source_digest = detail.get("source_digest")
            if isinstance(source_digest, str) and source_digest:
                self.source_digest = source_digest
            measurement = detail.get("measurement")
            if isinstance(measurement, Mapping):
                self.selected_measurement = measurement
            measured_frontier = detail.get("measured_frontier")
            if isinstance(measured_frontier, Mapping):
                self.measured_frontier = measured_frontier
            resource_preflight = detail.get("resource_preflight")
            if isinstance(resource_preflight, Mapping):
                self.resource_preflight = resource_preflight
            selected_candidate_ids = detail.get("selected_candidate_ids")
            if isinstance(selected_candidate_ids, list) and all(
                isinstance(item, str) for item in selected_candidate_ids
            ):
                candidates = self.candidates or self._enumerate()
                selected_by_id = {item.candidate_id: item for item in candidates}
                hydrated = tuple(
                    selected_by_id[item]
                    for item in selected_candidate_ids
                    if item in selected_by_id
                )
                if hydrated:
                    self.selected_candidates = hydrated
            channels = detail.get("channels")
            if isinstance(channels, list) and all(
                isinstance(channel, int) and not isinstance(channel, bool) and channel >= 0
                for channel in channels
            ):
                self.selected_channels = tuple(channels)
                if self.selected_channels:
                    self.selected_channel = self.selected_channels[0]
            if stage.artifact_digest is not None:
                artifact_value = detail.get("artifact")
                if isinstance(artifact_value, str) and artifact_value:
                    artifact = Path(artifact_value).expanduser().absolute().resolve(strict=False)
                    if artifact.is_file() and _artifact_digest(
                        artifact.parent
                    ) == stage.artifact_digest:
                        self.artifact = artifact
                        self.reloaded_model = self._reload(artifact)
            baseline = detail.get("baseline")
            candidate = detail.get("candidate")
            if isinstance(baseline, Mapping) and isinstance(candidate, Mapping):
                self.deployment_baseline = baseline
                self.deployment = candidate
                self.deployment_constraints_passed = stage.constraints_passed

    def _surgery_steps(self) -> int:
        root = _mapping(self.plan.resolved_config, "resolved configuration")
        search_config = root.get("search")
        configured = (
            None
            if search_config is None
            else _mapping(search_config, "search").get("max_surgery_steps")
        )
        if configured is None:
            return 2 if self.plan.budget.evaluations > 1 else 1
        steps = _integer(configured, 1, "search.max_surgery_steps")
        if steps < 1:
            raise FirstPartyOptimizeRuntimeError("search.max_surgery_steps must be positive")
        return min(steps, self.plan.budget.evaluations)

    def _candidate_scopes(self) -> tuple[CandidateScope, ...]:
        root = _mapping(self.plan.resolved_config, "resolved configuration")
        raw_search = root.get("search")
        raw_scopes = (
            (CandidateScope.MLP_CHANNEL.value,)
            if raw_search is None
            else _mapping(raw_search, "search").get(
                "scopes", (CandidateScope.MLP_CHANNEL.value,)
            )
        )
        if not isinstance(raw_scopes, (list, tuple)) or not raw_scopes:
            raise FirstPartyOptimizeRuntimeError("search.scopes must be a non-empty list")
        by_name = {
            "mlp_channel": CandidateScope.MLP_CHANNEL,
            "channel": CandidateScope.MLP_CHANNEL,
            "attention_head": CandidateScope.ATTENTION_HEAD,
            "head": CandidateScope.ATTENTION_HEAD,
            "transformer_layer": CandidateScope.TRANSFORMER_LAYER,
            "layer": CandidateScope.TRANSFORMER_LAYER,
            "low_rank": CandidateScope.LOW_RANK,
        }
        try:
            scopes = tuple(by_name[str(item)] for item in raw_scopes)
        except KeyError as error:
            raise FirstPartyOptimizeRuntimeError(
                f"unsupported first-party candidate scope: {error.args[0]}"
            ) from error
        if len(scopes) != len(set(scopes)):
            raise FirstPartyOptimizeRuntimeError("search.scopes must be unique")
        unsupported = set(scopes) - {
            CandidateScope.MLP_CHANNEL,
            CandidateScope.ATTENTION_HEAD,
            CandidateScope.TRANSFORMER_LAYER,
            CandidateScope.LOW_RANK,
        }
        if unsupported:
            raise FirstPartyOptimizeRuntimeError(
                "configured candidate scope is not supported by the HF runtime"
            )
        return scopes

    def _low_rank_rank(self) -> int:
        raw_search = _mapping(
            _mapping(self.plan.resolved_config, "resolved configuration").get("search", {}),
            "search",
        )
        rank = _integer(raw_search.get("low_rank_rank"), 4, "search.low_rank_rank")
        if rank <= 0:
            raise FirstPartyOptimizeRuntimeError("search.low_rank_rank must be positive")
        return rank

    def _meta_rank(self, candidates: tuple[MutationCandidate, ...]) -> tuple[str, ...]:
        from modelsurgeon.surgeon.matrix import transform_inference_record
        from modelsurgeon.surgeon.pretrained_registry import (
            PretrainedRegistryError,
            PretrainedSurgeonRegistry,
        )

        resolved_config = _mapping(self.plan.resolved_config, "resolved configuration")
        raw_surgeon = resolved_config.get("surgeon")
        if raw_surgeon is None:
            self.meta_guidance = {
                "status": "not_configured",
                "measurement_authority": "physical_evaluation",
            }
            return ()
        if any(candidate.scope is not CandidateScope.MLP_CHANNEL for candidate in candidates):
            self.meta_guidance = {
                "status": "unsupported_scope",
                "measurement_authority": "physical_evaluation",
                "reason": "the verified Meta-Surgeon bundle is trained for MLP-channel features",
            }
            return ()
        surgeon = _mapping(raw_surgeon, "surgeon")
        card_digest = surgeon.get("card_digest")
        if card_digest is None:
            self.meta_guidance = {
                "status": "not_configured",
                "measurement_authority": "physical_evaluation",
            }
            return ()
        registry_root = surgeon.get("registry_root")
        secret_env = surgeon.get("signing_env")
        if not isinstance(registry_root, str) or not registry_root.strip():
            raise FirstPartyOptimizeRuntimeError("surgeon.registry_root is required")
        if not isinstance(card_digest, str) or not card_digest.strip():
            raise FirstPartyOptimizeRuntimeError("surgeon.card_digest is required")
        if not isinstance(secret_env, str) or not secret_env.strip():
            raise FirstPartyOptimizeRuntimeError("surgeon.signing_env is required")
        secret = os.environ.get(secret_env)
        if not secret:
            raise FirstPartyOptimizeRuntimeError(
                f"Meta-Surgeon signing secret environment variable is missing: {secret_env}"
            )
        try:
            registry = PretrainedSurgeonRegistry(registry_root)
            card, bundle = registry.resolve(
                card_digest,
                secret=secret.encode("utf-8"),
                expected_feature_schema_version=1,
                expected_target_schema_version=1,
            )
        except (OSError, PretrainedRegistryError) as error:
            raise FirstPartyOptimizeRuntimeError(
                f"Meta-Surgeon bundle could not be verified or loaded: {error}"
            ) from error

        proof = self._ensure_loaded()
        predictions: dict[str, float] = {}
        for candidate in candidates:
            raw_features = self.candidate_features.get(candidate.candidate_id)
            if raw_features is None:
                raise FirstPartyOptimizeRuntimeError(
                    "Meta-Surgeon inference is missing candidate feature evidence"
                )
            record = {
                "example_id": candidate.candidate_id,
                "model": proof._model_target.to_record(),
                "mutation": {"plan": {"request": candidate.request.to_record()}},
                "pre_mutation_features": list(raw_features),
                "versions": {
                    "feature_schema_version": bundle.preprocessor.source_feature_schema_version
                },
            }
            try:
                row = transform_inference_record(
                    record, bundle.preprocessor, refuse_missing=False
                )
                predicted = bundle.model.predict((row,))
            except Exception as error:
                raise FirstPartyOptimizeRuntimeError(
                    f"Meta-Surgeon inference failed for {candidate.candidate_id}: {error}"
                ) from error
            prediction_value = predicted[0] if len(predicted) == 1 else None
            if not isinstance(prediction_value, (int, float)):
                raise FirstPartyOptimizeRuntimeError(
                    f"Meta-Surgeon returned an invalid prediction for {candidate.candidate_id}"
                )
            predictions[candidate.candidate_id] = float(cast(int | float, prediction_value))

        higher_is_better = bundle.card.target_name in {"behavior", "safe_mutation"}
        ordered = tuple(
            candidate_id
            for candidate_id, _ in sorted(
                predictions.items(),
                key=lambda item: (
                    -item[1] if higher_is_better else item[1],
                    item[0],
                ),
            )
        )
        self.meta_predictions = predictions
        self.meta_order = ordered
        self.meta_guidance = {
            "status": "verified_bundle",
            "card_digest": card_digest,
            "bundle_digest": card.bundle_digest,
            "target": bundle.card.target_name,
            "direction": "higher_is_better" if higher_is_better else "lower_is_better",
            "predictions": dict(sorted(predictions.items())),
            "ordered_candidates": list(ordered),
            "measurement_authority": "physical_evaluation",
        }
        return ordered

    def _feature_cache_root(self) -> Path:
        resolved_config = _mapping(self.plan.resolved_config, "resolved configuration")
        features = resolved_config.get("features")
        configured = None
        if features is not None:
            configured = _mapping(features, "features").get("cache_dir")
        if configured is not None:
            if not isinstance(configured, str) or not configured.strip():
                raise FirstPartyOptimizeRuntimeError("features.cache_dir must be a path")
            return Path(configured).expanduser().absolute().resolve(strict=False)
        artifact_dir = resolved_config.get("artifact_dir", "artifacts")
        if not isinstance(artifact_dir, str) or not artifact_dir.strip():
            raise FirstPartyOptimizeRuntimeError("artifact_dir must be a path")
        return (Path(artifact_dir) / "feature-cache").expanduser().absolute().resolve(
            strict=False
        )

    def _evidence_store_root(self) -> Path:
        resolved_config = _mapping(self.plan.resolved_config, "resolved configuration")
        artifact_dir = resolved_config.get("artifact_dir", "artifacts")
        if not isinstance(artifact_dir, str) or not artifact_dir.strip():
            raise FirstPartyOptimizeRuntimeError("artifact_dir must be a path")
        return (Path(artifact_dir) / "optimization-evidence").expanduser().absolute().resolve(
            strict=False
        )

    @staticmethod
    def _existing_probe_path(path: Path) -> Path:
        probe_path = path.expanduser().absolute().resolve(strict=False)
        while not probe_path.exists() and probe_path != probe_path.parent:
            probe_path = probe_path.parent
        return probe_path if probe_path.exists() else Path.cwd()

    def _collect_runtime_hardware(self, probe_path: Path | None = None) -> HardwareInventory:
        if probe_path is None:
            resolved_config = _mapping(self.plan.resolved_config, "resolved configuration")
            artifact_dir = resolved_config.get("artifact_dir", "artifacts")
            if not isinstance(artifact_dir, str) or not artifact_dir.strip():
                raise FirstPartyOptimizeRuntimeError("artifact_dir must be a path")
            probe_path = Path(artifact_dir)
        return collect_hardware_inventory(self._existing_probe_path(probe_path))

    def _runtime_hardware_record(self) -> Mapping[str, object]:
        if self.runtime_hardware is not None:
            return self.runtime_hardware
        self.runtime_hardware = self._collect_runtime_hardware().to_record()
        return self.runtime_hardware

    @staticmethod
    def _failure_classification(error: BaseException | str) -> str:
        text = str(error).lower()
        if "out of memory" in text or "oom" in text:
            return "oom"
        if "timeout" in text or "timed out" in text:
            return "timeout"
        if any(
            marker in text
            for marker in ("unknown", "unavailable", "unobservable", "not observable")
        ):
            return "unknown"
        if "unsupported" in text:
            return "unsupported"
        return "runtime_failure"

    def _record_runtime_boundary(
        self,
        stage: OptimizeStage,
        outcome: OptimizationEvidenceOutcome,
        reason: str,
        *,
        error: BaseException | None = None,
    ) -> Mapping[str, object] | None:
        """Retain non-measurement outcomes even when a stage cannot execute."""

        try:
            proof = self.proof
            model: Mapping[str, object]
            dataset: Mapping[str, object]
            hardware: dict[str, object]
            run_id: str
            if proof is not None:
                model = proof._model_target.to_record()
                dataset = proof._dataset.to_record()
                hardware = dict(proof._hardware.to_record())
                run_id = proof.run_id
            else:
                resolved_config = _mapping(self.plan.resolved_config, "resolved configuration")
                model = _mapping(resolved_config.get("model"), "model")
                dataset = _mapping(resolved_config.get("calibration"), "calibration")
                hardware = {}
                run_id = "runtime_" + hashlib.sha256(self.plan.plan_id.encode()).hexdigest()
            hardware["runtime_inventory"] = dict(self._runtime_hardware_record())
            failure = {
                "reason": reason,
                "classification": self._failure_classification(error or reason),
            }
            identity = {
                "run_id": run_id,
                "stage": stage.value,
                "outcome": outcome.value,
                "reason": reason,
                "failure": failure,
            }
            observation_id = "obs_" + hashlib.sha256(
                json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            record = OptimizationEvidenceRecord(
                observation_id=observation_id,
                run_id=run_id,
                stage=stage.value,
                state_id="boundary:" + stage.value,
                outcome=outcome,
                model=model,
                dataset=dataset,
                hardware=hardware,
                versions={
                    "runtime": "first_party_hf_physical_search",
                    "config_digest": self.plan.config_digest,
                    "plan_digest": self.plan.plan_id,
                    "evidence_schema_version": 1,
                },
                lineage={"boundary": True, "reason": reason},
                failure=failure,
            )
            published = OptimizationEvidenceStore(self._evidence_store_root()).publish(record)
            reference = published.to_record()
            self.boundary_evidence[observation_id] = reference
            self.evidence_publications[observation_id] = reference
            return reference
        except Exception:
            return None

    def _publish_feature_partition(self, partition: FeaturePartition) -> Mapping[str, object]:
        cache = FeaturePartitionCache(self._feature_cache_root())
        existing = cache.load(partition.key)
        if existing is None:
            published = cache.write(partition.key, partition.records)
        elif existing != partition:
            raise FirstPartyOptimizeRuntimeError(
                "feature cache identity already contains conflicting evidence"
            )
        else:
            published = existing
        return {
            "path": str(cache.path_for(published.key)),
            "key": published.key.to_record(),
            "records_sha256": published.records_sha256,
            "record_count": len(published.records),
        }

    def _record_optimization_observation(
        self,
        *,
        stage: str,
        state_id: str,
        outcome: OptimizationEvidenceOutcome,
        candidate_id: str | None,
        mutation_id: str | None,
        measurement: Mapping[str, object] | None,
        prediction: Mapping[str, object] | None = None,
        artifact: Mapping[str, object] | None = None,
        failure: Mapping[str, object] | None = None,
        feature_records: tuple[Mapping[str, object], ...] = (),
        feature_cache: Mapping[str, object] | None = None,
        lineage: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        proof = self._ensure_loaded()
        identity = {
            "run_id": proof.run_id,
            "stage": stage,
            "state_id": state_id,
            "candidate_id": candidate_id,
            "mutation_id": mutation_id,
            "outcome": outcome.value,
            "measurement": None if measurement is None else dict(measurement),
            "artifact": None if artifact is None else dict(artifact),
            "failure": None if failure is None else dict(failure),
        }
        observation_id = "obs_" + hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        resolved_lineage = dict(lineage or {})
        if feature_cache is not None:
            resolved_lineage["feature_cache"] = dict(feature_cache)
        record = OptimizationEvidenceRecord(
            observation_id=observation_id,
            run_id=proof.run_id,
            stage=stage,
            state_id=state_id,
            outcome=outcome,
            model=proof._model_target.to_record(),
            dataset=proof._dataset.to_record(),
            hardware={
                **proof._hardware.to_record(),
                "runtime_inventory": dict(self._runtime_hardware_record()),
            },
            versions={
                "runtime": "first_party_hf_physical_search",
                "config_digest": self.plan.config_digest,
                "plan_digest": self.plan.plan_id,
                "source_artifact_digest": self.source_digest,
                "evidence_schema_version": 1,
            },
            lineage=resolved_lineage,
            candidate_id=candidate_id,
            mutation_id=mutation_id,
            features=feature_records,
            prediction=prediction,
            measurement=measurement,
            artifact=artifact,
            failure=failure,
        )
        published = OptimizationEvidenceStore(self._evidence_store_root()).publish(record)
        reference = published.to_record()
        self.evidence_publications[observation_id] = reference
        if stage == "active_search":
            self.search_evidence_publications[observation_id] = reference
        return reference

    def _build_search_comparison(
        self,
        candidates: tuple[MutationCandidate, ...],
        ranked: list[tuple[float, float, str, MutationCandidate, Mapping[str, object]]],
    ) -> None:
        from modelsurgeon.surgeon.ranking import rank_random

        # Keep rejected-but-measured candidates visible. A baseline that chose
        # one must be recorded as measured-but-ineligible, not unmeasured.
        actual = {
            candidate_id: _number(measurement["perplexity_delta"], "perplexity_delta")
            for candidate_id, measurement in self.actual_measurements.items()
        }
        best_delta = ranked[0][0]
        allowed = self.plan.quality_profile.max_perplexity_delta
        allowed = 0.05 if allowed is None else allowed
        magnitudes: dict[str, float] = {}
        for candidate in candidates:
            values = [
                float(cast(int | float, item["value"]))
                for item in self.candidate_features.get(candidate.candidate_id, ())
                if item.get("name") == "weight_l1_norm"
                and isinstance(item.get("value"), (int, float))
            ]
            if values:
                magnitudes[candidate.candidate_id] = statistics.fmean(values)
        magnitude_order = tuple(
            item[0]
            for item in sorted(
                magnitudes.items(), key=lambda item: (item[1], item[0])
            )
        )
        seed = _integer(_config_value(self.plan, "calibration", "seed"), 0, "calibration.seed")
        random_order = tuple(
            item.candidate_id
            for item in rank_random(candidates, seed=seed).entries
        )

        def baseline_record(name: str, candidate_id: str | None) -> dict[str, object]:
            if candidate_id is None or candidate_id not in actual:
                return {"method": name, "candidate_id": None, "measured": False}
            measured_delta = actual[candidate_id]
            return {
                "method": name,
                "candidate_id": candidate_id,
                "measured": True,
                "constraints_passed": measured_delta <= allowed,
                "measured_perplexity_delta": measured_delta,
                "regret_vs_measured_best": measured_delta - best_delta,
            }

        self.search_comparison = {
            "measured_authority": "physical_evaluation",
            "measured_best": ranked[0][3].candidate_id,
            "baselines": [
                baseline_record("meta_surgeon", self.meta_order[0] if self.meta_order else None),
                baseline_record("magnitude", magnitude_order[0] if magnitude_order else None),
                baseline_record("random", random_order[0] if random_order else None),
                baseline_record("no_guidance", ranked[0][3].candidate_id),
            ],
        }

    def _model_source(self) -> str:
        value = _config_value(self.plan, "model", "path")
        if not isinstance(value, str) or not value.strip():
            raise FirstPartyOptimizeRuntimeError("model.path is required for first-party execution")
        return value

    def _calibration_text(self) -> Path:
        value = _config_value(self.plan, "calibration", "dataset")
        if not isinstance(value, str) or not value.strip():
            raise FirstPartyOptimizeRuntimeError(
                "calibration.dataset must be a local UTF-8 text file for the first-party HF runtime"
            )
        path = Path(value).expanduser().absolute().resolve(strict=False)
        if not path.is_file():
            raise FirstPartyOptimizeRuntimeError(
                f"calibration.dataset must name an existing local UTF-8 text file: {path}"
            )
        return path

    def _new_proof_runtime(
        self,
        model: str,
        revision: str,
        *,
        local_files_only: bool,
    ) -> HuggingFaceMLPProofRuntime:
        calibration_config = _mapping(
            _mapping(self.plan.resolved_config, "resolved configuration").get("calibration"),
            "calibration",
        )
        samples = _integer(calibration_config.get("samples"), 512, "calibration.samples")
        sequence_length = _integer(
            calibration_config.get("max_sequence_length"),
            2048,
            "calibration.max_sequence_length",
        )
        requested_tokens = max(sequence_length * samples, sequence_length * 2)
        return HuggingFaceMLPProofRuntime(
            HuggingFaceMLPProofConfig(
                model=model,
                calibration_text=self._calibration_text(),
                revision=revision,
                device_map="cpu" if self.plan.hardware_profile.device == "cpu" else "auto",
                dtype=_hf_dtype(_config_value(self.plan, "model", "dtype")),
                trust_remote_code=False,
                local_files_only=local_files_only,
                sequence_length=min(sequence_length, 2048),
                max_tokens=requested_tokens,
                safe_perplexity_delta=self.plan.quality_profile.max_perplexity_delta or 0.05,
                seed=_integer(calibration_config.get("seed"), 0, "calibration.seed"),
            )
        )

    def _ensure_loaded(self) -> HuggingFaceMLPProofRuntime:
        if self.proof is not None:
            return self.proof
        if _config_value(self.plan, "model", "format") not in {
            ModelFormat.HUGGING_FACE.value,
            ModelFormat.HUGGING_FACE,
            None,
        }:
            raise FirstPartyOptimizeRuntimeError(
                "the first-party executor currently supports Hugging Face models only"
            )
        model = self._model_source()
        source_path = Path(model).expanduser().absolute().resolve(strict=False)
        if source_path.exists():
            self.source_digest = _artifact_digest(source_path)
        revision = _config_value(self.plan, "model", "revision")
        if not isinstance(revision, str) or not revision.strip():
            raise FirstPartyOptimizeRuntimeError(
                "model.revision must be pinned before first-party execution"
            )
        self.proof = self._new_proof_runtime(
            model,
            revision,
            local_files_only=source_path.exists(),
        )
        return self.proof

    @staticmethod
    def _model_storage_bytes(model: Any) -> int:
        total = 0
        parameter_count = 0
        try:
            parameters = model.parameters()
            for parameter in parameters:
                numel = getattr(parameter, "numel", None)
                element_size = getattr(parameter, "element_size", None)
                if not callable(numel) or not callable(element_size):
                    raise FirstPartyOptimizeRuntimeError(
                        "model parameter storage size is unknown"
                    )
                parameter_bytes = int(numel()) * int(element_size())
                if parameter_bytes < 0:
                    raise FirstPartyOptimizeRuntimeError(
                        "model parameter storage size is invalid"
                    )
                total += parameter_bytes
                parameter_count += 1
        except FirstPartyOptimizeRuntimeError:
            raise
        except Exception as error:
            raise FirstPartyOptimizeRuntimeError(
                f"model parameter storage size is unknown: {type(error).__name__}"
            ) from error
        if parameter_count == 0 or total <= 0:
            raise FirstPartyOptimizeRuntimeError("model parameter storage size is unknown")
        return total

    @staticmethod
    def _model_cuda_devices(model: Any) -> tuple[int, ...]:
        devices: set[int] = set()
        try:
            for parameter in model.parameters():
                device = getattr(parameter, "device", None)
                if getattr(device, "type", None) != "cuda":
                    continue
                index = getattr(device, "index", None)
                devices.add(0 if index is None else int(index))
        except Exception as error:
            raise FirstPartyOptimizeRuntimeError(
                f"model CUDA placement is unknown: {type(error).__name__}"
            ) from error
        return tuple(sorted(devices))

    def _preflight_resources(
        self,
        model: Any,
        destination_root: Path,
        *,
        phase: str,
    ) -> Mapping[str, object]:
        """Require live capacity before search or any physical artifact write."""

        if not phase.strip():
            raise FirstPartyOptimizeRuntimeError("resource preflight phase is required")
        inventory = self._collect_runtime_hardware(destination_root)
        inventory_record = inventory.to_record()
        self.runtime_hardware = inventory_record

        available_ram = inventory.memory.available_bytes
        if not isinstance(available_ram, int) or available_ram <= 0:
            raise FirstPartyOptimizeRuntimeError(
                "available system memory is unknown; refusing first-party execution"
            )
        model_bytes = self._model_storage_bytes(model)
        working_set_bytes = model_bytes * 2
        if working_set_bytes > self.plan.budget.max_ram_bytes:
            raise FirstPartyOptimizeRuntimeError(
                "estimated candidate working set exceeds max_ram_bytes"
            )
        if working_set_bytes > available_ram:
            raise FirstPartyOptimizeRuntimeError(
                "insufficient available system memory for candidate evaluation"
            )

        source_path = Path(self._model_source()).expanduser().absolute().resolve(strict=False)
        source_bytes = _directory_bytes(source_path) if source_path.exists() else None
        estimated_artifact_bytes = max(model_bytes, source_bytes or 0)
        if estimated_artifact_bytes > self.plan.budget.max_artifact_bytes:
            raise FirstPartyOptimizeRuntimeError(
                "estimated candidate artifact exceeds max_artifact_bytes"
            )
        artifact_count = self._surgery_steps()
        estimated_disk_bytes = estimated_artifact_bytes * artifact_count
        if estimated_disk_bytes > self.plan.budget.max_disk_bytes:
            raise FirstPartyOptimizeRuntimeError(
                "estimated retained candidate artifacts exceed max_disk_bytes"
            )
        if estimated_disk_bytes > inventory.disk.free_bytes:
            raise FirstPartyOptimizeRuntimeError(
                "insufficient disk headroom for retained candidate artifacts"
            )

        cuda_devices = self._model_cuda_devices(model)
        requires_cuda = self.plan.hardware_profile.device.lower().startswith("cuda")
        if requires_cuda and not inventory.cuda.available:
            raise FirstPartyOptimizeRuntimeError(
                "CUDA runtime is unsupported for the requested hardware profile"
            )
        if requires_cuda and not inventory.cuda.devices:
            raise FirstPartyOptimizeRuntimeError(
                "CUDA device capacity is unknown for the requested hardware profile"
            )

        free_vram_bytes: int | None = None
        estimated_vram_bytes = 0
        if cuda_devices:
            if not inventory.cuda.available:
                raise FirstPartyOptimizeRuntimeError(
                    "CUDA runtime availability is unknown for the loaded model"
                )
            torch = self._ensure_loaded()._torch
            free_samples: list[int] = []
            try:
                mem_get_info = torch.cuda.mem_get_info
                for device_index in cuda_devices:
                    free, _ = mem_get_info(device_index)
                    free_samples.append(int(free))
            except Exception as error:
                raise FirstPartyOptimizeRuntimeError(
                    f"available VRAM is unknown: {type(error).__name__}"
                ) from error
            if not free_samples or any(value <= 0 for value in free_samples):
                raise FirstPartyOptimizeRuntimeError(
                    "available VRAM is unknown; refusing CUDA candidate evaluation"
                )
            free_vram_bytes = sum(free_samples)
            estimated_vram_bytes = model_bytes
            if self.plan.budget.max_vram_bytes is not None and (
                estimated_vram_bytes > self.plan.budget.max_vram_bytes
            ):
                raise FirstPartyOptimizeRuntimeError(
                    "estimated CUDA candidate working set exceeds max_vram_bytes"
                )
            if estimated_vram_bytes > free_vram_bytes:
                raise FirstPartyOptimizeRuntimeError(
                    "insufficient available VRAM for candidate evaluation"
                )

        result: Mapping[str, object] = {
            "status": "ready",
            "phase": phase,
            "authority": "live_hardware_inventory_and_model_tensor_storage",
            "hardware_inventory": inventory_record,
            "model_parameter_bytes": model_bytes,
            "estimated_working_set_bytes": working_set_bytes,
            "estimated_candidate_artifact_bytes": estimated_artifact_bytes,
            "source_artifact_bytes": source_bytes,
            "estimated_retained_disk_bytes": estimated_disk_bytes,
            "planned_artifact_count": artifact_count,
            "available_ram_bytes": available_ram,
            "available_disk_bytes": inventory.disk.free_bytes,
            "cuda_devices": list(cuda_devices),
            "estimated_vram_bytes": estimated_vram_bytes,
            "available_vram_bytes": free_vram_bytes,
            "budget": self.plan.budget.to_record(),
        }
        self.resource_preflight = result
        return result

    def _repair_settings(self) -> Mapping[str, object]:
        root = _mapping(self.plan.resolved_config, "resolved configuration")
        return _mapping(root.get("repair", {}), "repair")

    def _quantization_settings(self) -> Mapping[str, object]:
        root = _mapping(self.plan.resolved_config, "resolved configuration")
        return _mapping(root.get("quantization", {}), "quantization")

    def _repair_examples(self) -> tuple[dict[str, Any], ...]:
        proof = self._ensure_loaded()
        torch = proof._torch
        examples: list[dict[str, Any]] = []
        for chunk in proof._chunks[: min(len(proof._chunks), 8)]:
            input_ids = torch.tensor((chunk,), dtype=torch.long)
            examples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                    "labels": input_ids.clone(),
                }
            )
        if not examples:
            raise FirstPartyOptimizeRuntimeError(
                "repair calibration examples are unavailable"
            )
        return tuple(examples)

    def _repair_targets(self, model: Any) -> tuple[str, ...]:
        configured = self._repair_settings().get("target_modules", ())
        if not isinstance(configured, (list, tuple)) or not all(
            isinstance(item, str) for item in configured
        ):
            raise FirstPartyOptimizeRuntimeError("repair.target_modules must be a list of paths")
        if configured:
            targets = tuple(configured)
        else:
            torch = self._ensure_loaded()._torch
            targets = tuple(
                sorted(
                    name
                    for name, module in model.named_modules()
                    if isinstance(module, torch.nn.Linear)
                    and (name.endswith(".down_proj") or name.endswith(".o_proj"))
                )[:2]
            )
        if not targets:
            raise FirstPartyOptimizeRuntimeError(
                "no supported Linear modules are available for repair"
            )
        torch = self._ensure_loaded()._torch
        for name in targets:
            module = dict(model.named_modules()).get(name)
            if not isinstance(module, torch.nn.Linear):
                raise FirstPartyOptimizeRuntimeError(
                    f"repair target {name!r} is not a supported Linear module"
                )
        return tuple(sorted(set(targets)))

    @staticmethod
    def _checkpoint_id(digest: str) -> str:
        value = digest.removeprefix("sha256:")
        if len(value) != 64:
            value = hashlib.sha256(digest.encode()).hexdigest()
        return "checkpoint_" + value

    def _repair(self, context: StageContext) -> StageResult:
        settings = self._repair_settings()
        method = settings.get("method", "none")
        if not isinstance(method, str) or not method.strip():
            raise FirstPartyOptimizeRuntimeError("repair.method must be text")
        if method == "none":
            if self.artifact is None or self.artifact_digest is None:
                return self._unsupported(
                    OptimizeStage.REPAIR,
                    "repair was not requested and no physical artifact is available",
                )
            return self._result(
                OptimizeStage.REPAIR,
                json.dumps(
                    {
                        "status": "not_requested",
                        "method": method,
                        "artifact": str(self.artifact),
                        "artifact_digest": self.artifact_digest,
                    },
                    sort_keys=True,
                ),
                artifact_digest=self.artifact_digest,
                candidate=self.selected,
                constraints_passed=True,
                transaction_state=TransactionState.COMMITTED,
            )
        if method not in {"lora", "distillation"}:
            return self._unsupported(
                OptimizeStage.REPAIR,
                f"first-party Hugging Face repair method {method!r} is unsupported",
            )
        if self.artifact is None or self.artifact_digest is None or self.reloaded_model is None:
            return self._unsupported(
                OptimizeStage.REPAIR,
                f"{method} repair requires a published reloaded physical artifact",
            )

        proof = self._ensure_loaded()
        artifact_dir = _mapping(self.plan.resolved_config, "resolved configuration").get(
            "artifact_dir", "artifacts"
        )
        if not isinstance(artifact_dir, str) or not artifact_dir.strip():
            raise FirstPartyOptimizeRuntimeError("artifact_dir must be a path")
        repair_root = (
            Path(artifact_dir)
            / "optimize"
            / context.run.run_id
            / "hf-lora-repair"
        )
        self._preflight_resources(self.reloaded_model, repair_root, phase="repair")
        parent_model = self.reloaded_model
        parent_measurement = proof.measure_model(
            parent_model,
            repetitions=self.plan.quality_profile.evaluation_repetitions,
        )
        repair_model = copy.deepcopy(parent_model)
        targets = self._repair_targets(repair_model)
        max_steps_value = settings.get("max_steps")
        max_steps = (
            self.plan.budget.repair_steps
            if max_steps_value is None
            else _integer(max_steps_value, 1, "repair.max_steps")
        )
        seed = _integer(_config_value(self.plan, "calibration", "seed"), 0, "calibration.seed")
        examples = self._repair_examples()
        if method == "lora":
            lora_config = LoRARepairConfig(
                rank=_integer(settings.get("rank"), 4, "repair.rank"),
                alpha=_number(settings.get("alpha", 8.0), "repair.alpha"),
                dropout=_number(settings.get("dropout", 0.0), "repair.dropout"),
                learning_rate=_number(
                    settings.get("learning_rate", 1e-3), "repair.learning_rate"
                ),
                max_steps=min(max_steps, self.plan.budget.repair_steps),
                seed=seed,
                output_mode=LoRAOutputMode.MERGED,
            )
            lora_result = run_bounded_lora_repair(
                repair_model,
                examples,
                targets,
                lora_config,
                source_checkpoint_id=self._checkpoint_id(self.source_digest or self.plan.plan_id),
                candidate_checkpoint_id=self._checkpoint_id(self.artifact_digest),
            )
            repair_record = lora_result.to_record()
        else:
            parameter_names = tuple(
                sorted(
                    name
                    for target in targets
                    for name in (f"{target}.weight", f"{target}.bias")
                    if name in dict(repair_model.named_parameters())
                )
            )
            if not parameter_names:
                return self._unsupported(
                    OptimizeStage.REPAIR,
                    "distillation targets expose no trainable weight or bias parameters",
                )
            tokenizer_signature = TokenizerSignature.from_tokenizer(proof._tokenizer)
            teacher_model = copy.deepcopy(parent_model)
            distillation_config = DistillationRepairConfig(
                parameter_names=parameter_names,
                learning_rate=_number(
                    settings.get("learning_rate", 1e-3), "repair.learning_rate"
                ),
                max_steps=min(max_steps, self.plan.budget.repair_steps),
                seed=seed,
            )
            distillation_result = run_distillation_repair(
                repair_model,
                examples,
                distillation_config,
                teacher_tokenizer=tokenizer_signature,
                candidate_tokenizer=tokenizer_signature,
                source_checkpoint_id=self._checkpoint_id(self.source_digest or self.plan.plan_id),
                candidate_parent_checkpoint_id=self._checkpoint_id(self.artifact_digest),
                teacher_model=teacher_model,
            )
            repair_record = distillation_result.to_record()
        repaired_measurement = proof.measure_model(
            repair_model,
            repetitions=self.plan.quality_profile.evaluation_repetitions,
        )
        parent_perplexity = _number(parent_measurement["perplexity"], "parent.perplexity")
        repaired_perplexity = _number(
            repaired_measurement["perplexity"], "repaired.perplexity"
        )
        accepted = (
            repaired_perplexity <= parent_perplexity
        )
        detail: dict[str, object] = {
            "method": method,
            "status": "accepted" if accepted else "rejected",
            "repair": repair_record,
            "targets": list(targets),
            "parent_measurement": dict(parent_measurement),
            "repaired_measurement": dict(repaired_measurement),
            "resource_preflight": self.resource_preflight,
            "artifact": str(self.artifact),
            "artifact_digest": self.artifact_digest,
        }
        if not accepted:
            self._record_optimization_observation(
                stage="repair",
                state_id="rejected:" + self.artifact_digest,
                outcome=OptimizationEvidenceOutcome.REJECTED,
                candidate_id=(None if self.selected is None else self.selected.candidate_id),
                mutation_id=None,
                measurement={
                    "parent": dict(parent_measurement),
                    "repaired": dict(repaired_measurement),
                    "measurement_authority": "physical_repair_evaluation",
                },
                failure={"reason": "repair did not improve the measured parent artifact"},
                lineage={"method": method, "targets": list(targets)},
            )
            return self._result(
                OptimizeStage.REPAIR,
                json.dumps(detail, sort_keys=True),
                measured=True,
                constraints_passed=True,
                artifact_digest=self.artifact_digest,
                candidate=self.selected,
                transaction_state=TransactionState.COMMITTED,
            )

        destination = repair_root / "candidate"
        repaired_artifact = self._publish(repair_model, destination)
        reloaded = self._reload(repaired_artifact)
        if not self._generation_smoke(reloaded):
            raise FirstPartyOptimizeRuntimeError("repaired artifact failed inference smoke")
        reloaded_measurement = proof.measure_model(
            reloaded,
            repetitions=self.plan.quality_profile.evaluation_repetitions,
        )
        reloaded_perplexity = _number(
            reloaded_measurement["perplexity"], "reloaded_repaired.perplexity"
        )
        if reloaded_perplexity > parent_perplexity:
            detail.update(
                {
                    "status": "rejected",
                    "reloaded_measurement": dict(reloaded_measurement),
                    "rejected_artifact": str(repaired_artifact),
                    "rejected_artifact_digest": _artifact_digest(repaired_artifact.parent),
                    "rejection_reason": "reloaded repair was worse than its parent",
                }
            )
            self._record_optimization_observation(
                stage="repair",
                state_id="rejected_reload:" + self.artifact_digest,
                outcome=OptimizationEvidenceOutcome.REJECTED,
                candidate_id=(None if self.selected is None else self.selected.candidate_id),
                mutation_id=None,
                measurement={
                    "parent": dict(parent_measurement),
                    "repaired": dict(reloaded_measurement),
                    "measurement_authority": "physical_reloaded_repair_evaluation",
                },
                artifact={
                    "path": str(repaired_artifact.parent),
                    "digest": detail["rejected_artifact_digest"],
                },
                failure={"reason": detail["rejection_reason"]},
                lineage={"method": method, "targets": list(targets)},
            )
            return self._result(
                OptimizeStage.REPAIR,
                json.dumps(detail, sort_keys=True),
                measured=True,
                constraints_passed=True,
                artifact_digest=self.artifact_digest,
                candidate=self.selected,
                transaction_state=TransactionState.COMMITTED,
            )

        self.artifact = repaired_artifact
        self.artifact_digest = _artifact_digest(repaired_artifact.parent)
        self.reloaded_model = reloaded
        detail.update(
            {
                "status": "accepted",
                "artifact": str(repaired_artifact),
                "artifact_digest": self.artifact_digest,
                "reloaded_measurement": dict(reloaded_measurement),
            }
        )
        self._record_optimization_observation(
            stage="repair",
            state_id="accepted:" + self.artifact_digest,
            outcome=OptimizationEvidenceOutcome.ACCEPTED,
            candidate_id=(None if self.selected is None else self.selected.candidate_id),
            mutation_id=None,
            measurement={
                "parent": dict(parent_measurement),
                "repaired": dict(reloaded_measurement),
                "measurement_authority": "physical_reloaded_repair_evaluation",
            },
            artifact={
                "path": str(repaired_artifact.parent),
                "digest": self.artifact_digest,
            },
            lineage={"method": method, "targets": list(targets)},
        )
        return self._result(
            OptimizeStage.REPAIR,
            json.dumps(detail, sort_keys=True),
            measured=True,
            constraints_passed=True,
            artifact_digest=self.artifact_digest,
            candidate=self.selected,
            transaction_state=TransactionState.COMMITTED,
        )

    def _quantization(self, context: StageContext) -> StageResult:
        settings = self._quantization_settings()
        method = settings.get("method", "none")
        if not isinstance(method, str) or not method.strip():
            raise FirstPartyOptimizeRuntimeError("quantization.method must be text")
        if method == "none":
            reason = "quantization was not requested"
            reference = self._record_runtime_boundary(
                OptimizeStage.QUANTIZATION, OptimizationEvidenceOutcome.UNSUPPORTED, reason
            )
            return self._result(
                OptimizeStage.QUANTIZATION,
                json.dumps(
                    {
                        "status": "unsupported",
                        "method": method,
                        "reason": reason,
                        "evidence": reference,
                    },
                    sort_keys=True,
                ),
            )
        if method != "dynamic_int8":
            return self._unsupported(
                OptimizeStage.QUANTIZATION,
                f"first-party Hugging Face quantization method {method!r} is unsupported",
            )
        if self.artifact is None or self.artifact_digest is None or self.reloaded_model is None:
            return self._unsupported(
                OptimizeStage.QUANTIZATION,
                "dynamic-int8 quantization requires a published reloaded physical artifact",
            )

        proof = self._ensure_loaded()
        artifact_dir = _mapping(self.plan.resolved_config, "resolved configuration").get(
            "artifact_dir", "artifacts"
        )
        if not isinstance(artifact_dir, str) or not artifact_dir.strip():
            raise FirstPartyOptimizeRuntimeError("artifact_dir must be a path")
        quantization_root = (
            Path(artifact_dir) / "optimize" / context.run.run_id / "hf-dynamic-int8"
        )
        self._preflight_resources(self.reloaded_model, quantization_root, phase="quantization")
        parent_model = self.reloaded_model
        parent_measurement = proof.measure_model(
            parent_model,
            repetitions=self.plan.quality_profile.evaluation_repetitions,
        )
        quantized_model = copy.deepcopy(parent_model)
        try:
            quantized_model, quantization_report = quantize_huggingface_dynamic_int8(
                quantized_model
            )
        except HuggingFaceQuantizationError as error:
            return self._unsupported(OptimizeStage.QUANTIZATION, str(error))
        quantized_measurement = proof.measure_model(
            quantized_model,
            repetitions=self.plan.quality_profile.evaluation_repetitions,
        )
        parent_perplexity = _number(parent_measurement["perplexity"], "parent.perplexity")
        quantized_perplexity = _number(
            quantized_measurement["perplexity"], "quantized.perplexity"
        )
        allowed = self.plan.quality_profile.max_perplexity_delta
        allowed = 0.05 if allowed is None else allowed
        accepted = (
            quantized_perplexity - parent_perplexity <= allowed
            and quantization_report.storage_delta_bytes < 0
        )
        detail: dict[str, object] = {
            "method": method,
            "status": "accepted" if accepted else "rejected",
            "quantization": quantization_report.to_record(),
            "parent_measurement": dict(parent_measurement),
            "quantized_measurement": dict(quantized_measurement),
            "resource_preflight": self.resource_preflight,
            "artifact": str(self.artifact),
            "artifact_digest": self.artifact_digest,
        }
        if not accepted:
            reason = (
                "quantized quality exceeded the declared limit"
                if quantized_perplexity - parent_perplexity > allowed
                else "quantized storage did not decrease"
            )
            self._record_optimization_observation(
                stage="quantization",
                state_id="rejected:" + self.artifact_digest,
                outcome=OptimizationEvidenceOutcome.REJECTED,
                candidate_id=(None if self.selected is None else self.selected.candidate_id),
                mutation_id=None,
                measurement={
                    "parent": dict(parent_measurement),
                    "quantized": dict(quantized_measurement),
                    "measurement_authority": "physical_quantization_evaluation",
                },
                failure={"reason": reason},
                lineage={"method": method},
            )
            detail["rejection_reason"] = reason
            return self._result(
                OptimizeStage.QUANTIZATION,
                json.dumps(detail, sort_keys=True),
                measured=True,
                constraints_passed=True,
                artifact_digest=self.artifact_digest,
                candidate=self.selected,
                transaction_state=TransactionState.COMMITTED,
            )

        destination = quantization_root / "candidate"
        try:
            quantized_artifact, published_report = publish_huggingface_dynamic_int8(
                quantized_model,
                destination,
                source_weight_bytes=quantization_report.source_weight_bytes,
            )
        except HuggingFaceQuantizationError as error:
            raise FirstPartyOptimizeRuntimeError(str(error)) from error
        tokenizer = proof._tokenizer
        save_tokenizer = getattr(tokenizer, "save_pretrained", None)
        if not callable(save_tokenizer):
            raise FirstPartyOptimizeRuntimeError(
                "quantized Hugging Face publication requires a reloadable tokenizer"
            )
        save_tokenizer(destination)
        reloaded = self._reload(quantized_artifact)
        if not self._generation_smoke(reloaded):
            raise FirstPartyOptimizeRuntimeError(
                "reloaded dynamic-int8 artifact failed inference smoke"
            )
        reloaded_measurement = proof.measure_model(
            reloaded,
            repetitions=self.plan.quality_profile.evaluation_repetitions,
        )
        reloaded_perplexity = _number(
            reloaded_measurement["perplexity"], "reloaded_quantized.perplexity"
        )
        if reloaded_perplexity - parent_perplexity > allowed:
            detail.update(
                {
                    "status": "rejected",
                    "reloaded_measurement": dict(reloaded_measurement),
                    "rejected_artifact": str(quantized_artifact),
                    "rejected_artifact_digest": _artifact_digest(quantized_artifact.parent),
                    "rejection_reason": "reloaded quantized model exceeded the quality limit",
                }
            )
            self._record_optimization_observation(
                stage="quantization",
                state_id="rejected_reload:" + self.artifact_digest,
                outcome=OptimizationEvidenceOutcome.REJECTED,
                candidate_id=(None if self.selected is None else self.selected.candidate_id),
                mutation_id=None,
                measurement={
                    "parent": dict(parent_measurement),
                    "quantized": dict(reloaded_measurement),
                    "measurement_authority": "physical_reloaded_quantization_evaluation",
                },
                artifact={
                    "path": str(quantized_artifact.parent),
                    "digest": detail["rejected_artifact_digest"],
                },
                failure={"reason": detail["rejection_reason"]},
                lineage={"method": method},
            )
            return self._result(
                OptimizeStage.QUANTIZATION,
                json.dumps(detail, sort_keys=True),
                measured=True,
                constraints_passed=True,
                artifact_digest=self.artifact_digest,
                candidate=self.selected,
                transaction_state=TransactionState.COMMITTED,
            )

        self.artifact = quantized_artifact
        self.artifact_digest = _artifact_digest(quantized_artifact.parent)
        self.reloaded_model = reloaded
        detail.update(
            {
                "status": "accepted",
                "artifact": str(quantized_artifact),
                "artifact_digest": self.artifact_digest,
                "reloaded_measurement": dict(reloaded_measurement),
                "published_quantization": published_report.to_record(),
            }
        )
        self._record_optimization_observation(
            stage="quantization",
            state_id="accepted:" + self.artifact_digest,
            outcome=OptimizationEvidenceOutcome.ACCEPTED,
            candidate_id=(None if self.selected is None else self.selected.candidate_id),
            mutation_id=None,
            measurement={
                "parent": dict(parent_measurement),
                "quantized": dict(reloaded_measurement),
                "measurement_authority": "physical_reloaded_quantization_evaluation",
            },
            artifact={
                "path": str(quantized_artifact.parent),
                "digest": self.artifact_digest,
            },
            lineage={"method": method},
        )
        return self._result(
            OptimizeStage.QUANTIZATION,
            json.dumps(detail, sort_keys=True),
            measured=True,
            constraints_passed=True,
            artifact_digest=self.artifact_digest,
            candidate=self.selected,
            transaction_state=TransactionState.COMMITTED,
        )

    def _result(
        self,
        stage: OptimizeStage,
        detail: str,
        *,
        measured: bool = False,
        constraints_passed: bool = False,
        artifact_digest: str | None = None,
        candidate: MutationCandidate | None = None,
        evaluation_id: str | None = None,
        transaction_state: TransactionState | None = None,
        alternatives: tuple[str, ...] = (),
    ) -> StageResult:
        run_key = f"{self.plan.plan_id}:{stage.value}:{detail}"
        evidence = "evidence_" + hashlib.sha256(run_key.encode()).hexdigest()
        return StageResult(
            WorkflowOutcome.SUPPORTED,
            evidence,
            detail,
            measured=measured,
            complete=True,
            constraints_passed=constraints_passed,
            artifact_digest=artifact_digest,
            candidate_id=(
                None
                if candidate is None
                else "candidate_" + candidate.candidate_id.removeprefix("cand_")
            ),
            candidate_state_id=(
                None
                if candidate is None
                else "state_" + hashlib.sha256(candidate.candidate_id.encode()).hexdigest()
            ),
            evaluation_id=evaluation_id,
            transaction_state=transaction_state,
            artifact_immutable=artifact_digest is not None,
            alternatives=alternatives,
        )

    def _unsupported(self, stage: OptimizeStage, detail: str) -> StageResult:
        run_key = f"{self.plan.plan_id}:{stage.value}:{detail}"
        classification = self._failure_classification(detail)
        outcome = (
            OptimizationEvidenceOutcome.UNKNOWN
            if classification == "unknown"
            else OptimizationEvidenceOutcome.UNSUPPORTED
        )
        workflow_outcome = (
            WorkflowOutcome.UNKNOWN
            if classification == "unknown"
            else WorkflowOutcome.UNSUPPORTED
        )
        reference = self._record_runtime_boundary(
            stage, outcome, detail
        )
        return StageResult(
            workflow_outcome,
            ("unknown_" if classification == "unknown" else "unsupported_")
            + hashlib.sha256(run_key.encode()).hexdigest(),
            json.dumps(
                {"status": workflow_outcome.value, "reason": detail, "evidence": reference},
                sort_keys=True,
            ),
            complete=True,
        )

    def _enumerate_for_proof(
        self, proof: HuggingFaceMLPProofRuntime
    ) -> tuple[MutationCandidate, ...]:
        seed = _integer(_config_value(self.plan, "calibration", "seed"), 0, "calibration.seed")
        maximum = max(1, self.plan.budget.evaluations)
        scopes = self._candidate_scopes()
        candidates: list[MutationCandidate] = []
        base_scopes = tuple(scope for scope in scopes if scope is not CandidateScope.LOW_RANK)
        if base_scopes:
            report = enumerate_mutation_candidates(
                proof.component_graph,
                proof.run_id,
                CandidateEnumeratorConfig(
                    seed=seed,
                    filters=CandidateFilter(scopes=base_scopes),
                    max_candidates=maximum,
                ),
            )
            candidates.extend(report.candidates)
        if CandidateScope.LOW_RANK in scopes:
            report = enumerate_mutation_candidates(
                proof.component_graph,
                proof.run_id,
                CandidateEnumeratorConfig(
                    seed=seed,
                    filters=CandidateFilter(
                        scopes=(CandidateScope.COMPONENT,), include_kinds=("projection",)
                    ),
                    max_candidates=maximum,
                ),
            )
            rank = self._low_rank_rank()
            for candidate in report.candidates:
                request = MutationRequest(
                    MutationKind.LOW_RANK,
                    (candidate.component_id,),
                    (
                        ("module_path", str(candidate.component_id)),
                        ("rank", rank),
                    ),
                )
                candidates.append(
                    MutationCandidate(
                        derive_candidate_identity(proof.run_id, request.mutation_id).candidate_id,
                        CandidateScope.LOW_RANK,
                        candidate.component_id,
                        candidate.node_kind,
                        candidate.layer_index,
                        request,
                        candidate.affected_components,
                        candidate.constraint_ids,
                    )
                )
        self.candidates = tuple(sorted(candidates, key=lambda item: item.candidate_id))
        return self.candidates

    def _enumerate(self) -> tuple[MutationCandidate, ...]:
        return self._enumerate_for_proof(self._ensure_loaded())

    @staticmethod
    def _channel(candidate: MutationCandidate) -> int:
        parameters = dict(candidate.request.parameters)
        value = parameters.get("channel_index")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise FirstPartyOptimizeRuntimeError("candidate has no valid MLP channel index")
        return value

    @staticmethod
    def _parameter(candidate: MutationCandidate, name: str) -> int:
        value = dict(candidate.request.parameters).get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise FirstPartyOptimizeRuntimeError(
                f"candidate has no valid {name} parameter"
            )
        return value

    @staticmethod
    def _low_rank_module(candidate: MutationCandidate) -> str:
        value = dict(candidate.request.parameters).get("module_path")
        if not isinstance(value, str) or not value.startswith("model."):
            raise FirstPartyOptimizeRuntimeError("low-rank candidate has no valid module path")
        return value

    @staticmethod
    def _low_rank_candidate_rank(candidate: MutationCandidate) -> int:
        value = dict(candidate.request.parameters).get("rank")
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise FirstPartyOptimizeRuntimeError("low-rank candidate has no valid rank")
        return value

    def _operation_key(
        self, candidate: MutationCandidate, model: Any
    ) -> tuple[str, tuple[int, ...]]:
        scope = candidate.scope
        if scope is CandidateScope.MLP_CHANNEL:
            return scope.value, (self._channel(candidate),)
        if scope is CandidateScope.TRANSFORMER_LAYER:
            return scope.value, (self._parameter(candidate, "layer_index"),)
        if scope is CandidateScope.ATTENTION_HEAD:
            query_heads = getattr(getattr(model, "config", None), "num_attention_heads", None)
            kv_heads = getattr(getattr(model, "config", None), "num_key_value_heads", query_heads)
            if (
                not isinstance(query_heads, int)
                or not isinstance(kv_heads, int)
                or query_heads <= 0
                or kv_heads <= 0
                or query_heads % kv_heads != 0
            ):
                raise FirstPartyOptimizeRuntimeError("attention metadata is not divisible")
            head = self._parameter(candidate, "head_index")
            group_size = query_heads // kv_heads
            group = head // group_size
            return scope.value, tuple(
                range(group * group_size, (group + 1) * group_size)
            )
        if scope is CandidateScope.LOW_RANK:
            return scope.value, (
                int.from_bytes(
                    hashlib.sha256(self._low_rank_module(candidate).encode()).digest()[:8],
                    "big",
                ),
                self._low_rank_candidate_rank(candidate),
            )
        raise FirstPartyOptimizeRuntimeError(f"unsupported candidate scope: {scope.value}")

    def _apply_candidate(self, model: Any, candidate: MutationCandidate) -> object:
        if candidate.scope is CandidateScope.MLP_CHANNEL:
            return remove_huggingface_mlp_channels(model, (self._channel(candidate),))
        if candidate.scope is CandidateScope.ATTENTION_HEAD:
            _, heads = self._operation_key(candidate, model)
            return remove_huggingface_attention_heads(model, heads)
        if candidate.scope is CandidateScope.TRANSFORMER_LAYER:
            return remove_huggingface_transformer_layers(
                model, (self._parameter(candidate, "layer_index"),)
            )
        if candidate.scope is CandidateScope.LOW_RANK:
            from modelsurgeon.adapters.huggingface.low_rank import (
                replace_huggingface_linears_low_rank,
            )

            return replace_huggingface_linears_low_rank(
                model,
                ((self._low_rank_module(candidate), self._low_rank_candidate_rank(candidate)),),
            )
        raise FirstPartyOptimizeRuntimeError(
            f"unsupported candidate scope: {candidate.scope.value}"
        )

    def _structural_feature_partition(
        self, proof: HuggingFaceMLPProofRuntime, candidate: MutationCandidate
    ) -> FeaturePartition:
        layer = candidate.layer_index
        if layer is None:
            raise FirstPartyOptimizeRuntimeError("structural candidate has no layer index")
        if candidate.scope is not CandidateScope.LOW_RANK:
            layer = self._parameter(candidate, "layer_index")
        if candidate.scope is CandidateScope.ATTENTION_HEAD:
            module_path = f"model.layers.{layer}.self_attn"
        elif candidate.scope is CandidateScope.TRANSFORMER_LAYER:
            module_path = f"model.layers.{layer}"
        elif candidate.scope is CandidateScope.LOW_RANK:
            module_path = self._low_rank_module(candidate)
        else:
            raise FirstPartyOptimizeRuntimeError("structural features require a non-MLP candidate")
        module = dict(proof.model.named_modules()).get(module_path)
        if module is None:
            raise FirstPartyOptimizeRuntimeError(f"candidate module is missing: {module_path}")
        parameters = tuple(module.parameters())
        if not parameters:
            raise FirstPartyOptimizeRuntimeError(
                f"candidate module has no parameters: {module_path}"
            )
        parameter_count = sum(int(parameter.numel()) for parameter in parameters)
        l1_norm = sum(
            float(parameter.detach().float().abs().sum().cpu().item())
            for parameter in parameters
        )
        precision = PrecisionProvenance(
            PrecisionSource.HIGH_PRECISION,
            storage_dtype="model_storage",
            compute_dtype="float32",
        )
        records = tuple(
            FeatureRecord(
                component_id=candidate.component_id,
                name=name,
                kind=FeatureKind.SCALAR,
                value=float(value),
                dtype="float32",
                extractor="first_party_structural",
                extractor_version="1",
                precision=precision,
                metadata=(
                    ("module_path", module_path),
                    ("scope", candidate.scope.value),
                ),
            )
            for name, value in (
                ("parameter_count", parameter_count),
                ("weight_l1_norm", l1_norm),
                ("layer_index", layer),
                *(
                    (("rank", self._low_rank_candidate_rank(candidate)),)
                    if candidate.scope is CandidateScope.LOW_RANK
                    else ()
                ),
            )
        )
        key = FeaturePartitionKey(
            model_revision=str(proof._model_target.revision),
            input_revision=str(proof._dataset.revision),
            component_id=candidate.component_id,
            extractor="first_party_structural",
            extractor_version="1",
        )
        checksum = hashlib.sha256(
            json.dumps(
                [item.to_record() for item in records],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()
        return FeaturePartition(key, records, checksum)

    def _prepare_candidate_features(
        self, proof: HuggingFaceMLPProofRuntime, candidate: MutationCandidate
    ) -> None:
        if candidate.scope is CandidateScope.MLP_CHANNEL:
            partitions = proof.pre_mutation_feature_partitions(candidate)
            if len(partitions) != 1:
                raise FirstPartyOptimizeRuntimeError(
                    "HF feature extraction must produce one candidate partition"
                )
            partition = partitions[0]
        else:
            partition = self._structural_feature_partition(proof, candidate)
        self.feature_cache_entries[candidate.candidate_id] = self._publish_feature_partition(
            partition
        )
        self.candidate_features[candidate.candidate_id] = tuple(
            item.to_record() for item in partition.records
        )

    def _measure_candidate(
        self, proof: HuggingFaceMLPProofRuntime, candidate: MutationCandidate
    ) -> Mapping[str, object]:
        if self.baseline_runtime_measurement is None:
            self.baseline_runtime_measurement = proof.measure_model(
                proof.model,
                repetitions=self.plan.quality_profile.evaluation_repetitions,
            )
        candidate_model = copy.deepcopy(proof.model)
        baseline_parameter_count = sum(
            int(parameter.numel()) for parameter in proof.model.parameters()
        )
        baseline_storage_bytes = sum(
            int(parameter.numel()) * int(parameter.element_size())
            for parameter in proof.model.parameters()
        )
        mutation = self._apply_candidate(candidate_model, candidate)
        measured = proof.measure_model(
            candidate_model,
            repetitions=self.plan.quality_profile.evaluation_repetitions,
        )
        candidate_parameter_count = sum(
            int(parameter.numel()) for parameter in candidate_model.parameters()
        )
        candidate_storage_bytes = sum(
            int(parameter.numel()) * int(parameter.element_size())
            for parameter in candidate_model.parameters()
        )
        baseline = self.baseline_runtime_measurement
        return {
            "scope": candidate.scope.value,
            "mutation": (
                mutation.to_record() if hasattr(mutation, "to_record") else str(mutation)
            ),
            "measurement_authority": "physical_model_evaluation",
            "baseline_perplexity": baseline["perplexity"],
            "candidate_perplexity": measured["perplexity"],
            "perplexity_delta": _number(measured["perplexity"], "candidate.perplexity")
            - _number(baseline["perplexity"], "baseline.perplexity"),
            "baseline_median_seconds": baseline["median_seconds"],
            "candidate_median_seconds": measured["median_seconds"],
            "latency_delta_seconds": _number(
                measured["median_seconds"], "candidate.median_seconds"
            )
            - _number(baseline["median_seconds"], "baseline.median_seconds"),
            "baseline_parameter_count": baseline_parameter_count,
            "candidate_parameter_count": candidate_parameter_count,
            "parameter_delta": candidate_parameter_count - baseline_parameter_count,
            "baseline_storage_bytes": baseline_storage_bytes,
            "candidate_storage_bytes": candidate_storage_bytes,
            "storage_delta_bytes": candidate_storage_bytes - baseline_storage_bytes,
            "measurement_wall_seconds": measured["median_seconds"],
            "repetitions": measured["repetitions"],
            "token_count": measured["token_count"],
        }

    @staticmethod
    def _objective_value(
        metric: OptimizeMetric, measurement: Mapping[str, object]
    ) -> tuple[float, float] | None:
        if metric is OptimizeMetric.QUALITY:
            baseline = _number(measurement["baseline_perplexity"], "baseline.perplexity")
            candidate = _number(measurement["candidate_perplexity"], "candidate.perplexity")
            return baseline / candidate, 1.0
        if metric is OptimizeMetric.PERPLEXITY:
            return (
                _number(measurement["candidate_perplexity"], "candidate.perplexity"),
                _number(measurement["baseline_perplexity"], "baseline.perplexity"),
            )
        if metric is OptimizeMetric.PARAMETER_COUNT:
            return (
                _number(measurement["candidate_parameter_count"], "candidate.parameter_count"),
                _number(measurement["baseline_parameter_count"], "baseline.parameter_count"),
            )
        if metric is OptimizeMetric.LATENCY:
            return (
                _number(measurement["candidate_median_seconds"], "candidate.median_seconds"),
                _number(measurement["baseline_median_seconds"], "baseline.median_seconds"),
            )
        return None

    def _build_measured_frontier(
        self,
        candidates: tuple[MutationCandidate, ...],
        measurements: Mapping[str, Mapping[str, object]],
    ) -> Mapping[str, object]:
        from modelsurgeon.search.objectives import ObjectiveObservation, objectives_from_config
        from modelsurgeon.search.pareto import (
            ParetoArchive,
            ParetoCandidate,
            ParetoObjectiveValue,
        )

        root = _mapping(self.plan.resolved_config, "resolved configuration")
        objective_config = ObjectiveConfig.model_validate(
            _mapping(root.get("objective", {}), "objective")
        )
        objective_set = objectives_from_config(objective_config)
        values_by_candidate: dict[str, dict[OptimizeMetric, tuple[float, float]]] = {}
        missing: set[OptimizeMetric] = set()
        for candidate in candidates:
            measurement = measurements.get(candidate.candidate_id)
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
        objective_record = objective_set.to_record()
        if missing:
            result: Mapping[str, object] = {
                "status": "unknown",
                "reason": "declared Pareto objectives require runtime measurements not "
                "available during in-memory candidate evaluation",
                "missing_metrics": sorted(metric.value for metric in missing),
                "objective_set": objective_record,
                "candidate_count": len(values_by_candidate),
            }
            self.measured_frontier = result
            return result
        if not values_by_candidate:
            raise FirstPartyOptimizeRuntimeError(
                "no measured candidate has complete declared Pareto objectives"
            )

        artifact_dir = root.get("artifact_dir", "artifacts")
        if not isinstance(artifact_dir, str) or not artifact_dir.strip():
            raise FirstPartyOptimizeRuntimeError("artifact_dir must be a path")
        campaign_id = self.campaign_run_id or self.plan.plan_id
        archive_path = (
            Path(artifact_dir).expanduser().absolute().resolve(strict=False)
            / "optimize"
            / campaign_id
            / "measured-candidate-frontier.sqlite"
        )
        candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
        with ParetoArchive(archive_path, objective_set) as archive:
            for candidate_id in sorted(values_by_candidate):
                candidate = candidate_by_id[candidate_id]
                measurement = measurements[candidate_id]
                values = values_by_candidate[candidate_id]
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
                            "candidate": candidate.to_record(),
                            "measurement": dict(measurement),
                            "artifact_status": "physical_in_memory_only",
                        },
                    )
                )
            frontier_ids = tuple(
                entry.candidate.candidate_id for entry in archive.entries(frontier_only=True)
            )
        scores = {
            candidate_id: objective_set.score(
                tuple(
                    ObjectiveObservation(metric, value[0], value[1])
                    for metric, value in sorted(
                        values_by_candidate[candidate_id].items(),
                        key=lambda item: item[0].value,
                    )
                )
            ).reward
            for candidate_id in frontier_ids
        }
        preferred_id = sorted(frontier_ids, key=lambda item: (-scores[item], item))[0]
        result = {
            "status": "measured",
            "archive_path": str(archive_path),
            "archive_sha256": _digest_file(archive_path),
            "objective_set": objective_record,
            "candidate_count": len(values_by_candidate),
            "frontier_candidate_ids": list(frontier_ids),
            "preferred_candidate_id": preferred_id,
            "artifact_status": "frontier_candidates_require_reload_and_deployment_validation",
        }
        self.measured_frontier = result
        return result

    def _search(self) -> MutationCandidate:
        proof = self._ensure_loaded()
        artifact_dir = _mapping(self.plan.resolved_config, "resolved configuration").get(
            "artifact_dir", "artifacts"
        )
        if not isinstance(artifact_dir, str) or not artifact_dir.strip():
            raise FirstPartyOptimizeRuntimeError("artifact_dir must be a path")
        self._preflight_resources(
            proof.model,
            Path(artifact_dir),
            phase="active_search",
        )
        candidates = self.candidates or self._enumerate()
        if not candidates:
            raise FirstPartyOptimizeRuntimeError("no adapter-supported candidates exist")
        allowed = self.plan.quality_profile.max_perplexity_delta
        allowed = 0.05 if allowed is None else allowed
        unique: dict[tuple[str, tuple[int, ...]], MutationCandidate] = {}
        for candidate in candidates:
            unique.setdefault(self._operation_key(candidate, proof.model), candidate)
        candidate_pool = tuple(sorted(unique.values(), key=lambda item: item.candidate_id))
        for candidate in candidate_pool:
            self._prepare_candidate_features(proof, candidate)
        meta_order = self._meta_rank(candidate_pool)
        by_id = {candidate.candidate_id: candidate for candidate in candidate_pool}
        evaluation_order = (
            tuple(by_id[candidate_id] for candidate_id in meta_order if candidate_id in by_id)
            if meta_order
            else candidate_pool
        )
        scored: list[tuple[float, float, str, MutationCandidate, Mapping[str, object]]] = []
        for candidate in evaluation_order:
            try:
                measurement = self._measure_candidate(proof, candidate)
            except Exception as error:
                self._record_optimization_observation(
                    stage="active_search",
                    state_id="source:" + (self.source_digest or self.plan.plan_id),
                    outcome=OptimizationEvidenceOutcome.FAILED,
                    candidate_id=candidate.candidate_id,
                    mutation_id=candidate.request.mutation_id,
                    measurement=None,
                    failure={
                        "reason": str(error),
                        "classification": self._failure_classification(error),
                    },
                    feature_records=self.candidate_features.get(candidate.candidate_id, ()),
                    feature_cache=self.feature_cache_entries.get(candidate.candidate_id),
                    lineage={
                        "parent_state": "source",
                        "mutation": candidate.request.to_record(),
                    },
                )
                continue
            self.actual_measurements[candidate.candidate_id] = measurement
            delta = _number(measurement["perplexity_delta"], "perplexity_delta")
            prediction_value = self.meta_predictions.get(candidate.candidate_id)
            prediction = (
                None
                if prediction_value is None
                else {
                    "value": prediction_value,
                    "target": (
                        None
                        if self.meta_guidance is None
                        else self.meta_guidance.get("target")
                    ),
                    "authority": "prediction_only",
                }
            )
            self._record_optimization_observation(
                stage="active_search",
                state_id="source:" + (self.source_digest or self.plan.plan_id),
                outcome=(
                    OptimizationEvidenceOutcome.ACCEPTED
                    if delta <= allowed
                    else OptimizationEvidenceOutcome.REJECTED
                ),
                candidate_id=candidate.candidate_id,
                mutation_id=candidate.request.mutation_id,
                measurement=measurement,
                prediction=prediction,
                feature_records=self.candidate_features.get(candidate.candidate_id, ()),
                feature_cache=self.feature_cache_entries.get(candidate.candidate_id),
                lineage={
                    "parent_state": "source",
                    "mutation": candidate.request.to_record(),
                },
            )
            if delta <= allowed:
                scored.append(
                    (
                        delta,
                        _number(measurement["latency_delta_seconds"], "latency_delta_seconds"),
                        candidate.request.mutation_id,
                        candidate,
                        measurement,
                    )
                )
        if not scored:
            raise FirstPartyOptimizeRuntimeError(
                f"no measured candidate met max perplexity delta {allowed:.12g}"
            )
        ranked = sorted(scored, key=lambda item: (item[0], item[1], item[2]))
        self._build_search_comparison(candidate_pool, ranked)
        _, _, _, candidate, measurement = ranked[0]
        self.selected = candidate
        self.selected_channel = (
            self._channel(candidate)
            if candidate.scope is CandidateScope.MLP_CHANNEL
            else None
        )
        self.selected_measurement = measurement
        self.selected_candidates = tuple(item[3] for item in ranked[: self._surgery_steps()])
        self.selected_channels = tuple(
            self._channel(item)
            for item in self.selected_candidates
            if item.scope is CandidateScope.MLP_CHANNEL
        )
        frontier = self._build_measured_frontier(
            tuple(item[3] for item in scored),
            {item[3].candidate_id: item[4] for item in scored},
        )
        preferred_id = frontier.get("preferred_candidate_id")
        if isinstance(preferred_id, str) and preferred_id in by_id:
            candidate = by_id[preferred_id]
            self.selected = candidate
            self.selected_measurement = self.actual_measurements[preferred_id]
            ordered = [candidate]
            ordered.extend(
                item[3]
                for item in ranked
                if item[3].candidate_id != preferred_id
            )
            self.selected_candidates = tuple(ordered[: self._surgery_steps()])
            self.selected_channel = (
                self._channel(candidate)
                if candidate.scope is CandidateScope.MLP_CHANNEL
                else None
            )
            self.selected_channels = tuple(
                self._channel(item)
                for item in self.selected_candidates
                if item.scope is CandidateScope.MLP_CHANNEL
            )
        return candidate

    def _publish(self, model: Any, destination: Path) -> Path:
        if destination.exists():
            raise FirstPartyOptimizeRuntimeError(
                f"refusing to overwrite candidate artifact: {destination}"
            )
        self._preflight_resources(model, destination.parent, phase="artifact_write")
        low_rank_modules = tuple(
            sorted(
                name
                for name, module in model.named_modules()
                if getattr(module, "_modelsurgeon_low_rank", False)
            )
        )
        files: tuple[Path, ...]
        if low_rank_modules:
            try:
                files = (publish_huggingface_low_rank(model, destination)[0],)
            except HuggingFaceLowRankError as error:
                raise FirstPartyOptimizeRuntimeError(str(error)) from error
        else:
            destination.mkdir(parents=True)
            model.save_pretrained(destination, safe_serialization=True)
            files = tuple(destination.glob("*.safetensors"))
        tokenizer = self._ensure_loaded()._tokenizer
        save_tokenizer = getattr(tokenizer, "save_pretrained", None)
        if not callable(save_tokenizer):
            raise FirstPartyOptimizeRuntimeError(
                "Hugging Face publication requires a reloadable tokenizer"
            )
        save_tokenizer(destination)
        files = tuple(sorted(files))
        if len(files) != 1:
            raise FirstPartyOptimizeRuntimeError(
                "Hugging Face publication must produce exactly one safetensors weight file"
            )
        artifact_bytes = _directory_bytes(destination)
        if artifact_bytes > self.plan.budget.max_artifact_bytes:
            raise FirstPartyOptimizeRuntimeError(
                "published candidate artifact exceeds max_artifact_bytes"
            )
        if self.resource_preflight is not None:
            self.resource_preflight = {
                **self.resource_preflight,
                "published_artifact_bytes": artifact_bytes,
            }
        return files[0]

    def _reload(self, artifact: Path) -> Any:
        low_rank_manifest = artifact.parent / "modelsurgeon-low-rank.json"
        quantization_manifest = artifact.parent / "modelsurgeon-quantization.json"
        if quantization_manifest.is_file():
            trust_remote_code = _config_value(
                self.plan, "safety", "trust_remote_code"
            )
            if not isinstance(trust_remote_code, bool):
                raise FirstPartyOptimizeRuntimeError(
                    "safety.trust_remote_code must be boolean"
                )
            try:
                return load_huggingface_dynamic_int8(
                    artifact.parent, trust_remote_code=trust_remote_code
                )
            except HuggingFaceQuantizationError as error:
                raise FirstPartyOptimizeRuntimeError(str(error)) from error
        if low_rank_manifest.is_file():
            trust_remote_code = _config_value(self.plan, "safety", "trust_remote_code")
            if not isinstance(trust_remote_code, bool):
                raise FirstPartyOptimizeRuntimeError(
                    "safety.trust_remote_code must be boolean"
                )
            try:
                return load_huggingface_low_rank(
                    artifact.parent, trust_remote_code=trust_remote_code
                )
            except HuggingFaceLowRankError as error:
                raise FirstPartyOptimizeRuntimeError(str(error)) from error
        result = load_causal_lm(
            HuggingFaceLoadRequest(
                model=str(artifact.parent),
                revision=str(artifact.parent.resolve()),
                device_map=("cpu" if self.plan.hardware_profile.device == "cpu" else "auto"),
                dtype=_hf_dtype(_config_value(self.plan, "model", "dtype")),
                local_files_only=True,
            )
        )
        result.model.eval()
        return result.model

    @staticmethod
    def _synchronize(torch: Any, model: Any) -> None:
        try:
            device = next(iter(model.parameters())).device
        except StopIteration:
            return
        if getattr(device, "type", None) == "cuda":
            torch.cuda.synchronize(device)

    def _deployment_metrics(
        self,
        loader: Callable[[], Any],
        artifact_path: Path | None,
        *,
        label: str,
    ) -> dict[str, object]:
        """Measure a model load and generation path with retained uncertainty."""

        proof = self._ensure_loaded()
        torch = proof._torch
        load_samples: list[float] = []
        model: Any | None = None
        for index in range(7):
            started = time.perf_counter()
            loaded = loader()
            loaded.eval()
            self._synchronize(torch, loaded)
            load_samples.append(time.perf_counter() - started)
            if model is not None:
                del model
                gc.collect()
            model = loaded
            if index < 6:
                del loaded
                gc.collect()
        if model is None:
            raise FirstPartyOptimizeRuntimeError(f"{label} did not load a model")

        tokenizer = proof._tokenizer
        encoded = tokenizer("ModelSurgeon deployment benchmark", return_tensors="pt")
        embeddings = model.get_input_embeddings()
        input_ids = encoded["input_ids"].to(embeddings.weight.device)
        prompt_tokens = int(input_ids.shape[-1])
        if prompt_tokens <= 0:
            raise FirstPartyOptimizeRuntimeError("deployment benchmark prompt is empty")
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if not isinstance(pad_token_id, int):
            raise FirstPartyOptimizeRuntimeError(
                "deployment benchmark tokenizer has no pad or EOS token"
            )

        def run_generation() -> dict[str, float | int]:
            self._synchronize(torch, model)
            prefill_started = time.perf_counter()
            with torch.inference_mode():
                output = model(input_ids=input_ids, use_cache=True)
            self._synchronize(torch, model)
            prefill_seconds = time.perf_counter() - prefill_started
            del output
            decode_started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    input_ids,
                    do_sample=False,
                    max_new_tokens=8,
                    pad_token_id=pad_token_id,
                    use_cache=True,
                )
            self._synchronize(torch, model)
            decode_seconds = time.perf_counter() - decode_started
            generated_tokens = int(generated.shape[-1]) - prompt_tokens
            del generated
            if generated_tokens <= 0 or prefill_seconds <= 0 or decode_seconds <= 0:
                raise FirstPartyOptimizeRuntimeError(
                    "deployment benchmark produced no positive token/timing sample"
                )
            return {
                "prefill_tokens_per_second": prompt_tokens / prefill_seconds,
                "decode_tokens_per_second": generated_tokens / decode_seconds,
                "latency_seconds": prefill_seconds + decode_seconds,
                "prompt_tokens": prompt_tokens,
                "decode_tokens": generated_tokens,
            }

        for _ in range(2):
            run_generation()
        inference_samples: list[dict[str, float | int]] = []
        peak_ram_samples: list[int] = []
        peak_vram_samples: list[int] = []
        cuda_provider: TorchCudaMemoryProvider | None = None
        try:
            device = next(iter(model.parameters())).device
            if getattr(device, "type", None) == "cuda":
                cuda_provider = TorchCudaMemoryProvider(device=device)
        except (StopIteration, MemoryTelemetryError):
            cuda_provider = None
        for repetition in range(7):
            sample_box: list[dict[str, float | int]] = []

            def operation(box: list[dict[str, float | int]] = sample_box) -> None:
                box.append(run_generation())

            report = collect_memory_telemetry(
                f"hf-deployment-{label}-{repetition}",
                operation,
                MemoryTelemetryConfig(sampling_enabled=True, sample_interval_seconds=0.01),
                cuda=cuda_provider,
            )
            if len(sample_box) != 1:
                raise FirstPartyOptimizeRuntimeError(
                    "deployment benchmark did not produce exactly one inference sample"
                )
            inference_samples.append(sample_box[0])
            if report.peak_rss_bytes is not None:
                peak_ram_samples.append(report.peak_rss_bytes)
            if report.peak_cuda_allocated_bytes is not None:
                peak_vram_samples.append(report.peak_cuda_allocated_bytes)

        result: dict[str, object] = {
            "label": label,
            "load_time_samples": load_samples,
            "load_time_seconds": statistics.median(load_samples),
            "prefill_tokens_per_second_samples": [
                float(item["prefill_tokens_per_second"]) for item in inference_samples
            ],
            "decode_tokens_per_second_samples": [
                float(item["decode_tokens_per_second"]) for item in inference_samples
            ],
            "latency_seconds_samples": [
                float(item["latency_seconds"]) for item in inference_samples
            ],
            "peak_ram_bytes_samples": peak_ram_samples,
            "peak_vram_bytes_samples": peak_vram_samples or None,
            "prompt_tokens": prompt_tokens,
            "decode_tokens": int(
                statistics.median([int(item["decode_tokens"]) for item in inference_samples])
            ),
            "disk_bytes": None if artifact_path is None else _directory_bytes(artifact_path),
        }
        result["prefill_tokens_per_second"] = statistics.median(
            cast(list[float], result["prefill_tokens_per_second_samples"])
        )
        result["decode_tokens_per_second"] = statistics.median(
            cast(list[float], result["decode_tokens_per_second_samples"])
        )
        result["latency_seconds"] = statistics.median(
            cast(list[float], result["latency_seconds_samples"])
        )
        result["peak_ram_bytes"] = max(peak_ram_samples) if peak_ram_samples else None
        result["peak_vram_bytes"] = max(peak_vram_samples) if peak_vram_samples else None
        return result

    def _generation_smoke(self, model: Any) -> bool:
        proof = self._ensure_loaded()
        tokenizer = proof._tokenizer
        encoded = tokenizer("ModelSurgeon", return_tensors="pt", add_special_tokens=True)
        embeddings = model.get_input_embeddings()
        input_device = embeddings.weight.device
        input_ids = encoded["input_ids"].to(input_device)
        with proof._torch.inference_mode():
            output = model(input_ids=input_ids, use_cache=False)
        logits = getattr(output, "logits", None)
        if logits is None:
            return False
        return bool(proof._torch.is_tensor(logits) and logits.ndim == 3 and logits.shape[-1] > 0)

    def _surgery(self, context: StageContext) -> StageResult:
        proof = self._ensure_loaded()
        candidate = self.selected or self._search()
        if not self.selected_candidates:
            self._search()
        if not self.selected_candidates:
            raise FirstPartyOptimizeRuntimeError(
                "active search did not select a physical candidate"
            )
        artifact_dir = _mapping(self.plan.resolved_config, "resolved configuration").get(
            "artifact_dir", "artifacts"
        )
        artifact_root = Path(cast(str, artifact_dir))
        sequence_root = artifact_root / "optimize" / context.run.run_id / "hf-physical-sequence"
        self._preflight_resources(proof.model, sequence_root, phase="physical_surgery")
        original_channels = self.selected_channels
        first_candidate = self.selected_candidates[0]

        def apply_first(
            model: Any, selected_candidate: MutationCandidate = first_candidate
        ) -> object:
            return self._apply_candidate(model, selected_candidate)

        edits = (
            HuggingFaceEdit(
                f"{first_candidate.scope.value}-{first_candidate.request.mutation_id[:12]}",
                f"remove_{first_candidate.scope.value}",
                apply_first,
            ),
        )
        try:
            sequence = run_huggingface_cumulative_sequence(
                copy.deepcopy(proof.model),
                edits,
                output_root=sequence_root,
                source_outcome_id=self.source_digest or self.plan.plan_id,
                publish=self._publish,
                reload=self._reload,
                generate=self._generation_smoke,
                evaluate=self._evaluate_reloaded_child,
                on_accept=self._next_state_edit,
                max_stages=self._surgery_steps(),
            )
        except Exception as error:
            candidate_id = self._stage_candidate_id(0)
            self._record_optimization_observation(
                stage="physical_surgery",
                state_id="failed:" + (self.source_digest or self.plan.plan_id),
                outcome=OptimizationEvidenceOutcome.FAILED,
                candidate_id=candidate_id,
                mutation_id=(
                    None
                    if candidate_id is None
                    else self.selected_candidates[0].request.mutation_id
                ),
                measurement=None,
                failure={
                    "failed_index": 0,
                    "reason": str(error),
                    "classification": self._failure_classification(error),
                },
                feature_records=(
                    ()
                    if candidate_id is None
                    else self.candidate_features.get(candidate_id, ())
                ),
                feature_cache=self._stage_feature_cache(0, candidate_id),
                lineage={"sequence_root": str(sequence_root)},
            )
            raise
        self.surgery_sequence = sequence
        evidence_observations = self._record_surgery_observations(sequence)
        artifact = sequence.stages[-1].artifact
        reloaded = sequence.final_model
        self.artifact = artifact
        self.artifact_digest = _artifact_digest(artifact.parent)
        self.reloaded_model = reloaded
        return self._result(
            OptimizeStage.SURGERY,
            json.dumps(
                {
                    "operation": f"cumulative_remove_{first_candidate.scope.value}",
                    "candidate_scopes": [
                        item.scope.value for item in self.selected_candidates
                    ],
                    "channels": list(original_channels),
                    "stages": [item.to_record() for item in sequence.stages],
                    "failed_index": sequence.failed_index,
                    "failure_reason": sequence.failure_reason,
                    "failed_mutation_id": sequence.failed_mutation_id,
                    "failed_evaluation": sequence.failed_evaluation,
                    "state_updates": list(self.state_updates),
                    "evidence_observations": list(evidence_observations),
                    "resource_preflight": self.resource_preflight,
                    "source_digest": self.source_digest,
                    "artifact": str(artifact),
                    "artifact_manifest": _tree_entries(artifact.parent),
                },
                sort_keys=True,
            ),
            measured=True,
            constraints_passed=True,
            artifact_digest=self.artifact_digest,
            candidate=candidate,
            transaction_state=TransactionState.COMMITTED,
        )

    def _evaluate_reloaded_child(self, model: Any) -> Mapping[str, object]:
        proof = self._ensure_loaded()
        baseline = self.baseline or proof.baseline_measurement()
        measurement = proof.measure_model(
            model,
            repetitions=self.plan.quality_profile.evaluation_repetitions,
        )
        baseline_perplexity = _number(baseline["perplexity"], "baseline.perplexity")
        perplexity = _number(measurement["perplexity"], "child.perplexity")
        delta = perplexity - baseline_perplexity
        allowed = self.plan.quality_profile.max_perplexity_delta
        allowed = 0.05 if allowed is None else allowed
        return {
            "measurement": dict(measurement),
            "baseline_perplexity": baseline_perplexity,
            "perplexity_delta": delta,
            "max_perplexity_delta": allowed,
            "accepted": delta <= allowed,
            "measurement_authority": "physical_reloaded_child",
        }

    def _rediscover_child_states(
        self, stages: tuple[HuggingFaceStageEvidence, ...]
    ) -> tuple[Mapping[str, object], ...]:
        updates: list[Mapping[str, object]] = []
        for stage in stages:
            artifact = Path(stage.artifact).parent.absolute().resolve(strict=True)
            child_proof = self._new_proof_runtime(
                str(artifact),
                str(artifact),
                local_files_only=True,
            )
            child_candidates = self._enumerate_for_proof(child_proof)
            state_candidates: list[dict[str, object]] = []
            for candidate in child_candidates:
                self._prepare_candidate_features(child_proof, candidate)
                state_candidates.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "component_id": str(candidate.component_id),
                        "scope": candidate.scope.value,
                        "features": list(self.candidate_features[candidate.candidate_id]),
                        "cache": self.feature_cache_entries[candidate.candidate_id],
                    }
                )
            shape = child_proof._discovery.shape
            updates.append(
                {
                    "stage_index": stage.index,
                    "artifact": str(artifact),
                    "artifact_digest": stage.outcome.artifact.digest,
                    "model_revision": str(artifact),
                    "architecture": {
                        "layers": shape.layers,
                        "attention_heads": shape.attention_heads,
                        "kv_heads": shape.kv_heads,
                        "intermediate_size": shape.intermediate_size,
                    },
                    "baseline": child_proof.baseline_measurement(),
                    "candidate_count": len(state_candidates),
                    "candidates": state_candidates,
                    "state_authority": "reloaded_child_runtime",
                }
            )
        return tuple(updates)

    def _next_state_edit(
        self,
        model: Any,
        stage_index: int,
        stages: tuple[HuggingFaceStageEvidence, ...],
    ) -> HuggingFaceEdit | None:
        del model, stage_index
        if not stages:
            raise FirstPartyOptimizeRuntimeError(
                "state-conditioned search requires an accepted stage"
            )
        state = dict(self._rediscover_child_states((stages[-1],))[0])
        artifact = Path(stages[-1].artifact).parent.absolute().resolve(strict=True)
        child_proof = self._new_proof_runtime(
            str(artifact),
            str(artifact),
            local_files_only=True,
        )
        child_candidates = self._enumerate_for_proof(child_proof)
        allowed = self.plan.quality_profile.max_perplexity_delta
        allowed = 0.05 if allowed is None else allowed
        measurements: list[
            tuple[float, float, str, MutationCandidate, Mapping[str, object]]
        ] = []
        failed_measurements: list[dict[str, object]] = []
        for candidate in child_candidates:
            try:
                measured = self._measure_candidate(child_proof, candidate)
            except Exception as error:
                failed_measurements.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "scope": candidate.scope.value,
                        "perplexity_delta": None,
                        "latency_delta_seconds": None,
                        "accepted_by_quality_gate": False,
                        "failure": {
                            "reason": str(error),
                            "classification": self._failure_classification(error),
                        },
                    }
                )
                continue
            delta = _number(measured["perplexity_delta"], "state.perplexity_delta")
            latency = _number(measured["latency_delta_seconds"], "state.latency_delta_seconds")
            measurements.append(
                (delta, latency, candidate.request.mutation_id, candidate, measured)
            )
        eligible = [item for item in measurements if item[0] <= allowed]
        ranked = sorted(eligible, key=lambda item: (item[0], item[1], item[2]))
        state["candidate_measurements"] = [
            {
                "candidate_id": item[3].candidate_id,
                "scope": item[3].scope.value,
                **(
                    {"channel": self._channel(item[3])}
                    if item[3].scope is CandidateScope.MLP_CHANNEL
                    else {}
                ),
                "perplexity_delta": item[0],
                "latency_delta_seconds": item[1],
                "accepted_by_quality_gate": item in eligible,
            }
            for item in sorted(measurements, key=lambda item: item[3].candidate_id)
        ] + sorted(failed_measurements, key=lambda item: str(item["candidate_id"]))
        if not ranked:
            state["next_edit"] = None
            state["stopping_reason"] = "no state-conditioned candidate met the quality gate"
            self.state_updates = (*self.state_updates, state)
            return None
        selected = ranked[0][3]
        selected_identifier = selected.request.mutation_id
        state["next_edit"] = {
            "candidate_id": selected.candidate_id,
            "scope": selected.scope.value,
            "mutation_id": selected_identifier,
            **(
                {"channel": self._channel(selected)}
                if selected.scope is CandidateScope.MLP_CHANNEL
                else {}
            ),
            "selection_authority": "reloaded_child_physical_evaluation",
        }
        self.state_updates = (*self.state_updates, state)

        def apply_next(
            next_model: Any, selected_candidate: MutationCandidate = selected
        ) -> object:
            return self._apply_candidate(next_model, selected_candidate)

        return HuggingFaceEdit(
            (
                f"state-mlp-channel-{self._channel(selected)}"
                if selected.scope is CandidateScope.MLP_CHANNEL
                else f"state-{selected.scope.value}-{selected_identifier[:12]}"
            ),
            f"remove_{selected.scope.value}_from_rediscovered_state",
            apply_next,
        )

    def _stage_candidate_id(self, index: int) -> str | None:
        if index == 0 and self.selected_candidates:
            return self.selected_candidates[0].candidate_id
        state_index = index - 1
        if 0 <= state_index < len(self.state_updates):
            next_edit = self.state_updates[state_index].get("next_edit")
            if isinstance(next_edit, Mapping):
                candidate_id = next_edit.get("candidate_id")
                if isinstance(candidate_id, str) and candidate_id:
                    return candidate_id
        return None

    def _stage_feature_cache(
        self, index: int, candidate_id: str | None
    ) -> Mapping[str, object] | None:
        if candidate_id is None:
            return None
        cached = self.feature_cache_entries.get(candidate_id)
        if cached is not None:
            return cached
        state_index = index - 1
        if 0 <= state_index < len(self.state_updates):
            candidates = self.state_updates[state_index].get("candidates", ())
            if not isinstance(candidates, (list, tuple)):
                return None
            for item in candidates:
                if not isinstance(item, Mapping):
                    continue
                if item.get("candidate_id") == candidate_id:
                    cache = item.get("cache")
                    if isinstance(cache, Mapping):
                        return cache
        return None

    def _record_surgery_observations(
        self, sequence: HuggingFaceCumulativeRun
    ) -> tuple[Mapping[str, object], ...]:
        references: list[Mapping[str, object]] = []
        for stage in sequence.stages:
            candidate_id = self._stage_candidate_id(stage.index)
            references.append(
                self._record_optimization_observation(
                    stage="physical_surgery",
                    state_id="accepted_child:" + stage.outcome.outcome_id,
                    outcome=OptimizationEvidenceOutcome.ACCEPTED,
                    candidate_id=candidate_id,
                    mutation_id=stage.mutation_id,
                    measurement=stage.evaluation,
                    prediction=(
                        None
                        if candidate_id is None or candidate_id not in self.meta_predictions
                        else {
                            "value": self.meta_predictions[candidate_id],
                            "authority": "prediction_only",
                        }
                    ),
                    artifact={
                        "path": str(stage.artifact.parent),
                        "digest": stage.outcome.artifact.digest,
                        "size_bytes": stage.outcome.artifact.size_bytes,
                        "reloadable": stage.reloadable,
                        "generation_smoke": stage.generation_smoke,
                    },
                    feature_records=(
                        ()
                        if candidate_id is None
                        else self.candidate_features.get(candidate_id, ())
                    ),
                    feature_cache=self._stage_feature_cache(stage.index, candidate_id),
                    lineage={
                        "sequence_id": sequence.sequence_id,
                        "stage_index": stage.index,
                        "parent_outcome_id": stage.outcome.parent_outcome_id,
                        "outcome_id": stage.outcome.outcome_id,
                    },
                )
            )
        if sequence.failed_index is not None:
            failed_index = sequence.failed_index
            candidate_id = self._stage_candidate_id(failed_index)
            references.append(
                self._record_optimization_observation(
                    stage="physical_surgery",
                    state_id="rolled_back:" + sequence.sequence_id,
                    outcome=(
                        OptimizationEvidenceOutcome.ROLLED_BACK
                        if sequence.failed_evaluation is not None
                        else OptimizationEvidenceOutcome.FAILED
                    ),
                    candidate_id=candidate_id,
                    mutation_id=sequence.failed_mutation_id,
                    measurement=sequence.failed_evaluation,
                    failure={
                        "failed_index": failed_index,
                        "reason": sequence.failure_reason,
                        "classification": self._failure_classification(
                            sequence.failure_reason or ""
                        ),
                    },
                    feature_records=(
                        ()
                        if candidate_id is None
                        else self.candidate_features.get(candidate_id, ())
                    ),
                    feature_cache=self._stage_feature_cache(failed_index, candidate_id),
                    lineage={
                        "sequence_id": sequence.sequence_id,
                        "failed_index": failed_index,
                    },
                )
            )
        return tuple(references)

    def run_stage(self, context: StageContext) -> StageResult:
        stage = context.stage
        try:
            self._hydrate(context)
            if stage is OptimizeStage.PROFILE:
                proof = self._ensure_loaded()
                return self._result(
                    stage,
                    json.dumps(
                        {
                            "model": proof._model_target.to_record(),
                            "hardware_inventory": dict(self._runtime_hardware_record()),
                            "source_digest": self.source_digest,
                        },
                        sort_keys=True,
                    ),
                    measured=True,
                    constraints_passed=True,
                )
            if stage is OptimizeStage.CAPABILITY:
                self._ensure_loaded()
                return self._result(
                    stage,
                    json.dumps(
                        {
                            "supported_scopes": sorted(
                                scope.value for scope in self._candidate_scopes()
                            ),
                            "physical_measurements_authoritative": True,
                        },
                        sort_keys=True,
                    ),
                )
            if stage is OptimizeStage.BASELINE:
                self.baseline = self._ensure_loaded().baseline_measurement()
                return self._result(
                    stage,
                    json.dumps(
                        {
                            "baseline": dict(self.baseline),
                            "hardware_inventory": dict(self._runtime_hardware_record()),
                        },
                        sort_keys=True,
                    ),
                    measured=True,
                    constraints_passed=True,
                )
            if stage is OptimizeStage.CANDIDATE_GENERATION:
                candidates = self._enumerate()
                return self._result(
                    stage,
                    json.dumps(
                        {
                            "candidate_count": len(candidates),
                            "scopes": sorted({item.scope.value for item in candidates}),
                            "source": "component_graph",
                        },
                        sort_keys=True,
                    ),
                    measured=True,
                    constraints_passed=bool(candidates),
                    alternatives=tuple(sorted(item.candidate_id for item in candidates[:8])),
                )
            if stage is OptimizeStage.ACTIVE_SEARCH:
                candidate = self._search()
                assert self.selected_measurement is not None
                evaluation_id = (
                    "evaluation_"
                    + hashlib.sha256(
                        json.dumps(dict(self.selected_measurement), sort_keys=True).encode()
                    ).hexdigest()
                )
                return self._result(
                    stage,
                    json.dumps(
                        {
                            "candidate_id": candidate.candidate_id,
                            "candidate_scope": candidate.scope.value,
                            "selected_candidate_ids": [
                                item.candidate_id for item in self.selected_candidates
                            ],
                            "channels": list(self.selected_channels),
                            "feature_evidence": [
                                {
                                    "candidate_id": candidate_id,
                                    "features": list(features),
                                    "cache": self.feature_cache_entries.get(candidate_id),
                                }
                                for candidate_id, features in sorted(
                                    self.candidate_features.items()
                                )
                            ],
                            "feature_cache_root": str(self._feature_cache_root()),
                            "meta_guidance": self.meta_guidance,
                            "search_comparison": self.search_comparison,
                            "measured_frontier": self.measured_frontier,
                            "resource_preflight": self.resource_preflight,
                            "measurement": dict(self.selected_measurement),
                            "evidence_observations": [
                                reference
                                for _, reference in sorted(
                                    self.search_evidence_publications.items()
                                )
                            ],
                        },
                        sort_keys=True,
                    ),
                    measured=True,
                    constraints_passed=True,
                    candidate=candidate,
                    evaluation_id=evaluation_id,
                    alternatives=tuple(
                        sorted(
                            item.candidate_id for item in self.candidates if item != candidate
                        )[:8]
                    ),
                )
            if stage is OptimizeStage.SURGERY:
                return self._surgery(context)
            if stage is OptimizeStage.REPAIR:
                return self._repair(context)
            if stage is OptimizeStage.QUANTIZATION:
                return self._quantization(context)
            if stage is OptimizeStage.DEPLOYMENT_BENCHMARK:
                if self.reloaded_model is None or self.artifact_digest is None:
                    return self._unsupported(
                        stage, "deployment benchmark requires a published reloaded artifact"
                    )
                proof = self._ensure_loaded()
                candidate_quality = proof.measure_model(
                    self.reloaded_model,
                    repetitions=self.plan.quality_profile.evaluation_repetitions,
                )
                baseline = self.baseline or proof.baseline_measurement()
                model_source = self._model_source()
                revision = _config_value(self.plan, "model", "revision")
                if not isinstance(revision, str) or not revision.strip():
                    return self._unsupported(
                        stage, "deployment benchmark requires a pinned model revision"
                    )
                source_path = Path(model_source).expanduser().absolute().resolve(strict=False)
                device_map = "cpu" if self.plan.hardware_profile.device == "cpu" else "auto"

                def load_source() -> Any:
                    return load_causal_lm(
                        HuggingFaceLoadRequest(
                            model=model_source,
                            revision=revision,
                            device_map=device_map,
                            dtype=_hf_dtype(_config_value(self.plan, "model", "dtype")),
                            local_files_only=source_path.exists(),
                        )
                    ).model

                self.deployment_baseline = self._deployment_metrics(
                    load_source,
                    source_path if source_path.exists() else None,
                    label="baseline",
                )
                assert self.artifact is not None
                artifact = self.artifact
                self.deployment = self._deployment_metrics(
                    lambda: self._reload(artifact),
                    artifact.parent,
                    label="candidate",
                )
                quality_delta = _number(
                    candidate_quality["perplexity"], "deployment.perplexity"
                ) - _number(baseline["perplexity"], "baseline.perplexity")
                allowed = self.plan.quality_profile.max_perplexity_delta or 0.05
                constraints = _mapping(
                    _mapping(self.plan.resolved_config, "resolved configuration").get(
                        "constraints"
                    ),
                    "constraints",
                )
                candidate_latency = _number(
                    self.deployment["latency_seconds"], "candidate.latency_seconds"
                )
                baseline_latency = _number(
                    self.deployment_baseline["latency_seconds"],
                    "baseline.latency_seconds",
                )
                latency_gain = (
                    0.0
                    if baseline_latency <= 0
                    else (baseline_latency - candidate_latency) / baseline_latency
                )
                max_ram = constraints.get("max_ram_bytes")
                max_vram = constraints.get("max_vram_bytes")
                min_latency_gain = constraints.get("min_latency_gain_ratio")
                passed = quality_delta <= allowed
                peak_ram = self.deployment["peak_ram_bytes"]
                if isinstance(max_ram, int) and (
                    not isinstance(peak_ram, int) or peak_ram > max_ram
                ):
                    passed = False
                peak_vram = self.deployment["peak_vram_bytes"]
                if isinstance(max_vram, int) and (
                    not isinstance(peak_vram, int) or peak_vram > max_vram
                ):
                    passed = False
                if isinstance(min_latency_gain, (int, float)) and latency_gain < float(
                    min_latency_gain
                ):
                    passed = False
                disk_bytes = self.deployment["disk_bytes"]
                if (
                    not isinstance(disk_bytes, int)
                    or disk_bytes > self.plan.budget.max_artifact_bytes
                ):
                    passed = False
                self.deployment_constraints_passed = passed
                deployment_evidence = {
                    "baseline": dict(self.deployment_baseline),
                    "candidate": dict(self.deployment),
                    "candidate_quality": dict(candidate_quality),
                    "hardware_inventory": dict(self._runtime_hardware_record()),
                    "quality_delta": quality_delta,
                    "latency_gain_ratio": latency_gain,
                }
                return self._result(
                    stage,
                    json.dumps(deployment_evidence, sort_keys=True),
                    measured=True,
                    constraints_passed=passed,
                    artifact_digest=self.artifact_digest,
                    candidate=self.selected,
                    evaluation_id="evaluation_"
                    + hashlib.sha256(
                        json.dumps(dict(self.deployment), sort_keys=True).encode()
                    ).hexdigest(),
                    transaction_state=TransactionState.COMMITTED,
                )
            if stage is OptimizeStage.PARETO:
                if (
                    self.artifact_digest is None
                    or self.selected is None
                    or self.selected_measurement is None
                ):
                    return self._unsupported(
                        stage, "no measured feasible artifact is available for Pareto publication"
                    )
                if self.deployment_constraints_passed is not True:
                    return self._unsupported(
                        stage,
                        "deployment measurements did not satisfy every declared hard constraint",
                    )
                if self.deployment is None or self.deployment_baseline is None:
                    return self._unsupported(
                        stage, "Pareto publication requires retained deployment evidence"
                    )
                if (
                    self.measured_frontier is None
                    or self.measured_frontier.get("status") != "measured"
                ):
                    return self._unsupported(
                        stage,
                        "declared Pareto objectives were not fully measured for candidate "
                        "selection",
                    )
                frontier_ids_raw = (
                    ()
                    if self.measured_frontier is None
                    else self.measured_frontier.get("frontier_candidate_ids", ())
                )
                frontier_ids = (
                    tuple(item for item in frontier_ids_raw if isinstance(item, str))
                    if isinstance(frontier_ids_raw, (list, tuple))
                    else ()
                )
                return self._result(
                    stage,
                    json.dumps(
                        {
                            "selection": "one measured feasible HF artifact selected from "
                            "the declared-objective candidate frontier",
                            "baseline_deployment": dict(self.deployment_baseline),
                            "candidate_deployment": dict(self.deployment),
                            "measured_frontier": self.measured_frontier,
                        },
                        sort_keys=True,
                    ),
                    measured=True,
                    constraints_passed=True,
                    artifact_digest=self.artifact_digest,
                    candidate=self.selected,
                    evaluation_id="evaluation_"
                    + hashlib.sha256(
                        json.dumps(dict(self.selected_measurement), sort_keys=True).encode()
                    ).hexdigest(),
                    transaction_state=TransactionState.COMMITTED,
                    alternatives=tuple(
                        item
                        for item in frontier_ids
                        if self.selected is not None and item != self.selected.candidate_id
                    ),
                )
            if stage is OptimizeStage.REPORT:
                return self._result(
                    stage, "first-party evidence report retained in the optimize run state"
                )
            raise FirstPartyOptimizeRuntimeError(f"unsupported optimize stage: {stage.value}")
        except FirstPartyOptimizeRuntimeError as error:
            return self._unsupported(stage, str(error))
        except Exception as error:  # durable failure evidence is safer than an implicit claim
            reference = self._record_runtime_boundary(
                stage,
                OptimizationEvidenceOutcome.FAILED,
                str(error),
                error=error,
            )
            return StageResult(
                WorkflowOutcome.FAILED,
                "failure_"
                + hashlib.sha256(f"{self.plan.plan_id}:{stage.value}:{error}".encode()).hexdigest(),
                json.dumps(
                    {
                        "status": "failed",
                        "reason": f"first-party runtime failed during {stage.value}: {error}",
                        "evidence": reference,
                    },
                    sort_keys=True,
                ),
                complete=True,
            )


def build_first_party_optimize_runtime(plan: OptimizePlan) -> OptimizeRuntime:
    """Build the default runtime selected by the public optimize command."""

    return HuggingFaceOptimizeRuntime(plan)


__all__ = ["HuggingFaceOptimizeRuntime", "build_first_party_optimize_runtime"]
