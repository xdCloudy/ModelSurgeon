"""First-party execution boundary for the verified Hugging Face optimize cell.

This module intentionally supports one narrow, real path first: local or pinned
Hugging Face causal LMs exposing the adapter-defined gated MLP layout.  It uses
the existing proof runtime for measurement and the existing physical surgery
and reload contracts for publication.  Other formats and operations return an
explicit unsupported outcome instead of being represented by placeholder data.
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
from modelsurgeon.adapters.huggingface.physical_mlp import (
    remove_huggingface_mlp_channels,
)
from modelsurgeon.adapters.huggingface.proof_runtime import (
    HuggingFaceMLPProofConfig,
    HuggingFaceMLPProofRuntime,
)
from modelsurgeon.experiments.candidates import (
    CandidateEnumeratorConfig,
    CandidateFilter,
    CandidateScope,
    MutationCandidate,
    enumerate_mutation_candidates,
)
from modelsurgeon.features.cache import FeaturePartition, FeaturePartitionCache
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
from modelsurgeon.surgery.contracts import TransactionState
from modelsurgeon.surgery.huggingface_sequence import (
    HuggingFaceCumulativeRun,
    HuggingFaceEdit,
    run_huggingface_cumulative_sequence,
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
    """Execute one real, bounded HF MLP-channel optimization workflow."""

    def __init__(self, plan: OptimizePlan) -> None:
        self.plan = plan
        self.proof: HuggingFaceMLPProofRuntime | None = None
        self.source_digest: str | None = None
        self.candidates: tuple[MutationCandidate, ...] = ()
        self.selected: MutationCandidate | None = None
        self.selected_channel: int | None = None
        self.selected_measurement: Mapping[str, object] | None = None
        self.selected_channels: tuple[int, ...] = ()
        self.selected_candidates: tuple[MutationCandidate, ...] = ()
        self.candidate_features: dict[str, tuple[dict[str, object], ...]] = {}
        self.feature_cache_entries: dict[str, Mapping[str, object]] = {}
        self.actual_measurements: dict[str, Mapping[str, object]] = {}
        self.meta_guidance: Mapping[str, object] | None = None
        self.meta_predictions: dict[str, float] = {}
        self.meta_order: tuple[str, ...] = ()
        self.search_comparison: Mapping[str, object] | None = None
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
                    self.selected_channel = self._channel(self.selected)
                    self.selected_channels = (self.selected_channel,)
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

    def _build_search_comparison(
        self,
        candidates: tuple[MutationCandidate, ...],
        ranked: list[tuple[float, float, int, MutationCandidate, Mapping[str, object]]],
    ) -> None:
        from modelsurgeon.surgeon.ranking import rank_random

        actual = {item[3].candidate_id: item[0] for item in ranked}
        best_delta = ranked[0][0]
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
        calibration = self._calibration_text()
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
        self.proof = HuggingFaceMLPProofRuntime(
            HuggingFaceMLPProofConfig(
                model=model,
                calibration_text=calibration,
                revision=revision,
                device_map="cpu" if self.plan.hardware_profile.device == "cpu" else "auto",
                dtype=_hf_dtype(_config_value(self.plan, "model", "dtype")),
                trust_remote_code=False,
                local_files_only=source_path.exists(),
                sequence_length=min(sequence_length, 2048),
                max_tokens=requested_tokens,
                safe_perplexity_delta=self.plan.quality_profile.max_perplexity_delta or 0.05,
                seed=_integer(calibration_config.get("seed"), 0, "calibration.seed"),
            )
        )
        return self.proof

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
        return StageResult(
            WorkflowOutcome.UNSUPPORTED,
            "unsupported_" + hashlib.sha256(run_key.encode()).hexdigest(),
            detail,
            complete=True,
        )

    def _enumerate(self) -> tuple[MutationCandidate, ...]:
        proof = self._ensure_loaded()
        report = enumerate_mutation_candidates(
            proof.component_graph,
            proof.run_id,
            CandidateEnumeratorConfig(
                seed=_integer(
                    _config_value(self.plan, "calibration", "seed"), 0, "calibration.seed"
                ),
                filters=CandidateFilter(scopes=(CandidateScope.MLP_CHANNEL,)),
                max_candidates=max(1, self.plan.budget.evaluations),
            ),
        )
        self.candidates = report.candidates
        return self.candidates

    @staticmethod
    def _channel(candidate: MutationCandidate) -> int:
        parameters = dict(candidate.request.parameters)
        value = parameters.get("channel_index")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise FirstPartyOptimizeRuntimeError("candidate has no valid MLP channel index")
        return value

    def _search(self) -> MutationCandidate:
        proof = self._ensure_loaded()
        candidates = self.candidates or self._enumerate()
        if not candidates:
            raise FirstPartyOptimizeRuntimeError(
                "no adapter-supported MLP channel candidates exist"
            )
        allowed = self.plan.quality_profile.max_perplexity_delta
        allowed = 0.05 if allowed is None else allowed
        unique: dict[int, MutationCandidate] = {}
        for candidate in candidates:
            unique.setdefault(self._channel(candidate), candidate)
        candidate_pool = tuple(sorted(unique.values(), key=lambda item: item.candidate_id))
        for candidate in candidate_pool:
            partitions = proof.pre_mutation_feature_partitions(candidate)
            if len(partitions) != 1:
                raise FirstPartyOptimizeRuntimeError(
                    "HF feature extraction must produce one candidate partition"
                )
            partition = partitions[0]
            self.feature_cache_entries[candidate.candidate_id] = (
                self._publish_feature_partition(partition)
            )
            self.candidate_features[candidate.candidate_id] = tuple(
                item.to_record() for item in partitions[0].records
            )
        meta_order = self._meta_rank(candidate_pool)
        by_id = {candidate.candidate_id: candidate for candidate in candidate_pool}
        evaluation_order = (
            tuple(by_id[candidate_id] for candidate_id in meta_order if candidate_id in by_id)
            if meta_order
            else candidate_pool
        )
        scored: list[tuple[float, float, int, MutationCandidate, Mapping[str, object]]] = []
        for candidate in evaluation_order:
            channel = self._channel(candidate)
            measurement: Mapping[str, object] = proof.measure_channel_set(
                tuple((layer, channel) for layer in range(proof._discovery.shape.layers)),
                repetitions=self.plan.quality_profile.evaluation_repetitions,
            ).to_record()
            self.actual_measurements[candidate.candidate_id] = measurement
            delta = _number(measurement["perplexity_delta"], "perplexity_delta")
            if delta <= allowed:
                scored.append(
                    (
                        delta,
                        _number(measurement["latency_delta_seconds"], "latency_delta_seconds"),
                        channel,
                        candidate,
                        measurement,
                    )
                )
        if not scored:
            raise FirstPartyOptimizeRuntimeError(
                f"no measured channel met max perplexity delta {allowed:.12g}"
            )
        ranked = sorted(scored, key=lambda item: (item[0], item[1], item[2]))
        self._build_search_comparison(candidate_pool, ranked)
        _, _, channel, candidate, measurement = ranked[0]
        self.selected = candidate
        self.selected_channel = channel
        self.selected_measurement = measurement
        self.selected_candidates = tuple(item[3] for item in ranked[: self._surgery_steps()])
        self.selected_channels = tuple(self._channel(item) for item in self.selected_candidates)
        return candidate

    def _publish(self, model: Any, destination: Path) -> Path:
        if destination.exists():
            raise FirstPartyOptimizeRuntimeError(
                f"refusing to overwrite candidate artifact: {destination}"
            )
        destination.mkdir(parents=True)
        model.save_pretrained(destination, safe_serialization=True)
        files = sorted(destination.glob("*.safetensors"))
        if len(files) != 1:
            raise FirstPartyOptimizeRuntimeError(
                "Hugging Face publication must produce exactly one safetensors weight file"
            )
        return files[0]

    def _reload(self, artifact: Path) -> Any:
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
        if not self.selected_channels:
            raise FirstPartyOptimizeRuntimeError("active search did not select a physical channel")
        artifact_dir = _mapping(self.plan.resolved_config, "resolved configuration").get(
            "artifact_dir", "artifacts"
        )
        artifact_root = Path(cast(str, artifact_dir))
        sequence_root = artifact_root / "optimize" / context.run.run_id / "hf-mlp-sequence"
        original_channels = self.selected_channels
        edits: list[HuggingFaceEdit] = []
        for index, original_channel in enumerate(original_channels):
            current_channel = original_channel - sum(
                prior < original_channel for prior in original_channels[:index]
            )

            def apply_edit(model: Any, channel: int = current_channel) -> object:
                return remove_huggingface_mlp_channels(model, (channel,))

            edits.append(
                HuggingFaceEdit(
                    f"mlp-channel-{original_channel}",
                    "remove_mlp_channel",
                    apply_edit,
                )
            )
        sequence = run_huggingface_cumulative_sequence(
            copy.deepcopy(proof.model),
            tuple(edits),
            output_root=sequence_root,
            source_outcome_id=self.source_digest or self.plan.plan_id,
            publish=self._publish,
            reload=self._reload,
            generate=self._generation_smoke,
            evaluate=self._evaluate_reloaded_child,
        )
        self.surgery_sequence = sequence
        artifact = sequence.stages[-1].artifact
        reloaded = sequence.final_model
        self.artifact = artifact
        self.artifact_digest = _artifact_digest(artifact.parent)
        self.reloaded_model = reloaded
        return self._result(
            OptimizeStage.SURGERY,
            json.dumps(
                {
                    "operation": "cumulative_remove_mlp_channels",
                    "channels": list(original_channels),
                    "stages": [item.to_record() for item in sequence.stages],
                    "failed_index": sequence.failed_index,
                    "failure_reason": sequence.failure_reason,
                    "failed_evaluation": sequence.failed_evaluation,
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

    def run_stage(self, context: StageContext) -> StageResult:
        stage = context.stage
        try:
            self._hydrate(context)
            if stage is OptimizeStage.PROFILE:
                proof = self._ensure_loaded()
                return self._result(
                    stage,
                    "loaded and profiled "
                    f"{proof._model_target.identifier} revision "
                    f"{proof._model_target.revision}",
                    measured=True,
                    constraints_passed=True,
                )
            if stage is OptimizeStage.CAPABILITY:
                self._ensure_loaded()
                return self._result(
                    stage, "HF gated-MLP physical channel removal is adapter-supported"
                )
            if stage is OptimizeStage.BASELINE:
                self.baseline = self._ensure_loaded().baseline_measurement()
                return self._result(
                    stage,
                    json.dumps(dict(self.baseline), sort_keys=True),
                    measured=True,
                    constraints_passed=True,
                )
            if stage is OptimizeStage.CANDIDATE_GENERATION:
                candidates = self._enumerate()
                return self._result(
                    stage,
                    f"enumerated {len(candidates)} real MLP-channel candidates "
                    "from the component graph",
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
                            "measurement": dict(self.selected_measurement),
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
            if stage in {OptimizeStage.REPAIR, OptimizeStage.QUANTIZATION}:
                return self._result(
                    stage,
                    "not requested by this verified HF MLP-channel plan; no operation was claimed",
                )
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
                return self._result(
                    stage,
                    json.dumps(
                        {
                            "selection": "one measured feasible HF artifact selected on "
                            "quality and physical parameter reduction",
                            "baseline_deployment": dict(self.deployment_baseline),
                            "candidate_deployment": dict(self.deployment),
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
                )
            if stage is OptimizeStage.REPORT:
                return self._result(
                    stage, "first-party evidence report retained in the optimize run state"
                )
            raise FirstPartyOptimizeRuntimeError(f"unsupported optimize stage: {stage.value}")
        except FirstPartyOptimizeRuntimeError as error:
            return self._unsupported(stage, str(error))
        except Exception as error:  # durable failure evidence is safer than an implicit claim
            return StageResult(
                WorkflowOutcome.FAILED,
                "failure_"
                + hashlib.sha256(f"{self.plan.plan_id}:{stage.value}:{error}".encode()).hexdigest(),
                f"first-party runtime failed during {stage.value}: {error}",
                complete=True,
            )


def build_first_party_optimize_runtime(plan: OptimizePlan) -> OptimizeRuntime:
    """Build the default runtime selected by the public optimize command."""

    return HuggingFaceOptimizeRuntime(plan)


__all__ = ["HuggingFaceOptimizeRuntime", "build_first_party_optimize_runtime"]
