"""First-party execution boundary for the verified Hugging Face optimize cell.

This module intentionally supports one narrow, real path first: local or pinned
Hugging Face causal LMs exposing the adapter-defined gated MLP layout.  It uses
the existing proof runtime for measurement and the existing physical surgery
and reload contracts for publication.  Other formats and operations return an
explicit unsupported outcome instead of being represented by placeholder data.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
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
from modelsurgeon.optimization import OptimizePlan
from modelsurgeon.optimization_orchestrator import (
    OptimizeRuntime,
    OptimizeStage,
    StageContext,
    StageResult,
    WorkflowOutcome,
)
from modelsurgeon.surgery.contracts import TransactionState


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
    if path.is_file():
        return _digest_file(path)
    if not path.is_dir():
        raise FirstPartyOptimizeRuntimeError(f"model source does not exist: {path}")
    entries: list[dict[str, object]] = []
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        entries.append(
            {
                "path": child.relative_to(path).as_posix(),
                "sha256": _digest_file(child),
                "size": child.stat().st_size,
            }
        )
    if not entries:
        raise FirstPartyOptimizeRuntimeError("model source directory is empty")
    return hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256(path: Path) -> str:
    return "sha256:" + _digest_file(path)


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
        self.artifact: Path | None = None
        self.artifact_digest: str | None = None
        self.reloaded_model: Any | None = None
        self.baseline: Mapping[str, object] | None = None
        self.deployment: Mapping[str, object] | None = None

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
            self.source_digest = _digest_tree(source_path)
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
        scored: list[tuple[float, float, int, MutationCandidate, Mapping[str, object]]] = []
        for channel, candidate in sorted(unique.items()):
            measurement: Mapping[str, object] = proof.measure_channel_set(
                tuple((layer, channel) for layer in range(proof._discovery.shape.layers)),
                repetitions=self.plan.quality_profile.evaluation_repetitions,
            ).to_record()
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
        _, _, channel, candidate, measurement = min(
            scored, key=lambda item: (item[0], item[1], item[2])
        )
        self.selected = candidate
        self.selected_channel = channel
        self.selected_measurement = measurement
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
                device_map="cpu",
                dtype=_hf_dtype(_config_value(self.plan, "model", "dtype")),
                local_files_only=True,
            )
        )
        result.model.eval()
        return result.model

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
        assert self.selected_channel is not None
        model = copy.deepcopy(proof.model)
        removal = remove_huggingface_mlp_channels(model, (self.selected_channel,))
        artifact_dir = _mapping(self.plan.resolved_config, "resolved configuration").get(
            "artifact_dir", "artifacts"
        )
        artifact_root = Path(cast(str, artifact_dir))
        destination = artifact_root / "optimize" / context.run.run_id / "hf-mlp-channel"
        artifact = self._publish(model, destination)
        reloaded = self._reload(artifact)
        if not self._generation_smoke(reloaded):
            raise FirstPartyOptimizeRuntimeError(
                "reloaded artifact failed the inference smoke test"
            )
        self.artifact = artifact
        self.artifact_digest = _sha256(artifact)
        self.reloaded_model = reloaded
        return self._result(
            OptimizeStage.SURGERY,
            json.dumps(
                {
                    "operation": "remove_mlp_channel",
                    "channel": self.selected_channel,
                    "old_intermediate_size": removal.old_intermediate_size,
                    "new_intermediate_size": removal.new_intermediate_size,
                    "source_digest": self.source_digest,
                    "artifact": str(artifact),
                },
                sort_keys=True,
            ),
            measured=True,
            constraints_passed=True,
            artifact_digest=self.artifact_digest,
            candidate=candidate,
            transaction_state=TransactionState.COMMITTED,
        )

    def run_stage(self, context: StageContext) -> StageResult:
        stage = context.stage
        try:
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
                    alternatives=tuple(item.candidate_id for item in candidates[:8]),
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
                    json.dumps(dict(self.selected_measurement), sort_keys=True),
                    measured=True,
                    constraints_passed=True,
                    candidate=candidate,
                    evaluation_id=evaluation_id,
                    alternatives=tuple(
                        item.candidate_id for item in self.candidates if item != candidate
                    )[:8],
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
                self.deployment = proof.measure_model(
                    self.reloaded_model,
                    repetitions=self.plan.quality_profile.evaluation_repetitions,
                )
                baseline = self.baseline or proof.baseline_measurement()
                quality_delta = _number(
                    self.deployment["perplexity"], "deployment.perplexity"
                ) - _number(baseline["perplexity"], "baseline.perplexity")
                allowed = self.plan.quality_profile.max_perplexity_delta or 0.05
                passed = quality_delta <= allowed
                return self._result(
                    stage,
                    json.dumps(
                        {"deployment": dict(self.deployment), "quality_delta": quality_delta},
                        sort_keys=True,
                    ),
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
                return self._result(
                    stage,
                    "one measured feasible HF artifact selected on quality and "
                    "physical parameter reduction",
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
