"""Pinned quantization-only and matched pruning-plus-quantization evidence."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import StrEnum
from itertools import product

from modelsurgeon.adapters.gguf.conformance import GGML_UPSTREAM_REVISION

QUANTIZED_BASELINE_SCHEMA_VERSION = 1


class QuantizedBaselineError(ValueError):
    """Raised when a quantized baseline contract is incomplete or inconsistent."""


class QuantizationProfile(StrEnum):
    BF16 = "BF16"
    F16 = "F16"
    Q8_0 = "Q8_0"
    Q6_K = "Q6_K"
    Q5_K_M = "Q5_K_M"
    Q4_K_M = "Q4_K_M"


class QuantizationArm(StrEnum):
    QUANTIZATION_ONLY = "quantization_only"
    PRUNING_PLUS_QUANTIZATION = "pruning_plus_quantization"


class QuantizedOutcome(StrEnum):
    MEASURED = "measured"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"
    UNKNOWN = "unknown"


class QuantizedCompatibility(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


def _text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise QuantizedBaselineError(f"{label} is required")


def _finite(value: float, label: str) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise QuantizedBaselineError(f"{label} must be numeric")
    if not math.isfinite(float(value)):
        raise QuantizedBaselineError(f"{label} must be finite")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class QuantizationMethodSpec:
    """Exact codec/tool identity used by a profile."""

    profile: QuantizationProfile
    upstream_revision: str
    license: str
    source_url: str
    recipe: str

    def __post_init__(self) -> None:
        for label, value in (
            ("quantizer revision", self.upstream_revision),
            ("quantizer license", self.license),
            ("quantizer source URL", self.source_url),
            ("quantizer recipe", self.recipe),
        ):
            _text(value, label)

    def to_record(self) -> dict[str, str]:
        return {
            "profile": self.profile.value,
            "upstream_revision": self.upstream_revision,
            "license": self.license,
            "source_url": self.source_url,
            "recipe": self.recipe,
        }


@dataclass(frozen=True, slots=True)
class QuantizedModel:
    rung: str
    identifier: str
    revision: str
    family: str
    source_license: str

    def __post_init__(self) -> None:
        for label, value in (
            ("model rung", self.rung),
            ("model identifier", self.identifier),
            ("model revision", self.revision),
            ("model family", self.family),
            ("model license", self.source_license),
        ):
            _text(value, label)

    def to_record(self) -> dict[str, str]:
        return {
            "rung": self.rung,
            "identifier": self.identifier,
            "revision": self.revision,
            "family": self.family,
            "source_license": self.source_license,
        }


@dataclass(frozen=True, slots=True)
class QuantizedCorpus:
    identifier: str
    revision: str
    split: str
    license: str
    token_count: int
    tokenizer_revision: str

    def __post_init__(self) -> None:
        for label, value in (
            ("corpus identifier", self.identifier),
            ("corpus revision", self.revision),
            ("corpus split", self.split),
            ("corpus license", self.license),
            ("tokenizer revision", self.tokenizer_revision),
        ):
            _text(value, label)
        if self.token_count <= 0:
            raise QuantizedBaselineError("corpus token_count must be positive")

    def to_record(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "revision": self.revision,
            "split": self.split,
            "license": self.license,
            "token_count": self.token_count,
            "tokenizer_revision": self.tokenizer_revision,
        }


@dataclass(frozen=True, slots=True)
class QuantizedBudget:
    optimization_tokens: int
    evaluation_tokens: int
    max_wall_seconds: float
    max_gpu_memory_gib: float

    def __post_init__(self) -> None:
        if self.optimization_tokens <= 0 or self.evaluation_tokens <= 0:
            raise QuantizedBaselineError("token budgets must be positive")
        if self.max_wall_seconds <= 0 or self.max_gpu_memory_gib <= 0:
            raise QuantizedBaselineError("resource budgets must be positive")

    def to_record(self) -> dict[str, int | float]:
        return {
            "optimization_tokens": self.optimization_tokens,
            "evaluation_tokens": self.evaluation_tokens,
            "max_wall_seconds": self.max_wall_seconds,
            "max_gpu_memory_gib": self.max_gpu_memory_gib,
        }


@dataclass(frozen=True, slots=True)
class QuantizedMetric:
    name: str
    unit: str
    value: float | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        _text(self.name, "metric name")
        _text(self.unit, "metric unit")
        if self.value is None:
            _text(self.reason or "", "unavailable metric reason")
        elif self.reason is not None:
            raise QuantizedBaselineError("measured metrics cannot carry a reason")
        else:
            _finite(self.value, "metric value")

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "unit": self.unit,
            "value": self.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class QuantizedArtifact:
    source_bytes: int | None
    output_bytes: int | None
    digest: str | None
    reloadable: bool | None
    generation_smoke: bool | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if any(
            value is not None and value <= 0
            for value in (self.source_bytes, self.output_bytes)
        ):
            raise QuantizedBaselineError("artifact byte counts must be positive")
        if self.digest is not None and len(self.digest) != 64:
            raise QuantizedBaselineError("artifact digest must be a SHA-256")
        if self.reloadable is True and self.generation_smoke is not True:
            raise QuantizedBaselineError("reloadable artifacts require generation smoke evidence")
        if self.output_bytes is None:
            _text(self.reason or "", "unavailable artifact reason")

    def to_record(self) -> dict[str, object]:
        return {
            "source_bytes": self.source_bytes,
            "output_bytes": self.output_bytes,
            "digest": self.digest,
            "reloadable": self.reloadable,
            "generation_smoke": self.generation_smoke,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class LossDecomposition:
    quantization_loss: float | None
    surgery_loss: float | None
    interaction_loss: float | None
    reason: str | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.quantization_loss, "quantization loss"),
            (self.surgery_loss, "surgery loss"),
            (self.interaction_loss, "interaction loss"),
        ):
            if value is not None:
                _finite(value, label)
        if any(
            value is None
            for value in (
                self.quantization_loss,
                self.surgery_loss,
                self.interaction_loss,
            )
        ):
            _text(self.reason or "", "loss decomposition reason")

    def to_record(self) -> dict[str, object]:
        return {
            "quantization_loss": self.quantization_loss,
            "surgery_loss": self.surgery_loss,
            "interaction_loss": self.interaction_loss,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class QuantizedCompatibilityCell:
    model_rung: str
    profile: QuantizationProfile
    model_format: str
    outcome: QuantizedCompatibility
    reason: str

    def __post_init__(self) -> None:
        for label, value in (
            ("compatibility model", self.model_rung),
            ("compatibility format", self.model_format),
            ("compatibility reason", self.reason),
        ):
            _text(value, label)

    def to_record(self) -> dict[str, str]:
        return {
            "model_rung": self.model_rung,
            "profile": self.profile.value,
            "model_format": self.model_format,
            "outcome": self.outcome.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class QuantizedEvidenceCell:
    model: QuantizedModel
    arm: QuantizationArm
    profile: QuantizationProfile
    target_fraction: float
    seed: int
    corpus: QuantizedCorpus
    budget: QuantizedBudget
    outcome: QuantizedOutcome
    artifact: QuantizedArtifact
    metrics: tuple[QuantizedMetric, ...]
    loss: LossDecomposition
    outcome_reason: str | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.target_fraction < 1:
            raise QuantizedBaselineError("target fraction must be in [0, 1)")
        if self.arm is QuantizationArm.QUANTIZATION_ONLY and self.target_fraction != 0:
            raise QuantizedBaselineError("quantization-only cells cannot have a surgery target")
        if self.arm is QuantizationArm.PRUNING_PLUS_QUANTIZATION and self.target_fraction != 0.2:
            raise QuantizedBaselineError("combined cells use the preregistered 20% target")
        if self.seed < 0:
            raise QuantizedBaselineError("seed must be unsigned")
        names = tuple(item.name for item in self.metrics)
        if len(names) != len(set(names)):
            raise QuantizedBaselineError("metric names must be unique")
        if self.outcome is QuantizedOutcome.MEASURED:
            if self.artifact.output_bytes is None or self.artifact.reloadable is not True:
                raise QuantizedBaselineError("measured cells require a reloadable artifact")
            if self.outcome_reason is not None:
                raise QuantizedBaselineError("measured cells cannot have a failure reason")
        else:
            _text(self.outcome_reason or "", "outcome reason")

    @property
    def cell_id(self) -> str:
        identity = {
            "schema_version": QUANTIZED_BASELINE_SCHEMA_VERSION,
            "model": self.model.to_record(),
            "arm": self.arm.value,
            "profile": self.profile.value,
            "target_fraction": self.target_fraction,
            "seed": self.seed,
            "corpus": self.corpus.to_record(),
            "budget": self.budget.to_record(),
        }
        return "quantized_" + hashlib.sha256(_canonical(identity).encode()).hexdigest()

    def to_record(self) -> dict[str, object]:
        return {
            "cell_id": self.cell_id,
            "schema_version": QUANTIZED_BASELINE_SCHEMA_VERSION,
            "model": self.model.to_record(),
            "arm": self.arm.value,
            "profile": self.profile.value,
            "target_fraction": self.target_fraction,
            "seed": self.seed,
            "corpus": self.corpus.to_record(),
            "budget": self.budget.to_record(),
            "outcome": self.outcome.value,
            "artifact": self.artifact.to_record(),
            "metrics": [metric.to_record() for metric in self.metrics],
            "loss_decomposition": self.loss.to_record(),
            "outcome_reason": self.outcome_reason,
        }


@dataclass(frozen=True, slots=True)
class QuantizedBaselineProtocol:
    models: tuple[QuantizedModel, ...]
    methods: tuple[QuantizationMethodSpec, ...]
    profiles: tuple[QuantizationProfile, ...]
    corpus: QuantizedCorpus
    budget: QuantizedBudget
    seeds: tuple[int, ...]
    cells: tuple[QuantizedEvidenceCell, ...]
    compatibility: tuple[QuantizedCompatibilityCell, ...]
    schema_version: int = QUANTIZED_BASELINE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != QUANTIZED_BASELINE_SCHEMA_VERSION:
            raise QuantizedBaselineError("unsupported quantized baseline schema version")
        if len(self.models) < 2 or len({model.family for model in self.models}) < 2:
            raise QuantizedBaselineError("protocol requires two model families")
        if len(self.profiles) < 3 or len(self.seeds) < 3:
            raise QuantizedBaselineError("protocol requires three profiles and seeds")
        if self.seeds != tuple(sorted(set(self.seeds))):
            raise QuantizedBaselineError("seeds must be sorted and unique")
        expected = {
            (model.rung, arm, profile, seed)
            for model, arm, profile, seed in product(
                self.models, QuantizationArm, self.profiles, self.seeds
            )
        }
        actual = {(cell.model.rung, cell.arm, cell.profile, cell.seed) for cell in self.cells}
        if actual != expected or len(actual) != len(self.cells):
            raise QuantizedBaselineError("protocol must cover every model/arm/profile/seed cell")
        compatibility_expected = {
            (model.rung, profile, model_format)
            for model, profile, model_format in product(
                self.models, self.profiles, ("safetensors", "GGUF")
            )
        }
        compatibility_actual = {
            (item.model_rung, item.profile, item.model_format) for item in self.compatibility
        }
        if compatibility_actual != compatibility_expected:
            raise QuantizedBaselineError("compatibility matrix is incomplete")

    @property
    def protocol_id(self) -> str:
        return "quantized_protocol_" + hashlib.sha256(
            _canonical(self._identity_record()).encode()
        ).hexdigest()

    def _identity_record(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "models": [model.to_record() for model in self.models],
            "methods": [method.to_record() for method in self.methods],
            "profiles": [profile.value for profile in self.profiles],
            "corpus": self.corpus.to_record(),
            "budget": self.budget.to_record(),
            "seeds": list(self.seeds),
            "cells": [cell.to_record() for cell in self.cells],
            "compatibility": [item.to_record() for item in self.compatibility],
        }

    def to_record(self) -> dict[str, object]:
        record = self._identity_record()
        record["protocol_id"] = self.protocol_id
        return record


def build_default_quantized_baseline_protocol() -> QuantizedBaselineProtocol:
    models = (
        QuantizedModel(
            "small_llama",
            "HuggingFaceTB/SmolLM2-135M",
            "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
            "llama",
            "Apache-2.0",
        ),
        QuantizedModel(
            "small_qwen",
            "Qwen/Qwen2.5-0.5B",
            "060db6499f32faf8b98477b0a26969ef7d8b9987",
            "qwen",
            "Apache-2.0",
        ),
    )
    profiles = tuple(QuantizationProfile)
    methods = tuple(
        QuantizationMethodSpec(
            profile,
            GGML_UPSTREAM_REVISION,
            "MIT",
            "https://github.com/ggerganov/llama.cpp",
            profile.value,
        )
        for profile in profiles
    )
    corpus = QuantizedCorpus(
        "wikitext-2-raw-v1",
        "modelsurgeon-benchmark-datasets-v1.1",
        "test",
        "CC-BY-SA-4.0",
        100000,
        "modelsurgeon-benchmark-tokenizer-v1.1",
    )
    budget = QuantizedBudget(100000, 100000, 3600, 24)
    reason = "pinned llama.cpp quantizer executable is not bundled in this evidence release"
    cells = tuple(
        QuantizedEvidenceCell(
            model,
            arm,
            profile,
            0 if arm is QuantizationArm.QUANTIZATION_ONLY else 0.2,
            seed,
            corpus,
            budget,
            QuantizedOutcome.UNSUPPORTED,
            QuantizedArtifact(None, None, None, None, None, reason),
            tuple(
                QuantizedMetric(name, unit, reason=reason)
                for name, unit in (
                    ("quality", "nats/token"),
                    ("artifact_size", "bytes"),
                    ("runtime", "seconds"),
                    ("peak_ram", "bytes"),
                    ("peak_vram", "bytes"),
                )
            ),
            LossDecomposition(None, None, None, reason),
            reason,
        )
        for model, arm, profile, seed in product(
            models, QuantizationArm, profiles, (11, 23, 47)
        )
    )
    compatibility = tuple(
        QuantizedCompatibilityCell(
            model.rung,
            profile,
            model_format,
            QuantizedCompatibility.UNKNOWN
            if model_format == "safetensors"
            else QuantizedCompatibility.UNSUPPORTED,
            (
                "physical Safetensors codec applicability requires a registered external quantizer"
                if model_format == "safetensors"
                else "GGUF publication is unavailable without the pinned llama.cpp executable"
            ),
        )
        for model, profile, model_format in product(
            models, profiles, ("safetensors", "GGUF")
        )
    )
    return QuantizedBaselineProtocol(
        models,
        methods,
        profiles,
        corpus,
        budget,
        (11, 23, 47),
        cells,
        compatibility,
    )


DEFAULT_QUANTIZED_BASELINE_PROTOCOL = build_default_quantized_baseline_protocol()


def render_quantized_baseline_protocol(
    protocol: QuantizedBaselineProtocol = DEFAULT_QUANTIZED_BASELINE_PROTOCOL,
    *,
    format: str = "json",
) -> str:
    if format == "json":
        return _canonical(protocol.to_record()) + "\n"
    if format == "markdown":
        return "\n".join(
            (
                f"# Quantized baseline protocol {protocol.protocol_id}",
                "",
                f"- Models: {len(protocol.models)} families",
                f"- Profiles: {', '.join(profile.value for profile in protocol.profiles)}",
                f"- Cells: {len(protocol.cells)}",
                "- Unsupported and unknown cells are retained; "
                "no unavailable artifact is a success.",
            )
        ) + "\n"
    raise QuantizedBaselineError(f"unsupported protocol format: {format}")
