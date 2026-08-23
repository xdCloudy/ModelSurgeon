"""Streaming diagonal and deterministic Nyström activation covariance estimates."""

from __future__ import annotations

import math
from array import array
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from modelsurgeon.features.schema import (
    FeatureKind,
    FeatureRecord,
    PrecisionProvenance,
    PrecisionSource,
)
from modelsurgeon.graph import ComponentId

ACTIVATION_COVARIANCE_EXTRACTOR_VERSION = "1"
_MASK64 = (1 << 64) - 1


class ActivationCovarianceError(ValueError):
    """Raised when covariance collection cannot preserve bounded-memory semantics."""


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return value ^ (value >> 31)


def _projection(seed: int, channel: int, column: int, rank: int) -> float:
    mixed = _splitmix64(
        (seed & _MASK64)
        ^ ((channel + 1) * 0xD6E8FEB86659FD93)
        ^ ((column + 1) * 0xA5A3564E27F8862B)
    )
    sign = 1.0 if mixed & 1 else -1.0
    return sign / math.sqrt(rank)


@dataclass(frozen=True, slots=True)
class ActivationCovarianceConfig:
    component_id: ComponentId
    channel_ids: tuple[ComponentId, ...]
    sketch_rank: int = 16
    seed: int = 0
    max_workspace_bytes: int = 64 * 1024 * 1024
    covariance_ddof: int = 1
    eigenvalue_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if not self.channel_ids or len(self.channel_ids) != len(set(self.channel_ids)):
            raise ActivationCovarianceError("covariance channels must be non-empty and unique")
        if self.sketch_rank <= 0:
            raise ActivationCovarianceError("covariance sketch rank must be positive")
        if self.max_workspace_bytes <= 0:
            raise ActivationCovarianceError("covariance workspace budget must be positive")
        if self.covariance_ddof not in (0, 1):
            raise ActivationCovarianceError("covariance ddof must be 0 or 1")
        if not math.isfinite(self.eigenvalue_tolerance) or self.eigenvalue_tolerance <= 0.0:
            raise ActivationCovarianceError("covariance eigenvalue tolerance must be positive")
        if self.planned_peak_workspace_bytes > self.max_workspace_bytes:
            raise ActivationCovarianceError(
                f"planned covariance workspace {self.planned_peak_workspace_bytes} exceeds "
                f"budget {self.max_workspace_bytes}"
            )

    @property
    def channel_count(self) -> int:
        return len(self.channel_ids)

    @property
    def planned_peak_workspace_bytes(self) -> int:
        channels = self.channel_count
        rank = min(self.sketch_rank, channels)
        # Permanent mean/diagonal/sketch buffers plus update scratch and summary
        # Y/factor/projection/eigendecomposition scratch. All numeric buffers are float64.
        float_count = 4 * channels * rank + 4 * channels + 8 * rank * rank + 4 * rank
        return float_count * 8


@dataclass(frozen=True, slots=True)
class ActivationCovarianceSummary:
    component_id: ComponentId
    channel_ids: tuple[ComponentId, ...]
    observation_count: int
    diagonal: tuple[float, ...]
    sketch_factor: tuple[float, ...]
    factor_rank: int
    seed: int
    configured_sketch_rank: int
    planned_peak_workspace_bytes: int
    workspace_budget_bytes: int
    covariance_ddof: int
    eigenvalue_tolerance: float
    numpy_version: str

    def __post_init__(self) -> None:
        channels = len(self.channel_ids)
        if self.observation_count <= self.covariance_ddof:
            raise ActivationCovarianceError("covariance summary has insufficient observations")
        if len(self.diagonal) != channels:
            raise ActivationCovarianceError("covariance diagonal does not align with channels")
        if self.factor_rank <= 0 or len(self.sketch_factor) != channels * self.factor_rank:
            raise ActivationCovarianceError("covariance sketch factor shape is invalid")
        if any(not math.isfinite(value) or value < 0.0 for value in self.diagonal):
            raise ActivationCovarianceError("covariance diagonal must be finite and non-negative")
        if any(not math.isfinite(value) for value in self.sketch_factor):
            raise ActivationCovarianceError("covariance sketch factor must be finite")
        if self.planned_peak_workspace_bytes > self.workspace_budget_bytes:
            raise ActivationCovarianceError("covariance summary exceeds its workspace budget")

    def to_record(self) -> dict[str, object]:
        return {
            "component_id": str(self.component_id),
            "channel_ids": [str(item) for item in self.channel_ids],
            "observation_count": self.observation_count,
            "diagonal": list(self.diagonal),
            "sketch_factor": list(self.sketch_factor),
            "factor_shape": [len(self.channel_ids), self.factor_rank],
            "seed": self.seed,
            "configured_sketch_rank": self.configured_sketch_rank,
            "planned_peak_workspace_bytes": self.planned_peak_workspace_bytes,
            "workspace_budget_bytes": self.workspace_budget_bytes,
            "covariance_ddof": self.covariance_ddof,
            "eigenvalue_tolerance": self.eigenvalue_tolerance,
            "numpy_version": self.numpy_version,
        }

    def feature_records(self) -> tuple[FeatureRecord, ...]:
        precision = PrecisionProvenance(
            PrecisionSource.HIGH_PRECISION,
            "activation",
            "float64",
        )
        channel_text = ",".join(str(item) for item in self.channel_ids)
        metadata = (
            ("observation_count", self.observation_count),
            ("channel_count", len(self.channel_ids)),
            ("channel_ids", channel_text),
            ("seed", self.seed),
            ("configured_sketch_rank", self.configured_sketch_rank),
            ("factor_rank", self.factor_rank),
            ("workspace_peak_bytes", self.planned_peak_workspace_bytes),
            ("workspace_budget_bytes", self.workspace_budget_bytes),
            ("covariance_ddof", self.covariance_ddof),
            ("eigenvalue_tolerance", self.eigenvalue_tolerance),
            ("numpy_version", self.numpy_version),
        )
        return (
            FeatureRecord(
                self.component_id,
                "activation_covariance_diagonal",
                FeatureKind.VECTOR,
                self.diagonal,
                "float64",
                "activation_covariance",
                ACTIVATION_COVARIANCE_EXTRACTOR_VERSION,
                precision,
                metadata=metadata,
            ),
            FeatureRecord(
                self.component_id,
                "activation_covariance_sketch_factor",
                FeatureKind.VECTOR,
                self.sketch_factor,
                "float64",
                "activation_covariance",
                ACTIVATION_COVARIANCE_EXTRACTOR_VERSION,
                precision,
                metadata=metadata,
            ),
        )


@dataclass(frozen=True, slots=True)
class CovarianceAccuracyReport:
    diagonal_max_abs_error: float
    sketch_relative_frobenius_error: float
    exact_frobenius_norm: float

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (
                self.diagonal_max_abs_error,
                self.sketch_relative_frobenius_error,
                self.exact_frobenius_norm,
            )
        ):
            raise ActivationCovarianceError("covariance accuracy metrics must be finite")


class ActivationCovarianceCollector:
    """Update diagonal covariance and one-pass covariance-action sketch in fixed memory."""

    def __init__(self, config: ActivationCovarianceConfig) -> None:
        self.config = config
        channels = config.channel_count
        rank = min(config.sketch_rank, channels)
        self.rank = rank
        self.count = 0
        self.mean = array("d", [0.0]) * channels
        self.m2_diagonal = array("d", [0.0]) * channels
        self.m2_projection = array("d", [0.0]) * (channels * rank)

    @property
    def planned_peak_workspace_bytes(self) -> int:
        return self.config.planned_peak_workspace_bytes

    def update(self, tokens: tuple[tuple[float, ...], ...], mask: tuple[bool, ...]) -> None:
        if len(tokens) != len(mask):
            raise ActivationCovarianceError("activation tokens and mask must align")
        channels = self.config.channel_count
        for token, include in zip(tokens, mask, strict=True):
            if len(token) != channels:
                raise ActivationCovarianceError(
                    f"activation width {len(token)} does not match covariance channels {channels}"
                )
            if not include:
                continue
            if any(not math.isfinite(value) for value in token):
                raise ActivationCovarianceError("covariance activations must be finite")
            next_count = self.count + 1
            delta = array("d", [0.0]) * channels
            projected_delta2 = array("d", [0.0]) * self.rank
            for channel, value in enumerate(token):
                difference = value - self.mean[channel]
                delta[channel] = difference
                self.mean[channel] += difference / next_count
                delta2 = value - self.mean[channel]
                self.m2_diagonal[channel] += difference * delta2
                for column in range(self.rank):
                    projected_delta2[column] += delta2 * _projection(
                        self.config.seed, channel, column, self.rank
                    )
            for channel in range(channels):
                base = channel * self.rank
                for column in range(self.rank):
                    self.m2_projection[base + column] += (
                        delta[channel] * projected_delta2[column]
                    )
            self.count = next_count

    @staticmethod
    def _numpy() -> Any:
        try:
            return import_module("numpy")
        except ModuleNotFoundError as error:
            raise ActivationCovarianceError(
                "covariance sketch summarization requires NumPy; install the HF or dev extra"
            ) from error

    def summary(self) -> ActivationCovarianceSummary:
        if self.count <= self.config.covariance_ddof:
            raise ActivationCovarianceError("covariance requires more observations than ddof")
        denominator = self.count - self.config.covariance_ddof
        channels = self.config.channel_count
        np = self._numpy()
        y = np.frombuffer(self.m2_projection, dtype=np.float64).reshape(
            channels, self.rank
        ) / float(denominator)
        omega_t_y = np.zeros((self.rank, self.rank), dtype=np.float64)
        for left in range(self.rank):
            for right in range(self.rank):
                omega_t_y[left, right] = math.fsum(
                    _projection(self.config.seed, channel, left, self.rank)
                    * float(y[channel, right])
                    for channel in range(channels)
                )
        omega_t_y = (omega_t_y + omega_t_y.T) * 0.5
        eigenvalues, eigenvectors = np.linalg.eigh(omega_t_y)
        scale = max(1.0, float(np.max(np.abs(eigenvalues))))
        cutoff = self.config.eigenvalue_tolerance * scale
        keep = eigenvalues > cutoff
        if bool(np.any(keep)):
            selected_values = eigenvalues[keep]
            selected_vectors = eigenvectors[:, keep]
            factor = (y @ selected_vectors) / np.sqrt(selected_values)[None, :]
            factor_rank = int(factor.shape[1])
        else:
            factor = np.zeros((channels, 1), dtype=np.float64)
            factor_rank = 1
        diagonal = tuple(
            max(0.0, value / denominator) for value in self.m2_diagonal
        )
        return ActivationCovarianceSummary(
            component_id=self.config.component_id,
            channel_ids=self.config.channel_ids,
            observation_count=self.count,
            diagonal=diagonal,
            sketch_factor=tuple(float(value) for value in factor.reshape(-1)),
            factor_rank=factor_rank,
            seed=self.config.seed,
            configured_sketch_rank=self.config.sketch_rank,
            planned_peak_workspace_bytes=self.config.planned_peak_workspace_bytes,
            workspace_budget_bytes=self.config.max_workspace_bytes,
            covariance_ddof=self.config.covariance_ddof,
            eigenvalue_tolerance=self.config.eigenvalue_tolerance,
            numpy_version=str(np.__version__),
        )


def evaluate_covariance_accuracy(
    summary: ActivationCovarianceSummary,
    exact_covariance: tuple[tuple[float, ...], ...],
) -> CovarianceAccuracyReport:
    """Compare one bounded summary with a small exact covariance reference."""

    channels = len(summary.channel_ids)
    if len(exact_covariance) != channels or any(
        len(row) != channels for row in exact_covariance
    ):
        raise ActivationCovarianceError("exact covariance reference shape is invalid")
    exact = tuple(tuple(float(value) for value in row) for row in exact_covariance)
    if any(not math.isfinite(value) for row in exact for value in row):
        raise ActivationCovarianceError("exact covariance reference must be finite")
    diagonal_error = max(
        abs(summary.diagonal[index] - exact[index][index]) for index in range(channels)
    )
    factor = summary.sketch_factor
    approximate = tuple(
        tuple(
            math.fsum(
                factor[row * summary.factor_rank + column]
                * factor[col * summary.factor_rank + column]
                for column in range(summary.factor_rank)
            )
            for col in range(channels)
        )
        for row in range(channels)
    )
    exact_norm_sq = math.fsum(value * value for row in exact for value in row)
    error_sq = math.fsum(
        (approximate[row][col] - exact[row][col]) ** 2
        for row in range(channels)
        for col in range(channels)
    )
    exact_norm = math.sqrt(exact_norm_sq)
    relative = 0.0 if exact_norm == 0.0 else math.sqrt(error_sq) / exact_norm
    return CovarianceAccuracyReport(diagonal_error, relative, exact_norm)
