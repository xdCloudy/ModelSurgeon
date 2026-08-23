"""Deterministic exact spectral features for explicitly bounded weight matrices."""

from __future__ import annotations

import math
from dataclasses import dataclass

from modelsurgeon.features.schema import (
    FeatureKind,
    FeatureRecord,
    PrecisionProvenance,
    PrecisionSource,
)
from modelsurgeon.features.weight_statistics import WeightTensor, _host_values
from modelsurgeon.graph import ComponentId

EXACT_SPECTRAL_EXTRACTOR_VERSION = "1"


class ExactSpectralError(ValueError):
    """Raised when an accepted matrix cannot produce a valid exact spectral result."""


@dataclass(frozen=True, slots=True)
class ExactSpectralConfig:
    max_elements: int = 262_144
    max_minor_dimension: int = 128
    convergence_tolerance: float = 1e-12
    max_sweeps: int = 100
    energy_thresholds: tuple[float, ...] = (0.9, 0.95, 0.99)
    compute_dtype: str = "float64"

    def __post_init__(self) -> None:
        if self.max_elements <= 0 or self.max_minor_dimension <= 0 or self.max_sweeps <= 0:
            raise ExactSpectralError("spectral size and iteration limits must be positive")
        if (
            not math.isfinite(self.convergence_tolerance)
            or self.convergence_tolerance <= 0.0
        ):
            raise ExactSpectralError("spectral convergence tolerance must be positive and finite")
        if self.compute_dtype != "float64":
            raise ExactSpectralError("exact spectral extraction currently requires float64 compute")
        if not self.energy_thresholds:
            raise ExactSpectralError("at least one energy threshold is required")
        if any(
            not math.isfinite(value) or not 0.0 < value <= 1.0
            for value in self.energy_thresholds
        ):
            raise ExactSpectralError("energy thresholds must be finite values within (0, 1]")
        if tuple(sorted(self.energy_thresholds)) != self.energy_thresholds or len(
            set(self.energy_thresholds)
        ) != len(self.energy_thresholds):
            raise ExactSpectralError("energy thresholds must be strictly increasing")


@dataclass(frozen=True, slots=True)
class ExactSpectralFeatures:
    component_id: ComponentId
    shape: tuple[int, int]
    storage_dtype: str
    source_device: str
    singular_values: tuple[float, ...]
    singular_value_decay: tuple[float, ...]
    spectral_norm: float
    effective_rank: float
    stable_rank: float
    energy_ranks: tuple[tuple[float, int], ...]
    convergence_tolerance: float
    compute_dtype: str
    sweeps: int

    def __post_init__(self) -> None:
        if any(dimension <= 0 for dimension in self.shape):
            raise ExactSpectralError("spectral matrix dimensions must be positive")
        if len(self.singular_values) != min(self.shape) or not self.singular_values:
            raise ExactSpectralError("singular-value count must equal the matrix minor dimension")
        if len(self.singular_value_decay) != len(self.singular_values):
            raise ExactSpectralError("singular-value decay must align with singular values")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (*self.singular_values, *self.singular_value_decay)
        ):
            raise ExactSpectralError("spectral vectors must be finite and non-negative")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (self.spectral_norm, self.effective_rank, self.stable_rank)
        ):
            raise ExactSpectralError("spectral scalars must be finite and non-negative")
        if not self.compute_dtype or self.sweeps < 0:
            raise ExactSpectralError("spectral compute provenance is invalid")

    def to_record(self) -> dict[str, object]:
        return {
            "component_id": str(self.component_id),
            "shape": list(self.shape),
            "storage_dtype": self.storage_dtype,
            "source_device": self.source_device,
            "singular_values": list(self.singular_values),
            "singular_value_decay": list(self.singular_value_decay),
            "spectral_norm": self.spectral_norm,
            "effective_rank": self.effective_rank,
            "stable_rank": self.stable_rank,
            "energy_ranks": [
                {"threshold": threshold, "rank": rank}
                for threshold, rank in self.energy_ranks
            ],
            "convergence_tolerance": self.convergence_tolerance,
            "compute_dtype": self.compute_dtype,
            "sweeps": self.sweeps,
        }

    def feature_records(self) -> tuple[FeatureRecord, ...]:
        precision = PrecisionProvenance(
            PrecisionSource.HIGH_PRECISION,
            self.storage_dtype,
            self.compute_dtype,
        )
        metadata = (
            ("shape", f"{self.shape[0]}x{self.shape[1]}"),
            ("source_device", self.source_device),
            ("convergence_tolerance", self.convergence_tolerance),
            ("compute_dtype", self.compute_dtype),
            ("jacobi_sweeps", self.sweeps),
        )
        energy_thresholds = ",".join(
            format(threshold, ".17g") for threshold, _ in self.energy_ranks
        )
        return (
            FeatureRecord(
                self.component_id,
                "singular_values",
                FeatureKind.VECTOR,
                self.singular_values,
                self.compute_dtype,
                "spectral_exact",
                EXACT_SPECTRAL_EXTRACTOR_VERSION,
                precision,
                metadata=metadata,
            ),
            FeatureRecord(
                self.component_id,
                "singular_value_decay",
                FeatureKind.VECTOR,
                self.singular_value_decay,
                self.compute_dtype,
                "spectral_exact",
                EXACT_SPECTRAL_EXTRACTOR_VERSION,
                precision,
                metadata=metadata,
            ),
            FeatureRecord(
                self.component_id,
                "spectral_norm",
                FeatureKind.SCALAR,
                self.spectral_norm,
                self.compute_dtype,
                "spectral_exact",
                EXACT_SPECTRAL_EXTRACTOR_VERSION,
                precision,
                metadata=metadata,
            ),
            FeatureRecord(
                self.component_id,
                "effective_rank",
                FeatureKind.SCALAR,
                self.effective_rank,
                self.compute_dtype,
                "spectral_exact",
                EXACT_SPECTRAL_EXTRACTOR_VERSION,
                precision,
                metadata=metadata,
            ),
            FeatureRecord(
                self.component_id,
                "stable_rank",
                FeatureKind.SCALAR,
                self.stable_rank,
                self.compute_dtype,
                "spectral_exact",
                EXACT_SPECTRAL_EXTRACTOR_VERSION,
                precision,
                metadata=metadata,
            ),
            FeatureRecord(
                self.component_id,
                "energy_ranks",
                FeatureKind.VECTOR,
                tuple(float(rank) for _, rank in self.energy_ranks),
                "int64",
                "spectral_exact",
                EXACT_SPECTRAL_EXTRACTOR_VERSION,
                precision,
                metadata=(*metadata, ("energy_thresholds", energy_thresholds)),
            ),
        )


@dataclass(frozen=True, slots=True)
class ExactSpectralOutcome:
    features: ExactSpectralFeatures | None
    decline_reason: str | None

    def __post_init__(self) -> None:
        if (self.features is None) == (self.decline_reason is None):
            raise ExactSpectralError("spectral outcome must contain either features or a decline reason")

    @property
    def accepted(self) -> bool:
        return self.features is not None

    def to_record(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "decline_reason": self.decline_reason,
            "features": None if self.features is None else self.features.to_record(),
        }


def _gram(values: tuple[float, ...], rows: int, columns: int) -> list[list[float]]:
    if rows >= columns:
        size = columns
        gram = [[0.0] * size for _ in range(size)]
        for row in range(rows):
            offset = row * columns
            for left in range(columns):
                left_value = values[offset + left]
                for right in range(left, columns):
                    gram[left][right] += left_value * values[offset + right]
        for left in range(size):
            for right in range(left):
                gram[left][right] = gram[right][left]
        return gram

    size = rows
    gram = [[0.0] * size for _ in range(size)]
    for upper in range(rows):
        upper_offset = upper * columns
        for lower in range(upper, rows):
            lower_offset = lower * columns
            value = math.fsum(
                values[upper_offset + column] * values[lower_offset + column]
                for column in range(columns)
            )
            gram[upper][lower] = value
            gram[lower][upper] = value
    return gram


def _jacobi_eigenvalues(
    matrix: list[list[float]], tolerance: float, max_sweeps: int
) -> tuple[tuple[float, ...], int]:
    size = len(matrix)
    if size == 1:
        return (matrix[0][0],), 0
    scale = max(1.0, max(abs(matrix[row][column]) for row in range(size) for column in range(size)))
    threshold = tolerance * scale
    for sweep in range(1, max_sweeps + 1):
        largest = max(
            abs(matrix[row][column])
            for row in range(size)
            for column in range(row + 1, size)
        )
        if largest <= threshold:
            return tuple(matrix[index][index] for index in range(size)), sweep - 1
        for row in range(size - 1):
            for column in range(row + 1, size):
                off_diagonal = matrix[row][column]
                if abs(off_diagonal) <= threshold:
                    continue
                diagonal_delta = matrix[column][column] - matrix[row][row]
                angle = 0.5 * math.atan2(2.0 * off_diagonal, diagonal_delta)
                cosine = math.cos(angle)
                sine = math.sin(angle)
                row_diagonal = matrix[row][row]
                column_diagonal = matrix[column][column]
                matrix[row][row] = (
                    cosine * cosine * row_diagonal
                    - 2.0 * sine * cosine * off_diagonal
                    + sine * sine * column_diagonal
                )
                matrix[column][column] = (
                    sine * sine * row_diagonal
                    + 2.0 * sine * cosine * off_diagonal
                    + cosine * cosine * column_diagonal
                )
                matrix[row][column] = 0.0
                matrix[column][row] = 0.0
                for index in range(size):
                    if index in (row, column):
                        continue
                    row_value = matrix[index][row]
                    column_value = matrix[index][column]
                    rotated_row = cosine * row_value - sine * column_value
                    rotated_column = sine * row_value + cosine * column_value
                    matrix[index][row] = rotated_row
                    matrix[row][index] = rotated_row
                    matrix[index][column] = rotated_column
                    matrix[column][index] = rotated_column
    largest = max(
        abs(matrix[row][column])
        for row in range(size)
        for column in range(row + 1, size)
    )
    if largest > threshold:
        raise ExactSpectralError(
            f"Jacobi eigensolver did not converge within {max_sweeps} sweeps; residual={largest}"
        )
    return tuple(matrix[index][index] for index in range(size)), max_sweeps


def _effective_rank(singular_values: tuple[float, ...]) -> float:
    total = math.fsum(singular_values)
    if total == 0.0:
        return 0.0
    probabilities = tuple(value / total for value in singular_values if value > 0.0)
    entropy = -math.fsum(probability * math.log(probability) for probability in probabilities)
    return math.exp(entropy)


def _energy_ranks(
    singular_values: tuple[float, ...], thresholds: tuple[float, ...]
) -> tuple[tuple[float, int], ...]:
    energies = tuple(value * value for value in singular_values)
    total = math.fsum(energies)
    if total == 0.0:
        return tuple((threshold, 0) for threshold in thresholds)
    output: list[tuple[float, int]] = []
    cumulative = 0.0
    threshold_index = 0
    for rank, energy in enumerate(energies, start=1):
        cumulative += energy
        while threshold_index < len(thresholds) and cumulative / total >= thresholds[threshold_index]:
            output.append((thresholds[threshold_index], rank))
            threshold_index += 1
    while threshold_index < len(thresholds):
        output.append((thresholds[threshold_index], len(energies)))
        threshold_index += 1
    return tuple(output)


def extract_exact_spectral_features(
    component_id: ComponentId,
    tensor: WeightTensor,
    config: ExactSpectralConfig | None = None,
) -> ExactSpectralOutcome:
    """Compute exact singular-value features or explicitly decline an oversized matrix."""

    resolved = config or ExactSpectralConfig()
    values, raw_shape, storage_dtype, source_device = _host_values(tensor)
    if len(raw_shape) != 2:
        return ExactSpectralOutcome(None, f"expected a matrix, received shape {raw_shape}")
    rows, columns = raw_shape
    if rows <= 0 or columns <= 0:
        return ExactSpectralOutcome(None, "matrix dimensions must be positive")
    if len(values) > resolved.max_elements:
        return ExactSpectralOutcome(
            None,
            f"matrix has {len(values)} elements, exceeding limit {resolved.max_elements}",
        )
    minor = min(rows, columns)
    if minor > resolved.max_minor_dimension:
        return ExactSpectralOutcome(
            None,
            f"matrix minor dimension {minor} exceeds limit {resolved.max_minor_dimension}",
        )
    eigenvalues, sweeps = _jacobi_eigenvalues(
        _gram(values, rows, columns),
        resolved.convergence_tolerance,
        resolved.max_sweeps,
    )
    negative_tolerance = resolved.convergence_tolerance * max(
        1.0, max(abs(value) for value in eigenvalues)
    )
    if any(value < -negative_tolerance for value in eigenvalues):
        raise ExactSpectralError("Gram matrix produced a materially negative eigenvalue")
    singular_values = tuple(
        sorted((math.sqrt(max(0.0, value)) for value in eigenvalues), reverse=True)
    )
    spectral_norm = singular_values[0]
    if spectral_norm == 0.0:
        decay = (0.0,) * len(singular_values)
        stable_rank = 0.0
    else:
        decay = tuple(value / spectral_norm for value in singular_values)
        stable_rank = math.fsum(value * value for value in singular_values) / spectral_norm**2
    return ExactSpectralOutcome(
        ExactSpectralFeatures(
            component_id=component_id,
            shape=(rows, columns),
            storage_dtype=storage_dtype,
            source_device=source_device,
            singular_values=singular_values,
            singular_value_decay=decay,
            spectral_norm=spectral_norm,
            effective_rank=_effective_rank(singular_values),
            stable_rank=stable_rank,
            energy_ranks=_energy_ranks(singular_values, resolved.energy_thresholds),
            convergence_tolerance=resolved.convergence_tolerance,
            compute_dtype=resolved.compute_dtype,
            sweeps=sweeps,
        ),
        None,
    )
