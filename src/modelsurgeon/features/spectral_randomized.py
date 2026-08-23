"""Seeded randomized spectral features with explicit CPU workspace preflight."""

from __future__ import annotations

import math
from dataclasses import dataclass
from importlib import import_module
from typing import Any

from modelsurgeon.features.schema import (
    FeatureKind,
    FeatureRecord,
    PrecisionProvenance,
    PrecisionSource,
)
from modelsurgeon.features.weight_statistics import WeightTensor, _host_values, _shape
from modelsurgeon.graph import ComponentId

RANDOMIZED_SPECTRAL_EXTRACTOR_VERSION = "1"
RANDOMIZED_SPECTRAL_ALGORITHM = "gaussian_range_finder_qr_svd_v1"


class RandomizedSpectralError(ValueError):
    """Raised when randomized spectral extraction cannot preserve its contract."""


@dataclass(frozen=True, slots=True)
class RandomizedSpectralConfig:
    target_rank: int = 16
    oversampling: int = 8
    power_iterations: int = 1
    seed: int = 0
    max_workspace_bytes: int = 1 << 30
    reconstruction_ranks: tuple[int, ...] = (1, 4, 8, 16)
    compute_dtype: str = "float64"

    def __post_init__(self) -> None:
        if self.target_rank <= 0 or self.oversampling < 0:
            raise RandomizedSpectralError("target rank must be positive and oversampling non-negative")
        if self.power_iterations < 0 or self.power_iterations > 4:
            raise RandomizedSpectralError("power iterations must be between 0 and 4")
        if self.max_workspace_bytes <= 0:
            raise RandomizedSpectralError("workspace budget must be positive")
        if self.compute_dtype != "float64":
            raise RandomizedSpectralError("randomized spectral extraction currently requires float64")
        if not self.reconstruction_ranks:
            raise RandomizedSpectralError("at least one reconstruction rank is required")
        if any(rank <= 0 or rank > self.target_rank for rank in self.reconstruction_ranks):
            raise RandomizedSpectralError(
                "reconstruction ranks must be positive and no greater than target rank"
            )
        if tuple(sorted(self.reconstruction_ranks)) != self.reconstruction_ranks or len(
            set(self.reconstruction_ranks)
        ) != len(self.reconstruction_ranks):
            raise RandomizedSpectralError("reconstruction ranks must be strictly increasing")


@dataclass(frozen=True, slots=True)
class RandomizedWorkspacePlan:
    input_shape: tuple[int, int]
    oriented_shape: tuple[int, int]
    target_rank: int
    sketch_rank: int
    estimated_peak_bytes: int
    budget_bytes: int
    transposed: bool

    @property
    def fits(self) -> bool:
        return self.estimated_peak_bytes <= self.budget_bytes

    def to_record(self) -> dict[str, object]:
        return {
            "input_shape": list(self.input_shape),
            "oriented_shape": list(self.oriented_shape),
            "target_rank": self.target_rank,
            "sketch_rank": self.sketch_rank,
            "estimated_peak_bytes": self.estimated_peak_bytes,
            "budget_bytes": self.budget_bytes,
            "transposed": self.transposed,
            "fits": self.fits,
        }


@dataclass(frozen=True, slots=True)
class RandomizedSpectralFeatures:
    component_id: ComponentId
    shape: tuple[int, int]
    storage_dtype: str
    source_device: str
    singular_values: tuple[float, ...]
    reconstruction_errors: tuple[tuple[int, float], ...]
    seed: int
    power_iterations: int
    algorithm: str
    workspace: RandomizedWorkspacePlan
    compute_dtype: str
    numpy_version: str

    def __post_init__(self) -> None:
        if not self.singular_values:
            raise RandomizedSpectralError("randomized extraction requires singular values")
        if any(
            not math.isfinite(value) or value < 0.0 for value in self.singular_values
        ):
            raise RandomizedSpectralError("approximate singular values must be finite and non-negative")
        if any(
            rank <= 0 or not math.isfinite(error) or not 0.0 <= error <= 1.0 + 1e-12
            for rank, error in self.reconstruction_errors
        ):
            raise RandomizedSpectralError("reconstruction errors must be finite values in [0, 1]")
        if not self.workspace.fits:
            raise RandomizedSpectralError("completed randomized extraction exceeded its workspace plan")

    def to_record(self) -> dict[str, object]:
        return {
            "component_id": str(self.component_id),
            "shape": list(self.shape),
            "storage_dtype": self.storage_dtype,
            "source_device": self.source_device,
            "singular_values": list(self.singular_values),
            "reconstruction_errors": [
                {"rank": rank, "relative_frobenius_error": error}
                for rank, error in self.reconstruction_errors
            ],
            "seed": self.seed,
            "power_iterations": self.power_iterations,
            "algorithm": self.algorithm,
            "workspace": self.workspace.to_record(),
            "compute_dtype": self.compute_dtype,
            "numpy_version": self.numpy_version,
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
            ("seed", self.seed),
            ("power_iterations", self.power_iterations),
            ("algorithm", self.algorithm),
            ("workspace_peak_bytes", self.workspace.estimated_peak_bytes),
            ("workspace_budget_bytes", self.workspace.budget_bytes),
            ("numpy_version", self.numpy_version),
        )
        ranks = ",".join(str(rank) for rank, _ in self.reconstruction_errors)
        return (
            FeatureRecord(
                self.component_id,
                "randomized_singular_values",
                FeatureKind.VECTOR,
                self.singular_values,
                self.compute_dtype,
                "spectral_randomized",
                RANDOMIZED_SPECTRAL_EXTRACTOR_VERSION,
                precision,
                metadata=metadata,
            ),
            FeatureRecord(
                self.component_id,
                "low_rank_reconstruction_errors",
                FeatureKind.VECTOR,
                tuple(error for _, error in self.reconstruction_errors),
                self.compute_dtype,
                "spectral_randomized",
                RANDOMIZED_SPECTRAL_EXTRACTOR_VERSION,
                precision,
                metadata=(*metadata, ("reconstruction_ranks", ranks)),
            ),
        )


@dataclass(frozen=True, slots=True)
class RandomizedSpectralOutcome:
    features: RandomizedSpectralFeatures | None
    decline_reason: str | None
    workspace: RandomizedWorkspacePlan | None = None

    def __post_init__(self) -> None:
        if (self.features is None) == (self.decline_reason is None):
            raise RandomizedSpectralError(
                "randomized spectral outcome needs either features or a decline reason"
            )

    @property
    def accepted(self) -> bool:
        return self.features is not None


def _estimate_peak(rows: int, columns: int, sketch: int, power_iterations: int) -> int:
    float_bytes = 8
    input_and_snapshot = rows * columns * float_bytes * 2
    omega = columns * sketch * float_bytes
    y_and_q = rows * sketch * float_bytes * 3
    range_peak = input_and_snapshot + omega + y_and_q
    b = sketch * columns * float_bytes
    svd_scratch = b * 3 + sketch * sketch * float_bytes * 8
    reduction_peak = input_and_snapshot + rows * sketch * float_bytes + svd_scratch
    if power_iterations:
        z = columns * sketch * float_bytes
        power_peak = input_and_snapshot + z + rows * sketch * float_bytes * 2
    else:
        power_peak = 0
    return max(range_peak, reduction_peak, power_peak)


def plan_randomized_spectral_workspace(
    shape: tuple[int, int], config: RandomizedSpectralConfig
) -> RandomizedWorkspacePlan:
    rows, columns = shape
    if rows <= 0 or columns <= 0:
        raise RandomizedSpectralError("randomized spectral matrix dimensions must be positive")
    target = min(config.target_rank, rows, columns)
    sketch = min(target + config.oversampling, rows, columns)
    direct = _estimate_peak(rows, columns, sketch, config.power_iterations)
    transposed = _estimate_peak(columns, rows, sketch, config.power_iterations)
    use_transpose = transposed < direct
    oriented = (columns, rows) if use_transpose else (rows, columns)
    peak = transposed if use_transpose else direct
    return RandomizedWorkspacePlan(
        input_shape=shape,
        oriented_shape=oriented,
        target_rank=target,
        sketch_rank=sketch,
        estimated_peak_bytes=peak,
        budget_bytes=config.max_workspace_bytes,
        transposed=use_transpose,
    )


def _preflight_shape(tensor: WeightTensor) -> tuple[int, tuple[int, ...]]:
    try:
        detached = tensor.detach()
        return int(detached.numel()), _shape(detached.shape)
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise RandomizedSpectralError("object does not expose a valid tensor surface") from error


def _numpy() -> Any:
    try:
        return import_module("numpy")
    except ModuleNotFoundError as error:
        raise RandomizedSpectralError(
            "randomized spectral extraction requires NumPy; install ModelSurgeon with its HF or dev extra"
        ) from error


def _relative_errors(
    singular_values: tuple[float, ...], total_energy: float, ranks: tuple[int, ...]
) -> tuple[tuple[int, float], ...]:
    if total_energy == 0.0:
        return tuple((rank, 0.0) for rank in ranks)
    cumulative = 0.0
    by_rank: dict[int, float] = {}
    wanted = set(ranks)
    for rank, value in enumerate(singular_values, start=1):
        cumulative += value * value
        if rank in wanted:
            residual = max(0.0, total_energy - cumulative)
            by_rank[rank] = min(1.0, math.sqrt(residual / total_energy))
    return tuple((rank, by_rank.get(rank, by_rank[max(by_rank)])) for rank in ranks)


def extract_randomized_spectral_features(
    component_id: ComponentId,
    tensor: WeightTensor,
    config: RandomizedSpectralConfig | None = None,
) -> RandomizedSpectralOutcome:
    """Run a seeded randomized SVD only after its dense CPU workspace fits."""

    resolved = config or RandomizedSpectralConfig()
    count, raw_shape = _preflight_shape(tensor)
    if len(raw_shape) != 2:
        return RandomizedSpectralOutcome(None, f"expected a matrix, received shape {raw_shape}")
    shape = (raw_shape[0], raw_shape[1])
    if math.prod(shape) != count:
        raise RandomizedSpectralError("matrix shape does not match tensor numel")
    workspace = plan_randomized_spectral_workspace(shape, resolved)
    if not workspace.fits:
        return RandomizedSpectralOutcome(
            None,
            f"estimated peak workspace {workspace.estimated_peak_bytes} exceeds budget "
            f"{workspace.budget_bytes}",
            workspace,
        )

    values, loaded_shape, storage_dtype, source_device = _host_values(tensor)
    if loaded_shape != raw_shape:
        raise RandomizedSpectralError("tensor shape changed between preflight and snapshot")
    np = _numpy()
    matrix = np.asarray(values, dtype=np.float64).reshape(shape)
    oriented = matrix.T if workspace.transposed else matrix
    total_energy = float(np.sum(oriented * oriented, dtype=np.float64))
    target = workspace.target_rank
    sketch = workspace.sketch_rank
    if total_energy == 0.0:
        singular = (0.0,) * target
    else:
        rng = np.random.Generator(np.random.PCG64(resolved.seed))
        omega = rng.standard_normal((oriented.shape[1], sketch), dtype=np.float64)
        y = oriented @ omega
        del omega
        for _ in range(resolved.power_iterations):
            y = oriented @ (oriented.T @ y)
        q, _ = np.linalg.qr(y, mode="reduced")
        del y
        b = q.T @ oriented
        singular = tuple(float(value) for value in np.linalg.svd(b, compute_uv=False)[:target])
    effective_ranks = tuple(rank for rank in resolved.reconstruction_ranks if rank <= target)
    if not effective_ranks:
        effective_ranks = (target,)
    errors = _relative_errors(singular, total_energy, effective_ranks)
    features = RandomizedSpectralFeatures(
        component_id=component_id,
        shape=shape,
        storage_dtype=storage_dtype,
        source_device=source_device,
        singular_values=singular,
        reconstruction_errors=errors,
        seed=resolved.seed,
        power_iterations=resolved.power_iterations,
        algorithm=RANDOMIZED_SPECTRAL_ALGORITHM,
        workspace=workspace,
        compute_dtype=resolved.compute_dtype,
        numpy_version=str(np.__version__),
    )
    return RandomizedSpectralOutcome(features, None, workspace)
