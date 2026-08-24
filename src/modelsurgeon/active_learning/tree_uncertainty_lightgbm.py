"""Bounded LightGBM producers for the tree-uncertainty comparison contract."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

from modelsurgeon.surgeon.matrix import SurgeonMatrix

from .tree_uncertainty import (
    DEFAULT_TREE_UNCERTAINTY_BUDGET,
    TreeMethodEvidence,
    TreeUncertaintyBudget,
    TreeUncertaintyError,
    TreeUncertaintyMethod,
    TreeUncertaintyStudy,
    compare_tree_uncertainty,
    estimate_from_members,
    estimate_from_quantiles,
)


@dataclass(frozen=True, slots=True)
class LightGBMTreeUncertaintyConfig:
    confidence: float = 0.9
    members: int = 5
    num_leaves: int = 31
    learning_rate: float = 0.05
    max_rounds: int = 500
    early_stopping_rounds: int = 30
    seed: int = 0
    budget: TreeUncertaintyBudget = DEFAULT_TREE_UNCERTAINTY_BUDGET

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence < 1.0:
            raise TreeUncertaintyError("LightGBM uncertainty confidence must be within (0, 1)")
        if self.members < 2 or self.members > self.budget.max_fits_per_method:
            raise TreeUncertaintyError("LightGBM uncertainty members exceed the fit budget")
        if self.budget.max_fits_per_method < 3:
            raise TreeUncertaintyError("quantile uncertainty requires a three-fit budget")
        if self.num_leaves < 2 or self.max_rounds <= 0 or self.early_stopping_rounds <= 0:
            raise TreeUncertaintyError("LightGBM uncertainty tree/round limits are invalid")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise TreeUncertaintyError("LightGBM uncertainty learning rate must be positive")
        if isinstance(self.seed, bool) or not 0 <= self.seed < 1 << 31:
            raise TreeUncertaintyError(
                "LightGBM uncertainty seed must be a non-negative 31-bit integer"
            )


DEFAULT_LIGHTGBM_TREE_UNCERTAINTY_CONFIG = LightGBMTreeUncertaintyConfig()


def run_lightgbm_tree_uncertainty_study(
    train: SurgeonMatrix,
    validation: SurgeonMatrix,
    *,
    config: LightGBMTreeUncertaintyConfig = DEFAULT_LIGHTGBM_TREE_UNCERTAINTY_CONFIG,
) -> TreeUncertaintyStudy:
    """Fit all three methods and compare them using validation labels only."""

    train_x, train_y, train_weights = _measured_rows(train)
    validation_x, validation_y, validation_weights = _measured_rows(validation)
    if train.feature_names != validation.feature_names or not train.feature_names:
        raise TreeUncertaintyError("LightGBM uncertainty train/validation schemas differ")
    if train.target_name != validation.target_name:
        raise TreeUncertaintyError("LightGBM uncertainty train/validation targets differ")
    if len(validation_y) * config.members > config.budget.max_prediction_values:
        raise TreeUncertaintyError("LightGBM uncertainty members exceed prediction memory budget")

    lightgbm = _lightgbm()
    backend_names = [f"feature_{index}" for index in range(len(train.feature_names))]
    technique = f"lightgbm-{_lightgbm_version()}"

    ensemble_start = time.process_time()
    ensemble_predictions: list[tuple[float, ...]] = []
    ensemble_bytes = 0
    for offset in range(config.members):
        predictions, model_bytes = _fit_predict(
            lightgbm,
            train_x,
            train_y,
            train_weights,
            validation_x,
            validation_y,
            validation_weights,
            backend_names,
            config,
            seed=config.seed + offset,
            objective="regression",
            alpha=None,
            stochastic=True,
        )
        ensemble_predictions.append(predictions)
        ensemble_bytes += model_bytes
    ensemble_cpu = time.process_time() - ensemble_start

    bootstrap_start = time.process_time()
    bootstrap_predictions: list[tuple[float, ...]] = []
    bootstrap_bytes = 0
    for offset in range(config.members):
        randomizer = random.Random(config.seed + 10_000 + offset)
        indexes = tuple(randomizer.randrange(len(train_y)) for _ in train_y)
        sampled_x = tuple(train_x[index] for index in indexes)
        sampled_y = tuple(train_y[index] for index in indexes)
        sampled_weights = tuple(train_weights[index] for index in indexes)
        predictions, model_bytes = _fit_predict(
            lightgbm,
            sampled_x,
            sampled_y,
            sampled_weights,
            validation_x,
            validation_y,
            validation_weights,
            backend_names,
            config,
            seed=config.seed + 20_000 + offset,
            objective="regression",
            alpha=None,
            stochastic=False,
        )
        bootstrap_predictions.append(predictions)
        bootstrap_bytes += model_bytes
    bootstrap_cpu = time.process_time() - bootstrap_start

    quantile_start = time.process_time()
    alpha = (1.0 - config.confidence) / 2.0
    quantile_predictions: list[tuple[float, ...]] = []
    quantile_bytes = 0
    for offset, (objective, quantile) in enumerate(
        (("quantile", alpha), ("regression", None), ("quantile", 1.0 - alpha))
    ):
        predictions, model_bytes = _fit_predict(
            lightgbm,
            train_x,
            train_y,
            train_weights,
            validation_x,
            validation_y,
            validation_weights,
            backend_names,
            config,
            seed=config.seed + 30_000 + offset,
            objective=objective,
            alpha=quantile,
            stochastic=False,
        )
        quantile_predictions.append(predictions)
        quantile_bytes += model_bytes
    quantile_cpu = time.process_time() - quantile_start

    evidence = (
        TreeMethodEvidence(
            TreeUncertaintyMethod.ENSEMBLE,
            estimate_from_members(
                TreeUncertaintyMethod.ENSEMBLE,
                ensemble_predictions,
                confidence=config.confidence,
                max_prediction_values=config.budget.max_prediction_values,
            ),
            config.members,
            ensemble_cpu,
            ensemble_bytes,
            f"{technique}:seeded-subsample-v1",
        ),
        TreeMethodEvidence(
            TreeUncertaintyMethod.BOOTSTRAP,
            estimate_from_members(
                TreeUncertaintyMethod.BOOTSTRAP,
                bootstrap_predictions,
                confidence=config.confidence,
                max_prediction_values=config.budget.max_prediction_values,
            ),
            config.members,
            bootstrap_cpu,
            bootstrap_bytes,
            f"{technique}:row-bootstrap-v1",
        ),
        TreeMethodEvidence(
            TreeUncertaintyMethod.QUANTILE,
            estimate_from_quantiles(
                quantile_predictions[0],
                quantile_predictions[1],
                quantile_predictions[2],
                max_prediction_values=config.budget.max_prediction_values,
            ),
            3,
            quantile_cpu,
            quantile_bytes,
            f"{technique}:lower-median-upper-v1",
        ),
    )
    return compare_tree_uncertainty(
        validation_y,
        evidence,
        confidence=config.confidence,
        budget=config.budget,
    )


def _measured_rows(
    matrix: SurgeonMatrix,
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...], tuple[float, ...]]:
    rows: list[tuple[float, ...]] = []
    targets: list[float] = []
    weights: list[float] = []
    for row, target, present, weight in zip(
        matrix.values,
        matrix.target_values,
        matrix.target_mask,
        matrix.sample_weights,
        strict=True,
    ):
        if present:
            rows.append(row)
            targets.append(target)
            weights.append(weight)
    if len(rows) < 2:
        raise TreeUncertaintyError("LightGBM uncertainty requires at least two measured rows")
    return tuple(rows), tuple(targets), tuple(weights)


def _fit_predict(
    lightgbm: Any,
    train_x: Sequence[Sequence[float]],
    train_y: Sequence[float],
    train_weights: Sequence[float],
    validation_x: Sequence[Sequence[float]],
    validation_y: Sequence[float],
    validation_weights: Sequence[float],
    feature_names: Sequence[str],
    config: LightGBMTreeUncertaintyConfig,
    *,
    seed: int,
    objective: str,
    alpha: float | None,
    stochastic: bool,
) -> tuple[tuple[float, ...], int]:
    train_matrix = np.asarray(train_x, dtype=np.float64)
    validation_matrix = np.asarray(validation_x, dtype=np.float64)
    params: dict[str, object] = {
        "objective": objective,
        "metric": "quantile" if objective == "quantile" else "l2",
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "num_threads": config.budget.num_threads,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    if alpha is not None:
        params["alpha"] = alpha
    if stochastic:
        params.update({"feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1})
    training = lightgbm.Dataset(
        train_matrix,
        label=list(train_y),
        weight=list(train_weights),
        feature_name=list(feature_names),
        free_raw_data=False,
    )
    validating = lightgbm.Dataset(
        validation_matrix,
        label=list(validation_y),
        weight=list(validation_weights),
        reference=training,
        feature_name=list(feature_names),
        free_raw_data=False,
    )
    booster = lightgbm.train(
        params,
        training,
        num_boost_round=config.max_rounds,
        valid_sets=[validating],
        valid_names=["validation"],
        callbacks=[lightgbm.early_stopping(config.early_stopping_rounds, verbose=False)],
    )
    predictions = tuple(
        float(value)
        for value in booster.predict(validation_matrix, num_iteration=booster.best_iteration)
    )
    if any(not math.isfinite(value) for value in predictions):
        raise TreeUncertaintyError("LightGBM uncertainty produced non-finite predictions")
    model_string = str(booster.model_to_string(num_iteration=booster.best_iteration))
    return predictions, len(model_string.encode("utf-8"))


def _lightgbm() -> Any:
    try:
        return import_module("lightgbm")
    except ImportError as error:
        raise TreeUncertaintyError("LightGBM uncertainty requires the lightgbm extra") from error


def _lightgbm_version() -> str:
    try:
        return version("lightgbm")
    except PackageNotFoundError as error:
        raise TreeUncertaintyError("LightGBM package version is unavailable") from error
