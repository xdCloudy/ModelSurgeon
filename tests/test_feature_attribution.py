from __future__ import annotations

import math

import lightgbm
import numpy as np
import pytest

from modelsurgeon.explain import (
    AttributionError,
    AttributionReport,
    AttributionUnavailable,
    attribute_predictions,
)
from modelsurgeon.surgeon.models import (
    LightGBMConfig,
    LightGBMSurgeonModel,
    LinearConfig,
    LinearSurgeonModel,
    MLPConfig,
    MLPSurgeonModel,
    ModelTask,
)


def test_linear_contributions_reconcile_and_preserve_missingness_provenance() -> None:
    model = LinearSurgeonModel(
        ("num:weight", "missing:weight", "cat:family=llama"),
        "perplexity",
        (2.0, -3.0, 0.5),
        1.25,
        LinearConfig(),
        3,
    )
    result = attribute_predictions(model, ((0.0, 1.0, 1.0),))
    assert isinstance(result, AttributionReport)
    prediction = result.predictions[0]
    assert prediction.prediction == pytest.approx(-1.25)
    assert prediction.reconstructed_prediction == pytest.approx(prediction.prediction)
    assert tuple(item.contribution for item in prediction.contributions) == (0.0, -3.0, 0.5)
    assert tuple(item.missing for item in prediction.contributions) == (True, True, False)
    assert prediction.contributions[2].provenance.to_record() == {
        "feature_name": "cat:family=llama",
        "source_kind": "categorical",
        "source_name": "family",
        "category": "llama",
    }


def _tree_model(task: ModelTask) -> LightGBMSurgeonModel:
    x = np.asarray(((0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)))
    y = np.asarray((0.0, 1.0, 2.0, 3.0) if task is ModelTask.REGRESSION else (0, 0, 1, 1))
    objective = "regression" if task is ModelTask.REGRESSION else "binary"
    booster = lightgbm.train(
        {
            "objective": objective,
            "verbosity": -1,
            "seed": 7,
            "num_threads": 1,
            "min_data_in_leaf": 1,
            "num_leaves": 4,
        },
        lightgbm.Dataset(x, label=y, feature_name=("feature_0", "feature_1")),
        num_boost_round=4,
    )
    return LightGBMSurgeonModel(
        task,
        ("num:first", "missing:first"),
        "perplexity" if task is ModelTask.REGRESSION else "safe_mutation",
        booster.model_to_string(),
        LightGBMConfig(task, num_threads=1, max_rounds=4),
        4,
        1.0,
    )


@pytest.mark.parametrize("task", (ModelTask.REGRESSION, ModelTask.CLASSIFICATION))
def test_tree_shap_contributions_reconcile_in_declared_raw_space(task: ModelTask) -> None:
    model = _tree_model(task)
    result = attribute_predictions(model, ((0.0, 1.0), (1.0, 0.0)))
    assert isinstance(result, AttributionReport)
    assert result.technique == "lightgbm_tree_shap"
    assert result.output_space == "raw_score"
    assert all(
        item.absolute_reconciliation_error <= result.tolerance
        for item in result.predictions
    )
    assert all(math.isfinite(item.prediction) for item in result.predictions)


def test_unsupported_model_identifies_fallback_and_invalid_rows_fail() -> None:
    mlp = MLPSurgeonModel(
        ModelTask.REGRESSION,
        ("feature",),
        "perplexity",
        (1,),
        "unused",
        MLPConfig(ModelTask.REGRESSION, hidden_sizes=(1,)),
        1,
        4,
        0,
    )
    fallback = attribute_predictions(mlp, ((1.0,),))
    assert isinstance(fallback, AttributionUnavailable)
    assert fallback.available_fallbacks == ("permutation",)
    linear = LinearSurgeonModel(("x",), "y", (1.0,), 0.0, LinearConfig(), 1)
    with pytest.raises(AttributionError, match="finite"):
        attribute_predictions(linear, ((math.nan,),))
