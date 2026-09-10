"""Тесты оценки по типу задачи и приведения SHAP без Spark."""

from __future__ import annotations

import numpy as np
import pytest

from fmlib.feature_selection.config import FeatureSelectionConfig
from fmlib.feature_selection.exceptions import ConfigError, ExecutionError
from fmlib.feature_selection.schema import FeatureSchema
from fmlib.feature_selection.utils.task_runtime import (
    lgbm_objective_params,
    normalize_binary_shap_values,
    optuna_direction,
    resolve_task,
    score_model,
    shap_mean_abs,
)


def test_resolve_task_encodes_multiclass_and_keeps_binary_labels() -> None:
    binary = resolve_task("binary_classification", np.array([1, 0, 1, 0]), method_name="lightgbm")
    assert binary.is_binary
    assert list(binary.encoded_target) == [1, 0, 1, 0]

    multi = resolve_task("classification", np.array(["b", "a", "c", "a"]), method_name="lightgbm")
    assert multi.n_classes == 3
    assert set(multi.encoded_target.tolist()) == {0, 1, 2}

    regression = resolve_task("regression", np.array([1.5, 2.0, 0.5]), method_name="lightgbm")
    assert regression.is_regression
    assert regression.encoded_target.dtype == float

    raw = np.array(["b", "a", "c", "a"])
    kept = resolve_task(
        "classification",
        raw,
        method_name="catboost_rfe",
        encode_labels=False,
    )
    assert list(kept.encoded_target) == ["b", "a", "c", "a"]
    assert kept.n_classes == 3


def test_binary_requires_two_classes() -> None:
    with pytest.raises(ExecutionError, match="exactly two"):
        resolve_task("binary_classification", np.array([0, 0, 0]), method_name="lightgbm")


def test_lgbm_objectives_and_optuna_direction() -> None:
    binary = resolve_task("binary_classification", np.array([0, 1]), method_name="t")
    multi = resolve_task("classification", np.array([0, 1, 2]), method_name="t")
    regression = resolve_task("regression", np.array([0.1, 0.2]), method_name="t")

    assert lgbm_objective_params(binary) == {"objective": "binary", "metric": "auc"}
    assert lgbm_objective_params(multi) == {
        "objective": "multiclass",
        "metric": "multi_logloss",
        "num_class": 3,
    }
    assert lgbm_objective_params(regression) == {"objective": "regression", "metric": "rmse"}
    assert optuna_direction(binary) == "maximize"
    assert optuna_direction(multi) == "maximize"
    assert optuna_direction(regression) == "minimize"


def test_shap_binary_uses_positive_class() -> None:
    class_zero = np.ones((3, 2))
    class_one = np.array([[1.0, -2.0], [3.0, -4.0], [5.0, -6.0]])
    matrix = normalize_binary_shap_values([class_zero, class_one])
    np.testing.assert_array_equal(matrix, class_one)
    importance = shap_mean_abs("binary_classification", [class_zero, class_one])
    np.testing.assert_allclose(importance, np.abs(class_one).mean(axis=0))


def test_shap_multiclass_averages_abs_over_classes() -> None:
    class_a = np.array([[1.0, 0.0], [1.0, 0.0]])
    class_b = np.array([[0.0, 2.0], [0.0, 2.0]])
    class_c = np.array([[0.0, 0.0], [0.0, 0.0]])
    importance = shap_mean_abs("classification", [class_a, class_b, class_c])
    # mean |SHAP| over classes then rows: feature0 = 1/3, feature1 = 2/3
    np.testing.assert_allclose(importance, [1.0 / 3.0, 2.0 / 3.0])

    stacked = np.stack([class_a, class_b, class_c], axis=-1)
    np.testing.assert_allclose(shap_mean_abs("classification", stacked), importance)


def test_shap_regression_is_mean_abs_over_rows() -> None:
    values = np.array([[1.0, -2.0], [3.0, -4.0]])
    np.testing.assert_allclose(shap_mean_abs("regression", values), [2.0, 3.0])


def test_score_model_matches_task_metric_direction() -> None:
    binary = resolve_task("binary_classification", np.array([0, 1, 0, 1]), method_name="t")
    multi = resolve_task("classification", np.array(["a", "b", "c", "a"]), method_name="t")
    regression = resolve_task("regression", np.array([0.0, 1.0, 2.0]), method_name="t")

    class _Binary:
        def predict_proba(self, features: object) -> np.ndarray:
            del features
            return np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])

    class _Multi:
        classes_ = np.array(["a", "b", "c"])

        def predict_proba(self, features: object) -> np.ndarray:
            del features
            return np.array(
                [
                    [0.8, 0.1, 0.1],
                    [0.1, 0.8, 0.1],
                    [0.1, 0.1, 0.8],
                    [0.7, 0.2, 0.1],
                ],
            )

    class _Reg:
        def predict(self, features: object) -> np.ndarray:
            del features
            return np.array([0.0, 1.0, 2.0])

    dummy = np.zeros((4, 1))
    assert score_model(binary, _Binary(), dummy, binary.encoded_target) > 0.5
    assert score_model(multi, _Multi(), dummy[:4], np.array(["a", "b", "c", "a"])) > 0.5
    assert score_model(regression, _Reg(), dummy[:3], regression.encoded_target) == 0.0
    assert optuna_direction(binary) == "maximize"
    assert optuna_direction(multi) == "maximize"
    assert optuna_direction(regression) == "minimize"


def test_execution_task_type_rejects_unknown_and_defaults_to_binary() -> None:
    config = FeatureSelectionConfig.from_dict({"execution": {"seed": 1}})
    assert config.execution.task_type == "binary_classification"

    with pytest.raises(ConfigError, match="execution.task_type"):
        FeatureSelectionConfig.from_dict({"execution": {"task_type": "clustering"}})


def test_order_prerequisites_require_matching_task_type() -> None:
    from fmlib.feature_selection.base import StageContext
    from fmlib.feature_selection.runner import validate_order_prerequisites

    config = FeatureSelectionConfig.from_dict(
        {
            "order": [{"lightgbm": {}}],
            "execution": {"task_type": "regression"},
        },
    )
    schema = FeatureSchema(
        categorical=(),
        continuous=("a",),
        target="y",
        task_type="binary_classification",
    )
    context = StageContext(
        spark=None,
        datasets={"train": None},
        schema=schema,
        config=config,
        seed=0,
        candidates=["a"],
    )
    with pytest.raises(ConfigError, match="execution.task_type must match"):
        validate_order_prerequisites(context)
