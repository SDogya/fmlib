"""Spark-free tests for load-time model parameter validation."""

from __future__ import annotations

import pytest

from fmlib.feature_selection.exceptions import ConfigError
from fmlib.feature_selection.utils.model_param_validate import validate_model_parameters
from fmlib.feature_selection.utils.optuna_space import validate_parameter_spec


def test_catboost_alias_clash() -> None:
    with pytest.raises(ConfigError, match="aliases together"):
        validate_model_parameters(
            {"iterations": 10, "n_estimators": 20},
            library="catboost",
            method_name="catboost_rfe",
        )


def test_lightgbm_alias_clash() -> None:
    with pytest.raises(ConfigError, match="aliases together"):
        validate_model_parameters(
            {"n_estimators": 8, "num_iterations": 8},
            library="lightgbm",
            method_name="lightgbm",
        )


def test_unknown_name_suggests_close_match() -> None:
    with pytest.raises(ConfigError, match="n_estimators"):
        validate_model_parameters(
            {"n_estmators": 8},
            library="lightgbm",
            method_name="lightgbm",
        )


def test_empty_catboost_parameters_rejected() -> None:
    with pytest.raises(ConfigError, match="must not be empty"):
        validate_model_parameters(
            {},
            library="catboost",
            method_name="catboost_rfe",
            require_non_empty=True,
        )


def test_rf_rejects_lightgbm_only_key() -> None:
    with pytest.raises(ConfigError, match="unknown parameter"):
        validate_model_parameters(
            {"num_leaves": 16},
            library="random_forest",
            method_name="boruta_shap",
        )


def test_bootstrap_type_ignored_for_boruta_lgbm() -> None:
    validate_model_parameters(
        {"bootstrap_type": "MVS", "n_estimators": 8},
        library="lightgbm",
        method_name="boruta_shap",
        ignore_keys=frozenset({"bootstrap_type"}),
    )


def test_revalidate_boruta_lgbm_ignores_bootstrap_type() -> None:
    from fmlib.feature_selection.runner import _revalidate_model_parameters

    _revalidate_model_parameters(
        "boruta_shap",
        {
            "model_type": "lgbm",
            "parameters": {"bootstrap_type": "MVS", "n_estimators": 8},
        },
    )


def test_validate_parameter_spec_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="must define 'min' and 'max'"):
        validate_parameter_spec(
            "depth",
            {"type": "int"},
            method_name="catboost_rfe",
        )
