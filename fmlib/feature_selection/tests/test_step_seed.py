"""Тесты разрешения params.seed и параметров seed в LightGBM без Spark."""

from __future__ import annotations

from typing import Any

from fmlib.feature_selection.base import StageContext
from fmlib.feature_selection.config import FeatureSelectionConfig
from fmlib.feature_selection.model_based import lightgbm as lightgbm_module
from fmlib.feature_selection.model_based.catboost_rfe import CatBoostRfeSelector
from fmlib.feature_selection.model_based.lightgbm import LightGbmSelector
from fmlib.feature_selection.model_based.boruta_shap import BorutaShapSelector
from fmlib.feature_selection.schema import FeatureSchema


def _schema() -> FeatureSchema:
    return FeatureSchema(
        categorical=(),
        continuous=("first",),
        target="response",
        task_type="binary_classification",
    )


def _context(
    *,
    method: str,
    params: dict[str, Any],
    seed: int = 42,
    extra: dict[str, Any] | None = None,
) -> StageContext:
    payload: dict[str, Any] = {
        "execution": {"seed": seed, "max_local_rows": 1_000},
    }
    model: dict[str, Any] = {
        "enabled": True,
        "method": method,
        "params": params,
    }
    if extra:
        model.update(extra)
    payload["model"] = model
    config = FeatureSelectionConfig.from_dict(payload)
    return StageContext(
        spark=None,
        datasets={"train": None},
        schema=_schema(),
        config=config,
        seed=seed,
        candidates=["first"],
    )


def test_lightgbm_options_prefer_params_seed() -> None:
    context = _context(
        method="lightgbm",
        params={"n_folds": 2, "seed": 17},
        seed=42,
    )
    options = LightGbmSelector(context.config.model)._resolve_options(context)
    assert options["seed"] == 17


def test_lightgbm_options_fall_back_to_execution_seed() -> None:
    context = _context(method="lightgbm", params={"n_folds": 2}, seed=42)
    options = LightGbmSelector(context.config.model)._resolve_options(context)
    assert options["seed"] == 42


def test_catboost_options_prefer_params_seed() -> None:
    context = _context(
        method="catboost_rfe",
        params={"seed": 17, "parameters": {"iterations": 10}},
        seed=42,
        extra={"selection": {"max_features": 5}},
    )
    options = CatBoostRfeSelector(context.config.model)._resolve_options(context)
    assert options["seed"] == 17


def test_boruta_options_prefer_params_seed() -> None:
    context = _context(
        method="boruta_shap",
        params={"seed": 17, "optuna_params": {"enabled": False}},
        seed=42,
    )
    options = BorutaShapSelector(context.config.model)._resolve_options(context)
    assert options["seed"] == 17


def test_finalize_parameters_pins_library_seeds() -> None:
    params = lightgbm_module._finalize_parameters(
        {"n_estimators": 8, "force_col_wise": True},
        seed=17,
        n_jobs=1,
    )
    assert params["random_state"] == 17
    assert params["bagging_seed"] == 17
    assert params["feature_fraction_seed"] == 17
    assert params["data_random_seed"] == 17
    assert params["extra_seed"] == 17
    assert params["deterministic"] is True
    assert params["force_row_wise"] is True
    assert "force_col_wise" not in params
    assert params["n_jobs"] == 1


def test_boruta_lgbm_build_pins_library_seeds() -> None:
    class _FakeModel:
        def __init__(self, **params: Any) -> None:
            self.params = params

    lgbm = BorutaShapSelector._build_model(_FakeModel, "lgbm", {"n_estimators": 8}, 17)
    assert lgbm.params["random_state"] == 17
    assert lgbm.params["bagging_seed"] == 17
    assert lgbm.params["feature_fraction_seed"] == 17
    assert lgbm.params["data_random_seed"] == 17
    assert lgbm.params["extra_seed"] == 17
    assert lgbm.params["deterministic"] is True
    assert lgbm.params["force_row_wise"] is True

    forest = BorutaShapSelector._build_model(_FakeModel, "rf", {}, 17)
    assert forest.params["random_state"] == 17
    assert "bagging_seed" not in forest.params
    assert "deterministic" not in forest.params
