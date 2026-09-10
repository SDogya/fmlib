"""Тесты эвристик бустинга LightAutoML без Spark."""

from __future__ import annotations

from fmlib.feature_selection.utils.default_model_param_spaces import (
    BORUTA_LGBM_SEARCH_SPACE,
    CATBOOST_RFE_SEARCH_SPACE,
    LIGHTGBM_SEARCH_SPACE,
)
from fmlib.feature_selection.utils.lama_boost_defaults import (
    apply_boost_heuristics,
    boost_fixed_params,
    fit_lgbm_with_early_stopping,
    split_lgbm_early_stopping,
)
from fmlib.feature_selection.utils.optuna_space import resolve_tuning_space

_METHOD = "catboost_rfe"


def test_lightgbm_table_matches_binary_lama_buckets() -> None:
    params = boost_fixed_params(15_000, library="lightgbm")

    assert params["learning_rate"] == 0.02
    assert params["n_estimators"] == 3000
    assert params["early_stopping_rounds"] == 200


def test_catboost_table_matches_binary_lama_buckets() -> None:
    params = boost_fixed_params(10_000, library="catboost")

    assert params["learning_rate"] == 0.035
    assert params["iterations"] == 5000
    assert params["early_stopping_rounds"] == 100
    assert params["use_best_model"] is True


def test_catboost_table_matches_multiclass_and_regression_lama() -> None:
    multi = boost_fixed_params(
        10_000,
        library="catboost",
        task_type="classification",
    )
    assert multi["learning_rate"] == 0.03
    assert multi["iterations"] == 3000
    assert multi["early_stopping_rounds"] == 100

    multi_large = boost_fixed_params(
        200_000,
        library="catboost",
        task_type="classification",
    )
    assert multi_large["iterations"] == 4000

    regression = boost_fixed_params(
        10_000,
        library="catboost",
        task_type="regression",
    )
    assert regression["learning_rate"] == 0.05
    assert regression["iterations"] == 2000
    assert regression["early_stopping_rounds"] == 300


def test_one_mapping_still_gets_table_lr_and_patience() -> None:
    _fixed, search_space = resolve_tuning_space(
        {"depth": {"type": "int", "min": 3, "max": 7}},
        defaults=CATBOOST_RFE_SEARCH_SPACE,
        enabled=True,
        method_name=_METHOD,
    )

    fixed, space = apply_boost_heuristics(
        _fixed,
        search_space,
        n_rows=10_000,
        library="catboost",
    )

    assert space == {"depth": {"type": "int", "min": 3, "max": 7}}
    assert "learning_rate" not in space
    assert "early_stopping_rounds" not in space
    assert fixed["learning_rate"] == 0.035
    assert fixed["early_stopping_rounds"] == 100
    assert fixed["iterations"] == 5000
    assert fixed["use_best_model"] is True


def test_learning_rate_mapping_is_stripped_and_table_wins() -> None:
    _fixed, search_space = resolve_tuning_space(
        {
            "depth": {"type": "int", "min": 3, "max": 7},
            "learning_rate": {"type": "float", "min": 0.01, "max": 0.3, "log": True},
        },
        defaults=CATBOOST_RFE_SEARCH_SPACE,
        enabled=True,
        method_name=_METHOD,
    )

    fixed, space = apply_boost_heuristics(
        _fixed,
        search_space,
        n_rows=10_000,
        library="catboost",
    )

    assert "learning_rate" not in space
    assert "eta" not in space
    assert space["depth"]["max"] == 7
    assert fixed["learning_rate"] == 0.035


def test_yaml_scalar_learning_rate_is_kept_table_fills_the_rest() -> None:
    _fixed, search_space = resolve_tuning_space(
        {
            "learning_rate": 0.1,
            "depth": {"type": "int", "min": 3, "max": 7},
        },
        defaults=CATBOOST_RFE_SEARCH_SPACE,
        enabled=True,
        method_name=_METHOD,
    )

    fixed, space = apply_boost_heuristics(
        _fixed,
        search_space,
        n_rows=10_000,
        library="catboost",
    )

    assert "learning_rate" not in space
    assert fixed["learning_rate"] == 0.1
    assert fixed["early_stopping_rounds"] == 100
    assert fixed["iterations"] == 5000
    assert fixed["use_best_model"] is True
    assert _fixed["learning_rate"] == 0.1


def test_lightgbm_yaml_scalar_lr_is_kept_and_cap_filled() -> None:
    fixed, space = apply_boost_heuristics(
        {"learning_rate": 0.1},
        LIGHTGBM_SEARCH_SPACE,
        n_rows=15_000,
        library="lightgbm",
    )

    assert fixed["learning_rate"] == 0.1
    assert fixed["n_estimators"] == 3000
    assert fixed["early_stopping_rounds"] == 200
    assert "learning_rate" not in space
    assert "n_estimators" not in space
    assert "num_leaves" in space


def test_eta_scalar_blocks_table_learning_rate() -> None:
    fixed, _space = apply_boost_heuristics(
        {"eta": 0.05},
        LIGHTGBM_SEARCH_SPACE,
        n_rows=15_000,
        library="lightgbm",
    )

    assert fixed["eta"] == 0.05
    assert "learning_rate" not in fixed
    assert fixed["early_stopping_rounds"] == 200
    assert fixed["n_estimators"] == 3000


def test_pinned_use_best_model_false_is_kept() -> None:
    fixed, _space = apply_boost_heuristics(
        {"use_best_model": False, "iterations": 40},
        {"depth": {"type": "int", "min": 3, "max": 7}},
        n_rows=10_000,
        library="catboost",
    )

    assert fixed["use_best_model"] is False
    assert fixed["iterations"] == 40
    assert fixed["learning_rate"] == 0.035


def test_zero_early_stopping_is_kept_and_disables_patience() -> None:
    fixed, _space = apply_boost_heuristics(
        {"n_estimators": 500, "learning_rate": 0.05, "early_stopping_rounds": 0},
        {},
        n_rows=15_000,
        library="lightgbm",
    )

    assert fixed["early_stopping_rounds"] == 0
    ctor_params, rounds = split_lgbm_early_stopping(fixed)
    assert rounds is None
    assert "early_stopping_rounds" not in ctor_params

    class _Recorder:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] = {}

        def fit(self, features: object, target: object, **kwargs: object) -> None:
            self.kwargs = kwargs

    model = _Recorder()
    fit_lgbm_with_early_stopping(
        model,
        [[0.0]],
        [0],
        eval_set=[([[0.0]], [0])],
        early_stopping_rounds=0,
    )
    assert "callbacks" not in model.kwargs
    assert "early_stopping_rounds" not in model.kwargs
    assert "eval_set" in model.kwargs


def test_scalar_tree_cap_is_kept() -> None:
    fixed, _space = apply_boost_heuristics(
        {"iterations": 3000},
        {"depth": {"type": "int", "min": 3, "max": 7}},
        n_rows=10_000,
        library="catboost",
    )

    assert fixed["iterations"] == 3000
    assert fixed["learning_rate"] == 0.035


def test_default_spaces_omit_lr_and_tree_count() -> None:
    for space in (
        LIGHTGBM_SEARCH_SPACE,
        CATBOOST_RFE_SEARCH_SPACE,
        BORUTA_LGBM_SEARCH_SPACE,
    ):
        assert "learning_rate" not in space
        assert "n_estimators" not in space
        assert "iterations" not in space
        assert "early_stopping_rounds" not in space

    assert set(BORUTA_LGBM_SEARCH_SPACE) == set(LIGHTGBM_SEARCH_SPACE)
    assert set(LIGHTGBM_SEARCH_SPACE) == {
        "num_leaves",
        "colsample_bytree",
        "subsample",
        "min_child_weight",
    }
    assert set(CATBOOST_RFE_SEARCH_SPACE) == {
        "depth",
        "l2_leaf_reg",
        "min_data_in_leaf",
    }


def test_pinned_scalars_leave_the_lama_fallback_minus_lr() -> None:
    fixed, space = resolve_tuning_space(
        {"depth": 4, "iterations": 20},
        defaults=CATBOOST_RFE_SEARCH_SPACE,
        enabled=True,
        method_name=_METHOD,
    )

    assert fixed == {"depth": 4, "iterations": 20}
    assert "depth" not in space
    assert space["l2_leaf_reg"] == CATBOOST_RFE_SEARCH_SPACE["l2_leaf_reg"]
    assert "learning_rate" not in space
    assert "iterations" not in space
