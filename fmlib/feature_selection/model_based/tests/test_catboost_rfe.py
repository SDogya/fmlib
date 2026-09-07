"""Tests for the CatBoost RFE selector and its helpers.

Guards, out-of-time split, mixed materialization and YAML stay local. The
decision-building test trains a tiny real CatBoost RFE; missing CatBoost fails
the run instead of skipping.
"""

from __future__ import annotations

from itertools import pairwise

import pandas as pd
import pytest

from fmlib.feature_selection.base import StageContext
from fmlib.feature_selection.config import FeatureSelectionConfig
from fmlib.feature_selection.exceptions import ConfigError, ExecutionError
from fmlib.feature_selection.model_based.catboost_rfe import (
    CatBoostRfeSelector,
    _pop_elimination_schedule,
    constant_drop_targets,
    run_catboost_rfe,
    split_out_of_time,
)
from fmlib.feature_selection.schema import FeatureSchema
from fmlib.feature_selection.utils.conftest import require_spark_session
from fmlib.feature_selection.utils.default_model_param_spaces import CATBOOST_RFE_SEARCH_SPACE
from fmlib.feature_selection.utils.local_data import prepare_mixed_frame

_PARAMETERS = {"depth": 4, "iterations": 20}


def _require_catboost() -> None:
    try:
        import catboost  # noqa: F401
    except ImportError as exc:
        pytest.fail(f"Install the catboost extra. Root cause: {exc}")


def _frame(n_rows: int = 60, months: tuple[str, ...] = ("2024-01", "2024-02", "2024-03")) -> pd.DataFrame:
    rows = []
    for index in range(n_rows):
        rows.append(
            {
                "num_a": float(index),
                "num_b": float(index % 7),
                "cat_a": f"c{index % 3}",
                "month_part": months[index % len(months)],
                "target": index % 2,
            },
        )
    return pd.DataFrame(rows)


def _schema(time: str | None = "month_part") -> FeatureSchema:
    return FeatureSchema(
        categorical=("cat_a",),
        continuous=("num_a", "num_b"),
        target="target",
        task_type="binary_classification",
        time=time,
    )


def _context(
    frame: pd.DataFrame,
    *,
    schema: FeatureSchema,
    params: dict | None = None,
    max_features: int = 1,
) -> StageContext:
    config = FeatureSelectionConfig.from_dict(
        {
            "model": {
                "method": "catboost_rfe",
                "params": params if params is not None else {"parameters": _PARAMETERS},
                "selection": {"max_features": max_features},
            },
            "execution": {"seed": 42, "task_type": schema.task_type},
        },
    )
    return StageContext(
        spark=require_spark_session(),
        datasets={"train": frame},
        schema=schema,
        config=config,
        seed=42,
        candidates=["cat_a", "num_a", "num_b"],
    )


# --- out-of-time split -----------------------------------------------------


def test_split_out_of_time_allows_unique_regression_target() -> None:
    frame = _frame()
    frame["target"] = 1.0
    fit, evaluation, periods = split_out_of_time(
        frame,
        time_col="month_part",
        target_col="target",
        eval_months=1,
        method_name="catboost_rfe",
        task_type="regression",
    )
    assert periods == ["2024-03"]
    assert not fit.empty
    assert not evaluation.empty


def test_split_out_of_time_reserves_latest_periods() -> None:
    fit, evaluation, periods = split_out_of_time(
        _frame(),
        time_col="month_part",
        target_col="target",
        eval_months=1,
        method_name="catboost_rfe",
    )
    assert periods == ["2024-03"]
    assert set(evaluation["month_part"]) == {"2024-03"}
    assert set(fit["month_part"]) == {"2024-01", "2024-02"}
    assert len(fit) + len(evaluation) == 60


def test_split_out_of_time_requires_enough_periods() -> None:
    with pytest.raises(ExecutionError, match="distinct periods"):
        split_out_of_time(
            _frame(months=("2024-01",)),
            time_col="month_part",
            target_col="target",
            eval_months=1,
            method_name="catboost_rfe",
        )


def test_split_out_of_time_rejects_missing_periods() -> None:
    frame = _frame()
    frame.loc[0, "month_part"] = None
    with pytest.raises(ExecutionError, match="contains missing values"):
        split_out_of_time(
            frame,
            time_col="month_part",
            target_col="target",
            eval_months=1,
            method_name="catboost_rfe",
        )


# --- mixed materialization -------------------------------------------------


def test_prepare_mixed_frame_keeps_categories_and_nans() -> None:
    frame = _frame()
    frame.loc[0, "num_a"] = None
    frame.loc[1, "cat_a"] = None

    local = prepare_mixed_frame(
        frame,
        target_col="target",
        feature_cols=["cat_a", "num_a", "num_b"],
        categorical_cols=["cat_a"],
        extra_cols=("month_part",),
        max_rows=1_000,
        sample_fraction=None,
        seed=42,
        method_name="catboost_rfe",
    )

    assert "month_part" in local.columns
    # dtype differs between pandas 2.x (object) and 3.x (str); values are what matters
    assert all(isinstance(value, str) for value in local["cat_a"])
    assert "None" in set(local["cat_a"])
    # numeric NaNs survive: CatBoost handles them natively
    assert local["num_a"].isna().sum() == 1


def test_prepare_mixed_frame_rejects_non_numeric_continuous() -> None:
    frame = _frame()
    frame["num_a"] = "abc"
    with pytest.raises(ExecutionError, match="could not be converted"):
        prepare_mixed_frame(
            frame,
            target_col="target",
            feature_cols=["num_a"],
            categorical_cols=[],
            extra_cols=(),
            max_rows=1_000,
            sample_fraction=None,
            seed=42,
            method_name="catboost_rfe",
        )


# --- selector guards -------------------------------------------------------


def test_select_requires_time_column() -> None:
    context = _context(_frame(), schema=_schema(time=None))
    selector = CatBoostRfeSelector(context.config.model)
    with pytest.raises(ExecutionError, match="FeatureSchema.time is required"):
        selector.select(context, ["cat_a", "num_a", "num_b"])


def test_select_runs_classification_and_regression() -> None:
    _require_catboost()
    params = {
        "parameters": _PARAMETERS,
        "optuna_params": {"enabled": False},
    }

    def pandas_context(frame: pd.DataFrame, schema: FeatureSchema) -> StageContext:
        config = FeatureSelectionConfig.from_dict(
            {
                "model": {
                    "method": "catboost_rfe",
                    "params": params,
                    "selection": {"max_features": 1},
                },
                "execution": {"seed": 42, "task_type": schema.task_type},
            },
        )
        return StageContext(
            spark=None,
            datasets={"train": frame},
            schema=schema,
            config=config,
            seed=42,
            candidates=["cat_a", "num_a", "num_b"],
        )

    class_frame = _frame()
    # month_part is index % 3; cycle the label on a different axis so every
    # out-of-time part still has all three classes.
    class_frame["target"] = [(index // 3) % 3 for index in range(len(class_frame))]
    class_schema = FeatureSchema(
        categorical=("cat_a",),
        continuous=("num_a", "num_b"),
        target="target",
        task_type="classification",
        time="month_part",
    )
    class_context = pandas_context(class_frame, class_schema)
    class_decisions = CatBoostRfeSelector(class_context.config.model).select(
        class_context,
        ["cat_a", "num_a", "num_b"],
    )
    assert {item.feature for item in class_decisions} == {"cat_a", "num_a", "num_b"}
    assert sum(item.keep for item in class_decisions) == 1

    reg_frame = _frame()
    reg_frame["target"] = [float(index) for index in range(len(reg_frame))]
    reg_schema = FeatureSchema(
        categorical=("cat_a",),
        continuous=("num_a", "num_b"),
        target="target",
        task_type="regression",
        time="month_part",
    )
    reg_context = pandas_context(reg_frame, reg_schema)
    reg_decisions = CatBoostRfeSelector(reg_context.config.model).select(
        reg_context,
        ["cat_a", "num_a", "num_b"],
    )
    assert {item.feature for item in reg_decisions} == {"cat_a", "num_a", "num_b"}
    assert class_context.scores["catboost_rfe"]["best_params"]["loss_function"] == "MultiClass"
    assert reg_context.scores["catboost_rfe"]["best_params"]["loss_function"] == "RMSE"


def test_select_requires_parameters() -> None:
    context = _context(_frame(), schema=_schema(), params={})
    selector = CatBoostRfeSelector(context.config.model)
    with pytest.raises(ExecutionError, match="model.params.parameters is required"):
        selector.select(context, ["cat_a", "num_a", "num_b"])


def test_select_is_noop_when_candidates_fit_target() -> None:
    context = _context(_frame(), schema=_schema(), max_features=10)
    selector = CatBoostRfeSelector(context.config.model)

    decisions = selector.select(context, ["cat_a", "num_a", "num_b"])

    assert decisions == []
    assert context.scores["catboost_rfe"]["skipped"] is True


def test_select_builds_decisions_and_scores() -> None:
    _require_catboost()
    context = _context(
        _frame(),
        schema=_schema(),
        params={
            "parameters": _PARAMETERS,
            "optuna_params": {"enabled": False},
        },
        max_features=1,
    )
    selector = CatBoostRfeSelector(context.config.model)
    decisions = selector.select(context, ["cat_a", "num_a", "num_b"])

    kept = [item.feature for item in decisions if item.keep]
    dropped = [item.feature for item in decisions if not item.keep]
    assert len(kept) == 1
    assert set(dropped) == {"cat_a", "num_a", "num_b"} - set(kept)
    assert {item.feature for item in decisions} == {"cat_a", "num_a", "num_b"}
    assert all(item.reason == "catboost_rfe_selected" for item in decisions if item.keep)
    assert all(item.reason == "catboost_rfe_eliminated" for item in decisions if not item.keep)
    assert all(item.value is not None and item.value >= 1.0 for item in decisions if not item.keep)

    scores = context.scores["catboost_rfe"]
    assert scores["eval_strategy"] == "out_of_time"
    assert scores["eval_periods"] == ["2024-03"]
    assert scores["categorical_evaluated"] == ["cat_a"]
    assert set(scores["selected_features"]) == set(kept)


# --- loss curve ------------------------------------------------------------


def test_scores_carry_the_loss_curve() -> None:
    """The eval loss per elimination step must reach the caller.

    Paired with ``elimination_order`` the curve is what lets
    ``selection.max_features`` be read off a measured cutoff instead of
    guessed, so it has to survive into ``scores`` and stay JSON-friendly.
    """
    _require_catboost()
    steps = 2
    context = _context(
        _frame(),
        schema=_schema(),
        params={
            "parameters": _PARAMETERS,
            "optuna_params": {"enabled": False},
            "feature_selection_params": {"steps": steps},
        },
        max_features=1,
    )
    CatBoostRfeSelector(context.config.model).select(
        context,
        ["cat_a", "num_a", "num_b"],
    )

    scores = context.scores["catboost_rfe"]
    assert scores["steps"] == steps
    assert scores["n_candidates"] == 3

    graph = scores["loss_graph"]
    assert set(graph) >= {"removed_features_count", "loss_values", "main_indices"}
    assert len(graph["loss_values"]) == len(graph["removed_features_count"])
    # An unmeasured curve is two points; steps > 1 has to add real measurements.
    assert len(graph["main_indices"]) > 1
    assert max(graph["main_indices"]) < len(graph["loss_values"])
    # The x axis counts removals, so it starts at zero and never decreases.
    removed = graph["removed_features_count"]
    assert removed[0] == 0
    assert removed == sorted(removed)


def test_steps_default_leaves_more_than_one_measurement() -> None:
    """Without an explicit ``steps`` CatBoost eliminates in one pass.

    That yields a two-point curve with no interior measurement, so the module
    pins its own default rather than inheriting CatBoost's.
    """
    _require_catboost()
    context = _context(
        _frame(),
        schema=_schema(),
        params={
            "parameters": _PARAMETERS,
            "optuna_params": {"enabled": False},
        },
        max_features=1,
    )
    CatBoostRfeSelector(context.config.model).select(
        context,
        ["cat_a", "num_a", "num_b"],
    )

    scores = context.scores["catboost_rfe"]
    assert scores["steps"] > 1
    assert len(scores["loss_graph"]["main_indices"]) > 1


def test_non_positive_steps_are_rejected() -> None:
    """``steps`` below one cannot describe an elimination schedule."""
    with pytest.raises(ConfigError, match="feature_selection_params.steps"):
        FeatureSelectionConfig.from_dict(
            {
                "model": {
                    "enabled": True,
                    "method": "catboost_rfe",
                    "params": {
                        "parameters": _PARAMETERS,
                        "feature_selection_params": {"steps": 0},
                    },
                    "selection": {"max_features": 1},
                },
            },
        )


def test_constant_drop_targets_schedule() -> None:
    assert constant_drop_targets(10, 4, 3) == [7, 4]
    assert constant_drop_targets(8, 3, 10) == [3]
    assert constant_drop_targets(5, 5, 2) == []
    with pytest.raises(ValueError, match="feature_drop_per_step"):
        constant_drop_targets(10, 4, 0)


def test_runtime_rejects_steps_and_feature_drop_per_step_together() -> None:
    with pytest.raises(ExecutionError, match="cannot set both"):
        _pop_elimination_schedule(
            {"steps": 5, "feature_drop_per_step": 3},
            method_name="catboost_rfe",
        )


def test_config_rejects_steps_and_feature_drop_per_step_together() -> None:
    with pytest.raises(ConfigError, match="not both"):
        FeatureSelectionConfig.from_dict(
            {
                "model": {
                    "enabled": True,
                    "method": "catboost_rfe",
                    "params": {
                        "parameters": _PARAMETERS,
                        "feature_selection_params": {
                            "steps": 5,
                            "feature_drop_per_step": 3,
                        },
                    },
                    "selection": {"max_features": 8},
                },
            },
        )


def test_config_accepts_feature_drop_per_step_without_steps() -> None:
    config = FeatureSelectionConfig.from_dict(
        {
            "model": {
                "enabled": True,
                "method": "catboost_rfe",
                "params": {
                    "parameters": _PARAMETERS,
                    "feature_selection_params": {"feature_drop_per_step": 3},
                },
                "selection": {"max_features": 8},
            },
        },
    )
    assert config.model.params["feature_selection_params"]["feature_drop_per_step"] == 3
    assert "steps" not in config.model.params["feature_selection_params"]


def test_config_accepts_steps_without_feature_drop_per_step() -> None:
    config = FeatureSelectionConfig.from_dict(
        {
            "model": {
                "enabled": True,
                "method": "catboost_rfe",
                "params": {
                    "parameters": _PARAMETERS,
                    "feature_selection_params": {"steps": 4},
                },
                "selection": {"max_features": 8},
            },
        },
    )
    assert config.model.params["feature_selection_params"]["steps"] == 4


def test_feature_drop_per_step_drops_a_constant_count() -> None:
    """Each round removes ``feature_drop_per_step`` names until max_features."""
    _require_catboost()
    frame = _frame()
    frame["num_c"] = frame["num_a"] * 0.5
    frame["num_d"] = frame["num_b"] + 1.0
    frame["num_e"] = (frame["num_a"] % 5).astype(float)
    features = ["cat_a", "num_a", "num_b", "num_c", "num_d", "num_e"]
    drop_per_step = 2
    max_features = 3
    details = run_catboost_rfe(
        frame,
        feature_cols=features,
        categorical_cols=["cat_a"],
        target_col="target",
        time_col="month_part",
        eval_months=1,
        parameters=_PARAMETERS,
        optuna_params={"enabled": False},
        feature_selection_params={
            "algorithm": "RecursiveByPredictionValuesChange",
            "feature_drop_per_step": drop_per_step,
        },
        num_features_to_select=max_features,
        seed=0,
    )
    assert details["elimination_mode"] == "feature_drop_per_step"
    assert details["feature_drop_per_step"] == drop_per_step
    assert details["steps"] == 2
    assert len(details["eliminated_features"]) == len(features) - max_features
    assert len(details["selected_features"]) == max_features
    removed = details["loss_graph"]["removed_features_count"]
    assert removed[0] == 0
    deltas = [later - earlier for earlier, later in pairwise(removed)]
    assert deltas[:-1] == [drop_per_step] * (len(deltas) - 1)
    assert 1 <= deltas[-1] <= drop_per_step


def test_steps_mode_keeps_a_single_select_features_call() -> None:
    """The geometric ``steps`` path still goes through one CatBoost elimination."""
    _require_catboost()
    details = run_catboost_rfe(
        _frame(),
        feature_cols=["cat_a", "num_a", "num_b"],
        categorical_cols=["cat_a"],
        target_col="target",
        time_col="month_part",
        eval_months=1,
        parameters=_PARAMETERS,
        optuna_params={"enabled": False},
        feature_selection_params={
            "algorithm": "RecursiveByPredictionValuesChange",
            "steps": 2,
        },
        num_features_to_select=1,
        seed=0,
    )
    assert details["elimination_mode"] == "steps"
    assert details["steps"] == 2
    assert "feature_drop_per_step" not in details
    assert len(details["eliminated_features"]) == 2
    assert len(details["selected_features"]) == 1


# --- search space ----------------------------------------------------------


def test_scalar_parameters_fall_back_to_the_default_grid() -> None:
    context = _context(_frame(), schema=_schema())
    options = CatBoostRfeSelector(context.config.model)._resolve_options(context)

    assert options["optuna_enabled"] is True
    assert options["fixed_params"] == _PARAMETERS
    assert "depth" not in options["search_space"]
    assert options["search_space"]["l2_leaf_reg"] == CATBOOST_RFE_SEARCH_SPACE["l2_leaf_reg"]
    assert "learning_rate" not in options["search_space"]
    assert "iterations" not in options["search_space"]


def test_a_yaml_mapping_replaces_the_default_grid() -> None:
    context = _context(
        _frame(),
        schema=_schema(),
        params={
            "parameters": {
                "iterations": 10,
                "depth": {"type": "int", "min": 4, "max": 6},
            },
        },
    )
    options = CatBoostRfeSelector(context.config.model)._resolve_options(context)

    assert options["search_space"] == {"depth": {"type": "int", "min": 4, "max": 6}}
    assert "learning_rate" not in options["search_space"]
    assert options["fixed_params"] == {"iterations": 10}


def test_yaml_scalar_learning_rate_is_kept() -> None:
    _require_catboost()
    details = run_catboost_rfe(
        _frame(),
        feature_cols=["cat_a", "num_a", "num_b"],
        categorical_cols=["cat_a"],
        target_col="target",
        time_col="month_part",
        eval_months=1,
        parameters={**_PARAMETERS, "learning_rate": 0.1},
        optuna_params={"enabled": False},
        feature_selection_params={
            "algorithm": "RecursiveByPredictionValuesChange",
            "steps": 2,
        },
        num_features_to_select=1,
        seed=0,
    )

    assert details["fit_rows"] == 40
    assert details["best_params"]["learning_rate"] == 0.1
    assert details["best_params"]["early_stopping_rounds"] == 100
    assert details["best_params"]["iterations"] == 20
    assert details["best_params"]["use_best_model"] is True
    assert details["best_params"]["depth"] == 4


def test_optuna_disabled_clears_the_search_space() -> None:
    context = _context(
        _frame(),
        schema=_schema(),
        params={
            "parameters": _PARAMETERS,
            "optuna_params": {"enabled": False},
        },
    )
    options = CatBoostRfeSelector(context.config.model)._resolve_options(context)

    assert options["optuna_enabled"] is False
    assert options["search_space"] == {}
    assert options["fixed_params"] == _PARAMETERS


# --- config validation -----------------------------------------------------


def test_config_rejects_unknown_algorithm() -> None:
    with pytest.raises(ConfigError, match="feature_selection_params.algorithm"):
        FeatureSelectionConfig.from_dict(
            {
                "model": {
                    "enabled": True,
                    "method": "catboost_rfe",
                    "params": {
                        "parameters": _PARAMETERS,
                        "feature_selection_params": {"algorithm": "Greedy"},
                    },
                },
            },
        )


def test_config_rejects_bad_numeric_options() -> None:
    with pytest.raises(ConfigError, match="eval_months"):
        FeatureSelectionConfig.from_dict(
            {
                "model": {
                    "enabled": True,
                    "method": "catboost_rfe",
                    "params": {"parameters": _PARAMETERS, "eval_months": 0},
                },
            },
        )
    with pytest.raises(ConfigError, match="sample_fraction"):
        FeatureSelectionConfig.from_dict(
            {
                "model": {
                    "enabled": True,
                    "method": "catboost_rfe",
                    "params": {"parameters": _PARAMETERS, "sample_fraction": 1.5},
                },
            },
        )


def test_config_rejects_unknown_sampler() -> None:
    with pytest.raises(ConfigError, match="sampler"):
        FeatureSelectionConfig.from_dict(
            {
                "model": {
                    "enabled": True,
                    "method": "catboost_rfe",
                    "params": {
                        "parameters": _PARAMETERS,
                        "optuna_params": {"sampler": "CMAES"},
                    },
                },
            },
        )
