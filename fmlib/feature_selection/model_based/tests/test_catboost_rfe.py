"""Tests for the CatBoost RFE selector and its helpers.

Guards, out-of-time split, mixed materialization and YAML stay local. The
decision-building test trains a tiny real CatBoost RFE; missing CatBoost fails
the run instead of skipping.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fmlib.feature_selection.base import StageContext
from fmlib.feature_selection.config import FeatureSelectionConfig
from fmlib.feature_selection.utils.conftest import require_spark_session
from fmlib.feature_selection.exceptions import ConfigError, ExecutionError
from fmlib.feature_selection.model_based.catboost_rfe import CatBoostRfeSelector, split_out_of_time
from fmlib.feature_selection.schema import FeatureSchema
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


def test_select_requires_binary_classification() -> None:
    schema = FeatureSchema(
        categorical=("cat_a",),
        continuous=("num_a",),
        target="target",
        task_type="regression",
        time="month_part",
    )
    context = _context(_frame(), schema=schema)
    selector = CatBoostRfeSelector(context.config.model)
    with pytest.raises(ExecutionError, match="binary_classification"):
        selector.select(context, ["cat_a", "num_a"])


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


# --- search space ----------------------------------------------------------


def test_scalar_parameters_fall_back_to_the_default_grid() -> None:
    context = _context(_frame(), schema=_schema())
    options = CatBoostRfeSelector(context.config.model)._resolve_options(context)

    assert options["optuna_enabled"] is True
    assert options["fixed_params"] == _PARAMETERS
    assert "depth" not in options["search_space"]
    assert options["search_space"]["learning_rate"] == CATBOOST_RFE_SEARCH_SPACE["learning_rate"]


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
