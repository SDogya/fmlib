"""Tests for LightGbmSelector."""

from __future__ import annotations

import sys
from decimal import Decimal
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

import fmlib.feature_selection.model_based.lightgbm as lightgbm_module
from fmlib.feature_selection.base import StageContext
from fmlib.feature_selection.config import FeatureSelectionConfig
from fmlib.feature_selection.utils.conftest import require_spark_session
from fmlib.feature_selection.exceptions import BackendError, ExecutionError
from fmlib.feature_selection.model_based.lightgbm import (
    DEFAULT_SEARCH_SPACE,
    LightGbmSelector,
    normalize_binary_shap_values,
)
from fmlib.feature_selection.schema import FeatureSchema
from fmlib.feature_selection.utils.local_data import sample_size


def _context(
    frame: object,
    *,
    categorical: tuple[str, ...] = ("category",),
    continuous: tuple[str, ...] = ("first", "second"),
    task_type: str = "binary_classification",
    params: dict[str, Any] | None = None,
    max_local_rows: int = 1_000,
    seed: int = 17,
) -> StageContext:
    config = FeatureSelectionConfig.from_dict(
        {
            "model": {
                "method": "lightgbm",
                "params": {
                    "n_trials": 1,
                    "n_folds": 2,
                    **(params or {}),
                },
            },
            "execution": {
                "seed": seed,
                "max_local_rows": max_local_rows,
            },
        },
    )
    schema = FeatureSchema(
        categorical=categorical,
        continuous=continuous,
        target="response",
        task_type=task_type,
    )
    candidates = schema.candidate_features()
    return StageContext(
        spark=require_spark_session(),
        datasets={"train": frame},
        schema=schema,
        config=config,
        seed=seed,
        candidates=candidates,
    )


def _frame(n_rows: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category": ["a", "b"] * (n_rows // 2),
            "first": np.arange(n_rows, dtype=float),
            "second": np.arange(n_rows, dtype=float) * 2.0,
            "not_a_candidate": np.arange(n_rows, dtype=float) * 3.0,
            "response": [0, 1] * (n_rows // 2),
        },
    )


def _mock_details() -> dict[str, Any]:
    return {
        "selected_features": ["first"],
        "global_best_params": {"n_estimators": 120},
        "fold_best_params": {
            "1": {"n_estimators": 120},
            "2": {"n_estimators": 120},
        },
        "importances_df": pd.DataFrame(
            {
                "feature": ["first", "second"],
                "lgbm_norm": [0.6, 0.4],
                "shap_norm": [0.7, 0.3],
                "lgbm_cumsum": [0.6, 1.0],
                "shap_cumsum": [0.7, 1.0],
            },
        ),
    }


def _mock_backends(selector: LightGbmSelector, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        selector,
        "_load_backends",
        lambda *args, **kwargs: (object(), object(), object()),
    )


def _require_ml_backends() -> None:
    """Fail immediately when the LightGBM extra is missing. Do not skip."""
    try:
        import lightgbm  # noqa: F401
        import optuna  # noqa: F401
        import shap  # noqa: F401
        import sklearn  # noqa: F401
    except ImportError as exc:
        pytest.fail(f"Install the lightgbm extra (lightgbm, shap, optuna, sklearn). Root cause: {exc}")
    if lightgbm_module.shap is None or lightgbm_module.optuna is None:
        pytest.fail("Install the lightgbm extra (lightgbm, shap, optuna, sklearn).")


_TINY_FIXED_PARAMS = {
    "n_estimators": 8,
    "num_leaves": 8,
    "learning_rate": 0.1,
    "max_depth": 2,
    "min_child_samples": 1,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
}


def test_selector_is_lightgbm() -> None:
    context = _context(_frame())

    selector = LightGbmSelector(context.config.model)

    assert isinstance(selector, LightGbmSelector)
    assert selector.method_name == "lightgbm"


def test_select_scopes_training_to_continuous_candidates_and_passes_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(_frame())
    selector = LightGbmSelector(context.config.model)
    captured: dict[str, Any] = {}
    _mock_backends(selector, monkeypatch)

    def fake_select(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return _mock_details()

    monkeypatch.setattr(selector, "_select_robust_features", fake_select)

    decisions = selector.select(
        context,
        ["category", "first", "second"],
    )

    assert captured["feature_cols"] == ["first", "second"]
    assert captured["seed"] == 17
    assert [decision.feature for decision in decisions] == ["first", "second"]
    assert decisions[0].keep is True
    assert decisions[0].reason == "passed_lgbm_shap_selection"
    assert decisions[0].stage == "model"
    assert decisions[0].method == "lightgbm"
    assert decisions[0].value == pytest.approx(0.7 / 0.85)
    assert decisions[0].threshold == 1.0
    assert decisions[1].keep is False
    assert decisions[1].reason == "failed_lgbm_shap_selection"
    assert "category" not in {decision.feature for decision in decisions}


def test_select_stores_flat_json_compatible_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(_frame())
    selector = LightGbmSelector(context.config.model)
    _mock_backends(selector, monkeypatch)
    monkeypatch.setattr(
        selector,
        "_select_robust_features",
        lambda **_kwargs: _mock_details(),
    )

    selector.select(context, context.candidates)

    scores = context.scores["lightgbm"]
    assert "lightgbm" not in scores
    assert scores["importances"] == {"first": 0.6, "second": 0.4}
    assert scores["shap_importances"] == {"first": 0.7, "second": 0.3}
    assert scores["lgbm_threshold"] == 0.85
    assert scores["shap_threshold"] == 0.85
    assert scores["selection_mode"] == "aggregated"
    assert scores["min_set_share"] == 1.0
    assert scores["optuna_mode"] == "global"
    assert scores["fold_execution"] == "driver"
    assert scores["global_best_params"] == {"n_estimators": 120}
    assert set(scores["fold_best_params"]) == {"1", "2"}


def test_only_categorical_candidates_pass_through_without_dependencies() -> None:
    frame = pd.DataFrame(
        {
            "category": ["a", "b"],
            "response": [0, 1],
        },
    )
    context = _context(
        frame,
        categorical=("category",),
        continuous=(),
    )

    decisions = LightGbmSelector(context.config.model).select(
        context,
        ["category"],
    )

    assert decisions == []


def test_empty_candidates_and_missing_train_are_handled_before_dependencies() -> None:
    context = _context(_frame())
    selector = LightGbmSelector(context.config.model)

    assert selector.select(context, []) == []

    context.datasets = {}
    with pytest.raises(ExecutionError, match="'train' split"):
        selector.select(context, context.candidates)


def test_rejects_non_binary_task_before_loading_dependencies() -> None:
    context = _context(_frame(), task_type="regression")
    selector = LightGbmSelector(context.config.model)

    with pytest.raises(
        ExecutionError,
        match="only task_type='binary_classification'",
    ):
        selector.select(
            context,
            context.candidates,
        )


def test_options_use_typed_fallbacks_and_execution_row_cap() -> None:
    context = _context(
        _frame(),
        params={"max_rows": 500, "n_folds": 4},
        max_local_rows=7,
    )
    selector = LightGbmSelector(context.config.model)

    options = selector._resolve_options(context)

    assert options["max_rows"] == 7
    assert options["n_folds"] == 4
    assert options["n_trials"] == 1
    assert options["optuna_mode"] == "global"
    assert options["selection_mode"] == "aggregated"
    assert options["min_set_share"] == 1.0
    assert options["n_jobs"] == -1
    assert options["shap_max_rows"] == 5_000
    assert options["seed"] == 17


def test_options_enable_per_fold_nested_tuning() -> None:
    context = _context(
        _frame(),
        params={
            "optuna_mode": "per_fold",
            "n_jobs": 2,
            "shap_max_rows": 500,
        },
    )

    options = LightGbmSelector(
        context.config.model,
    )._resolve_options(context)

    assert options["optuna_mode"] == "per_fold"
    assert options["n_jobs"] == 2
    assert options["shap_max_rows"] == 500


def test_options_enable_vote_selection() -> None:
    context = _context(
        _frame(),
        params={
            "selection_mode": "vote",
            "min_set_share": 0.5,
        },
    )

    options = LightGbmSelector(
        context.config.model,
    )._resolve_options(context)

    assert options["selection_mode"] == "vote"
    assert options["min_set_share"] == 0.5


def test_options_accept_legacy_driver_n_jobs_alias() -> None:
    context = _context(
        _frame(),
        params={"driver_n_jobs": 3},
    )

    options = LightGbmSelector(
        context.config.model,
    )._resolve_options(context)

    assert options["n_jobs"] == 3


def test_options_prefer_n_jobs_over_legacy_driver_n_jobs() -> None:
    context = _context(
        _frame(),
        params={"n_jobs": 4, "driver_n_jobs": 2},
    )

    options = LightGbmSelector(
        context.config.model,
    )._resolve_options(context)

    assert options["n_jobs"] == 4


@pytest.mark.parametrize(
    ("mode", "expected_driver_calls"),
    [("global", 1), ("per_fold", 0)],
)
def test_optuna_mode_controls_driver_tuning_and_fold_payloads(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_driver_calls: int,
) -> None:
    _require_ml_backends()
    context = _context(
        _frame(),
        params={
            "optuna_mode": mode,
            "n_trials": 1,
            "n_folds": 2,
            "n_jobs": 1,
            "shap_max_rows": 32,
            "parameters": {
                "n_estimators": {"type": "int", "min": 8, "max": 10},
                "num_leaves": 8,
                "learning_rate": 0.1,
                "max_depth": 2,
                "min_child_samples": 1,
            },
            "optuna_params": {"n_trials": 1, "n_startup_trials": 1},
        },
    )
    selector = LightGbmSelector(context.config.model)
    options = selector._resolve_options(context)
    tune_calls: list[dict[str, Any]] = []
    captured_folds: list[dict[str, Any]] = []
    original_tune = lightgbm_module.tune_parameters
    original_run = selector._run_fold

    def wrapped_tune(
        matrix: np.ndarray,
        target: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        tune_calls.append(kwargs)
        return original_tune(matrix, target, **kwargs)

    def wrapped_run(
        matrix: np.ndarray,
        target: np.ndarray,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        captured_folds.append(kwargs)
        return original_run(matrix, target, **kwargs)

    monkeypatch.setattr(lightgbm_module, "tune_parameters", wrapped_tune)
    monkeypatch.setattr(selector, "_run_fold", wrapped_run)

    details = selector._select_robust_features(
        df=_frame(),
        target_col="response",
        feature_cols=["first", "second"],
        n_trials=1,
        n_folds=2,
        max_rows_limit=20,
        sample_fraction=None,
        lgbm_threshold=0.85,
        shap_threshold=0.85,
        seed=17,
        optuna_mode=mode,
        n_jobs=1,
        shap_max_rows=32,
        search_space=options["search_space"],
        fixed_params=options["fixed_params"],
        n_startup_trials=1,
        return_importances=True,
    )

    assert len(tune_calls) == expected_driver_calls
    assert len(captured_folds) == 2
    assert all(fold["optuna_mode"] == mode for fold in captured_folds)
    assert all(fold["n_jobs"] == 1 for fold in captured_folds)
    assert [fold["seed"] for fold in captured_folds] == [18, 19]
    if mode == "global":
        assert tune_calls[0]["n_jobs"] == 1
        assert all(fold["global_params"] is not None for fold in captured_folds)
        assert details["global_best_params"] is not None
        assert 8 <= details["global_best_params"]["n_estimators"] <= 10
    else:
        assert all(fold["global_params"] is None for fold in captured_folds)
        assert details["global_best_params"] is None
    assert set(details["fold_best_params"]) == {"1", "2"}


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"lgbm_threshold": 0.0}, "lgbm_threshold"),
        ({"shap_threshold": 1.1}, "shap_threshold"),
        ({"n_trials": 0}, "n_trials"),
        ({"n_folds": 1}, "n_folds"),
        ({"sample_fraction": 0.0}, "sample_fraction"),
        ({"optuna_mode": "driver"}, "optuna_mode"),
        ({"selection_mode": "mean"}, "selection_mode"),
        ({"min_set_share": 0.0}, "min_set_share"),
        ({"n_jobs": 0}, "n_jobs"),
        ({"driver_n_jobs": 0}, "n_jobs"),
        ({"n_jobs": -2}, "n_jobs"),
        ({"shap_max_rows": 0}, "shap_max_rows"),
    ],
)
def test_invalid_options_have_actionable_errors(
    params: dict[str, Any],
    message: str,
) -> None:
    context = _context(_frame(), params=params)

    with pytest.raises(ExecutionError, match=message):
        LightGbmSelector(context.config.model)._resolve_options(context)


def test_pandas_preparation_is_stratified_bounded_and_deterministic() -> None:
    context = _context(_frame())
    selector = LightGbmSelector(context.config.model)

    first_matrix, first_target, features = selector._extract_and_prep_data(
        _frame(),
        "response",
        ["first", "second"],
        max_rows=8,
        sample_fraction=None,
        seed=19,
    )
    second_matrix, second_target, _ = selector._extract_and_prep_data(
        _frame(),
        "response",
        ["first", "second"],
        max_rows=8,
        sample_fraction=None,
        seed=19,
    )

    assert features == ["first", "second"]
    assert first_matrix.shape == (8, 2)
    assert np.array_equal(first_matrix, second_matrix)
    assert np.array_equal(first_target, second_target)
    classes, counts = np.unique(first_target, return_counts=True)
    assert set(classes) == {0, 1}
    assert int(counts.sum()) == 8
    assert int(counts.min()) >= 1


def test_pandas_preparation_keeps_numeric_nulls() -> None:
    frame = _frame()
    frame.loc[0, "first"] = np.nan
    selector = LightGbmSelector(
        FeatureSelectionConfig.from_dict(
            {
                "model": {
                    "method": "lightgbm",
                    "params": {"n_trials": 1, "n_folds": 2},
                },
                "execution": {"seed": 17, "max_local_rows": 1_000},
            },
        ).model,
    )

    matrix, _, _ = selector._extract_and_prep_data(
        frame,
        "response",
        ["first", "second"],
        max_rows=len(frame),
        sample_fraction=None,
        seed=17,
    )

    assert int(np.isnan(matrix).sum()) == 1


def test_pandas_preparation_accepts_decimal_numeric_objects() -> None:
    frame = _frame()
    frame["first"] = [
        Decimal(str(value))
        for value in frame["first"]
    ]
    selector = LightGbmSelector(_context(frame).config.model)

    matrix, _, _ = selector._extract_and_prep_data(
        frame,
        "response",
        ["first", "second"],
        max_rows=len(frame),
        sample_fraction=None,
        seed=17,
    )

    assert matrix.dtype == np.float64
    # Row order is canonical, not input order, so compare the column's contents.
    assert sorted(matrix[:, 0].tolist()) == sorted(
        float(value) for value in frame["first"]
    )


def test_pandas_continuous_columns_must_exist_and_be_numeric() -> None:
    selector = LightGbmSelector(_context(_frame()).config.model)

    with pytest.raises(ExecutionError, match="columns missing.*missing"):
        selector._extract_and_prep_data(
            _frame(),
            "response",
            ["missing"],
            max_rows=20,
            sample_fraction=None,
            seed=17,
        )

    non_numeric = _frame().assign(first="text")
    with pytest.raises(
        ExecutionError,
        match=r"FeatureSchema\.continuous.*first",
    ):
        selector._extract_and_prep_data(
            non_numeric,
            "response",
            ["first", "second"],
            max_rows=20,
            sample_fraction=None,
            seed=17,
        )


def test_preparation_rejects_unsupported_input_and_missing_target() -> None:
    selector = LightGbmSelector(_context(_frame()).config.model)

    with pytest.raises(ExecutionError, match="unsupported train split type"):
        selector._extract_and_prep_data(
            object(),
            "response",
            ["first"],
            max_rows=20,
            sample_fraction=None,
            seed=17,
        )

    missing_target = _frame()
    missing_target.loc[0, "response"] = np.nan
    with pytest.raises(ExecutionError, match="target column contains missing"):
        selector._extract_and_prep_data(
            missing_target,
            "response",
            ["first", "second"],
            max_rows=20,
            sample_fraction=None,
            seed=17,
        )


def test_aggregate_importances_preserves_cumulative_intersection() -> None:
    details = LightGbmSelector._aggregate_importances(
        ["first", "second", "third", "fourth"],
        np.array([0.4, 0.3, 0.2, 0.1]),
        np.array([0.5, 0.25, 0.15, 0.1]),
        lgbm_threshold=0.75,
        shap_threshold=0.75,
    )

    assert details["lgbm_selected"] == ["first", "second"]
    assert details["shap_selected"] == ["first", "second"]
    assert details["selected_features"] == ["first", "second"]


def test_vote_importances_requires_all_sets_when_share_is_one() -> None:
    features = ["first", "second", "third"]
    # threshold 0.8 keeps the prefix whose cumsum is <= 0.8.
    # [0.5, 0.3, 0.2] -> first+second; [0.6, 0.3, 0.1] -> first only.
    vote = LightGbmSelector._vote_importances(
        features,
        [
            np.array([0.5, 0.3, 0.2]),
            np.array([0.5, 0.3, 0.2]),
        ],
        [
            np.array([0.5, 0.3, 0.2]),
            np.array([0.6, 0.3, 0.1]),
        ],
        lgbm_threshold=0.8,
        shap_threshold=0.8,
        min_set_share=1.0,
    )

    assert vote["n_sets"] == 4
    assert vote["set_presence"] == {
        "first": 1.0,
        "second": 0.75,
        "third": 0.0,
    }
    assert vote["selected_features"] == ["first"]
    assert vote["fold_sets"]["1"]["lgbm"] == ["first", "second"]
    assert vote["fold_sets"]["2"]["shap"] == ["first"]


def test_vote_importances_keeps_features_that_hit_the_share() -> None:
    features = ["first", "second", "third"]
    vote = LightGbmSelector._vote_importances(
        features,
        [
            np.array([0.5, 0.3, 0.2]),
            np.array([0.5, 0.3, 0.2]),
        ],
        [
            np.array([0.5, 0.3, 0.2]),
            np.array([0.6, 0.3, 0.1]),
        ],
        lgbm_threshold=0.8,
        shap_threshold=0.8,
        min_set_share=0.75,
    )

    assert vote["selected_features"] == ["first", "second"]
    assert "third" not in vote["selected_features"]


def test_vote_importances_rejects_zero_fold_totals() -> None:
    with pytest.raises(ExecutionError, match="non-positive total"):
        LightGbmSelector._vote_importances(
            ["first"],
            [np.array([0.0]), np.array([1.0])],
            [np.array([1.0]), np.array([1.0])],
            lgbm_threshold=0.85,
            shap_threshold=0.85,
            min_set_share=1.0,
        )


def test_select_vote_uses_set_presence_and_vote_reasons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(
        _frame(),
        params={"selection_mode": "vote", "min_set_share": 0.75},
    )
    selector = LightGbmSelector(context.config.model)
    _mock_backends(selector, monkeypatch)
    details = _mock_details()
    details["selected_features"] = ["first"]
    details["set_presence"] = {"first": 1.0, "second": 0.5}
    details["n_sets"] = 4
    details["fold_sets"] = {
        "1": {"lgbm": ["first"], "shap": ["first"]},
        "2": {"lgbm": ["first"], "shap": ["first", "second"]},
    }
    monkeypatch.setattr(
        selector,
        "_select_robust_features",
        lambda **_kwargs: details,
    )

    decisions = selector.select(context, ["first", "second"])

    assert decisions[0].keep is True
    assert decisions[0].reason == "passed_lgbm_shap_vote"
    assert decisions[0].value == pytest.approx(1.0)
    assert decisions[0].threshold == pytest.approx(0.75)
    assert decisions[1].keep is False
    assert decisions[1].reason == "failed_lgbm_shap_vote"
    assert decisions[1].value == pytest.approx(0.5)
    scores = context.scores["lightgbm"]
    assert scores["selection_mode"] == "vote"
    assert scores["min_set_share"] == 0.75
    assert scores["n_sets"] == 4
    assert scores["set_presence"] == {"first": 1.0, "second": 0.5}
    assert scores["fold_sets"]["2"]["shap"] == ["first", "second"]


def test_aggregate_importances_rejects_zero_totals() -> None:
    with pytest.raises(ExecutionError, match="non-positive total"):
        LightGbmSelector._aggregate_importances(
            ["first"],
            np.array([0.0]),
            np.array([1.0]),
            lgbm_threshold=0.85,
            shap_threshold=0.85,
        )
    with pytest.raises(ExecutionError, match="SHAP importances"):
        LightGbmSelector._aggregate_importances(
            ["first"],
            np.array([1.0]),
            np.array([0.0]),
            lgbm_threshold=0.85,
            shap_threshold=0.85,
        )


def test_shap_output_normalization_supports_known_binary_shapes() -> None:
    class_zero = np.zeros((3, 2))
    class_one = np.ones((3, 2))
    three_dimensional = np.stack([class_zero, class_one], axis=2)

    assert np.array_equal(
        normalize_binary_shap_values([class_zero, class_one]),
        class_one,
    )
    assert np.array_equal(
        normalize_binary_shap_values(three_dimensional),
        class_one,
    )

    class Explanation:
        values = class_one

    assert np.array_equal(
        normalize_binary_shap_values([class_zero, Explanation()]),
        class_one,
    )
    with pytest.raises(ExecutionError, match="unsupported SHAP output shape"):
        normalize_binary_shap_values(np.ones(5))


def test_missing_shap_and_optuna_raise_backend_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector = LightGbmSelector(_context(_frame()).config.model)
    monkeypatch.setitem(sys.modules, "lightgbm", ModuleType("lightgbm"))
    monkeypatch.setattr(lightgbm_module, "shap", None)

    with pytest.raises(BackendError, match="SHAP"):
        selector._load_backends()

    monkeypatch.setattr(lightgbm_module, "shap", object())
    monkeypatch.setattr(lightgbm_module, "optuna", None)
    with pytest.raises(BackendError, match="Optuna"):
        selector._load_backends()


def test_sample_size_applies_fraction_and_capacity_caps() -> None:
    assert sample_size(100, max_rows=80, sample_fraction=None) == 80
    assert sample_size(100, max_rows=80, sample_fraction=0.25) == 25
    assert sample_size(100, max_rows=10, sample_fraction=0.25) == 10


def test_pandas_end_to_end_with_ml_backends() -> None:
    _require_ml_backends()
    rng = np.random.default_rng(23)
    target = np.tile([0, 1], 50)
    frame = pd.DataFrame(
        {
            "category": np.where(target == 1, "positive", "negative"),
            "signal": target + rng.normal(0.0, 0.05, len(target)),
            "noise": rng.normal(0.0, 1.0, len(target)),
            "response": target,
        },
    )
    context = _context(
        frame,
        categorical=("category",),
        continuous=("signal", "noise"),
        params={
            "n_trials": 1,
            "n_folds": 2,
            "max_rows": len(frame),
            "optuna_mode": "global",
            "n_jobs": 1,
            "parameters": dict(_TINY_FIXED_PARAMS),
            "optuna_params": {"enabled": False},
        },
    )
    decisions = LightGbmSelector(context.config.model).select(
        context,
        context.candidates,
    )

    assert [decision.feature for decision in decisions] == [
        "signal",
        "noise",
    ]
    assert "category" not in {
        decision.feature for decision in decisions
    }
    assert context.scores["lightgbm"]["fold_execution"] == "driver"
    assert set(context.scores["lightgbm"]["importances"]) == {
        "signal",
        "noise",
    }


def test_spark_preparation_uses_declared_columns_and_supports_dots(spark: Any) -> None:
    frame = spark.createDataFrame(
        [
            (float(index), float(index * 10), index % 2)
            for index in range(10)
        ],
        ["foo.bar", "unused", "response"],
    )
    selector = LightGbmSelector(
        _context(
            pd.DataFrame(),
            categorical=(),
            continuous=("foo.bar",),
        ).config.model,
    )

    matrix, target, features = selector._extract_and_prep_data(
        frame,
        "response",
        ["foo.bar"],
        max_rows=20,
        sample_fraction=None,
        seed=17,
    )

    assert features == ["foo.bar"]
    assert matrix.shape == (10, 1)
    assert target.shape == (10,)
    # Row order is canonical, not partition order, so compare as a set.
    assert sorted(matrix[:, 0].tolist()) == [float(index) for index in range(10)]


def test_search_space_defaults_to_established_ranges() -> None:
    context = _context(_frame())
    selector = LightGbmSelector(context.config.model)

    options = selector._resolve_options(context)

    assert options["search_space"] == DEFAULT_SEARCH_SPACE
    assert options["fixed_params"] == {}
    assert options["optuna_enabled"] is True


def test_search_space_overrides_come_from_config() -> None:
    context = _context(
        _frame(),
        params={
            "parameters": {
                "num_leaves": {"type": "int", "min": 16, "max": 128},
                "min_child_samples": 20,
            },
        },
    )
    selector = LightGbmSelector(context.config.model)

    options = selector._resolve_options(context)

    assert options["search_space"]["num_leaves"] == {"type": "int", "min": 16, "max": 128}
    assert options["search_space"] == {"num_leaves": {"type": "int", "min": 16, "max": 128}}
    assert options["fixed_params"] == {"min_child_samples": 20}
    assert "learning_rate" not in options["search_space"]
    assert "min_child_samples" not in options["search_space"]


def test_optuna_params_block_is_read_like_the_other_selectors() -> None:
    context = _context(
        _frame(),
        params={
            "optuna_params": {
                "n_trials": 25,
                "n_startup_trials": 4,
                "sampler": "random",
                "timeout": 120,
            },
        },
    )
    selector = LightGbmSelector(context.config.model)

    options = selector._resolve_options(context)

    assert options["n_trials"] == 25
    assert options["n_startup_trials"] == 4
    assert options["sampler"] == "RANDOM"
    assert options["timeout"] == 120
    assert options["optuna_enabled"] is True


def test_optuna_disabled_clears_the_search_space() -> None:
    context = _context(
        _frame(),
        params={"optuna_params": {"enabled": False}},
    )
    selector = LightGbmSelector(context.config.model)

    options = selector._resolve_options(context)

    assert options["optuna_enabled"] is False
    assert options["search_space"] == {}


def test_optuna_disabled_skips_driver_tuning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_ml_backends()
    context = _context(
        _frame(),
        params={
            "parameters": dict(_TINY_FIXED_PARAMS),
            "optuna_params": {"enabled": False},
            "n_folds": 2,
            "n_jobs": 1,
            "shap_max_rows": 32,
        },
    )
    selector = LightGbmSelector(context.config.model)
    tune_calls: list[dict[str, Any]] = []
    original_tune = lightgbm_module.tune_parameters

    def wrapped_tune(
        matrix: np.ndarray,
        target: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        tune_calls.append(kwargs)
        return original_tune(matrix, target, **kwargs)

    monkeypatch.setattr(lightgbm_module, "tune_parameters", wrapped_tune)

    options = selector._resolve_options(context)
    selector._select_robust_features(
        df=_frame(),
        target_col="response",
        feature_cols=["first", "second"],
        n_trials=2,
        n_folds=2,
        max_rows_limit=20,
        sample_fraction=None,
        lgbm_threshold=0.85,
        shap_threshold=0.85,
        seed=17,
        optuna_mode="global",
        n_jobs=1,
        shap_max_rows=32,
        search_space=options["search_space"],
        fixed_params=options["fixed_params"],
        return_importances=True,
    )

    assert tune_calls == []


def test_legacy_n_trials_still_works_as_the_fallback() -> None:
    legacy_context = _context(_frame(), params={"n_trials": 7})

    legacy = LightGbmSelector(legacy_context.config.model)._resolve_options(
        legacy_context,
    )

    assert legacy["n_trials"] == 7
    assert legacy["sampler"] == "TPE"


def test_optuna_params_win_over_the_legacy_n_trials_key() -> None:
    context = _context(
        _frame(),
        params={"n_trials": 7, "optuna_params": {"n_trials": 25}},
    )

    options = LightGbmSelector(context.config.model)._resolve_options(context)

    assert options["n_trials"] == 25


def test_n_trials_falls_back_to_the_shared_default() -> None:
    config = FeatureSelectionConfig.from_dict(
        {"model": {"method": "lightgbm", "params": {"n_folds": 2}}},
    )
    context = _context(_frame())
    context.config = config

    options = LightGbmSelector(config.model)._resolve_options(context)

    assert options["n_trials"] == 20


def test_scalar_overrides_a_default_range_instead_of_being_ignored() -> None:
    context = _context(
        _frame(),
        params={"parameters": {"learning_rate": 0.05, "max_depth": 4}},
    )
    selector = LightGbmSelector(context.config.model)

    options = selector._resolve_options(context)

    # Pinned parameters must leave the search space: otherwise Optuna keeps
    # suggesting values for them and the suggestion overrides the constant.
    assert options["fixed_params"] == {"learning_rate": 0.05, "max_depth": 4}
    assert "learning_rate" not in options["search_space"]
    assert "max_depth" not in options["search_space"]
    assert "num_leaves" in options["search_space"]


def test_configured_search_space_reaches_tuning_and_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_ml_backends()
    context = _context(
        _frame(),
        params={
            "n_trials": 1,
            "n_folds": 2,
            "n_jobs": 1,
            "shap_max_rows": 32,
            "parameters": {
                "num_leaves": {"type": "int", "min": 8, "max": 16},
                "n_estimators": 8,
                "learning_rate": 0.1,
                "max_depth": 2,
                "min_child_samples": 1,
            },
            "optuna_params": {"n_trials": 1, "n_startup_trials": 1},
        },
    )
    selector = LightGbmSelector(context.config.model)
    seen: dict[str, Any] = {}
    original_tune = lightgbm_module.tune_parameters
    original_run = selector._run_fold

    def wrapped_tune(
        matrix: np.ndarray,
        target: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        seen["tune_space"] = kwargs["search_space"]
        seen["tune_fixed"] = kwargs["fixed_params"]
        return original_tune(matrix, target, **kwargs)

    def wrapped_run(
        matrix: np.ndarray,
        target: np.ndarray,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        seen.setdefault("fold_space", kwargs["search_space"])
        return original_run(matrix, target, **kwargs)

    monkeypatch.setattr(lightgbm_module, "tune_parameters", wrapped_tune)
    monkeypatch.setattr(selector, "_run_fold", wrapped_run)

    options = selector._resolve_options(context)
    selector._select_robust_features(
        df=_frame(),
        target_col="response",
        feature_cols=["first", "second"],
        n_trials=1,
        n_folds=2,
        max_rows_limit=20,
        sample_fraction=None,
        lgbm_threshold=0.85,
        shap_threshold=0.85,
        seed=17,
        optuna_mode="global",
        n_jobs=1,
        shap_max_rows=32,
        search_space=options["search_space"],
        fixed_params=options["fixed_params"],
        n_startup_trials=1,
        return_importances=True,
    )

    assert seen["tune_space"]["num_leaves"]["max"] == 16
    assert seen["fold_space"]["num_leaves"]["max"] == 16
    assert seen["tune_fixed"]["n_estimators"] == 8
    assert seen["tune_fixed"]["learning_rate"] == 0.1
    assert seen["tune_fixed"]["early_stopping_rounds"] == 200
    assert "learning_rate" not in seen["tune_space"]


def test_trial_parameters_preserve_search_space_and_execution_limits() -> None:
    _require_ml_backends()
    import optuna

    captured: dict[str, Any] = {}

    def objective(trial: Any) -> float:
        captured["params"] = lightgbm_module.build_trial_parameters(
            trial,
            seed=7,
            n_jobs=2,
            search_space={
                "n_estimators": {"type": "int", "min": 8, "max": 12},
                "learning_rate": {"type": "float", "min": 0.01, "max": 0.2, "log": True},
            },
        )
        return 0.0

    optuna.create_study(direction="maximize").optimize(objective, n_trials=1)
    params = captured["params"]
    assert 8 <= params["n_estimators"] <= 12
    assert 0.01 <= params["learning_rate"] <= 0.2
    assert params["random_state"] == 7
    assert params["bagging_seed"] == 7
    assert params["feature_fraction_seed"] == 7
    assert params["data_random_seed"] == 7
    assert params["extra_seed"] == 7
    assert params["deterministic"] is True
    assert params["force_row_wise"] is True
    assert params["n_jobs"] == 2
    assert params["objective"] == "binary"


def test_folds_run_in_order_and_importances_are_averaged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_ml_backends()
    context = _context(
        _frame(),
        params={
            "parameters": dict(_TINY_FIXED_PARAMS),
            "optuna_params": {"enabled": False},
            "n_folds": 2,
            "n_jobs": 1,
            "shap_max_rows": 32,
        },
    )
    selector = LightGbmSelector(context.config.model)
    options = selector._resolve_options(context)
    call_order: list[int] = []
    original_run = selector._run_fold

    def wrapped_run(
        matrix: np.ndarray,
        target: np.ndarray,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        call_order.append(kwargs["fold_index"])
        return original_run(matrix, target, **kwargs)

    monkeypatch.setattr(selector, "_run_fold", wrapped_run)

    details = selector._select_robust_features(
        df=_frame(),
        target_col="response",
        feature_cols=["first", "second"],
        n_trials=1,
        n_folds=2,
        max_rows_limit=20,
        sample_fraction=None,
        lgbm_threshold=0.85,
        shap_threshold=0.85,
        seed=17,
        optuna_mode="global",
        n_jobs=1,
        shap_max_rows=32,
        search_space=options["search_space"],
        fixed_params=options["fixed_params"],
        return_importances=True,
    )

    assert call_order == [1, 2]
    assert set(details["fold_best_params"]) == {"1", "2"}
    assert details["global_best_params"]["learning_rate"] == 0.1
    assert details["global_best_params"]["early_stopping_rounds"] == 200
    assert details["global_best_params"]["n_estimators"] == 8
    importances = details["importances_df"].set_index("feature")
    assert importances["lgbm_norm"].sum() == pytest.approx(1.0)
    assert importances["shap_norm"].sum() == pytest.approx(1.0)


def test_global_fold_uses_shared_params_with_driver_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_ml_backends()
    import lightgbm as lgb

    original = lgb.LGBMClassifier
    seen: list[dict[str, Any]] = []

    def spy_classifier(**params: Any) -> Any:
        seen.append(dict(params))
        return original(**params)

    monkeypatch.setattr(lgb, "LGBMClassifier", spy_classifier)
    matrix = np.arange(40, dtype=float).reshape(20, 2)
    target = np.array([0, 1] * 10)

    lgbm_importances, shap_importances, best_params = LightGbmSelector._run_fold(
        matrix,
        target,
        fold_index=1,
        valid_indices=np.array([0, 1], dtype=np.int64),
        seed=31,
        optuna_mode="global",
        n_trials=1,
        n_jobs=1,
        shap_max_rows=8,
        global_params={**_TINY_FIXED_PARAMS, "n_estimators": 12, "n_jobs": -1},
        search_space={},
        fixed_params=_TINY_FIXED_PARAMS,
    )

    assert lgbm_importances.shape == (2,)
    assert shap_importances.shape == (2,)
    assert np.all(lgbm_importances >= 0.0)
    assert best_params["n_estimators"] == 12
    assert seen[-1]["n_jobs"] == 1
    assert seen[-1]["random_state"] == 31
    assert seen[-1]["n_estimators"] == 12


def test_per_fold_mode_tunes_only_on_outer_train_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_ml_backends()
    matrix = np.arange(40, dtype=float).reshape(20, 2)
    target = np.array([0, 1] * 10)
    captured: dict[str, Any] = {}
    original_tune = lightgbm_module.tune_parameters

    def wrapped_tune(
        train_matrix: np.ndarray,
        train_target: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        captured["matrix"] = train_matrix
        captured["target"] = train_target
        captured.update(kwargs)
        return original_tune(train_matrix, train_target, **kwargs)

    monkeypatch.setattr(lightgbm_module, "tune_parameters", wrapped_tune)

    _lgbm, _shap, best_params = LightGbmSelector._run_fold(
        matrix,
        target,
        fold_index=2,
        valid_indices=np.array([0, 1], dtype=np.int64),
        seed=42,
        optuna_mode="per_fold",
        n_trials=1,
        n_jobs=1,
        shap_max_rows=16,
        n_startup_trials=1,
        search_space={"n_estimators": {"type": "int", "min": 8, "max": 10}},
        fixed_params={
            "num_leaves": 8,
            "learning_rate": 0.1,
            "max_depth": 2,
        },
    )

    assert captured["matrix"].shape == (18, 2)
    assert captured["target"].tolist() == target[2:].tolist()
    assert captured["n_trials"] == 1
    assert captured["seed"] == 42
    assert captured["n_jobs"] == 1
    assert 8 <= best_params["n_estimators"] <= 10


def test_fold_without_global_params_reports_missing_tuning() -> None:
    _require_ml_backends()
    matrix = np.arange(12, dtype=float).reshape(6, 2)
    target = np.array([0, 1, 0, 1, 0, 1])

    with pytest.raises(ExecutionError, match="no global Optuna"):
        LightGbmSelector._run_fold(
            matrix,
            target,
            fold_index=3,
            valid_indices=np.array([0], dtype=np.int64),
            seed=5,
            optuna_mode="global",
            n_trials=1,
            n_jobs=1,
            shap_max_rows=5,
            global_params=None,
        )


# --- selection_mode end to end ---------------------------------------------


_VOTE_CONTINUOUS = ("driver_a", "driver_b", "noise_0", "noise_1", "noise_2")


def _vote_frame(n_rows: int = 240) -> pd.DataFrame:
    """Frame whose target follows two drivers plus noise.

    The signal is deliberately spread over two features: a single dominant
    feature would exceed the cumulative threshold on its own and leave the
    cut empty, which says nothing about ``selection_mode``.
    """
    rng = np.random.default_rng(0)
    driver_a = rng.normal(0.0, 1.0, n_rows)
    driver_b = rng.normal(0.0, 1.0, n_rows)
    logit = 0.6 * driver_a + 0.4 * driver_b + rng.normal(0.0, 0.8, n_rows)
    columns: dict[str, Any] = {
        "category": ["a", "b"] * (n_rows // 2),
        "driver_a": driver_a,
        "driver_b": driver_b,
        "response": (logit > 0.0).astype(int),
    }
    for index in range(3):
        columns[f"noise_{index}"] = rng.normal(0.0, 1.0, n_rows)
    return pd.DataFrame(columns)


def _run_selection_mode(
    mode: str,
    *,
    min_set_share: float = 1.0,
    n_folds: int = 3,
) -> tuple[list[str], dict[str, Any], list[Any]]:
    """Run a real LightGBM selection in ``mode`` and return kept/scores/decisions."""
    frame = _vote_frame()
    context = _context(
        frame,
        categorical=("category",),
        continuous=_VOTE_CONTINUOUS,
        params={
            "n_trials": 1,
            "n_folds": n_folds,
            "max_rows": len(frame),
            "optuna_mode": "global",
            "n_jobs": 1,
            "selection_mode": mode,
            "min_set_share": min_set_share,
            "parameters": dict(_TINY_FIXED_PARAMS),
            "optuna_params": {"enabled": False},
        },
        max_local_rows=len(frame),
    )
    decisions = LightGbmSelector(context.config.model).select(
        context,
        context.candidates,
    )
    kept = [decision.feature for decision in decisions if decision.keep]
    return kept, context.scores["lightgbm"], decisions


def test_both_selection_modes_run_and_record_their_mode() -> None:
    """Each mode reaches the scores payload and decides every continuous candidate."""
    _require_ml_backends()
    for mode in ("aggregated", "vote"):
        kept, scores, decisions = _run_selection_mode(mode)

        assert scores["selection_mode"] == mode
        # Categorical candidates are skipped, continuous ones all get a decision.
        assert [decision.feature for decision in decisions] == list(_VOTE_CONTINUOUS)
        assert set(kept) <= set(_VOTE_CONTINUOUS)


def test_vote_mode_reports_one_set_per_fold_and_channel() -> None:
    """``vote`` cuts split and SHAP separately in every fold: ``2 * n_folds`` sets."""
    _require_ml_backends()
    n_folds = 3
    _, scores, _ = _run_selection_mode("vote", n_folds=n_folds)

    assert scores["n_sets"] == 2 * n_folds
    assert set(scores["fold_sets"]) == {"1", "2", "3"}
    for fold in scores["fold_sets"].values():
        assert set(fold) == {"lgbm", "shap"}
    assert set(scores["set_presence"]) == set(_VOTE_CONTINUOUS)
    assert all(0.0 <= share <= 1.0 for share in scores["set_presence"].values())


def test_aggregated_mode_reports_no_vote_payload() -> None:
    """The vote-only keys stay out of the scores payload in ``aggregated``."""
    _require_ml_backends()
    _, scores, _ = _run_selection_mode("aggregated")

    assert "n_sets" not in scores
    assert "set_presence" not in scores
    assert "fold_sets" not in scores


def test_lowering_min_set_share_admits_partially_present_features() -> None:
    """``min_set_share`` is a real cut, not a value carried through unused.

    The relaxed threshold is derived from the observed presence values rather
    than hard-coded: a feature that appears in some but not all sets must be
    admitted once the share drops to its own presence.
    """
    _require_ml_backends()
    strict, scores, _ = _run_selection_mode("vote", min_set_share=1.0)
    presence = scores["set_presence"]
    partial = {
        feature: share
        for feature, share in presence.items()
        if 0.0 < share < 1.0
    }
    assert partial, f"fixture produced no partially-present feature: {presence}"

    relaxed, _, _ = _run_selection_mode("vote", min_set_share=min(partial.values()))

    assert set(strict) < set(relaxed)
    assert set(partial) <= set(relaxed)


def test_vote_decisions_carry_presence_against_the_share_threshold() -> None:
    """In ``vote`` the reported value is set presence, not a cumulative ratio."""
    _require_ml_backends()
    min_set_share = 0.5
    _, scores, decisions = _run_selection_mode("vote", min_set_share=min_set_share)

    for decision in decisions:
        assert decision.threshold == min_set_share
        assert decision.value == pytest.approx(scores["set_presence"][decision.feature])
        expected = "passed_lgbm_shap_vote" if decision.keep else "failed_lgbm_shap_vote"
        assert decision.reason == expected
