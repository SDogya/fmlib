"""Tests for LightGbmSelector."""

from __future__ import annotations

import sys
from decimal import Decimal
from types import ModuleType
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

import fmlib.feature_selection.model_based.lightgbm as lightgbm_module
from fmlib.feature_selection.base import StageContext
from fmlib.feature_selection.config import FeatureSelectionConfig
from fmlib.feature_selection.exceptions import BackendError, ExecutionError
from fmlib.feature_selection.utils.local_data import sample_size
from fmlib.feature_selection.model_based.lightgbm import (
    DEFAULT_SEARCH_SPACE,
    LightGbmSelector,
    normalize_binary_shap_values,
)
from fmlib.feature_selection.schema import FeatureSchema
from fmlib.feature_selection.conftest import FakeSparkSession


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
        spark=FakeSparkSession(),
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
        lambda: (object(), object(), object()),
    )


def _pyspark_available() -> bool:
    try:
        import pyspark  # noqa: F401
    except ImportError:
        return False
    return True


def _ml_backends_available() -> bool:
    if lightgbm_module.shap is None or lightgbm_module.optuna is None:
        return False
    try:
        import lightgbm  # noqa: F401
        import sklearn  # noqa: F401
    except Exception:  # noqa: BLE001 - optional binary dependencies
        return False
    return True


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
    assert options["n_jobs"] == -1
    assert options["shap_max_rows"] == 5_000


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


class _FakeStratifiedKFold:
    def __init__(self: _FakeStratifiedKFold, **_kwargs: Any) -> None:
        return None

    def split(
        self: _FakeStratifiedKFold,
        _matrix: np.ndarray,
        _target: np.ndarray,
    ) -> Any:
        yield np.array([2, 3, 4, 5]), np.array([0, 1])
        yield np.array([0, 1, 4, 5]), np.array([2, 3])


def _install_fake_sklearn_model_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sklearn_module = ModuleType("sklearn")
    model_selection = ModuleType("sklearn.model_selection")
    model_selection.StratifiedKFold = _FakeStratifiedKFold
    sklearn_module.model_selection = model_selection
    monkeypatch.setitem(sys.modules, "sklearn", sklearn_module)
    monkeypatch.setitem(
        sys.modules,
        "sklearn.model_selection",
        model_selection,
    )


@pytest.mark.parametrize(
    ("mode", "expected_driver_calls"),
    [("global", 1), ("per_fold", 0)],
)
def test_optuna_mode_controls_driver_tuning_and_fold_payloads(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_driver_calls: int,
) -> None:
    _install_fake_sklearn_model_selection(monkeypatch)
    context = _context(_frame(), params={"optuna_mode": mode})
    selector = LightGbmSelector(context.config.model)
    matrix = np.arange(12, dtype=float).reshape(6, 2)
    target = np.array([0, 1, 0, 1, 0, 1])
    monkeypatch.setattr(
        selector,
        "_extract_and_prep_data",
        lambda *_args, **_kwargs: (
            matrix,
            target,
            ["first", "second"],
        ),
    )
    tune_calls: list[dict[str, Any]] = []
    captured_folds: list[dict[str, Any]] = []

    def fake_tune(
        _matrix: np.ndarray,
        _target: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        tune_calls.append(kwargs)
        return {"n_estimators": 120}

    def fake_run(
        _matrix: np.ndarray,
        _target: np.ndarray,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        captured_folds.append(kwargs)
        return (
            np.array([1.0, 0.5]),
            np.array([0.8, 0.4]),
            {"n_estimators": 120 + kwargs["fold_index"]},
        )

    monkeypatch.setattr(lightgbm_module, "tune_parameters", fake_tune)
    monkeypatch.setattr(selector, "_run_fold", fake_run)

    details = selector._select_robust_features(
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
        optuna_mode=mode,
        n_jobs=2,
        shap_max_rows=100,
        return_importances=True,
    )

    assert len(tune_calls) == expected_driver_calls
    assert len(captured_folds) == 2
    assert all(fold["optuna_mode"] == mode for fold in captured_folds)
    assert all(fold["n_jobs"] == 2 for fold in captured_folds)
    assert [fold["seed"] for fold in captured_folds] == [18, 19]
    if mode == "global":
        assert tune_calls[0]["n_jobs"] == 2
        assert all(
            fold["global_params"] == {"n_estimators": 120}
            for fold in captured_folds
        )
        assert details["global_best_params"] == {"n_estimators": 120}
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


def test_pandas_preparation_fills_numeric_nulls_with_median() -> None:
    frame = _frame()
    frame.loc[0, "first"] = np.nan
    selector = LightGbmSelector(_context(frame).config.model)

    matrix, _, _ = selector._extract_and_prep_data(
        frame,
        "response",
        ["first", "second"],
        max_rows=len(frame),
        sample_fraction=None,
        seed=17,
    )

    assert not np.isnan(matrix).any()


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
    assert matrix[0, 0] == 0.0


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


@pytest.mark.skipif(
    not _ml_backends_available(),
    reason="LightGBM, SHAP, Optuna, and sklearn are required",
)
def test_pandas_end_to_end_with_optional_ml_backends() -> None:
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


@pytest.mark.skipif(not _pyspark_available(), reason="pyspark not installed")
def test_spark_preparation_uses_declared_columns_and_supports_dots() -> None:
    from pyspark.sql import SparkSession

    try:
        spark = (
            SparkSession.builder.master("local[1]")
            .appName("test-lightgbm-preparation")
            .config("spark.ui.enabled", "false")
            .config("spark.driver.host", "127.0.0.1")
            .getOrCreate()
        )
    except Exception as exc:  # noqa: BLE001 - optional local Spark runtime
        pytest.skip(f"Spark runtime unavailable: {exc}")
    spark.sparkContext.setLogLevel("ERROR")
    try:
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
        assert matrix[:, 0].tolist() == [float(index) for index in range(10)]
    finally:
        spark.stop()


def test_search_space_defaults_to_established_ranges() -> None:
    context = _context(_frame())
    selector = LightGbmSelector(context.config.model)

    options = selector._resolve_options(context)

    assert options["search_space"] == DEFAULT_SEARCH_SPACE
    assert options["fixed_params"] == {}


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

    # overridden entry replaces the default
    assert options["search_space"]["num_leaves"] == {"type": "int", "min": 16, "max": 128}
    # untouched entries keep their defaults
    assert options["search_space"]["learning_rate"] == DEFAULT_SEARCH_SPACE["learning_rate"]
    # scalars are passed to the model instead of being tuned
    assert options["fixed_params"] == {"min_child_samples": 20}
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
    context = _context(
        _frame(),
        params={"parameters": {"num_leaves": {"type": "int", "min": 16, "max": 128}}},
    )
    selector = LightGbmSelector(context.config.model)
    seen: dict[str, Any] = {}

    def fake_tune(
        _matrix: np.ndarray,
        _target: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        seen["tune_space"] = kwargs["search_space"]
        seen["tune_fixed"] = kwargs["fixed_params"]
        return {"n_estimators": 100}

    def fake_run(
        _matrix: np.ndarray,
        _target: np.ndarray,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        seen.setdefault("fold_space", kwargs["search_space"])
        return (
            np.array([1.0, 0.5]),
            np.array([0.8, 0.4]),
            {},
        )

    monkeypatch.setattr(lightgbm_module, "tune_parameters", fake_tune)
    monkeypatch.setattr(selector, "_run_fold", fake_run)
    _install_fake_sklearn_model_selection(monkeypatch)

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
        shap_max_rows=5,
        search_space=options["search_space"],
        fixed_params=options["fixed_params"],
        return_importances=True,
    )

    assert seen["tune_space"]["num_leaves"]["max"] == 128
    assert seen["fold_space"]["num_leaves"]["max"] == 128


class _FakeTrial:
    def suggest_int(self: _FakeTrial, _name: str, minimum: int, _maximum: int) -> int:
        return minimum

    def suggest_float(
        self: _FakeTrial,
        _name: str,
        minimum: float,
        _maximum: float,
        *,
        log: bool = False,
    ) -> float:
        del log
        return minimum


class _FakeBooster:
    def feature_importance(self: _FakeBooster, *, importance_type: str) -> np.ndarray:
        assert importance_type == "split"
        return np.array([3.0, 1.0])


class _FakeModel:
    instances: ClassVar[list[_FakeModel]] = []

    def __init__(self: _FakeModel, **params: Any) -> None:
        self.params = params
        self.booster_ = _FakeBooster()
        self.__class__.instances.append(self)

    def fit(self: _FakeModel, _matrix: np.ndarray, _target: np.ndarray) -> None:
        return None


class _FakeExplainer:
    def __init__(self: _FakeExplainer, model: _FakeModel) -> None:
        self.model = model

    def shap_values(self: _FakeExplainer, matrix: np.ndarray) -> list[np.ndarray]:
        zeros = np.zeros((len(matrix), 2))
        positive = np.tile(np.array([0.5, 0.2]), (len(matrix), 1))
        return [zeros, positive]


def _install_fold_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_lightgbm = ModuleType("lightgbm")
    fake_lightgbm.LGBMClassifier = _FakeModel
    fake_shap = ModuleType("shap")
    fake_shap.TreeExplainer = _FakeExplainer
    monkeypatch.setitem(sys.modules, "lightgbm", fake_lightgbm)
    monkeypatch.setitem(sys.modules, "shap", fake_shap)


def test_trial_parameters_preserve_search_space_and_execution_limits() -> None:
    params = lightgbm_module.build_trial_parameters(
        _FakeTrial(),
        seed=7,
        n_jobs=2,
    )

    assert params["n_estimators"] == 100
    assert params["learning_rate"] == 0.01
    assert params["random_state"] == 7
    assert params["n_jobs"] == 2
    assert params["objective"] == "binary"


def test_folds_run_in_order_and_importances_are_averaged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_sklearn_model_selection(monkeypatch)
    context = _context(_frame())
    selector = LightGbmSelector(context.config.model)
    matrix = np.arange(12, dtype=float).reshape(6, 2)
    target = np.array([0, 1, 0, 1, 0, 1])
    monkeypatch.setattr(
        selector,
        "_extract_and_prep_data",
        lambda *_args, **_kwargs: (matrix, target, ["first", "second"]),
    )
    monkeypatch.setattr(
        lightgbm_module,
        "tune_parameters",
        lambda *_args, **_kwargs: {"n_estimators": 120},
    )
    call_order: list[int] = []

    def fake_run(
        received_matrix: np.ndarray,
        received_target: np.ndarray,
        **kwargs: Any,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        assert np.array_equal(received_matrix, matrix)
        assert np.array_equal(received_target, target)
        fold_index = kwargs["fold_index"]
        call_order.append(fold_index)
        return (
            np.array([2.0 * fold_index, 1.0]),
            np.array([0.5, 1.0 * fold_index]),
            {"fold": fold_index},
        )

    monkeypatch.setattr(selector, "_run_fold", fake_run)

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
        shap_max_rows=5,
        return_importances=True,
    )

    assert call_order == [1, 2]
    assert details["fold_best_params"] == {"1": {"fold": 1}, "2": {"fold": 2}}
    importances = details["importances_df"].set_index("feature")
    # (2*1 + 2*2) / 2 = 3.0 and (1 + 1) / 2 = 1.0 -> normalized over the total 4.0
    assert importances.loc["first", "lgbm_norm"] == pytest.approx(0.75)
    assert importances.loc["second", "lgbm_norm"] == pytest.approx(0.25)


def test_global_fold_uses_shared_params_with_driver_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeModel.instances = []
    _install_fold_backends(monkeypatch)
    matrix = np.arange(12, dtype=float).reshape(6, 2)
    target = np.array([0, 1, 0, 1, 0, 1])

    lgbm_importances, shap_importances, best_params = LightGbmSelector._run_fold(
        matrix,
        target,
        fold_index=1,
        valid_indices=np.array([0, 1], dtype=np.int64),
        seed=31,
        optuna_mode="global",
        n_trials=2,
        n_jobs=1,
        shap_max_rows=2,
        global_params={"n_estimators": 150, "n_jobs": -1},
    )

    assert lgbm_importances.tolist() == [3.0, 1.0]
    assert shap_importances.tolist() == pytest.approx([0.5, 0.2])
    assert best_params["n_estimators"] == 150
    assert _FakeModel.instances[-1].params["n_jobs"] == 1
    assert _FakeModel.instances[-1].params["random_state"] == 31


def test_per_fold_mode_tunes_only_on_outer_train_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeModel.instances = []
    _install_fold_backends(monkeypatch)
    matrix = np.arange(16, dtype=float).reshape(8, 2)
    target = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    captured: dict[str, Any] = {}

    def fake_tune(
        train_matrix: np.ndarray,
        train_target: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, Any]:
        captured["matrix"] = train_matrix
        captured["target"] = train_target
        captured.update(kwargs)
        return {"n_estimators": 99}

    monkeypatch.setattr(lightgbm_module, "tune_parameters", fake_tune)

    _lgbm, _shap, best_params = LightGbmSelector._run_fold(
        matrix,
        target,
        fold_index=2,
        valid_indices=np.array([0, 1], dtype=np.int64),
        seed=42,
        optuna_mode="per_fold",
        n_trials=3,
        n_jobs=1,
        shap_max_rows=10,
    )

    assert captured["matrix"].shape == (6, 2)
    assert captured["target"].tolist() == target[2:].tolist()
    assert captured["n_trials"] == 3
    assert captured["seed"] == 42
    assert captured["n_jobs"] == 1
    assert best_params["n_estimators"] == 99


def test_fold_without_global_params_reports_missing_tuning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fold_backends(monkeypatch)
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
