"""Tests for BorutaShapSelector."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pandas as pd
import pytest

import fmlib.feature_selection.precise.boruta_shap as boruta_module
from fmlib.feature_selection.base import StageContext
from fmlib.feature_selection.config import FeatureSelectionConfig, PreciseConfig
from fmlib.feature_selection.exceptions import BackendError, ExecutionError
from fmlib.feature_selection.precise.boruta_shap import (
    BorutaShapSelector,
    _Backends,
)
from fmlib.feature_selection.schema import FeatureSchema
from fmlib.feature_selection.utils.optuna_space import build_sampler
from fmlib.feature_selection.conftest import FakeSparkSession


def _frame(n_rows: int = 20) -> pd.DataFrame:
    target = np.tile([0, 1], n_rows // 2)
    return pd.DataFrame(
        {
            "category": np.where(target == 1, "positive", "negative"),
            "first": np.arange(n_rows, dtype=float),
            "second": np.arange(n_rows, dtype=float) * 2.0,
            "unused": np.arange(n_rows, dtype=float) * 3.0,
            "response": target,
        },
    )


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
            "precise": {
                "method": "boruta_shap",
                "params": params or {},
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
    return StageContext(
        spark=FakeSparkSession(),
        datasets={"train": frame},
        schema=schema,
        config=config,
        seed=seed,
        candidates=schema.candidate_features(),
    )


def _mock_details() -> dict[str, Any]:
    return {
        "accepted": ["first"],
        "rejected": ["second"],
        "tentative": [],
        "best_auc": 0.82,
        "best_params": {
            "n_estimators": np.int64(150),
            "max_depth": 5,
        },
    }


def _mock_selector_core(
    selector: BorutaShapSelector,
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, Any] | None = None,
) -> None:
    monkeypatch.setattr(selector, "_load_backends", lambda _model: object())

    def fake_run(**kwargs: Any) -> dict[str, Any]:
        if captured is not None:
            captured.update(kwargs)
        return _mock_details()

    monkeypatch.setattr(selector, "_run_boruta_selection", fake_run)


class _FakeTrial:
    def __init__(self: _FakeTrial) -> None:
        self.params: dict[str, Any] = {}

    def suggest_int(self: _FakeTrial, name: str, minimum: int, _maximum: int) -> int:
        self.params[name] = minimum
        return minimum

    def suggest_float(
        self: _FakeTrial,
        name: str,
        minimum: float,
        _maximum: float,
        *,
        log: bool,
    ) -> float:
        del log
        self.params[name] = minimum
        return minimum

    def suggest_categorical(
        self: _FakeTrial,
        name: str,
        values: list[Any],
    ) -> Any:
        self.params[name] = values[0]
        return values[0]


class _FakeStudy:
    def __init__(self: _FakeStudy) -> None:
        self.best_params: dict[str, Any] = {}
        self.best_value = 0.0

    def optimize(
        self: _FakeStudy,
        objective: Any,
        *,
        n_trials: int,
        show_progress_bar: bool,
        timeout: int | None = None,
    ) -> None:
        assert n_trials >= 1
        assert timeout is None or timeout >= 1
        assert show_progress_bar is False
        trial = _FakeTrial()
        self.best_value = objective(trial)
        self.best_params = dict(trial.params)


class _FakeSampler:
    def __init__(self: _FakeSampler, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeSamplers:
    TPESampler = _FakeSampler
    RandomSampler = _FakeSampler
    GridSampler = _FakeSampler


class _FakeOptuna:
    samplers = _FakeSamplers()
    last_study: _FakeStudy | None = None

    @classmethod
    def create_study(
        cls: type[_FakeOptuna],
        *,
        direction: str,
        sampler: Any,
    ) -> _FakeStudy:
        assert direction == "maximize"
        assert sampler is not None
        cls.last_study = _FakeStudy()
        return cls.last_study


class _FakeModel:
    instances: list[_FakeModel] = []

    def __init__(self: _FakeModel, **params: Any) -> None:
        self.params = params
        self.fit_kwargs: dict[str, Any] = {}
        self.__class__.instances.append(self)

    def fit(
        self: _FakeModel,
        _features: pd.DataFrame,
        _target: pd.Series,
        **kwargs: Any,
    ) -> _FakeModel:
        self.fit_kwargs = kwargs
        return self

    def predict_proba(self: _FakeModel, features: pd.DataFrame) -> np.ndarray:
        probabilities = np.linspace(0.2, 0.8, len(features))
        return np.column_stack([1.0 - probabilities, probabilities])


class _FakeBoruta:
    last_instance: _FakeBoruta | None = None

    def __init__(self: _FakeBoruta, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.fit_kwargs: dict[str, Any] = {}
        self.accepted = ["first"]
        self.rejected: list[str] = []
        self.tentative = ["second"]
        self.rough_called = False
        self.__class__.last_instance = self

    def fit(self: _FakeBoruta, **kwargs: Any) -> None:
        self.fit_kwargs = kwargs

    def TentativeRoughFix(self: _FakeBoruta) -> None:  # noqa: N802
        self.rough_called = True
        self.rejected = ["second"]


def _fake_train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    **kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    assert kwargs["stratify"] is target
    assert kwargs["test_size"] == 0.2
    return (
        features.iloc[:-4],
        features.iloc[-4:],
        target.iloc[:-4],
        target.iloc[-4:],
    )


def _fake_backends() -> _Backends:
    return _Backends(
        boruta_class=_FakeBoruta,
        model_class=_FakeModel,
        optuna_module=_FakeOptuna,
        roc_auc_score=lambda _target, _predictions: 0.81,
        train_test_split=_fake_train_test_split,
    )


def _boruta_stack_available() -> bool:
    if boruta_module.BorutaShap is None or boruta_module.optuna is None:
        return False
    try:
        import lightgbm  # noqa: F401
        import sklearn  # noqa: F401
    except Exception:  # noqa: BLE001 - optional binary dependencies
        return False
    return True


def _pyspark_available() -> bool:
    return importlib.util.find_spec("pyspark") is not None


@pytest.fixture(autouse=True)
def _reset_fake_state() -> None:
    _FakeModel.instances = []
    _FakeBoruta.last_instance = None


def test_selector_is_boruta_shap() -> None:
    context = _context(_frame())

    selector = BorutaShapSelector(context.config.precise)

    assert isinstance(selector, BorutaShapSelector)
    assert selector.method_name == "boruta_shap"


def test_select_scopes_to_continuous_and_passes_categories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(_frame())
    selector = BorutaShapSelector(context.config.precise)
    captured: dict[str, Any] = {}
    _mock_selector_core(selector, monkeypatch, captured)

    decisions = selector.select(context, context.candidates)

    assert captured["feature_cols"] == ["first", "second"]
    assert captured["seed"] == 17
    assert [decision.feature for decision in decisions] == ["first", "second"]
    assert decisions[0].keep is True
    assert decisions[0].reason == "boruta_accepted"
    assert decisions[1].keep is False
    assert decisions[1].reason == "boruta_rejected"
    assert all(decision.stage == "precise" for decision in decisions)
    assert "category" not in {decision.feature for decision in decisions}


def test_select_stores_flat_json_compatible_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(_frame(), params={"model_type": "rf"})
    selector = BorutaShapSelector(context.config.precise)
    _mock_selector_core(selector, monkeypatch)

    selector.select(context, context.candidates)

    scores = context.scores["boruta_shap"]
    assert "boruta_shap" not in scores
    assert scores["accepted"] == ["first"]
    assert scores["rejected"] == ["second"]
    assert scores["model_type"] == "rf"
    assert scores["best_auc"] == 0.82
    assert scores["best_params"]["n_estimators"] == 150


def test_unresolved_tentative_feature_is_reported_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(_frame())
    selector = BorutaShapSelector(context.config.precise)
    monkeypatch.setattr(selector, "_load_backends", lambda _model: object())
    monkeypatch.setattr(
        selector,
        "_run_boruta_selection",
        lambda **_kwargs: {
            "accepted": ["first"],
            "rejected": [],
            "tentative": ["second"],
            "best_auc": 0.8,
            "best_params": {},
        },
    )

    decisions = selector.select(context, context.candidates)

    assert decisions[1].reason == "boruta_tentative"
    assert decisions[1].keep is False
    assert context.scores["boruta_shap"]["tentative"] == ["second"]
    assert context.scores["boruta_shap"]["rejected"] == []


def test_empty_and_categorical_only_candidates_need_no_dependencies() -> None:
    context = _context(
        pd.DataFrame({"category": ["a", "b"], "response": [0, 1]}),
        continuous=(),
    )
    selector = BorutaShapSelector(context.config.precise)

    assert selector.select(context, []) == []
    assert selector.select(context, ["category"]) == []


def test_missing_train_and_non_binary_task_fail_before_dependencies() -> None:
    context = _context(_frame())
    selector = BorutaShapSelector(context.config.precise)
    context.datasets = {}
    with pytest.raises(ExecutionError, match="'train' split"):
        selector.select(context, context.candidates)

    regression = _context(_frame(), task_type="regression")
    with pytest.raises(ExecutionError, match="binary_classification"):
        BorutaShapSelector(regression.config.precise).select(
            regression,
            regression.candidates,
        )


@pytest.mark.parametrize("model_type", ["lgbm", "rf"])
def test_options_support_both_models_and_execution_cap(
    model_type: str,
) -> None:
    context = _context(
        _frame(),
        params={
            "model_type": model_type,
            "max_rows": 500,
            "boruta_trials": 7,
            "optuna_params": {
                "n_trials": 3,
                "n_startup_trials": 2,
                "sampler": "RANDOM",
            },
        },
        max_local_rows=11,
    )

    options = BorutaShapSelector(
        context.config.precise,
    )._resolve_options(context)

    assert options["model_type"] == model_type
    assert options["max_rows"] == 11
    assert options["n_trials"] == 3
    assert options["n_startup_trials"] == 2
    assert options["boruta_trials"] == 7
    assert options["sampler"] == "RANDOM"


def test_scalar_parameters_are_accepted_and_split_from_ranges() -> None:
    context = _context(
        _frame(),
        params={
            "parameters": {
                "boosting_type": "gbdt",
                "n_estimators": 300,
                "learning_rate": {
                    "type": "float",
                    "min": 0.01,
                    "max": 0.1,
                    "log": True,
                },
            },
        },
    )

    options = BorutaShapSelector(
        context.config.precise,
    )._resolve_options(context)

    assert options["fixed_params"] == {
        "boosting_type": "gbdt",
        "n_estimators": 300,
    }
    assert set(options["parameters"]) == {"learning_rate"}


def test_options_support_legacy_aliases() -> None:
    context = _context(
        _frame(),
        params={
            "max_rows_limit": 123,
            "optuna_trials": 9,
        },
    )

    options = BorutaShapSelector(
        context.config.precise,
    )._resolve_options(context)

    assert options["max_rows"] == 123
    assert options["n_trials"] == 9


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"model_type": "xgb"}, "model_type"),
        ({"boruta_trials": 0}, "boruta_trials"),
        ({"sample_fraction": 0.0}, "sample_fraction"),
        ({"tentative_fix_method": "median"}, "tentative_fix_method"),
    ],
)
def test_runtime_option_validation_for_direct_precise_config(
    params: dict[str, Any],
    message: str,
) -> None:
    context = _context(_frame())
    selector = BorutaShapSelector(
        PreciseConfig(method="boruta_shap", params=params),
    )

    with pytest.raises(ExecutionError, match=message):
        selector._resolve_options(context)


@pytest.mark.parametrize("model_type", ["lgbm", "rf"])
def test_mock_core_runs_tuning_and_boruta_for_both_models(
    model_type: str,
) -> None:
    context = _context(
        _frame(),
        params={
            "model_type": model_type,
            "n_trials": 1,
            "boruta_trials": 3,
            "max_rows": 20,
        },
    )
    selector = BorutaShapSelector(context.config.precise)
    options = selector._resolve_options(context)

    details = selector._run_boruta_selection(
        train=_frame(),
        target_col="response",
        feature_cols=["first", "second"],
        options=options,
        seed=17,
        backends=_fake_backends(),
    )

    assert details["accepted"] == ["first"]
    assert details["rejected"] == ["second"]
    assert details["best_auc"] == 0.81
    assert len(_FakeModel.instances) == 2
    assert all(model.params["random_state"] == 17 for model in _FakeModel.instances)
    if model_type == "lgbm":
        assert "eval_set" in _FakeModel.instances[0].fit_kwargs
        assert _FakeModel.instances[0].params["objective"] == "binary"
    else:
        assert _FakeModel.instances[0].fit_kwargs == {}
        assert "objective" not in _FakeModel.instances[0].params

    boruta = _FakeBoruta.last_instance
    assert boruta is not None
    assert boruta.init_kwargs["importance_measure"] == "shap"
    assert boruta.init_kwargs["classification"] is True
    assert boruta.init_kwargs["model"] is _FakeModel.instances[-1]
    assert boruta.fit_kwargs["n_trials"] == 3
    assert boruta.fit_kwargs["random_state"] == 17
    assert boruta.fit_kwargs["verbose"] is False
    assert boruta.fit_kwargs["X"].columns.tolist() == ["first", "second"]
    assert boruta.rough_called is True


def test_tentative_rough_fix_can_be_disabled() -> None:
    context = _context(
        _frame(),
        params={
            "tentative_fix_method": None,
            "n_trials": 1,
            "boruta_trials": 2,
        },
    )
    selector = BorutaShapSelector(context.config.precise)

    details = selector._run_boruta_selection(
        train=_frame(),
        target_col="response",
        feature_cols=["first", "second"],
        options=selector._resolve_options(context),
        seed=17,
        backends=_fake_backends(),
    )

    assert details["tentative"] == ["second"]
    assert _FakeBoruta.last_instance is not None
    assert _FakeBoruta.last_instance.rough_called is False


def test_search_spaces_match_models_and_drop_invalid_bootstrap() -> None:
    lgbm_space = BorutaShapSelector._build_search_space(
        "lgbm",
        {
            "bootstrap_type": {
                "type": "categorical",
                "values": ["MVS"],
            },
            "n_estimators": {"type": "int", "min": 10, "max": 20},
        },
        "TPE",
        {},
    )
    rf_space = BorutaShapSelector._build_search_space(
        "rf",
        {},
        "TPE",
        {},
    )

    assert "bootstrap_type" not in lgbm_space
    assert lgbm_space["n_estimators"]["min"] == 10
    assert "num_leaves" in lgbm_space
    assert "min_samples_split" in rf_space
    assert "num_leaves" not in rf_space


def test_scalar_parameters_are_pinned_and_leave_the_search_space() -> None:
    space = BorutaShapSelector._build_search_space(
        "lgbm",
        {"n_estimators": {"type": "int", "min": 10, "max": 20}},
        "TPE",
        {"boosting_type": "gbdt", "learning_rate": 0.05},
    )

    assert "boosting_type" not in space
    assert "learning_rate" not in space
    assert space["n_estimators"]["min"] == 10
    assert "num_leaves" in space


def test_grid_sampler_uses_finite_custom_values() -> None:
    search_space = {
        "max_depth": {
            "values": [3, 5],
        },
    }

    sampler = build_sampler(
        _FakeOptuna,
        sampler_name="GRID",
        search_space=search_space,
        seed=17,
        n_startup_trials=1,
        method_name="boruta_shap",
    )

    assert sampler.kwargs["search_space"] == {"max_depth": [3, 5]}


def test_missing_boruta_and_optuna_raise_backend_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector = BorutaShapSelector(_context(_frame()).config.precise)
    monkeypatch.setattr(boruta_module, "BorutaShap", None)
    with pytest.raises(BackendError, match="BorutaShap"):
        selector._load_backends("lgbm")

    monkeypatch.setattr(boruta_module, "BorutaShap", object())
    monkeypatch.setattr(boruta_module, "optuna", None)
    with pytest.raises(BackendError, match="Optuna"):
        selector._load_backends("lgbm")


def test_core_rejects_single_class_and_all_null_features() -> None:
    selector = BorutaShapSelector(_context(_frame()).config.precise)
    options = selector._resolve_options(_context(_frame()))
    single_class = _frame().assign(response=0)
    with pytest.raises(ExecutionError, match="exactly two"):
        selector._run_boruta_selection(
            train=single_class,
            target_col="response",
            feature_cols=["first", "second"],
            options=options,
            seed=17,
            backends=_fake_backends(),
        )

    all_null = _frame().assign(first=np.nan)
    with pytest.raises(ExecutionError, match="all-null"):
        selector._run_boruta_selection(
            train=all_null,
            target_col="response",
            feature_cols=["first", "second"],
            options=options,
            seed=17,
            backends=_fake_backends(),
        )

    tiny = _frame(4)
    with pytest.raises(ExecutionError, match="too small"):
        selector._run_boruta_selection(
            train=tiny,
            target_col="response",
            feature_cols=["first", "second"],
            options=options,
            seed=17,
            backends=_fake_backends(),
        )


@pytest.mark.skipif(not _pyspark_available(), reason="pyspark not installed")
def test_spark_core_uses_shared_materialization_and_supports_dots() -> None:
    from pyspark.sql import SparkSession

    try:
        spark = (
            SparkSession.builder.master("local[1]")
            .appName("test-boruta-shap-preparation")
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
                (float(index), float(index * 2), index % 2)
                for index in range(20)
            ],
            ["foo.bar", "second", "response"],
        )
        context = _context(
            pd.DataFrame(),
            categorical=(),
            continuous=("foo.bar", "second"),
            params={
                "model_type": "rf",
                "n_trials": 1,
                "boruta_trials": 2,
                "max_rows": 20,
            },
        )
        selector = BorutaShapSelector(context.config.precise)

        details = selector._run_boruta_selection(
            train=frame,
            target_col="response",
            feature_cols=["foo.bar", "second"],
            options=selector._resolve_options(context),
            seed=17,
            backends=_fake_backends(),
        )

        assert details["accepted"] == []
        boruta = _FakeBoruta.last_instance
        assert boruta is not None
        assert boruta.fit_kwargs["X"].columns.tolist() == [
            "foo.bar",
            "second",
        ]
    finally:
        spark.stop()


@pytest.mark.skipif(
    not _boruta_stack_available(),
    reason="BorutaShap, LightGBM, Optuna, and sklearn are required",
)
def test_pandas_end_to_end_with_optional_boruta_stack() -> None:
    rng = np.random.default_rng(31)
    target = np.tile([0, 1], 50)
    frame = pd.DataFrame(
        {
            "category": np.where(target == 1, "positive", "negative"),
            "signal": target + rng.normal(0.0, 0.03, len(target)),
            "noise": rng.normal(0.0, 1.0, len(target)),
            "response": target,
        },
    )
    context = _context(
        frame,
        categorical=("category",),
        continuous=("signal", "noise"),
        params={
            "model_type": "lgbm",
            "n_trials": 1,
            "boruta_trials": 3,
            "max_rows": len(frame),
        },
    )

    decisions = BorutaShapSelector(context.config.precise).select(
        context,
        context.candidates,
    )

    assert [decision.feature for decision in decisions] == ["signal", "noise"]
    assert "category" not in {decision.feature for decision in decisions}
    assert set(context.scores["boruta_shap"]["accepted"]) <= {
        "signal",
        "noise",
    }
