"""Тесты BorutaShapSelector."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import fmlib.feature_selection.model_based.boruta_shap as boruta_module
from fmlib.feature_selection.base import StageContext
from fmlib.feature_selection.config import FeatureSelectionConfig, ModelConfig
from fmlib.feature_selection.utils.conftest import require_spark_session
from fmlib.feature_selection.exceptions import BackendError, ExecutionError
from fmlib.feature_selection.model_based.boruta_shap import (
    BorutaShapSelector,
    _Backends,
)
from fmlib.feature_selection.schema import FeatureSchema
from fmlib.feature_selection.utils.optuna_space import build_sampler


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
            "model": {
                "method": "boruta_shap",
                "params": params or {},
            },
            "execution": {
                "seed": seed,
                "max_local_rows": max_local_rows,
                "task_type": task_type,
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
        spark=require_spark_session(),
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
    monkeypatch.setattr(
        selector,
        "_load_backends",
        lambda *_args, **_kwargs: object(),
    )

    def fake_run(**kwargs: Any) -> dict[str, Any]:
        if captured is not None:
            captured.update(kwargs)
        return _mock_details()

    monkeypatch.setattr(selector, "_run_boruta_selection", fake_run)


def _require_boruta_stack() -> None:
    """Немедленно вызывает ошибку при отсутствии дополнительной зависимости Boruta. Не пропускает тест."""
    try:
        import lightgbm  # noqa: F401
        import optuna  # noqa: F401
        import sklearn  # noqa: F401
        from BorutaShap import BorutaShap  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - missing extra must fail the run
        pytest.fail(
            "Install the boruta extra (BorutaShap, lightgbm, optuna, sklearn). "
            f"Root cause: {exc}",
        )
    if boruta_module.BorutaShap is None or boruta_module.optuna is None:
        pytest.fail("Install the boruta extra (BorutaShap, lightgbm, optuna, sklearn).")


def _tiny_boruta_params(model_type: str = "rf", **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_type": model_type,
        "n_trials": 1,
        "boruta_trials": 2,
        "max_rows": 40,
        "parameters": {"n_estimators": 8, "max_depth": 3},
        "optuna_params": {"enabled": False, "n_trials": 1, "n_startup_trials": 1},
    }
    payload.update(overrides)
    return payload


def test_selector_is_boruta_shap() -> None:
    context = _context(_frame())

    selector = BorutaShapSelector(context.config.model)

    assert isinstance(selector, BorutaShapSelector)
    assert selector.method_name == "boruta_shap"


def test_select_scopes_to_continuous_and_passes_categories(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(_frame())
    selector = BorutaShapSelector(context.config.model)
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
    assert all(decision.stage == "model" for decision in decisions)
    assert "category" not in {decision.feature for decision in decisions}


def test_select_stores_flat_json_compatible_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(_frame(), params={"model_type": "rf"})
    selector = BorutaShapSelector(context.config.model)
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
    selector = BorutaShapSelector(context.config.model)
    monkeypatch.setattr(
        selector,
        "_load_backends",
        lambda *_args, **_kwargs: object(),
    )
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
    selector = BorutaShapSelector(context.config.model)

    assert selector.select(context, []) == []
    assert selector.select(context, ["category"]) == []


def test_missing_train_fails_before_dependencies() -> None:
    config = FeatureSelectionConfig.from_dict(
        {
            "model": {"method": "boruta_shap", "params": {}},
            "execution": {"seed": 17, "task_type": "binary_classification"},
        },
    )
    schema = FeatureSchema(
        categorical=("category",),
        continuous=("first", "second"),
        target="response",
        task_type="binary_classification",
    )
    context = StageContext(
        spark=None,
        datasets={},
        schema=schema,
        config=config,
        seed=17,
        candidates=list(schema.candidate_features()),
    )
    selector = BorutaShapSelector(context.config.model)
    with pytest.raises(ExecutionError, match="'train' split"):
        selector.select(context, context.candidates)


def test_classification_and_regression_run_with_fake_boruta() -> None:
    class FakeModel:
        def __init__(self, **kwargs: Any) -> None:
            pass

    class FakeSelector:
        def __init__(self, **kwargs: Any) -> None:
            self.accepted = ["first"]
            self.rejected = ["second"]
            self.tentative: list[str] = []
            self.classification = kwargs.get("classification")

        def fit(self, **kwargs: Any) -> None:
            return None

        def TentativeRoughFix(self) -> None:
            return None

    def pandas_context(frame: pd.DataFrame, task_type: str) -> StageContext:
        config = FeatureSelectionConfig.from_dict(
            {
                "model": {
                    "method": "boruta_shap",
                    "params": {"optuna_params": {"enabled": False}},
                },
                "execution": {"seed": 17, "task_type": task_type},
            },
        )
        schema = FeatureSchema(
            categorical=("category",),
            continuous=("first", "second"),
            target="response",
            task_type=task_type,
        )
        return StageContext(
            spark=None,
            datasets={"train": frame},
            schema=schema,
            config=config,
            seed=17,
            candidates=schema.candidate_features(),
        )

    class_captured: dict[str, Any] = {}

    class ClassRecordingSelector(FakeSelector):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            class_captured["classification"] = kwargs.get("classification")

    backends = _Backends(
        boruta_class=ClassRecordingSelector,
        model_class=FakeModel,
        optuna_module=None,
        train_test_split=None,
    )
    class_frame = _frame(30)
    class_frame["response"] = [0, 1, 2] * 10
    class_context = pandas_context(class_frame, "classification")
    selector = BorutaShapSelector(class_context.config.model)
    details = selector._run_boruta_selection(
        train=class_frame,
        target_col="response",
        feature_cols=["first", "second"],
        options=selector._resolve_options(class_context),
        seed=17,
        backends=backends,
        context=class_context,
    )
    assert details["accepted"] == ["first"]
    assert class_captured["classification"] is True

    captured: dict[str, Any] = {}

    class RecordingSelector(FakeSelector):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            captured["classification"] = kwargs.get("classification")

    reg_frame = _frame(30)
    reg_frame["response"] = np.arange(30, dtype=float)
    reg_context = pandas_context(reg_frame, "regression")
    backends = _Backends(
        boruta_class=RecordingSelector,
        model_class=FakeModel,
        optuna_module=None,
        train_test_split=None,
    )
    selector = BorutaShapSelector(reg_context.config.model)
    selector._run_boruta_selection(
        train=reg_frame,
        target_col="response",
        feature_cols=["first", "second"],
        options=selector._resolve_options(reg_context),
        seed=17,
        backends=backends,
        context=reg_context,
    )
    assert captured["classification"] is False


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
        context.config.model,
    )._resolve_options(context)

    assert options["model_type"] == model_type
    assert options["max_rows"] == 11
    assert options["n_trials"] == 3
    assert options["n_startup_trials"] == 2
    assert options["boruta_trials"] == 7
    assert options["sampler"] == "RANDOM"
    assert options["optuna_enabled"] is True


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
        context.config.model,
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
        context.config.model,
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
def test_runtime_option_validation_for_direct_model_config(
    params: dict[str, Any],
    message: str,
) -> None:
    context = _context(_frame())
    selector = BorutaShapSelector(
        ModelConfig(method="boruta_shap", params=params),
    )

    with pytest.raises(ExecutionError, match=message):
        selector._resolve_options(context)


@pytest.mark.parametrize("model_type", ["lgbm", "rf"])
def test_core_runs_boruta_for_both_models(
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
) -> None:
    _require_boruta_stack()
    frame = _frame(40)
    context = _context(frame, params=_tiny_boruta_params(model_type))
    selector = BorutaShapSelector(context.config.model)
    options = selector._resolve_options(context)
    backends = selector._load_backends(model_type, require_optuna=False)
    original_boruta = backends.boruta_class
    built: list[dict[str, Any]] = []
    original_build = selector._build_model
    rough_calls = {"n": 0}
    fit_columns: list[list[str]] = []

    def spy_build(
        model_class: Any,
        received_type: str,
        parameters: Any,
        seed: int,
    ) -> Any:
        model = original_build(model_class, received_type, parameters, seed)
        built.append({"model_type": received_type, "params": dict(model.get_params())})
        return model

    def spy_boruta(**kwargs: Any) -> Any:
        instance = original_boruta(**kwargs)
        original_rough = instance.TentativeRoughFix
        original_fit = instance.fit

        def wrapped_rough(*args: Any, **inner: Any) -> Any:
            rough_calls["n"] += 1
            return original_rough(*args, **inner)

        def wrapped_fit(*args: Any, **inner: Any) -> Any:
            frame = inner.get("X", args[0] if args else None)
            if hasattr(frame, "columns"):
                fit_columns.append(list(frame.columns))
            return original_fit(*args, **inner)

        instance.TentativeRoughFix = wrapped_rough
        instance.fit = wrapped_fit
        return instance

    monkeypatch.setattr(selector, "_build_model", spy_build)
    details = selector._run_boruta_selection(
        train=frame,
        target_col="response",
        feature_cols=["first", "second"],
        options=options,
        seed=17,
        backends=_Backends(
            boruta_class=spy_boruta,
            model_class=backends.model_class,
            optuna_module=backends.optuna_module,
            train_test_split=backends.train_test_split,
        ),
    )

    covered = set(details["accepted"]) | set(details["rejected"]) | set(details["tentative"])
    assert covered == {"first", "second"}
    assert rough_calls["n"] == 1
    assert fit_columns == [["first", "second"]]
    assert built
    assert all(item["params"]["random_state"] == 17 for item in built)
    if model_type == "lgbm":
        assert built[0]["params"]["objective"] == "binary"
    else:
        assert "objective" not in built[0]["params"] or built[0]["params"]["objective"] is None


def test_tentative_rough_fix_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_boruta_stack()
    frame = _frame(40)
    context = _context(
        frame,
        params=_tiny_boruta_params(tentative_fix_method=None),
    )
    selector = BorutaShapSelector(context.config.model)
    backends = selector._load_backends("rf", require_optuna=False)
    original_boruta = backends.boruta_class
    rough_calls = {"n": 0}

    def spy_boruta(**kwargs: Any) -> Any:
        instance = original_boruta(**kwargs)
        original_rough = instance.TentativeRoughFix

        def wrapped_rough(*args: Any, **inner: Any) -> Any:
            rough_calls["n"] += 1
            return original_rough(*args, **inner)

        instance.TentativeRoughFix = wrapped_rough
        return instance

    details = selector._run_boruta_selection(
        train=frame,
        target_col="response",
        feature_cols=["first", "second"],
        options=selector._resolve_options(context),
        seed=17,
        backends=_Backends(
            boruta_class=spy_boruta,
            model_class=backends.model_class,
            optuna_module=backends.optuna_module,
            train_test_split=backends.train_test_split,
        ),
    )

    covered = set(details["accepted"]) | set(details["rejected"]) | set(details["tentative"])
    assert covered == {"first", "second"}
    assert rough_calls["n"] == 0


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
    assert set(lgbm_space) == {"n_estimators"}
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
    assert "num_leaves" not in space
    assert set(space) == {"n_estimators"}


def test_grid_sampler_uses_finite_custom_values() -> None:
    _require_boruta_stack()
    import optuna

    search_space = {
        "max_depth": {
            "values": [3, 5],
        },
    }

    sampler = build_sampler(
        optuna,
        sampler_name="GRID",
        search_space=search_space,
        seed=17,
        n_startup_trials=1,
        method_name="boruta_shap",
    )

    assert type(sampler).__name__ == "GridSampler"
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(lambda trial: float(trial.suggest_categorical("max_depth", [3, 5])), n_trials=2)
    assert study.best_params["max_depth"] in {3, 5}


def test_optuna_disabled_clears_the_search_space() -> None:
    context = _context(
        _frame(),
        params={"optuna_params": {"enabled": False}},
    )
    options = BorutaShapSelector(context.config.model)._resolve_options(context)

    assert options["optuna_enabled"] is False
    assert options["search_space"] == {}


def test_missing_boruta_and_optuna_raise_backend_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector = BorutaShapSelector(_context(_frame()).config.model)
    monkeypatch.setattr(boruta_module, "BorutaShap", None)
    with pytest.raises(BackendError, match="BorutaShap"):
        selector._load_backends("lgbm")

    monkeypatch.setattr(boruta_module, "BorutaShap", object())
    monkeypatch.setattr(boruta_module, "optuna", None)
    with pytest.raises(BackendError, match="Optuna"):
        selector._load_backends("lgbm")


def test_core_rejects_single_class_and_all_null_features() -> None:
    _require_boruta_stack()
    selector = BorutaShapSelector(_context(_frame()).config.model)
    options = selector._resolve_options(_context(_frame(), params=_tiny_boruta_params()))
    backends = selector._load_backends("rf", require_optuna=False)
    single_class = _frame().assign(response=0)
    with pytest.raises(ExecutionError, match="exactly two"):
        selector._run_boruta_selection(
            train=single_class,
            target_col="response",
            feature_cols=["first", "second"],
            options=options,
            seed=17,
            backends=backends,
        )

    all_null = _frame().assign(first=np.nan)
    with pytest.raises(ExecutionError, match="all-null"):
        selector._run_boruta_selection(
            train=all_null,
            target_col="response",
            feature_cols=["first", "second"],
            options=options,
            seed=17,
            backends=backends,
        )

    tiny = _frame(4)
    tiny_context = _context(
        tiny,
        params=_tiny_boruta_params(
            parameters={"n_estimators": {"type": "int", "min": 8, "max": 10}},
            optuna_params={"enabled": True, "n_trials": 1, "n_startup_trials": 1},
        ),
    )
    tiny_options = selector._resolve_options(tiny_context)
    with pytest.raises(ExecutionError, match="too small"):
        selector._run_boruta_selection(
            train=tiny,
            target_col="response",
            feature_cols=["first", "second"],
            options=tiny_options,
            seed=17,
            backends=selector._load_backends("rf", require_optuna=True),
        )


def test_core_keeps_partial_nans() -> None:
    frame = _frame()
    config = FeatureSelectionConfig.from_dict(
        {
            "model": {
                "method": "boruta_shap",
                "params": _tiny_boruta_params(),
            },
            "execution": {"seed": 17, "max_local_rows": 1_000},
        },
    )
    schema = FeatureSchema(
        categorical=("category",),
        continuous=("first", "second"),
        target="response",
        task_type="binary_classification",
    )
    selector = BorutaShapSelector(config.model)
    options = selector._resolve_options(
        StageContext(
            spark=None,
            datasets={"train": frame},
            schema=schema,
            config=config,
            seed=17,
            candidates=schema.candidate_features(),
        ),
    )

    class FakeModel:
        def __init__(self, **kwargs: Any) -> None:
            pass

    class FakeSelector:
        def __init__(self, **kwargs: Any) -> None:
            self.accepted = ["first"]
            self.rejected = ["second"]
            self.tentative: list[str] = []

        def fit(self, **kwargs: Any) -> None:
            return None

        def TentativeRoughFix(self) -> None:
            return None

    backends = _Backends(
        boruta_class=FakeSelector,
        model_class=FakeModel,
        optuna_module=None,
        train_test_split=None,
    )
    all_null = frame.assign(first=np.nan)
    with pytest.raises(ExecutionError, match="all-null"):
        selector._run_boruta_selection(
            train=all_null,
            target_col="response",
            feature_cols=["first", "second"],
            options=options,
            seed=17,
            backends=backends,
        )

    partial_null = frame.copy()
    partial_null.loc[0, "first"] = np.nan
    details = selector._run_boruta_selection(
        train=partial_null,
        target_col="response",
        feature_cols=["first", "second"],
        options=options,
        seed=17,
        backends=backends,
    )
    assert details["accepted"] == ["first"]
    assert details["rejected"] == ["second"]


def test_spark_core_uses_shared_materialization_and_supports_dots(spark: Any) -> None:
    _require_boruta_stack()
    frame = spark.createDataFrame(
        [
            (float(index), float(index * 2), index % 2)
            for index in range(40)
        ],
        ["foo.bar", "second", "response"],
    )
    context = _context(
        pd.DataFrame(),
        categorical=(),
        continuous=("foo.bar", "second"),
        params=_tiny_boruta_params(),
    )
    selector = BorutaShapSelector(context.config.model)
    backends = selector._load_backends("rf", require_optuna=False)

    details = selector._run_boruta_selection(
        train=frame,
        target_col="response",
        feature_cols=["foo.bar", "second"],
        options=selector._resolve_options(context),
        seed=17,
        backends=backends,
    )

    covered = set(details["accepted"]) | set(details["rejected"]) | set(details["tentative"])
    assert covered == {"foo.bar", "second"}


def test_pandas_end_to_end_with_boruta_stack() -> None:
    _require_boruta_stack()
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
        params=_tiny_boruta_params("lgbm", max_rows=len(frame), boruta_trials=3),
    )

    decisions = BorutaShapSelector(context.config.model).select(
        context,
        context.candidates,
    )

    assert [decision.feature for decision in decisions] == ["signal", "noise"]
    assert "category" not in {decision.feature for decision in decisions}
    assert set(context.scores["boruta_shap"]["accepted"]) <= {
        "signal",
        "noise",
    }
