"""Рекурсивное исключение признаков CatBoost с необязательным подбором параметров Optuna.

Метод отбора загружает ограниченную выборку из train в локальную память. Приватные функции
работают только с этим pandas DataFrame: не обращаются к Spark, не читают файлы и не проверяют
валидационную и тестовую выборки.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from fmlib.feature_selection.base import FeatureDecision, StageContext, resolve_step_seed
from fmlib.feature_selection.config import ModelConfig
from fmlib.feature_selection.exceptions import BackendError, ExecutionError
from fmlib.feature_selection.utils.default_model_param_spaces import (
    CATBOOST_RFE_SEARCH_SPACE,
)
from fmlib.feature_selection.utils.lama_boost_defaults import apply_boost_heuristics
from fmlib.feature_selection.utils.local_data import prepare_mixed_frame, root_cause
from fmlib.feature_selection.utils.optuna_space import (
    build_sampler,
    resolve_optuna_settings,
    resolve_tuning_space,
    suggest_parameter,
)
from fmlib.feature_selection.utils.task_runtime import (
    TaskRuntime,
    binary_task,
    catboost_estimator_class,
    catboost_loss_params,
    optuna_direction,
    resolve_task,
    score_model,
)

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ROWS = 500_000
_DEFAULT_EVAL_MONTHS = 1
# Elimination steps, i.e. how many points the reported loss curve carries.
# Each step retrains CatBoost, so this is also a direct time multiplier.
_DEFAULT_RFE_STEPS = 10


@dataclass(frozen=True)
class _Backends:
    """Объекты сторонних библиотек CatBoost и Optuna, используемые методом отбора."""

    estimator_class: Any
    pool_class: Any
    optuna_module: Any


def _constant_drop_targets(
    n_features: int,
    max_features: int,
    drop_per_step: int,
) -> list[int]:
    """Возвращает ``num_features_to_select`` после каждого раунда исключения фиксированного числа признаков.

    В последнем раунде может исключаться меньше ``drop_per_step`` признаков, чтобы их число
    не опустилось ниже ``max_features``.
    """
    if drop_per_step < 1:
        msg = "feature_drop_per_step must be >= 1."
        raise ValueError(msg)
    remaining = n_features
    targets: list[int] = []
    while remaining > max_features:
        drop_now = min(drop_per_step, remaining - max_features)
        remaining -= drop_now
        targets.append(remaining)
    return targets


_ALGORITHMS = frozenset(
    {
        "RecursiveByLossFunctionChange",
        "RecursiveByShapValues",
        "RecursiveByPredictionValuesChange",
    },
)

_DEFAULT_PARAMETERS: dict[str, Any] = {
    "verbose": False,
    "allow_writing_files": False,
}


class CatBoostRfeSelector:
    """Отбирает признаки через CatBoost ``select_features`` с разделением по времени.

    Из train формируется стратифицированная выборка ограниченного размера, которая затем разделяется
    по ``FeatureSchema.time``: последние ``eval_months`` периодов образуют eval
    для ранней остановки и расчёта важности при исключении, а все более ранние данные
    образуют train для обучения. Внешние выборки ``valid`` и ``test`` не читаются,
    поэтому остаются пригодными для независимой проверки отобранного набора признаков.

    Optuna запускается, если ``params.optuna_params.enabled`` равно true (по умолчанию).
    Если ``params.parameters`` содержит словари пространства поиска
    (``{"type": "int", "min": 4, "max": 8}``), они задают всю
    сетку. Если блок содержит только скаляры, используется пространство поиска по умолчанию из
    ``CATBOOST_RFE_SEARCH_SPACE``. ``enabled: false`` пропускает Optuna и
    передаёт скаляры в CatBoost без изменений. Отсутствующие ``learning_rate`` и
    ``early_stopping_rounds`` заполняются по таблице LightAutoML в зависимости от числа строк
    после определения train-части временного разбиения; скаляр из YAML сохраняется. Optuna
    не подбирает эти значения. Оцениваются и категориальные, и непрерывные
    кандидаты — категориальные передаются в CatBoost как
    ``cat_features``.

    Args:
        config: Настройки этапа модели.
    """

    method_name = "catboost_rfe"
    stage_name = "model"

    def __init__(self: CatBoostRfeSelector, config: ModelConfig) -> None:
        self.config = config

    def select(
        self: CatBoostRfeSelector,
        context: StageContext,
        candidates: Sequence[str],
    ) -> list[FeatureDecision]:
        """Выполняет рекурсивное исключение и возвращает решения о сохранении или исключении.

        Args:
            context: Общий контекст этапа с наборами данных, схемой, конфигурацией и seed.
            candidates: Признаки, которые ещё рассматриваются для отбора.

        Returns:
            Решения по каждому оценённому кандидату. Пустой список, если этап
            ничего не меняет, например если число кандидатов уже не превышает целевое.

        Raises:
            BackendError: Если необязательная зависимость для машинного обучения недоступна.
            ExecutionError: При некорректных входных данных, разбиении или выполнении модели.
        """
        features = list(candidates)
        if not features:
            return []

        target_col = context.schema.target
        if not target_col:
            msg = "catboost_rfe: FeatureSchema.target is required."
            raise ExecutionError(msg)

        time_col = context.schema.time
        if not time_col:
            msg = (
                "catboost_rfe: FeatureSchema.time is required for the out-of-time split. "
                "Set time to the period column (for example 'month_part')."
            )
            raise ExecutionError(msg)

        train = context.datasets.get("train")
        if train is None:
            msg = "catboost_rfe: context.datasets must contain a 'train' split."
            raise ExecutionError(msg)

        options = self._resolve_options(context)
        target_count = options["num_features_to_select"]
        if len(features) <= target_count:
            logger.info(
                "%s: %d candidates already fit num_features_to_select=%d, stage is a no-op",
                self.method_name,
                len(features),
                target_count,
            )
            context.scores[self.method_name] = {
                "skipped": True,
                "reason": "candidates_below_target",
                "candidate_count": len(features),
                "num_features_to_select": target_count,
            }
            return []

        categorical = [feature for feature in features if feature in set(context.schema.categorical)]

        try:
            local = prepare_mixed_frame(
                train,
                target_col=target_col,
                feature_cols=features,
                categorical_cols=categorical,
                extra_cols=(time_col,),
                max_rows=options["max_rows"],
                sample_fraction=options["sample_fraction"],
                seed=options["seed"],
                method_name=self.method_name,
                context=context,
            )
            details = _run_catboost_rfe(
                local,
                feature_cols=features,
                categorical_cols=categorical,
                target_col=target_col,
                time_col=time_col,
                eval_months=options["eval_months"],
                fixed_params=options["fixed_params"],
                search_space=options["search_space"],
                optuna_params=options["optuna_params"],
                feature_selection_params=options["feature_selection_params"],
                num_features_to_select=target_count,
                seed=options["seed"],
                method_name=self.method_name,
                task_type=context.schema.task_type,
            )
        except (BackendError, ExecutionError):
            raise
        except Exception as exc:  # noqa: BLE001 - normalize third-party failures
            msg = f"catboost_rfe: feature selection failed. Root cause: {root_cause(exc)}."
            raise ExecutionError(msg) from exc

        selected = set(details["selected_features"])
        elimination_rank = {name: index + 1 for index, name in enumerate(details["eliminated_features"])}

        decisions = [
            FeatureDecision(
                feature=feature,
                stage=self.stage_name,
                method=self.method_name,
                reason="catboost_rfe_selected" if feature in selected else "catboost_rfe_eliminated",
                value=None if feature in selected else float(elimination_rank.get(feature, 0)),
                threshold=float(target_count),
                keep=feature in selected,
            )
            for feature in features
        ]

        context.scores[self.method_name] = {
            "selected_features": list(details["selected_features"]),
            "elimination_order": list(details["eliminated_features"]),
            "best_params": details["best_params"],
            "best_metric": details["best_metric"],
            "optuna_enabled": options["optuna_enabled"],
            "optuna_trials": details["optuna_trials"],
            "fixed_params": details["fixed_params"],
            "search_space": details["search_space"],
            "algorithm": details["algorithm"],
            "num_features_to_select": details["num_features_to_select"],
            # CatBoost measures the eval loss after every elimination step.
            # Paired with elimination_order it shows where the loss starts
            # rising, so num_features_to_select can be picked from the curve
            # instead of guessed up front. n_candidates converts the curve's
            # "features removed" axis into "features left".
            "loss_graph": details["loss_graph"],
            "steps": details["steps"],
            "elimination_mode": details["elimination_mode"],
            "n_candidates": len(features),
            "eval_strategy": "out_of_time",
            "eval_periods": details["eval_periods"],
            "fit_rows": details["fit_rows"],
            "eval_rows": details["eval_rows"],
            "categorical_evaluated": categorical,
        }
        if details.get("feature_drop_per_step") is not None:
            context.scores[self.method_name]["feature_drop_per_step"] = details[
                "feature_drop_per_step"
            ]
        logger.info(
            "%s: evaluated %d features, kept %d",
            self.method_name,
            len(features),
            len(selected),
        )
        return decisions

    def _resolve_options(
        self: CatBoostRfeSelector,
        context: StageContext,
    ) -> dict[str, Any]:
        """Определяет параметры метода и лимиты ресурсов выполнения."""
        params = self.config.params
        parameters = params.get("parameters")
        if not isinstance(parameters, Mapping) or not parameters:
            msg = (
                "catboost_rfe: model.params.parameters is required. Provide CatBoost parameters as scalars "
                "(used as-is) and/or as search spaces such as {'type': 'int', 'min': 4, 'max': 8} (tuned by Optuna)."
            )
            raise ExecutionError(msg)

        optuna_params = params.get("optuna_params", {})
        if not isinstance(optuna_params, Mapping):
            msg = "catboost_rfe: model.params.optuna_params must be a mapping."
            raise ExecutionError(msg)

        feature_selection_params = params.get("feature_selection_params", {})
        if not isinstance(feature_selection_params, Mapping):
            msg = "catboost_rfe: model.params.feature_selection_params must be a mapping."
            raise ExecutionError(msg)

        target_count = self.config.selection.max_features
        if target_count is None:
            msg = (
                "catboost_rfe: model.selection.max_features is required; it sets "
                "num_features_to_select for recursive elimination."
            )
            raise ExecutionError(msg)

        try:
            optuna_settings = resolve_optuna_settings(
                optuna_params,
                method_name=self.method_name,
            )
            fixed, search_space = resolve_tuning_space(
                parameters,
                defaults=CATBOOST_RFE_SEARCH_SPACE,
                enabled=optuna_settings["enabled"],
                method_name=self.method_name,
            )
            options: dict[str, Any] = {
                "eval_months": int(params.get("eval_months", _DEFAULT_EVAL_MONTHS)),
                "max_rows": min(
                    int(params.get("max_rows", _DEFAULT_MAX_ROWS)),
                    int(context.config.execution.max_local_rows),
                ),
                "sample_fraction": params.get("sample_fraction"),
                "num_features_to_select": int(target_count),
                "fixed_params": fixed,
                "search_space": search_space,
                "optuna_enabled": optuna_settings["enabled"],
                "optuna_params": dict(optuna_params),
                "feature_selection_params": dict(feature_selection_params),
                "seed": resolve_step_seed(params, context),
            }
            if options["sample_fraction"] is not None:
                options["sample_fraction"] = float(options["sample_fraction"])
        except (TypeError, ValueError) as exc:
            msg = f"catboost_rfe: invalid numeric model parameter. Root cause: {exc}."
            raise ExecutionError(msg) from exc

        for name in ("eval_months", "max_rows", "num_features_to_select"):
            if options[name] < 1:
                msg = f"catboost_rfe: {name} must be at least 1."
                raise ExecutionError(msg)
        sample_fraction = options["sample_fraction"]
        if sample_fraction is not None and not 0.0 < sample_fraction <= 1.0:
            msg = "catboost_rfe: sample_fraction must be in (0, 1]."
            raise ExecutionError(msg)
        return options


def _split_out_of_time(
    frame: pd.DataFrame,
    *,
    time_col: str,
    target_col: str,
    eval_months: int,
    method_name: str,
    task_type: str = "binary_classification",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Разделяет DataFrame на train и eval по последним периодам временного столбца.

    Последние ``eval_months`` уникальных значений ``time_col`` образуют eval,
    а все более ранние — train. Это позволяет выполнять раннюю остановку и
    оценку важности RFE на будущих периодах, не используя внешнюю валидационную выборку.

    Args:
        frame: Локальный DataFrame со столбцами ``time_col`` и ``target_col``.
        time_col: Столбец с идентификатором периода.
        target_col: Целевой столбец; наличие классов проверяется в обеих частях.
        eval_months: Число последних периодов, выделяемых в eval.
        method_name: Имя метода отбора для сообщений об ошибках.
        task_type: Задача моделирования, используемая при проверке целевых классов.

    Returns:
        Кортеж ``(fit_frame, eval_frame, eval_periods)``.

    Raises:
        ExecutionError: Если разбиение невозможно или вырождено.
    """
    if time_col not in frame.columns:
        msg = f"{method_name}: time column {time_col!r} is missing from the materialized sample."
        raise ExecutionError(msg)
    if frame[time_col].isna().any():
        msg = f"{method_name}: time column {time_col!r} contains missing values; out-of-time split is not possible."
        raise ExecutionError(msg)

    try:
        periods = sorted(frame[time_col].unique())
    except TypeError as exc:
        msg = f"{method_name}: time column {time_col!r} holds values that cannot be ordered: {exc}."
        raise ExecutionError(msg) from exc

    if len(periods) <= eval_months:
        msg = (
            f"{method_name}: out-of-time split needs more than eval_months={eval_months} distinct periods "
            f"in {time_col!r}, found {len(periods)}. Widen the training window or lower eval_months."
        )
        raise ExecutionError(msg)

    eval_periods = periods[-eval_months:]
    eval_mask = frame[time_col].isin(eval_periods)
    fit_frame = frame.loc[~eval_mask]
    eval_frame = frame.loc[eval_mask]

    for name, part in (("fit", fit_frame), ("eval", eval_frame)):
        if part.empty:
            msg = f"{method_name}: the {name} part of the out-of-time split is empty."
            raise ExecutionError(msg)
        if task_type != "regression":
            classes = part[target_col].unique()
            if len(classes) < 2:
                msg = (
                    f"{method_name}: the {name} part of the out-of-time split contains a single target class. "
                    "Adjust eval_months or the sampling bounds."
                )
                raise ExecutionError(msg)

    return fit_frame, eval_frame, [str(period) for period in eval_periods]


def _load_backends(
    *,
    require_optuna: bool = True,
    task_type: str = "binary_classification",
) -> _Backends:
    """Лениво импортирует CatBoost, Optuna и метрики scikit-learn."""
    try:
        from catboost import Pool
    except ImportError as exc:
        msg = "catboost_rfe: CatBoost is required. Install the catboost optional dependency."
        raise BackendError(msg) from exc
    estimator_class = catboost_estimator_class(task_type)
    optuna_module: Any = None
    if require_optuna:
        try:
            import optuna as optuna_module
        except ImportError as exc:
            msg = "catboost_rfe: Optuna is required. Install the optuna optional dependency."
            raise BackendError(msg) from exc
        try:
            import sklearn.metrics  # noqa: F401
        except ImportError as exc:
            msg = "catboost_rfe: scikit-learn is required. Install the scikit-learn optional dependency."
            raise BackendError(msg) from exc
    return _Backends(
        estimator_class=estimator_class,
        pool_class=Pool,
        optuna_module=optuna_module,
    )


def _finalize_parameters(
    parameters: Mapping[str, Any],
    *,
    seed: int,
    task: TaskRuntime | None = None,
) -> dict[str, Any]:
    """Применяет настройки библиотеки по умолчанию, функцию потерь задачи и seed для воспроизводимости."""
    finalized = dict(_DEFAULT_PARAMETERS)
    finalized.update(parameters)
    finalized.update(catboost_loss_params(task if task is not None else binary_task()))
    finalized["random_seed"] = seed
    return finalized


def _tune_parameters(
    *,
    fit_pool: Any,
    eval_pool: Any,
    eval_labels: Any,
    fixed: Mapping[str, Any],
    search_space: Mapping[str, Mapping[str, Any]],
    optuna_params: Mapping[str, Any],
    seed: int,
    method_name: str,
    backends: _Backends,
    task: TaskRuntime,
) -> tuple[dict[str, Any], float, int]:
    """Подбирает параметры CatBoost через Optuna на eval из временного разбиения.

    Args:
        fit_pool: CatBoost ``Pool`` с train для обучения.
        eval_pool: CatBoost ``Pool`` с eval для ранней остановки и оценки.
        eval_labels: Истинные метки, соответствующие ``eval_pool``.
        fixed: Скалярные параметры, передаваемые без изменений.
        search_space: Спецификации параметров, подбираемых Optuna.
        optuna_params: ``n_trials``, ``n_startup_trials``, ``sampler``, ``timeout``.
        seed: Детерминированный seed для сэмплера и CatBoost.
        method_name: Имя метода отбора для сообщений об ошибках.
        backends: Результат :func:`_load_backends`.

    Returns:
        Кортеж ``(best_params, best_metric, completed_trials)``.

    Raises:
        ExecutionError: Если подбор завершился ошибкой или не дал ни одного пригодного испытания.
    """
    settings = resolve_optuna_settings(optuna_params, method_name=method_name)
    sampler = build_sampler(
        backends.optuna_module,
        sampler_name=settings["sampler"],
        search_space=search_space,
        seed=seed,
        n_startup_trials=settings["n_startup_trials"],
        method_name=method_name,
    )
    study = backends.optuna_module.create_study(
        direction=optuna_direction(task),
        sampler=sampler,
    )

    def objective(trial: Any) -> float:
        suggested = {
            name: suggest_parameter(trial, name, specification, method_name=method_name)
            for name, specification in search_space.items()
        }
        parameters = _finalize_parameters({**fixed, **suggested}, seed=seed, task=task)
        model = backends.estimator_class(**parameters)
        model.fit(fit_pool, eval_set=eval_pool)
        return score_model(task, model, eval_pool, eval_labels)

    try:
        study.optimize(
            objective,
            n_trials=settings["n_trials"],
            timeout=settings["timeout"],
            n_jobs=1,
        )
    except ExecutionError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize third-party tuning failures
        msg = f"catboost_rfe: Optuna tuning failed: {exc}"
        raise ExecutionError(msg) from exc

    completed = [trial for trial in study.trials if trial.value is not None]
    if not completed:
        msg = "catboost_rfe: Optuna finished without a completed trial. Increase n_trials or the timeout."
        raise ExecutionError(msg)

    best_params = _finalize_parameters(
        {**fixed, **study.best_params},
        seed=seed,
        task=task,
    )
    return best_params, float(study.best_value), len(completed)


def _run_catboost_rfe(
    frame: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    categorical_cols: Sequence[str],
    target_col: str,
    time_col: str,
    eval_months: int,
    fixed_params: Mapping[str, Any],
    search_space: Mapping[str, Mapping[str, Any]],
    optuna_params: Mapping[str, Any],
    feature_selection_params: Mapping[str, Any],
    num_features_to_select: int,
    seed: int,
    method_name: str = "catboost_rfe",
    task_type: str = "binary_classification",
) -> dict[str, Any]:
    """При необходимости подбирает параметры, затем выполняет рекурсивное исключение признаков CatBoost.

    Args:
        frame: Локальная выборка ограниченного размера с признаками, целевой переменной и временным столбцом.
        feature_cols: Признаки-кандидаты, оцениваемые методом отбора.
        categorical_cols: Подмножество ``feature_cols``, передаваемое в CatBoost как ``cat_features``.
        target_col: Имя целевого столбца.
        time_col: Столбец для разделения по времени.
        eval_months: Число последних периодов, выделяемых в eval.
        fixed_params: Скалярные параметры CatBoost, передаваемые без изменений.
        search_space: Спецификации параметров, подбираемых Optuna.
        optuna_params: Настройки Optuna, используемые при наличии пространства поиска.
        feature_selection_params: ``algorithm``, ``steps`` и другие параметры ``select_features``.
        num_features_to_select: Целевое число признаков после исключения.
        seed: Базовый seed для воспроизводимости.
        method_name: Имя метода отбора для сообщений об ошибках.
        task_type: Задача классификации или регрессии, передаваемая в CatBoost.

    Returns:
        Словарь с отобранными и исключёнными признаками, сведениями о подборе и размерами частей разбиения.

    Raises:
        BackendError: Если CatBoost, Optuna или scikit-learn недоступны.
        ExecutionError: При ошибке разбиения, подбора параметров или исключения признаков.
    """
    backends = _load_backends(require_optuna=False, task_type=task_type)
    estimator_class = backends.estimator_class
    pool_class = backends.pool_class
    task = resolve_task(
        task_type,
        frame[target_col].to_numpy(),
        method_name=method_name,
        encode_labels=False,
    )

    fit_frame, eval_frame, eval_periods = _split_out_of_time(
        frame,
        time_col=time_col,
        target_col=target_col,
        eval_months=eval_months,
        method_name=method_name,
        task_type=task.task_type,
    )

    features = list(feature_cols)
    categorical = [column for column in features if column in set(categorical_cols)]
    fit_pool = pool_class(
        fit_frame.loc[:, features],
        fit_frame[target_col],
        cat_features=categorical,
    )
    eval_pool = pool_class(
        eval_frame.loc[:, features],
        eval_frame[target_col],
        cat_features=categorical,
    )

    fixed, tuned_space = apply_boost_heuristics(
        fixed_params,
        search_space,
        n_rows=len(fit_frame),
        library="catboost",
        task_type=task.task_type,
    )
    if tuned_space:
        backends = _load_backends(require_optuna=True, task_type=task.task_type)
        best_params, best_metric, completed_trials = _tune_parameters(
            fit_pool=fit_pool,
            eval_pool=eval_pool,
            eval_labels=eval_frame[target_col].to_numpy(),
            fixed=fixed,
            search_space=tuned_space,
            optuna_params=optuna_params,
            seed=seed,
            method_name=method_name,
            backends=backends,
            task=task,
        )
        logger.info(
            "%s: Optuna finished %d trials, best eval metric %.5f",
            method_name,
            completed_trials,
            best_metric,
        )
    else:
        best_params = _finalize_parameters(fixed, seed=seed, task=task)
        best_metric = None
        completed_trials = 0
        logger.info("%s: Optuna disabled or search space empty, skipping tuning", method_name)

    selection_params = dict(feature_selection_params)
    algorithm = str(selection_params.pop("algorithm", "RecursiveByLossFunctionChange"))
    if algorithm not in _ALGORITHMS:
        msg = (
            f"{method_name}: unsupported algorithm={algorithm!r}. "
            f"Expected one of: {sorted(_ALGORITHMS)}."
        )
        raise ExecutionError(msg)
    selection_params.pop("num_features_to_select", None)
    selection_params.pop("train_final_model", None)
    mode, schedule_value = _pop_elimination_schedule(selection_params, method_name=method_name)

    if mode == "feature_drop_per_step":
        selected, eliminated, loss_graph, n_rounds = _eliminate_constant_drop(
            estimator_class=estimator_class,
            pool_class=pool_class,
            fit_frame=fit_frame,
            eval_frame=eval_frame,
            features=features,
            categorical_cols=categorical,
            target_col=target_col,
            best_params=best_params,
            extra_selection_params=selection_params,
            algorithm=algorithm,
            drop_per_step=schedule_value,
            num_features_to_select=num_features_to_select,
            method_name=method_name,
        )
        payload_steps = n_rounds
    else:
        model = estimator_class(**best_params)
        try:
            summary = model.select_features(
                fit_pool,
                eval_set=eval_pool,
                features_for_select=features,
                num_features_to_select=num_features_to_select,
                algorithm=algorithm,
                steps=schedule_value,
                train_final_model=False,
                **selection_params,
            )
        except Exception as exc:  # noqa: BLE001 - normalize CatBoost failures
            msg = f"{method_name}: CatBoost recursive elimination failed: {exc}"
            raise ExecutionError(msg) from exc
        selected = _names_from_summary(summary, features, "selected_features_names", "selected_features")
        eliminated = _names_from_summary(
            summary,
            features,
            "eliminated_features_names",
            "eliminated_features",
        )
        loss_graph = summary.get("loss_graph")
        payload_steps = schedule_value

    result: dict[str, Any] = {
        "selected_features": selected,
        "eliminated_features": eliminated,
        "best_params": best_params,
        "best_metric": best_metric,
        "optuna_trials": completed_trials,
        "fixed_params": fixed,
        "search_space": tuned_space,
        "algorithm": algorithm,
        "num_features_to_select": num_features_to_select,
        "steps": payload_steps,
        "elimination_mode": mode,
        "eval_periods": eval_periods,
        "fit_rows": len(fit_frame),
        "eval_rows": len(eval_frame),
        "loss_graph": loss_graph,
    }
    if mode == "feature_drop_per_step":
        result["feature_drop_per_step"] = schedule_value
    return result


def _pop_elimination_schedule(
    selection_params: dict[str, Any],
    *,
    method_name: str,
) -> tuple[str, int]:
    """Возвращает ``(mode, steps_or_drop)`` и удаляет эти ключи из ``selection_params``."""
    has_steps = "steps" in selection_params
    has_drop = "feature_drop_per_step" in selection_params
    if has_steps and has_drop:
        msg = (
            f"{method_name}: feature_selection_params cannot set both "
            "'steps' and 'feature_drop_per_step'."
        )
        raise ExecutionError(msg)
    if has_drop:
        raw = selection_params.pop("feature_drop_per_step")
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
            msg = f"{method_name}: feature_selection_params.feature_drop_per_step must be >= 1."
            raise ExecutionError(msg)
        return "feature_drop_per_step", raw
    raw_steps = selection_params.pop("steps", _DEFAULT_RFE_STEPS)
    steps = int(raw_steps)
    if steps < 1:
        msg = f"{method_name}: feature_selection_params.steps must be >= 1."
        raise ExecutionError(msg)
    return "steps", steps


def _names_from_summary(
    summary: Mapping[str, Any],
    remaining: Sequence[str],
    names_key: str,
    index_key: str,
) -> list[str]:
    """Использует списки имён из CatBoost, а при их отсутствии — индексы в ``remaining``."""
    names = summary.get(names_key)
    if names:
        return [str(name) for name in names]
    indices = summary.get(index_key) or []
    resolved: list[str] = []
    for index in indices:
        position = int(index)
        if 0 <= position < len(remaining):
            resolved.append(remaining[position])
    return resolved


def _eliminate_constant_drop(
    *,
    estimator_class: Any,
    pool_class: Any,
    fit_frame: pd.DataFrame,
    eval_frame: pd.DataFrame,
    features: Sequence[str],
    categorical_cols: Sequence[str],
    target_col: str,
    best_params: Mapping[str, Any],
    extra_selection_params: Mapping[str, Any],
    algorithm: str,
    drop_per_step: int,
    num_features_to_select: int,
    method_name: str,
) -> tuple[list[str], list[str], Any, int]:
    """Исключает фиксированное число признаков за каждый раунд ``select_features(..., steps=1)``."""
    remaining = list(features)
    eliminated: list[str] = []
    graphs: list[Any] = []
    cumulative_removed: list[int] = []
    try:
        targets = _constant_drop_targets(
            len(remaining),
            num_features_to_select,
            drop_per_step,
        )
    except ValueError as exc:
        msg = f"{method_name}: {exc}"
        raise ExecutionError(msg) from exc

    categorical_set = set(categorical_cols)
    for target in targets:
        cats = [name for name in remaining if name in categorical_set]
        fit_pool = pool_class(
            fit_frame.loc[:, remaining],
            fit_frame[target_col],
            cat_features=cats,
        )
        eval_pool = pool_class(
            eval_frame.loc[:, remaining],
            eval_frame[target_col],
            cat_features=cats,
        )
        model = estimator_class(**best_params)
        try:
            summary = model.select_features(
                fit_pool,
                eval_set=eval_pool,
                features_for_select=remaining,
                num_features_to_select=target,
                algorithm=algorithm,
                steps=1,
                train_final_model=False,
                **extra_selection_params,
            )
        except Exception as exc:
            msg = f"{method_name}: CatBoost recursive elimination failed: {exc}"
            raise ExecutionError(msg) from exc
        dropped = _names_from_summary(
            summary,
            remaining,
            "eliminated_features_names",
            "eliminated_features",
        )
        kept = _names_from_summary(
            summary,
            remaining,
            "selected_features_names",
            "selected_features",
        )
        if not kept:
            kept = [name for name in remaining if name not in set(dropped)]
        eliminated.extend(dropped)
        remaining = kept
        graphs.append(summary.get("loss_graph"))
        cumulative_removed.append(len(eliminated))

    return remaining, eliminated, _stitch_constant_drop_loss(graphs, cumulative_removed), len(targets)


def _stitch_constant_drop_loss(
    graphs: Sequence[Any],
    cumulative_removed: Sequence[int],
) -> dict[str, Any] | None:
    """Перестраивает loss_graph, чтобы ось x отражала число признаков, удалённых из исходного набора."""
    if not graphs:
        return None
    removed_counts: list[int] = [0]
    loss_values: list[Any] = []
    first = graphs[0] if isinstance(graphs[0], Mapping) else None
    if first and first.get("loss_values"):
        loss_values.append(first["loss_values"][0])
    else:
        loss_values.append(None)
    for graph, total_removed in zip(graphs, cumulative_removed, strict=False):
        if not isinstance(graph, Mapping) or not graph.get("loss_values"):
            continue
        values = list(graph["loss_values"])
        removed_counts.append(int(total_removed))
        loss_values.append(values[-1])
    return {
        "removed_features_count": removed_counts,
        "loss_values": loss_values,
        "main_indices": list(range(len(loss_values))),
    }
