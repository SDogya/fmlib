"""Метод отбора на основе LightGBM с оценкой важности SHAP.

Внешние фолды обрабатываются последовательно на драйвере: метод строит локальную числовую
матрицу ограниченного размера, при необходимости подбирает параметры LightGBM через Optuna, затем обучает по модели на
фолд. ``selection_mode="aggregated"`` усредняет важности по разбиениям и SHAP и
сохраняет пересечение наборов, отобранных по накопленному порогу. ``selection_mode="vote"`` применяет порог
к векторам важности по разбиениям и SHAP каждого фолда отдельно и сохраняет признаки, входящие
как минимум в долю ``min_set_share`` из этих ``2 * n_folds`` наборов.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from fmlib.feature_selection.base import FeatureDecision, StageContext, resolve_step_seed
from fmlib.feature_selection.config import ModelConfig
from fmlib.feature_selection.exceptions import BackendError, ExecutionError
from fmlib.feature_selection.utils.default_model_param_spaces import (
    LIGHTGBM_SEARCH_SPACE,
)
from fmlib.feature_selection.utils.lama_boost_defaults import (
    apply_boost_heuristics,
    fit_lgbm_with_early_stopping,
    split_lgbm_early_stopping,
)
from fmlib.feature_selection.utils.local_data import prepare_numeric_frame, root_cause
from fmlib.feature_selection.utils.optuna_space import (
    build_sampler,
    resolve_optuna_settings,
    resolve_tuning_space,
    suggest_parameter,
)
from fmlib.feature_selection.utils.task_runtime import (
    TaskRuntime,
    binary_task,
    lgbm_estimator_class,
    lgbm_objective_params,
    make_folds,
    normalize_binary_shap_values as normalize_binary_shap_values,
    optuna_direction,
    require_min_class_count,
    resolve_task,
    score_model,
    shap_mean_abs,
)

logger = logging.getLogger(__name__)

try:
    import optuna
except ImportError:
    optuna = None

try:
    import shap
except ImportError:
    shap = None

OPTUNA_MODES = frozenset({"global", "per_fold"})
SELECTION_MODES = frozenset({"aggregated", "vote"})

_DEFAULT_N_TRIALS = 20

DEFAULT_SEARCH_SPACE: dict[str, dict[str, Any]] = LIGHTGBM_SEARCH_SPACE


class LightGbmSelector:
    """Отбирает непрерывные признаки по оценкам важности LightGBM и SHAP.

    Алгоритм модели следует ``shap_lgbm_spark.py``: подбирает параметры LightGBM на
    отложенном разбиении (стратифицированном, кроме задач регрессии), обрабатывает внешние фолды
    последовательно на драйвере, затем либо усредняет важности по разбиениям и SHAP
    и сохраняет пересечение наборов, отобранных по накопленному порогу
    (``selection_mode="aggregated"``), либо применяет порог к векторам важности по разбиениям и SHAP каждого фолда
    отдельно и сохраняет признаки, входящие как минимум в долю
    ``min_set_share`` из этих ``2 * n_folds`` наборов (``selection_mode="vote"``).
    ``optuna_mode="global"``
    подбирает параметры один раз на драйвере; ``"per_fold"`` — независимо для каждого
    фолда, используя только строки его внешней обучающей части.

    Входные данные Spark стратифицируются до загрузки в локальную память. Для уже локальных
    данных pandas используется аналогичная стратифицированная выборка ограниченного размера. Оцениваются только текущие
    кандидаты, объявленные в ``FeatureSchema.continuous``;
    категориальные кандидаты проходят этап модели без изменений. При обучении фолдов
    код и пакеты никогда не отправляются исполнителям Spark.

    Для Optuna по умолчанию используется пространство поиска ``DEFAULT_SEARCH_SPACE``, если
    ``params.parameters`` не содержит словарей. Любой словарь в этом блоке
    полностью заменяет пространство поиска по умолчанию: неуказанные ключи по умолчанию не добавляются.
    Скаляр передаётся в LightGBM без изменений и не подбирается.
    Отсутствующие ``learning_rate`` и ``early_stopping_rounds`` заполняются по
    таблице LightAutoML в зависимости от числа строк после загрузки локальной выборки; скаляр из YAML
    сохраняется. Optuna не подбирает эти параметры.
    ``early_stopping_rounds: 0`` обучает модель до лимита деревьев без ранней остановки.
    ``params.optuna_params.enabled: false`` полностью отключает Optuna.

    Args:
        config: Настройки этапа модели. Параметры ``params`` конкретного метода переопределяют
            соответствующие настройки подбора и кросс-валидации по умолчанию.
    """

    method_name = "lightgbm"
    stage_name = "model"

    def __init__(self: LightGbmSelector, config: ModelConfig) -> None:
        self.config = config

    def select(
        self: LightGbmSelector,
        context: StageContext,
        candidates: Sequence[str],
    ) -> list[FeatureDecision]:
        """Оценивает непрерывные признаки-кандидаты с помощью LightGBM и SHAP.

        Args:
            context: Общий контекст этапа с обучающими данными, схемой, конфигурацией
                и seed для воспроизводимости.
            candidates: Признаки, которые ещё рассматриваются для отбора.

        Returns:
            Решения о сохранении или исключении оценённых непрерывных признаков. Признаки
            вне области действия этого метода проходят без решения и изменений.

        Raises:
            BackendError: Если необязательная зависимость для машинного обучения недоступна.
            ExecutionError: При некорректных входных данных или выполнении модели.
        """
        if not candidates:
            return []

        target_col = context.schema.target
        if not target_col:
            msg = "lightgbm: FeatureSchema.target is required."
            raise ExecutionError(msg)

        train = context.datasets.get("train")
        if train is None:
            msg = "lightgbm: context.datasets must contain a 'train' split."
            raise ExecutionError(msg)

        continuous = set(context.schema.continuous)
        feature_cols = [feature for feature in candidates if feature in continuous]
        if not feature_cols:
            return []

        options = self._resolve_options(context)
        self._load_backends(require_optuna=bool(options["search_space"]))

        try:
            details = self._select_robust_features(
                df=train,
                target_col=target_col,
                feature_cols=feature_cols,
                n_trials=options["n_trials"],
                n_startup_trials=options["n_startup_trials"],
                sampler=options["sampler"],
                timeout=options["timeout"],
                n_folds=options["n_folds"],
                max_rows_limit=options["max_rows"],
                sample_fraction=options["sample_fraction"],
                lgbm_threshold=options["lgbm_threshold"],
                shap_threshold=options["shap_threshold"],
                seed=options["seed"],
                optuna_mode=options["optuna_mode"],
                selection_mode=options["selection_mode"],
                min_set_share=options["min_set_share"],
                n_jobs=options["n_jobs"],
                shap_max_rows=options["shap_max_rows"],
                search_space=options["search_space"],
                fixed_params=options["fixed_params"],
                return_importances=True,
                context=context,
            )
        except (BackendError, ExecutionError):
            raise
        except Exception as exc:  # noqa: BLE001 - normalize third-party failures
            msg = f"lightgbm: feature selection failed. Root cause: {root_cause(exc)}."
            raise ExecutionError(msg) from exc

        importances = details["importances_df"]
        indexed_importances = importances.set_index("feature")
        lgbm_scores = indexed_importances["lgbm_norm"].to_dict()
        shap_scores = indexed_importances["shap_norm"].to_dict()
        lgbm_cumulative = indexed_importances["lgbm_cumsum"].to_dict()
        shap_cumulative = indexed_importances["shap_cumsum"].to_dict()
        selected = set(details["selected_features"])
        selection_mode = options["selection_mode"]
        min_set_share = options["min_set_share"]
        set_presence = {
            str(feature): float(share)
            for feature, share in dict(details.get("set_presence") or {}).items()
        }

        decisions = []
        for feature in feature_cols:
            if selection_mode == "vote":
                value = float(set_presence[feature])
                threshold = min_set_share
                reason = (
                    "passed_lgbm_shap_vote"
                    if feature in selected
                    else "failed_lgbm_shap_vote"
                )
            else:
                value = float(
                    max(
                        lgbm_cumulative[feature] / options["lgbm_threshold"],
                        shap_cumulative[feature] / options["shap_threshold"],
                    ),
                )
                threshold = 1.0
                reason = (
                    "passed_lgbm_shap_selection"
                    if feature in selected
                    else "failed_lgbm_shap_selection"
                )
            decisions.append(
                FeatureDecision(
                    feature=feature,
                    stage=self.stage_name,
                    method=self.method_name,
                    reason=reason,
                    value=value,
                    threshold=threshold,
                    keep=feature in selected,
                ),
            )

        scores: dict[str, Any] = {
            "importances": {key: float(value) for key, value in lgbm_scores.items()},
            "shap_importances": {
                key: float(value) for key, value in shap_scores.items()
            },
            "lgbm_threshold": options["lgbm_threshold"],
            "shap_threshold": options["shap_threshold"],
            "selection_mode": options["selection_mode"],
            "min_set_share": options["min_set_share"],
            "optuna_mode": options["optuna_mode"],
            "optuna_enabled": options["optuna_enabled"],
            "search_space": options["search_space"],
            "fixed_params": options["fixed_params"],
            "fold_execution": "driver",
            "global_best_params": details["global_best_params"],
            "fold_best_params": details["fold_best_params"],
        }
        if selection_mode == "vote":
            scores["n_sets"] = int(details["n_sets"])
            scores["set_presence"] = set_presence
            scores["fold_sets"] = details["fold_sets"]
        context.scores[self.method_name] = scores
        logger.info(
            "LightGbmSelector: evaluated %d features, kept %d",
            len(feature_cols),
            len(selected),
        )
        return decisions

    def _load_backends(
        self: LightGbmSelector,
        *,
        require_optuna: bool = True,
    ) -> tuple[Any, Any, Any]:
        """Загружает необязательные зависимости модели."""
        try:
            import lightgbm as lgb
        except ImportError as exc:
            msg = (
                "lightgbm: LightGBM is required. Install the lightgbm optional "
                "dependency."
            )
            raise BackendError(msg) from exc

        if shap is None:
            msg = "lightgbm: SHAP is required. Install the shap optional dependency."
            raise BackendError(msg)
        if require_optuna and optuna is None:
            msg = (
                "lightgbm: Optuna is required. Install the optuna optional "
                "dependency."
            )
            raise BackendError(msg)
        return lgb, shap, optuna

    def _resolve_options(
        self: LightGbmSelector,
        context: StageContext,
    ) -> dict[str, Any]:
        """Определяет и проверяет параметры метода, сохраняя значения по умолчанию из прототипа."""
        params = self.config.params
        try:
            legacy_n_trials = params.get("n_trials")
            optuna_settings = resolve_optuna_settings(
                params.get("optuna_params", {}),
                method_name=self.method_name,
                n_trials=(
                    _DEFAULT_N_TRIALS
                    if legacy_n_trials is None
                    else int(legacy_n_trials)
                ),
            )
            if "n_jobs" in params:
                n_jobs = int(params["n_jobs"])
            elif "driver_n_jobs" in params:
                n_jobs = int(params["driver_n_jobs"])
            else:
                n_jobs = -1
            options: dict[str, Any] = {
                "lgbm_threshold": float(params.get("lgbm_threshold", 0.85)),
                "shap_threshold": float(params.get("shap_threshold", 0.85)),
                "n_folds": int(
                    params.get("n_folds")
                    or self.config.cross_validation.folds
                ),
                "n_trials": optuna_settings["n_trials"],
                "n_startup_trials": optuna_settings["n_startup_trials"],
                "sampler": optuna_settings["sampler"],
                "timeout": optuna_settings["timeout"],
                "max_rows": min(
                    int(params.get("max_rows", 250_000)),
                    int(context.config.execution.max_local_rows),
                ),
                "sample_fraction": params.get("sample_fraction"),
                "optuna_mode": str(
                    params.get("optuna_mode", "global"),
                ).lower(),
                "selection_mode": str(
                    params.get("selection_mode", "aggregated"),
                ).lower(),
                "min_set_share": float(params.get("min_set_share", 1.0)),
                "optuna_enabled": optuna_settings["enabled"],
                "n_jobs": n_jobs,
                "shap_max_rows": int(params.get("shap_max_rows", 5_000)),
                "seed": resolve_step_seed(params, context),
            }
            fixed, search_space = resolve_tuning_space(
                params.get("parameters", {}),
                defaults=DEFAULT_SEARCH_SPACE,
                enabled=optuna_settings["enabled"],
                method_name=self.method_name,
            )
            options["fixed_params"] = fixed
            options["search_space"] = search_space
            sample_fraction = options["sample_fraction"]
            if sample_fraction is not None:
                options["sample_fraction"] = float(sample_fraction)
        except (TypeError, ValueError) as exc:
            msg = f"lightgbm: invalid numeric model parameter. Root cause: {exc}."
            raise ExecutionError(msg) from exc

        for name in ("lgbm_threshold", "shap_threshold"):
            if not 0.0 < options[name] <= 1.0:
                msg = f"lightgbm: {name} must be in (0, 1]; got {options[name]!r}."
                raise ExecutionError(msg)
        if options["n_folds"] < 2:
            msg = "lightgbm: n_folds must be at least 2."
            raise ExecutionError(msg)
        if options["max_rows"] < 1:
            msg = "lightgbm: max_rows must be at least 1."
            raise ExecutionError(msg)
        if options["optuna_mode"] not in OPTUNA_MODES:
            msg = (
                f"lightgbm: optuna_mode must be one of "
                f"{sorted(OPTUNA_MODES)}."
            )
            raise ExecutionError(msg)
        if options["selection_mode"] not in SELECTION_MODES:
            msg = (
                f"lightgbm: selection_mode must be one of "
                f"{sorted(SELECTION_MODES)}."
            )
            raise ExecutionError(msg)
        if not 0.0 < options["min_set_share"] <= 1.0:
            msg = (
                "lightgbm: min_set_share must be in (0, 1]; "
                f"got {options['min_set_share']!r}."
            )
            raise ExecutionError(msg)
        if options["n_jobs"] == 0 or options["n_jobs"] < -1:
            msg = "lightgbm: n_jobs must be -1 or a positive integer."
            raise ExecutionError(msg)
        if options["shap_max_rows"] < 1:
            msg = "lightgbm: shap_max_rows must be at least 1."
            raise ExecutionError(msg)
        if (
            sample_fraction is not None
            and not 0.0 < options["sample_fraction"] <= 1.0
        ):
            msg = "lightgbm: sample_fraction must be in (0, 1]."
            raise ExecutionError(msg)
        return options

    def _extract_and_prep_data(
        self: LightGbmSelector,
        df: Any,
        target_col: str,
        feature_cols: list[str],
        max_rows: int,
        sample_fraction: float | None,
        seed: int,
        context: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Строит локальную числовую матрицу ограниченного размера для алгоритма прототипа."""
        local = prepare_numeric_frame(
            df,
            target_col=target_col,
            feature_cols=feature_cols,
            max_rows=max_rows,
            sample_fraction=sample_fraction,
            seed=seed,
            method_name=self.method_name,
            context=context,
        )
        return (
            local.loc[:, feature_cols].to_numpy(dtype=float),
            local[target_col].to_numpy(),
            list(feature_cols),
        )

    def _select_robust_features(
        self: LightGbmSelector,
        *,
        df: Any,
        target_col: str,
        feature_cols: list[str],
        n_trials: int,
        n_folds: int,
        max_rows_limit: int,
        sample_fraction: float | None,
        lgbm_threshold: float,
        shap_threshold: float,
        seed: int,
        optuna_mode: str,
        n_jobs: int,
        shap_max_rows: int,
        search_space: dict[str, dict[str, Any]] | None = None,
        fixed_params: dict[str, Any] | None = None,
        n_startup_trials: int = 10,
        sampler: str = "TPE",
        timeout: int | None = None,
        selection_mode: str = "aggregated",
        min_set_share: float = 1.0,
        return_importances: bool = False,
        context: Any | None = None,
    ) -> list[str] | dict[str, Any]:
        """Подбирает параметры и обрабатывает каждый внешний фолд на драйвере."""
        feature_matrix, target, evaluated = self._extract_and_prep_data(
            df,
            target_col,
            feature_cols,
            max_rows_limit,
            sample_fraction,
            seed,
            context,
        )
        task_type = (
            context.schema.task_type
            if context is not None
            else "binary_classification"
        )
        task = resolve_task(task_type, target, method_name="lightgbm")
        target = task.encoded_target
        if task.is_regression:
            if len(target) < n_folds:
                msg = (
                    "lightgbm: regression sample is smaller than n_folds "
                    f"({len(target)} < {n_folds})."
                )
                raise ExecutionError(msg)
        else:
            require_min_class_count(
                task,
                min_count=n_folds,
                method_name="lightgbm",
            )

        global_best_params: dict[str, Any] | None = None
        fixed_params, effective_space = apply_boost_heuristics(
            fixed_params or {},
            DEFAULT_SEARCH_SPACE if search_space is None else search_space,
            n_rows=len(target),
            library="lightgbm",
            task_type=task.task_type,
        )
        if optuna_mode == "global":
            if effective_space:
                global_best_params = tune_parameters(
                    feature_matrix,
                    target,
                    n_trials=n_trials,
                    seed=seed,
                    n_jobs=n_jobs,
                    search_space=effective_space,
                    fixed_params=fixed_params,
                    n_startup_trials=n_startup_trials,
                    sampler=sampler,
                    timeout=timeout,
                    task=task,
                )
            else:
                global_best_params = _finalize_parameters(
                    fixed_params or {},
                    seed=seed,
                    n_jobs=n_jobs,
                    task=task,
                )
        folds = make_folds(task, n_folds=n_folds, seed=seed)
        logger.info(
            "LightGbmSelector: running %d folds on driver, optuna_mode=%s",
            n_folds,
            optuna_mode,
        )

        fold_lgbm: list[np.ndarray] = []
        fold_shap: list[np.ndarray] = []
        fold_best_params: dict[str, dict[str, Any]] = {}
        for fold_index, (_, valid_indices) in enumerate(
            folds.split(feature_matrix, target),
            start=1,
        ):
            lgbm_values, shap_values, best_params = self._run_fold(
                feature_matrix,
                target,
                fold_index=fold_index,
                valid_indices=np.asarray(valid_indices, dtype=np.int64),
                seed=seed + fold_index,
                optuna_mode=optuna_mode,
                n_trials=n_trials,
                n_startup_trials=n_startup_trials,
                sampler=sampler,
                timeout=timeout,
                n_jobs=n_jobs,
                shap_max_rows=shap_max_rows,
                global_params=global_best_params,
                search_space=effective_space,
                fixed_params=fixed_params,
                task=task,
            )
            fold_lgbm.append(lgbm_values)
            fold_shap.append(shap_values)
            fold_best_params[str(fold_index)] = dict(best_params)

        if not fold_lgbm:
            msg = "lightgbm: cross-validation produced no folds."
            raise ExecutionError(msg)
        if selection_mode not in SELECTION_MODES:
            msg = (
                f"lightgbm: selection_mode must be one of "
                f"{sorted(SELECTION_MODES)}."
            )
            raise ExecutionError(msg)

        lgbm_importances = np.sum(fold_lgbm, axis=0) / n_folds
        shap_importances = np.sum(fold_shap, axis=0) / n_folds

        details = self._aggregate_importances(
            evaluated,
            lgbm_importances,
            shap_importances,
            lgbm_threshold,
            shap_threshold,
        )
        if selection_mode == "vote":
            vote = self._vote_importances(
                evaluated,
                fold_lgbm,
                fold_shap,
                lgbm_threshold,
                shap_threshold,
                min_set_share,
            )
            details["selected_features"] = vote["selected_features"]
            details["set_presence"] = vote["set_presence"]
            details["fold_sets"] = vote["fold_sets"]
            details["n_sets"] = vote["n_sets"]
        details["global_best_params"] = (
            dict(global_best_params)
            if global_best_params is not None
            else None
        )
        details["fold_best_params"] = fold_best_params
        return details if return_importances else details["selected_features"]

    @staticmethod
    def _run_fold(
        feature_matrix: np.ndarray,
        target: np.ndarray,
        *,
        fold_index: int,
        valid_indices: np.ndarray,
        seed: int,
        optuna_mode: str,
        n_trials: int,
        n_jobs: int,
        shap_max_rows: int,
        n_startup_trials: int = 10,
        sampler: str = "TPE",
        timeout: int | None = None,
        global_params: dict[str, Any] | None = None,
        search_space: dict[str, dict[str, Any]] | None = None,
        fixed_params: dict[str, Any] | None = None,
        task: TaskRuntime | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """При необходимости подбирает параметры, обучает один фолд и вычисляет важности LGBM/SHAP.

        Args:
            feature_matrix: Полная локальная матрица признаков, общая для всех фолдов.
            target: Полный локальный вектор целевой переменной, общий для всех фолдов.
            fold_index: Номер фолда, начиная с 1, для сообщений об ошибках.
            valid_indices: Позиции строк, выделенных в валидационную часть этого фолда.
            seed: Детерминированный seed для подбора параметров, модели и формирования выборки SHAP.
            optuna_mode: ``"global"`` или ``"per_fold"``.
            n_trials: Число испытаний Optuna; используется только при ``optuna_mode="per_fold"``.
            n_jobs: Число потоков, принудительно задаваемое модели.
            shap_max_rows: Верхняя граница числа строк для объяснения с помощью SHAP.
            global_params: Параметры, подобранные один раз; обязательны, если подбор не выполняется по фолдам.
            search_space: Спецификации параметров, подбираемых Optuna.
            fixed_params: Скалярные параметры, передаваемые без изменений.

        Returns:
            Кортеж ``(lgbm_importances, shap_importances, best_params)``.

        Raises:
            BackendError: Если LightGBM или SHAP недоступны.
            ExecutionError: При ошибке подбора, обучения или расчёта SHAP для этого фолда.
        """
        try:
            import shap
        except ImportError as exc:
            msg = (
                "lightgbm: LightGBM and SHAP must be installed on the driver "
                "process running fold execution."
            )
            raise BackendError(msg) from exc

        resolved = (
            task
            if task is not None
            else resolve_task("binary_classification", target, method_name="lightgbm")
        )
        row_count = len(target)
        train_mask = np.ones(row_count, dtype=bool)
        train_mask[valid_indices] = False
        train_indices = np.flatnonzero(train_mask)
        train_matrix = feature_matrix[train_indices]
        train_target = target[train_indices]
        valid_matrix = feature_matrix[valid_indices]
        valid_target = target[valid_indices]

        if optuna_mode == "per_fold":
            fold_space = DEFAULT_SEARCH_SPACE if search_space is None else search_space
            if fold_space:
                best_params = tune_parameters(
                    train_matrix,
                    train_target,
                    n_trials=n_trials,
                    seed=seed,
                    n_jobs=n_jobs,
                    search_space=fold_space,
                    fixed_params=fixed_params,
                    n_startup_trials=n_startup_trials,
                    sampler=sampler,
                    timeout=timeout,
                    task=resolved,
                )
            else:
                best_params = _finalize_parameters(
                    fixed_params or {},
                    seed=seed,
                    n_jobs=n_jobs,
                    task=resolved,
                )
        elif global_params is not None:
            best_params = _finalize_parameters(
                global_params,
                seed=seed,
                n_jobs=n_jobs,
                task=resolved,
            )
        else:
            msg = (
                f"lightgbm: fold {fold_index} has no global Optuna "
                "parameters."
            )
            raise ExecutionError(msg)

        try:
            ctor_params, stopping_rounds = split_lgbm_early_stopping(best_params)
            model = lgbm_estimator_class(resolved)(**ctor_params)
            fit_lgbm_with_early_stopping(
                model,
                train_matrix,
                train_target,
                eval_set=[(valid_matrix, valid_target)],
                early_stopping_rounds=stopping_rounds,
            )
            lgbm_importances = np.asarray(
                model.booster_.feature_importance(importance_type="split"),
                dtype=float,
            )

            sample_size = min(len(valid_matrix), shap_max_rows)
            if len(valid_matrix) <= sample_size:
                shap_sample = valid_matrix
            else:
                rng = np.random.default_rng(seed)
                sample_indices = rng.choice(
                    len(valid_matrix),
                    sample_size,
                    replace=False,
                )
                shap_sample = valid_matrix[sample_indices]
            shap_values = shap.TreeExplainer(model).shap_values(shap_sample)
            shap_importances = shap_mean_abs(resolved, shap_values)
        except Exception as exc:  # noqa: BLE001 - model/SHAP failures
            msg = f"lightgbm: fold {fold_index} failed: {exc}"
            raise ExecutionError(msg) from exc

        return lgbm_importances, shap_importances, dict(best_params)

    @staticmethod
    def _aggregate_importances(
        feature_cols: list[str],
        lgbm_importances: np.ndarray,
        shap_importances: np.ndarray,
        lgbm_threshold: float,
        shap_threshold: float,
    ) -> dict[str, Any]:
        """Нормализует важности и пересекает наборы, достигающие накопленного порога."""
        lgbm_selected, lgbm_norm, lgbm_cumsum = _cumulative_select(
            lgbm_importances,
            feature_cols,
            lgbm_threshold,
            empty_total_message=(
                "lightgbm: split importances have a non-positive total."
            ),
        )
        shap_selected, shap_norm, shap_cumsum = _cumulative_select(
            shap_importances,
            feature_cols,
            shap_threshold,
            empty_total_message=(
                "lightgbm: SHAP importances have a non-positive total."
            ),
        )
        importances = pd.DataFrame(
            {
                "feature": feature_cols,
                "lgbm_imp": lgbm_importances,
                "shap_imp": shap_importances,
                "lgbm_norm": lgbm_norm,
                "shap_norm": shap_norm,
                "lgbm_cumsum": lgbm_cumsum,
                "shap_cumsum": shap_cumsum,
            }
        )
        selected = [
            feature
            for feature in feature_cols
            if feature in lgbm_selected and feature in shap_selected
        ]
        return {
            "selected_features": selected,
            "lgbm_selected": [
                feature for feature in feature_cols if feature in lgbm_selected
            ],
            "shap_selected": [
                feature for feature in feature_cols if feature in shap_selected
            ],
            "lgbm_dropped": [
                feature for feature in feature_cols if feature not in lgbm_selected
            ],
            "shap_dropped": [
                feature for feature in feature_cols if feature not in shap_selected
            ],
            "importances_df": importances,
        }

    @staticmethod
    def _vote_importances(
        feature_cols: list[str],
        fold_lgbm: Sequence[np.ndarray],
        fold_shap: Sequence[np.ndarray],
        lgbm_threshold: float,
        shap_threshold: float,
        min_set_share: float,
    ) -> dict[str, Any]:
        """Применяет порог к векторам важности по разбиениям и SHAP каждого фолда, затем отбирает по доле вхождений в наборы."""
        if len(fold_lgbm) != len(fold_shap):
            msg = "lightgbm: vote selection requires one SHAP vector per fold."
            raise ExecutionError(msg)
        n_folds = len(fold_lgbm)
        if n_folds < 1:
            msg = "lightgbm: vote selection requires at least one fold."
            raise ExecutionError(msg)
        n_sets = 2 * n_folds
        counts = {feature: 0 for feature in feature_cols}
        fold_sets: dict[str, dict[str, list[str]]] = {}
        for fold_index, (lgbm_values, shap_values) in enumerate(
            zip(fold_lgbm, fold_shap),
            start=1,
        ):
            lgbm_selected, _, _ = _cumulative_select(
                np.asarray(lgbm_values, dtype=float),
                feature_cols,
                lgbm_threshold,
                empty_total_message=(
                    "lightgbm: split importances have a non-positive total."
                ),
            )
            shap_selected, _, _ = _cumulative_select(
                np.asarray(shap_values, dtype=float),
                feature_cols,
                shap_threshold,
                empty_total_message=(
                    "lightgbm: SHAP importances have a non-positive total."
                ),
            )
            lgbm_kept = [
                feature for feature in feature_cols if feature in lgbm_selected
            ]
            shap_kept = [
                feature for feature in feature_cols if feature in shap_selected
            ]
            fold_sets[str(fold_index)] = {"lgbm": lgbm_kept, "shap": shap_kept}
            for feature in lgbm_kept:
                counts[feature] += 1
            for feature in shap_kept:
                counts[feature] += 1
        set_presence = {
            feature: counts[feature] / n_sets for feature in feature_cols
        }
        selected = [
            feature
            for feature in feature_cols
            if set_presence[feature] >= min_set_share
        ]
        return {
            "selected_features": selected,
            "set_presence": set_presence,
            "fold_sets": fold_sets,
            "n_sets": n_sets,
        }


def _cumulative_select(
    values: np.ndarray,
    feature_cols: list[str],
    threshold: float,
    *,
    empty_total_message: str,
) -> tuple[set[str], np.ndarray, np.ndarray]:
    """Нормализует вектор важности и сохраняет начальную часть по накопленной доле.

    Признаки ранжируются по убыванию доли. Сохраняется минимальный префикс,
    накопленная сумма которого достигает ``threshold``, включая признак,
    пересекающий порог. При положительной сумме важностей набор не пуст.
    Равные важности сохраняют исходный порядок кандидатов.
    """
    total = float(np.sum(values))
    if not np.isfinite(total) or total <= 0.0:
        raise ExecutionError(empty_total_message)

    ranked = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": np.asarray(values, dtype=float),
        }
    )
    ranked["norm"] = ranked["importance"] / total
    ordered = ranked.sort_values("norm", ascending=False, kind="stable").copy()
    ordered["cumsum"] = ordered["norm"].cumsum()
    cutoff = min(
        int(ordered["cumsum"].searchsorted(threshold, side="left")) + 1,
        len(ordered),
    )
    selected = set(ordered.iloc[:cutoff]["feature"])
    cumsum_by_feature = ordered.set_index("feature")["cumsum"]
    ranked["cumsum"] = ranked["feature"].map(cumsum_by_feature)
    return (
        selected,
        ranked["norm"].to_numpy(dtype=float),
        ranked["cumsum"].to_numpy(dtype=float),
    )


def build_trial_parameters(
    trial: Any,
    *,
    seed: int,
    n_jobs: int,
    search_space: Mapping[str, Mapping[str, Any]] | None = None,
    fixed_params: Mapping[str, Any] | None = None,
    task: TaskRuntime | None = None,
) -> dict[str, Any]:
    """Формирует параметры LightGBM для одного испытания Optuna."""
    space = DEFAULT_SEARCH_SPACE if search_space is None else search_space
    suggested = {
        name: suggest_parameter(trial, name, specification, method_name="lightgbm")
        for name, specification in space.items()
    }
    return _finalize_parameters(
        {**dict(fixed_params or {}), **suggested},
        seed=seed,
        n_jobs=n_jobs,
        task=task,
    )


def tune_parameters(
    feature_matrix: np.ndarray,
    target: np.ndarray,
    *,
    n_trials: int,
    seed: int,
    n_jobs: int,
    search_space: Mapping[str, Mapping[str, Any]] | None = None,
    fixed_params: Mapping[str, Any] | None = None,
    n_startup_trials: int = 10,
    sampler: str = "TPE",
    timeout: int | None = None,
    task: TaskRuntime | None = None,
) -> dict[str, Any]:
    """Подбирает один набор параметров LightGBM на отложенном разбиении 80/20."""
    try:
        import optuna
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        msg = (
            "lightgbm: LightGBM, Optuna, and scikit-learn must be installed "
            "on the process running tuning."
        )
        raise BackendError(msg) from exc

    resolved = (
        task
        if task is not None
        else resolve_task("binary_classification", target, method_name="lightgbm")
    )
    split_kwargs: dict[str, Any] = {"test_size": 0.2, "random_state": seed}
    if resolved.stratify:
        split_kwargs["stratify"] = target
    train_matrix, valid_matrix, train_target, valid_target = train_test_split(
        feature_matrix,
        target,
        **split_kwargs,
    )
    space = DEFAULT_SEARCH_SPACE if search_space is None else search_space
    study = optuna.create_study(
        direction=optuna_direction(resolved),
        sampler=build_sampler(
            optuna,
            sampler_name=sampler,
            search_space=space,
            seed=seed,
            n_startup_trials=n_startup_trials,
            method_name="lightgbm",
        ),
    )

    def objective(trial: Any) -> float:
        trial_params = build_trial_parameters(
            trial,
            seed=seed,
            n_jobs=n_jobs,
            search_space=space,
            fixed_params=fixed_params,
            task=resolved,
        )
        ctor_params, stopping_rounds = split_lgbm_early_stopping(trial_params)
        model = lgbm_estimator_class(resolved)(**ctor_params)
        fit_lgbm_with_early_stopping(
            model,
            train_matrix,
            train_target,
            eval_set=[(valid_matrix, valid_target)],
            early_stopping_rounds=stopping_rounds,
        )
        return score_model(resolved, model, valid_matrix, valid_target)

    try:
        study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=1)
    except ExecutionError:
        raise
    except Exception as exc:  # noqa: BLE001 - third-party tuning failures
        msg = f"lightgbm: Optuna tuning failed: {exc}"
        raise ExecutionError(msg) from exc

    completed = [trial for trial in study.trials if trial.value is not None]
    if not completed:
        msg = "lightgbm: Optuna finished without a completed trial. Increase n_trials or the timeout."
        raise ExecutionError(msg)

    return _finalize_parameters(
        {**dict(fixed_params or {}), **study.best_params},
        seed=seed,
        n_jobs=n_jobs,
        task=resolved,
    )


def _lightgbm_library_seeds(seed: int) -> dict[str, Any]:
    """Внутренние параметры генераторов LightGBM, привязанные к seed шага."""
    return {
        "random_state": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
        "extra_seed": seed,
        "deterministic": True,
        "force_row_wise": True,
    }


def _finalize_parameters(
    parameters: Mapping[str, Any],
    *,
    seed: int,
    n_jobs: int,
    task: TaskRuntime | None = None,
) -> dict[str, Any]:
    """Добавляет objective/metric для задачи и параметры выполнения."""
    resolved = task if task is not None else binary_task()
    finalized = {
        **dict(parameters),
        **lgbm_objective_params(resolved),
        "verbosity": -1,
        "n_jobs": n_jobs,
        **_lightgbm_library_seeds(seed),
    }
    finalized.pop("force_col_wise", None)
    return finalized
