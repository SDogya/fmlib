"""Исполнитель пайплайна с одним циклом, управляемый ``FeatureSelectionConfig.order``."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from fmlib.feature_selection.base import (
    FeatureDecision,
    StageContext,
    apply_drop_decisions,
    bind_process_rng,
    persist_step_artifact,
    resolve_step_seed,
    step_seed,
)
from fmlib.feature_selection.config import (
    ConstantsConfig,
    CorrelationConfig,
    CrossValidationConfig,
    FeatureDropConfig,
    IvConfig,
    LowVarianceConfig,
    METHOD_STAGE,
    ModelConfig,
    ModelSelectionRuleConfig,
    NullRateConfig,
    PipelineStepConfig,
    PsiConfig,
    RandomFeatureDropConfig,
    RowSampleConfig,
    _build_section,
    split_model_step_params,
)
from fmlib.feature_selection.exceptions import BackendError, ConfigError
from fmlib.feature_selection.model_based.catboost_rfe import CatBoostRfeSelector
from fmlib.feature_selection.model_based.lightgbm import LightGbmSelector
from fmlib.feature_selection.model_based.boruta_shap import BorutaShapSelector
from fmlib.feature_selection.statistical_filters.constants import (
    ConstantsSelector,
)
from fmlib.feature_selection.statistical_filters.correlation import (
    CorrelationSelector,
)
from fmlib.feature_selection.statistical_filters.iv import IvSelector
from fmlib.feature_selection.statistical_filters.low_variance import (
    LowVarianceSelector,
)
from fmlib.feature_selection.statistical_filters.null_rate import NullRateSelector
from fmlib.feature_selection.statistical_filters.psi import PsiSelector
from fmlib.feature_selection.utils.model_param_validate import validate_model_parameters
from fmlib.feature_selection.utils.statistics_cache import (
    CACHEABLE_METHODS,
    StatisticsMetricsCache,
    compute_data_fingerprint,
    compute_fingerprint,
    resolve_cache_path,
)
from fmlib.feature_selection.utils.steps import (
    FeatureDropStep,
    RandomFeatureDropStep,
    RowSampleStep,
)
from fmlib.feature_selection.utils.verbose import run_selector_logged

_PREPROCESSING_METHODS = frozenset(
    {"feature_drop", "random_feature_drop", "row_sample"},
)
_MODEL_METHODS = frozenset({"boruta_shap", "catboost_rfe", "lightgbm"})


def run_order(
    context: StageContext,
    candidates: Sequence[str],
) -> tuple[list[str], list[FeatureDecision]]:
    """Выполняет каждый шаг из ``context.config.order``.

    Args:
        context: Общий контекст пайплайна.
        candidates: Текущие имена признаков-кандидатов.

    Returns:
        Оставшиеся кандидаты и решения об исключении за этот запуск.
    """
    remaining = list(candidates)
    decisions: list[FeatureDecision] = []
    cache = _open_stats_cache(context)
    for step in context.config.order:
        before = len(context.decisions)
        remaining = _run_step(context, step, remaining, cache=cache)
        step_decisions = context.decisions[before:]
        decisions.extend(step_decisions)
        _relocate_step_scores(context, step.method, context.step_index)
        persist_step_artifact(
            context,
            remaining,
            stage_name=METHOD_STAGE[step.method],
            method_name=step.method,
        )
        context.step_index += 1
    return remaining, decisions


def validate_order_prerequisites(context: StageContext) -> None:
    """Вызывает ошибку до выполнения шагов, если схема, дополнительные зависимости или параметры не позволяют запуск.

    Проверяет весь ``order``, чтобы отсутствие целевой переменной или зависимости SHAP обнаружилось при
    запуске ``fit_select``, а не после нескольких часов вычисления статистик.
    """
    _require_matching_task_type(context)
    for index, step in enumerate(context.config.order):
        section = f"order[{index}].{step.method}"
        if step.method == "psi":
            settings = _build_section(PsiConfig, step.params, section)
            _validate_psi(context, settings)
        elif step.method == "iv":
            _validate_iv(context)
        elif step.method in _MODEL_METHODS:
            if context.schema.target is None:
                msg = "model stage requires FeatureSchema.target."
                raise ConfigError(msg)
            if step.method == "catboost_rfe" and not context.schema.time:
                msg = (
                    "catboost_rfe: FeatureSchema.time is required for the "
                    "out-of-time split. Set time to the period column."
                )
                raise ConfigError(msg)
            model_params, _, _ = split_model_step_params(step.params)
            _require_method_extras(step.method, model_params)
            _revalidate_model_parameters(step.method, model_params)


def _run_step(
    context: StageContext,
    step: PipelineStepConfig,
    candidates: Sequence[str],
    *,
    cache: StatisticsMetricsCache | None = None,
) -> list[str]:
    """Создаёт и выполняет один шаг из order."""
    context.run_seed = resolve_step_seed(step.params, context)
    bind_process_rng(context.run_seed)
    if step.method in _PREPROCESSING_METHODS:
        worker = _build_preprocessing_step(step)
        remaining = worker.run(context, candidates)
        context.candidates = remaining
        return remaining

    if step.method == "psi":
        settings = _build_section(PsiConfig, step.params, f"order.{step.method}")
        _validate_psi(context, settings)
        selector = PsiSelector(settings)
    elif step.method == "iv":
        _validate_iv(context)
        selector = IvSelector(_build_section(IvConfig, step.params, f"order.{step.method}"))
    elif step.method in _MODEL_METHODS:
        if context.schema.target is None:
            msg = "model stage requires FeatureSchema.target."
            raise ConfigError(msg)
        selector = _build_model_selector(step)
    elif step.method == "null_rate":
        selector = NullRateSelector(
            _build_section(NullRateConfig, step.params, f"order.{step.method}"),
        )
    elif step.method == "constants":
        selector = ConstantsSelector(
            _build_section(ConstantsConfig, step.params, f"order.{step.method}"),
        )
    elif step.method == "low_variance":
        selector = LowVarianceSelector(
            _build_section(LowVarianceConfig, step.params, f"order.{step.method}"),
        )
    elif step.method == "correlation":
        selector = CorrelationSelector(
            _build_section(CorrelationConfig, step.params, f"order.{step.method}"),
        )
    else:
        msg = f"Unknown method in order: {step.method!r}."
        raise ConfigError(msg)

    if (
        cache is not None
        and step.method in CACHEABLE_METHODS
        and hasattr(selector, "compute")
        and hasattr(selector, "apply")
    ):
        cached_selector = _CachedStatisticsSelector(selector, cache)
        step_decisions = run_selector_logged(cached_selector, context, candidates)
    else:
        step_decisions = run_selector_logged(selector, context, candidates)
    context.decisions.extend(step_decisions)
    remaining = apply_drop_decisions(candidates, step_decisions)
    context.candidates = remaining
    return remaining


def _build_preprocessing_step(step: PipelineStepConfig) -> Any:
    """Создаёт вспомогательный объект предобработки из шага order."""
    if step.method == "feature_drop":
        return FeatureDropStep(
            _build_section(FeatureDropConfig, step.params, "order.feature_drop"),
        )
    if step.method == "random_feature_drop":
        return RandomFeatureDropStep(
            _build_section(
                RandomFeatureDropConfig,
                step.params,
                "order.random_feature_drop",
            ),
        )
    return RowSampleStep(
        _build_section(RowSampleConfig, step.params, "order.row_sample"),
    )


def _build_model_selector(step: PipelineStepConfig) -> Any:
    """Создаёт метод отбора на основе модели из шага order."""
    model_params, selection_raw, cv_raw = split_model_step_params(step.params)
    config = ModelConfig(
        enabled=True,
        method=step.method,
        params=model_params,
        selection=_build_section(
            ModelSelectionRuleConfig,
            selection_raw,
            f"order.{step.method}.selection",
        ),
        cross_validation=_build_section(
            CrossValidationConfig,
            cv_raw,
            f"order.{step.method}.cross_validation",
        ),
    )
    if step.method == "lightgbm":
        return LightGbmSelector(config)
    if step.method == "catboost_rfe":
        return CatBoostRfeSelector(config)
    if step.method == "boruta_shap":
        return BorutaShapSelector(config)
    msg = f"Unsupported model method {step.method!r}."
    raise ConfigError(msg)


def _relocate_step_scores(
    context: StageContext,
    method_name: str,
    step_index: int,
) -> None:
    """Предотвращает перезапись оценок при повторном использовании методов."""
    if method_name not in context.scores:
        return
    context.scores[f"{method_name}#{step_index}"] = context.scores.pop(method_name)


def _validate_psi(context: StageContext, settings: PsiConfig) -> None:
    """Проверяет наличие схемы и данных, необходимых для режима PSI на этом шаге."""
    mode = settings.mode
    if mode == "month_over_month" and context.schema.time is None:
        msg = (
            "psi mode='month_over_month' requires FeatureSchema.time. "
            "Set time or remove psi from order."
        )
        raise ConfigError(msg)
    if mode == "train_valid":
        has_valid = "valid" in context.datasets
        has_split = context.schema.split is not None
        if not has_valid and not has_split:
            msg = (
                "psi mode='train_valid' requires a valid split "
                "(datasets['valid'] or FeatureSchema.split). "
                "Remove psi from order or provide valid data."
            )
            raise ConfigError(msg)
        month_col = settings.month_column
        is_in_schema = (
            month_col in context.schema.categorical
            or month_col in context.schema.continuous
        )
        is_split_col = month_col == context.schema.split
        if not is_in_schema and not is_split_col:
            msg = (
                f"psi month_column={month_col!r} is not in "
                "FeatureSchema.categorical or FeatureSchema.continuous, and is "
                "not equal to FeatureSchema.split."
            )
            raise ConfigError(msg)


def _validate_iv(context: StageContext) -> None:
    """Проверяет наличие бинарной целевой переменной для информационной ценности (IV)."""
    if not context.schema.target:
        msg = "iv requires FeatureSchema.target. Remove iv from order or set target."
        raise ConfigError(msg)
    if context.schema.task_type != "binary_classification":
        msg = (
            "iv requires FeatureSchema.task_type='binary_classification'. "
            f"Got {context.schema.task_type!r}."
        )
        raise ConfigError(msg)


def _open_stats_cache(context: StageContext) -> StatisticsMetricsCache | None:
    """Открывает кэш статистических метрик при включённом ``statistics.cache.enabled``."""
    settings = context.config.statistics.cache
    if not settings.enabled:
        return None
    if not isinstance(settings.dataset_version, str) or not settings.dataset_version.strip():
        msg = "statistics.cache requires a non-empty dataset_version."
        raise ConfigError(msg)
    path = resolve_cache_path(settings.path, output_dir=context.output_dir)
    return StatisticsMetricsCache.load(
        path,
        force_recompute=settings.force_recompute,
    )


def _run_cached_statistics(
    selector: Any,
    context: StageContext,
    remaining: Sequence[str],
    cache: StatisticsMetricsCache,
) -> list[FeatureDecision]:
    """Находит или вычисляет метрики для текущих кандидатов, затем применяет пороги."""
    columns = list(remaining)
    if not columns:
        return []
    method = selector.method_name
    method_fingerprint = compute_fingerprint(
        method,
        selector.config,
        max_local_rows=context.config.execution.max_local_rows,
        seed=step_seed(context),
        task_type=context.schema.task_type,
    )
    fingerprint = {
        "data": compute_data_fingerprint(context),
        "parameters": method_fingerprint,
        "candidates": columns,
    }
    force = bool(context.config.statistics.cache.force_recompute)
    metrics = None if force else cache.lookup(method, fingerprint)
    if metrics is None:
        metrics = selector.compute(context, columns)
        cache.upsert(method, fingerprint, metrics)
    return selector.apply(metrics, columns, context)


class _CachedStatisticsSelector:
    """Адаптер, сохраняющий подробное журналирование при вычислении и применении метрик с кэшем."""

    def __init__(
        self: _CachedStatisticsSelector,
        selector: Any,
        cache: StatisticsMetricsCache,
    ) -> None:
        self._selector = selector
        self._cache = cache
        self.method_name = selector.method_name

    def select(
        self: _CachedStatisticsSelector,
        context: StageContext,
        candidates: Sequence[str],
    ) -> list[FeatureDecision]:
        return _run_cached_statistics(
            self._selector,
            context,
            candidates,
            self._cache,
        )


def _require_matching_task_type(context: StageContext) -> None:
    """Проверяет совпадение YAML-параметра ``execution.task_type`` с ``FeatureSchema.task_type``."""
    yaml_task = context.config.execution.task_type
    schema_task = context.schema.task_type
    if yaml_task == schema_task:
        return
    msg = (
        "execution.task_type must match FeatureSchema.task_type "
        f"({yaml_task!r} vs {schema_task!r})."
    )
    raise ConfigError(msg)


def _revalidate_model_parameters(method: str, params: Mapping[str, Any]) -> None:
    """Повторно проверяет псевдонимы и неизвестные имена для конфигураций, созданных без ``from_dict``."""
    parameters = params.get("parameters", {})
    if method == "lightgbm":
        validate_model_parameters(
            parameters if isinstance(parameters, Mapping) else {},
            library="lightgbm",
            method_name=method,
        )
        return
    if method == "catboost_rfe":
        validate_model_parameters(
            parameters if isinstance(parameters, Mapping) else {},
            library="catboost",
            method_name=method,
            require_non_empty=True,
        )
        return
    if method == "boruta_shap":
        model_type = str(params.get("model_type", "lgbm")).lower()
        library = "random_forest" if model_type == "rf" else "lightgbm"
        validate_model_parameters(
            parameters if isinstance(parameters, Mapping) else {},
            library=library,
            method_name=method,
            ignore_keys=(
                frozenset({"bootstrap_type"})
                if library == "lightgbm"
                else frozenset()
            ),
        )


def _require_method_extras(method: str, params: Mapping[str, Any]) -> None:
    """Импортирует дополнительные зависимости шага до запуска фолдов или испытаний."""
    optuna_needed = _optuna_enabled(params)
    if method == "lightgbm":
        _import_or_fail("lightgbm", "lightgbm: LightGBM is required. Install the lightgbm optional dependency.")
        _import_or_fail("shap", "lightgbm: SHAP is required. Install the shap optional dependency.")
        if optuna_needed:
            _import_or_fail("optuna", "lightgbm: Optuna is required. Install the optuna optional dependency.")
        return
    if method == "catboost_rfe":
        try:
            from catboost import CatBoostClassifier  # noqa: F401
        except ImportError as exc:
            msg = "catboost_rfe: CatBoost is required. Install the catboost optional dependency."
            raise BackendError(msg) from exc
        if optuna_needed:
            _import_or_fail("optuna", "catboost_rfe: Optuna is required. Install the optuna optional dependency.")
        return
    if method == "boruta_shap":
        try:
            from BorutaShap import BorutaShap  # noqa: F401
        except ImportError as exc:
            msg = "boruta_shap: BorutaShap is required. Install the BorutaShap optional dependency."
            raise BackendError(msg) from exc
        model_type = str(params.get("model_type", "lgbm")).lower()
        if model_type == "rf":
            try:
                from sklearn.ensemble import RandomForestClassifier  # noqa: F401
            except ImportError as exc:
                msg = (
                    "boruta_shap: scikit-learn RandomForest is required for "
                    "model_type='rf'."
                )
                raise BackendError(msg) from exc
        else:
            _import_or_fail(
                "lightgbm",
                "boruta_shap: LightGBM is required for model_type='lgbm'.",
            )
        if optuna_needed:
            _import_or_fail("optuna", "boruta_shap: Optuna is required. Install the optuna optional dependency.")


def _optuna_enabled(params: Mapping[str, Any]) -> bool:
    """Будет ли шаг запускать Optuna (по умолчанию true, как в методах отбора)."""
    optuna_params = params.get("optuna_params", {})
    if not isinstance(optuna_params, Mapping):
        return True
    return optuna_params.get("enabled", True) is not False


def _import_or_fail(module: str, message: str) -> None:
    """Импортирует ``module`` или вызывает ``BackendError`` с ``message``."""
    try:
        __import__(module)
    except ImportError as exc:
        msg = message
        raise BackendError(msg) from exc
