"""Single-loop pipeline runner driven by ``FeatureSelectionConfig.order``."""

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
    PreciseConfig,
    PsiConfig,
    RandomFeatureDropConfig,
    RowSampleConfig,
    StabilityClassifierConfig,
    _build_section,
    split_model_step_params,
)
from fmlib.feature_selection.exceptions import BackendError, ConfigError, SchemaError
from fmlib.feature_selection.model_based.catboost_rfe import CatBoostRfeSelector
from fmlib.feature_selection.model_based.lasso import LassoSelector
from fmlib.feature_selection.model_based.lightgbm import LightGbmSelector
from fmlib.feature_selection.model_based.random_forest import RandomForestSelector
from fmlib.feature_selection.precise.boruta_shap import BorutaShapSelector
from fmlib.feature_selection.statistics.constants import ConstantsSelector
from fmlib.feature_selection.statistics.correlation import CorrelationSelector
from fmlib.feature_selection.statistics.iv import IvSelector
from fmlib.feature_selection.statistics.low_variance import LowVarianceSelector
from fmlib.feature_selection.statistics.null_rate import NullRateSelector
from fmlib.feature_selection.statistics.psi import PsiSelector
from fmlib.feature_selection.statistics.stability_classifier import StabilityClassifierSelector
from fmlib.feature_selection.utils.model_param_validate import validate_model_parameters
from fmlib.feature_selection.utils.statistics_cache import (
    CACHEABLE_METHODS,
    StatisticsMetricsCache,
    compute_fingerprint,
    resolve_cache_path,
)
from fmlib.feature_selection.utils.steps import (
    FeatureDropStep,
    RandomFeatureDropStep,
    RowSampleStep,
)
from fmlib.feature_selection.utils.verbose import run_selector_logged

STUB_METHODS = frozenset({"lasso", "random_forest", "stability_classifier"})

_PREPROCESSING_METHODS = frozenset(
    {"feature_drop", "random_feature_drop", "row_sample"},
)
_MODEL_METHODS = frozenset(
    {"lasso", "random_forest", "catboost_rfe", "lightgbm"},
)
_BINARY_MODEL_METHODS = frozenset({"lightgbm", "catboost_rfe", "boruta_shap"})


def run_order(
    context: StageContext,
    candidates: Sequence[str],
) -> tuple[list[str], list[FeatureDecision]]:
    """Execute every step in ``context.config.order``.

    Args:
        context: Shared pipeline context.
        candidates: Current candidate feature names.

    Returns:
        Remaining candidates and this run's drop decisions.
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
    """Fail before any step if schema, extras or parameters cannot run.

    Walks the whole ``order`` so a missing target or SHAP extra surfaces at
    ``fit_select`` start instead of after hours of statistics.
    """
    for index, step in enumerate(context.config.order):
        section = f"order[{index}].{step.method}"
        if step.method == "psi":
            settings = _build_section(PsiConfig, step.params, section)
            _validate_psi(context, settings)
        elif step.method == "iv":
            _validate_iv(context)
        elif step.method == "stability_classifier":
            _validate_stability(context)
        elif step.method in _MODEL_METHODS:
            if context.schema.target is None:
                msg = "model stage requires FeatureSchema.target."
                raise ConfigError(msg)
            if step.method in _BINARY_MODEL_METHODS:
                _require_binary_task(step.method, context)
            if step.method == "catboost_rfe" and not context.schema.time:
                msg = (
                    "catboost_rfe: FeatureSchema.time is required for the "
                    "out-of-time split. Set time to the period column."
                )
                raise ConfigError(msg)
            model_params, _, _ = split_model_step_params(step.params)
            _require_method_extras(step.method, model_params)
            _revalidate_model_parameters(step.method, model_params)
        elif step.method == "boruta_shap":
            if not context.schema.target:
                msg = "boruta_shap requires FeatureSchema.target."
                raise ConfigError(msg)
            _require_binary_task(step.method, context)
            _require_method_extras(step.method, step.params)
            _revalidate_model_parameters(step.method, step.params)


def _run_step(
    context: StageContext,
    step: PipelineStepConfig,
    candidates: Sequence[str],
    *,
    cache: StatisticsMetricsCache | None = None,
) -> list[str]:
    """Build and execute one order step."""
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
    elif step.method == "stability_classifier":
        _validate_stability(context)
        selector = StabilityClassifierSelector(
            _build_section(
                StabilityClassifierConfig,
                step.params,
                f"order.{step.method}",
            ),
        )
    elif step.method in _MODEL_METHODS:
        if context.schema.target is None:
            msg = "model stage requires FeatureSchema.target."
            raise ConfigError(msg)
        selector = _build_model_selector(step)
    elif step.method == "boruta_shap":
        if not context.schema.target:
            msg = "boruta_shap requires FeatureSchema.target."
            raise ConfigError(msg)
        selector = BorutaShapSelector(
            PreciseConfig(enabled=True, method="boruta_shap", params=dict(step.params)),
        )
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
    """Instantiate a preprocessing helper from an order step."""
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
    """Instantiate a model selector from an order step."""
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
    if step.method == "lasso":
        return LassoSelector(config)
    if step.method == "random_forest":
        return RandomForestSelector(config)
    msg = f"Unsupported model method {step.method!r}."
    raise ConfigError(msg)


def _relocate_step_scores(
    context: StageContext,
    method_name: str,
    step_index: int,
) -> None:
    """Keep repeated methods from overwriting each other's scores."""
    if method_name not in context.scores:
        return
    context.scores[f"{method_name}#{step_index}"] = context.scores.pop(method_name)


def _validate_psi(context: StageContext, settings: PsiConfig) -> None:
    """Require schema/data needed by the PSI mode of this step."""
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
    """Require a binary target for Information Value."""
    if not context.schema.target:
        msg = "iv requires FeatureSchema.target. Remove iv from order or set target."
        raise ConfigError(msg)
    if context.schema.task_type != "binary_classification":
        msg = (
            "iv requires FeatureSchema.task_type='binary_classification'. "
            f"Got {context.schema.task_type!r}."
        )
        raise ConfigError(msg)


def _validate_stability(context: StageContext) -> None:
    """Require at least two data sources for the stability classifier."""
    n_sources = len(context.datasets)
    if context.schema.split is not None and n_sources < 2:
        return
    if n_sources < 2 and context.schema.split is None:
        msg = (
            "stability_classifier requires at least two data sources "
            "(train/valid[/test] mapping or a split column). "
            "Remove stability_classifier from order or provide additional splits."
        )
        raise SchemaError(msg)


def _open_stats_cache(context: StageContext) -> StatisticsMetricsCache | None:
    """Open the statistics metrics cache when ``statistics.cache.enabled``."""
    settings = context.config.statistics.cache
    if not settings.enabled:
        return None
    path = resolve_cache_path(settings.path)
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
    """Lookup or compute metrics on the full candidate set, then apply thresholds."""
    method = selector.method_name
    fingerprint = compute_fingerprint(
        method,
        selector.config,
        max_local_rows=context.config.execution.max_local_rows,
        seed=step_seed(context),
        task_type=context.schema.task_type,
    )
    force = bool(context.config.statistics.cache.force_recompute)
    metrics = None if force else cache.lookup(method, fingerprint)
    full_features = list(context.schema.candidate_features())
    if metrics is None:
        metrics = selector.compute(context, full_features)
        cache.upsert(method, fingerprint, metrics)
    return selector.apply(metrics, remaining, context)


class _CachedStatisticsSelector:
    """Adapter so cached compute/apply still goes through verbose logging."""

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


def _require_binary_task(method: str, context: StageContext) -> None:
    """Require binary classification for model / precise selectors that need it."""
    if context.schema.task_type == "binary_classification":
        return
    msg = (
        f"{method}: only task_type='binary_classification' is supported; "
        f"got {context.schema.task_type!r}."
    )
    raise ConfigError(msg)


def _revalidate_model_parameters(method: str, params: Mapping[str, Any]) -> None:
    """Re-check aliases and unknown names for configs built without ``from_dict``."""
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
    """Import optional extras for a step before any fold or trial runs."""
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
    """Whether this step will run Optuna (default true, matching selectors)."""
    optuna_params = params.get("optuna_params", {})
    if not isinstance(optuna_params, Mapping):
        return True
    return optuna_params.get("enabled", True) is not False


def _import_or_fail(module: str, message: str) -> None:
    """Import ``module`` or raise ``BackendError`` with ``message``."""
    try:
        __import__(module)
    except ImportError as exc:
        msg = message
        raise BackendError(msg) from exc
