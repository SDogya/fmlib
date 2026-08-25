"""Single-loop pipeline runner driven by ``FeatureSelectionConfig.order``."""

from __future__ import annotations

from typing import Any, Sequence

from fmlib.feature_selection.base import (
    FeatureDecision,
    StageContext,
    apply_drop_decisions,
    bind_process_rng,
    persist_step_artifact,
    resolve_step_seed,
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
from fmlib.feature_selection.exceptions import ConfigError, SchemaError
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
from fmlib.feature_selection.utils.steps import (
    FeatureDropStep,
    RandomFeatureDropStep,
    RowSampleStep,
)
from fmlib.feature_selection.utils.verbose import run_selector_logged

STUB_METHODS = frozenset({"lasso", "random_forest"})

_PREPROCESSING_METHODS = frozenset(
    {"feature_drop", "random_feature_drop", "row_sample"},
)
_MODEL_METHODS = frozenset(
    {"lasso", "random_forest", "catboost_rfe", "lightgbm"},
)


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
    for step in context.config.order:
        before = len(context.decisions)
        remaining = _run_step(context, step, remaining)
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


def _run_step(
    context: StageContext,
    step: PipelineStepConfig,
    candidates: Sequence[str],
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
