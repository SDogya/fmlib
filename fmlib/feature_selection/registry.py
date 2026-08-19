"""Method registry for the feature-selection pipeline.

Utils run first when enabled, then ``statistics.order``, then one model
selector, then optional precise. Stage tags exist only for decision metadata
and intermediate artifact filenames.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol, Sequence

from fmlib.feature_selection.backends.spark import SPARK_CAPABILITIES
from fmlib.feature_selection.base import Selector, StageContext, apply_drop_decisions
from fmlib.feature_selection.config import FeatureSelectionConfig, PsiConfig
from fmlib.feature_selection.debug import run_selector_logged
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
from fmlib.feature_selection.statistics.stability_classifier import (
    StabilityClassifierSelector,
)
from fmlib.feature_selection.utils.steps import build_utils_steps

METHOD_STAGE: dict[str, str] = {
    "feature_drop": "preprocessing",
    "random_feature_drop": "preprocessing",
    "row_sample": "preprocessing",
    "null_rate": "statistics",
    "constants": "statistics",
    "low_variance": "statistics",
    "correlation": "statistics",
    "psi": "statistics",
    "iv": "statistics",
    "stability_classifier": "statistics",
    "lightgbm": "model",
    "lasso": "model",
    "random_forest": "model",
    "catboost_rfe": "model",
    "boruta_shap": "precise",
}

STUB_METHODS = frozenset({"lasso", "random_forest"})


class PipelineStep(Protocol):
    """One named step in the pipeline."""

    method_name: str

    def run(
        self: PipelineStep,
        context: StageContext,
        candidates: Sequence[str],
    ) -> list[str]:
        """Execute the step and return remaining candidates."""
        ...


def stage_backends_for_config(config: FeatureSelectionConfig) -> dict[str, str]:
    """Spark backend tag for every stage that will run."""
    backends: dict[str, str] = {}
    for step in build_pipeline_steps(config):
        backends[METHOD_STAGE[step.method_name]] = SPARK_CAPABILITIES.name
    return backends


def build_pipeline_steps(config: FeatureSelectionConfig) -> list[PipelineStep]:
    """Utils (if enabled), then statistics.order, then model, then precise."""
    steps: list[PipelineStep] = list(build_utils_steps(config))
    for name in config.statistics.order:
        steps.append(build_statistics_step(name, config))
    if config.model.enabled:
        steps.append(build_model_step(config))
    if config.precise.enabled:
        steps.append(build_precise_step(config))
    return steps


def build_statistics_step(name: str, config: FeatureSelectionConfig) -> PipelineStep:
    """Instantiate one statistics selector from ``statistics.order``."""
    settings = getattr(config.statistics, name)
    if name == "null_rate":
        return SelectorStep(NullRateSelector(settings))
    if name == "constants":
        return SelectorStep(ConstantsSelector(settings))
    if name == "low_variance":
        return SelectorStep(LowVarianceSelector(settings))
    if name == "correlation":
        return SelectorStep(CorrelationSelector(settings))
    if name == "psi":
        return SelectorStep(
            PsiSelector(settings),
            validate=lambda context: validate_psi(context, settings),
        )
    if name == "iv":
        return SelectorStep(
            IvSelector(settings),
            needs_target=True,
            validate=validate_iv,
        )
    if name == "stability_classifier":
        return SelectorStep(
            StabilityClassifierSelector(settings),
            validate=validate_stability,
        )
    msg = f"Unknown method in statistics.order: {name!r}."
    raise ConfigError(msg)


def build_model_step(config: FeatureSelectionConfig) -> PipelineStep:
    """Instantiate the single enabled model selector."""
    method = config.model.method
    if method == "lightgbm":
        return SelectorStep(LightGbmSelector(config.model), needs_target=True)
    if method == "lasso":
        return SelectorStep(LassoSelector(config.model), needs_target=True)
    if method == "random_forest":
        return SelectorStep(RandomForestSelector(config.model), needs_target=True)
    if method == "catboost_rfe":
        return SelectorStep(CatBoostRfeSelector(config.model), needs_target=True)
    msg = f"Unknown model.method: {method!r}."
    raise ConfigError(msg)


def build_precise_step(config: FeatureSelectionConfig) -> PipelineStep:
    """Instantiate the precise selector (BorutaSHAP)."""
    return SelectorStep(BorutaShapSelector(config.precise), needs_target=True)


class SelectorStep:
    """Adapter that runs an existing ``Selector`` as a pipeline step."""

    def __init__(
        self: SelectorStep,
        selector: Selector,
        *,
        needs_target: bool = False,
        validate: Optional[Callable[[StageContext], None]] = None,
    ) -> None:
        self.selector = selector
        self.method_name = selector.method_name
        self.needs_target = needs_target
        self._validate = validate

    def run(
        self: SelectorStep,
        context: StageContext,
        candidates: Sequence[str],
    ) -> list[str]:
        if self._validate is not None:
            self._validate(context)
        if self.needs_target and not context.schema.target:
            msg = f"{self.method_name} requires FeatureSchema.target."
            raise ConfigError(msg)
        decisions = run_selector_logged(self.selector, context, candidates)
        context.decisions.extend(decisions)
        remaining = apply_drop_decisions(candidates, decisions)
        context.candidates = remaining
        return remaining


def validate_iv(context: StageContext) -> None:
    """Require a binary target for Information Value."""
    if not context.schema.target:
        msg = (
            "iv requires FeatureSchema.target. "
            "Remove iv from statistics.order or set target."
        )
        raise ConfigError(msg)
    if context.schema.task_type != "binary_classification":
        msg = (
            "iv requires FeatureSchema.task_type='binary_classification'. "
            f"Got {context.schema.task_type!r}."
        )
        raise ConfigError(msg)


def validate_psi(
    context: StageContext,
    psi: PsiConfig | None = None,
) -> None:
    """Require time/split columns that the configured PSI mode needs."""
    settings = psi if psi is not None else context.config.statistics.psi
    mode = settings.mode
    if mode == "month_over_month" and context.schema.time is None:
        msg = (
            "statistics.psi.mode='month_over_month' requires FeatureSchema.time. "
            "Set time or remove psi from statistics.order."
        )
        raise ConfigError(msg)
    if mode == "train_valid":
        has_valid = "valid" in context.datasets
        has_split = context.schema.split is not None
        if not has_valid and not has_split:
            msg = (
                "statistics.psi.mode='train_valid' requires a valid split "
                "(datasets['valid'] or FeatureSchema.split). "
                "Remove psi from statistics.order or provide valid data."
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
                f"statistics.psi.month_column='{month_col}' is not in "
                "FeatureSchema.categorical or FeatureSchema.continuous, and is "
                "not equal to FeatureSchema.split. Add the column to the schema, "
                f"change month_column, or set split='{month_col}'."
            )
            raise ConfigError(msg)


def validate_stability(context: StageContext) -> None:
    """Require at least two sources for the stability-classifier stub."""
    n_sources = len(context.datasets)
    if context.schema.split is not None and n_sources < 2:
        return
    if n_sources < 2 and context.schema.split is None:
        msg = (
            "stability_classifier requires at least two data sources "
            "(train/valid[/test] mapping or a split column). "
            "Remove stability_classifier from statistics.order or provide "
            "additional splits."
        )
        raise SchemaError(msg)
