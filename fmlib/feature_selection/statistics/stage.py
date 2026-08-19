"""Statistical selection stage orchestration."""

from __future__ import annotations

from typing import Sequence

from fmlib.feature_selection.base import (
    FeatureDecision,
    Selector,
    StageContext,
    apply_drop_decisions,
    persist_step_artifact,
)
from fmlib.feature_selection.exceptions import ConfigError, SchemaError
from fmlib.feature_selection.statistics.constants import ConstantsSelector
from fmlib.feature_selection.statistics.correlation import CorrelationSelector
from fmlib.feature_selection.statistics.iv import IvSelector
from fmlib.feature_selection.statistics.low_variance import LowVarianceSelector
from fmlib.feature_selection.statistics.null_rate import NullRateSelector
from fmlib.feature_selection.statistics.psi import PsiSelector
from fmlib.feature_selection.statistics.stability_classifier import StabilityClassifierSelector
from fmlib.feature_selection.utils.verbose import run_selector_logged


class StatisticsStage:
    """Run statistical filters in ``statistics.order``.

    Args:
        selectors: Ordered selectors. When omitted, built from config.
    """

    stage_name = "statistics"

    def __init__(self: StatisticsStage, selectors: Sequence[Selector] | None = None) -> None:
        self._selectors = list(selectors) if selectors is not None else None

    def build_selectors(self: StatisticsStage, context: StageContext) -> list[Selector]:
        """Build selectors from ``statistics.order`` and validate prerequisites.

        Args:
            context: Shared stage context.

        Returns:
            Ordered list of enabled selectors.
        """
        if self._selectors is not None:
            return list(self._selectors)

        stats = context.config.statistics
        selectors: list[Selector] = []
        for name in stats.order:
            settings = getattr(stats, name)
            if name == "null_rate":
                selectors.append(NullRateSelector(settings))
            elif name == "constants":
                selectors.append(ConstantsSelector(settings))
            elif name == "low_variance":
                selectors.append(LowVarianceSelector(settings))
            elif name == "correlation":
                selectors.append(CorrelationSelector(settings))
            elif name == "psi":
                self._validate_psi(context)
                selectors.append(PsiSelector(settings))
            elif name == "iv":
                self._validate_iv(context)
                selectors.append(IvSelector(settings))
            elif name == "stability_classifier":
                self._validate_stability(context)
                selectors.append(StabilityClassifierSelector(settings))
            else:
                msg = f"Unknown method in statistics.order: {name!r}."
                raise ConfigError(msg)
        return selectors

    def run(
        self: StatisticsStage,
        context: StageContext,
        candidates: Sequence[str],
    ) -> tuple[list[str], list[FeatureDecision]]:
        """Execute enabled statistical filters.

        Args:
            context: Shared stage context.
            candidates: Current candidate features.

        Returns:
            Remaining candidates and accumulated drop decisions.
        """
        remaining = list(candidates)
        decisions: list[FeatureDecision] = []
        for selector in self.build_selectors(context):
            step_decisions = run_selector_logged(selector, context, remaining)
            decisions.extend(step_decisions)
            context.decisions.extend(step_decisions)
            remaining = apply_drop_decisions(remaining, step_decisions)
            context.candidates = remaining
            persist_step_artifact(
                context,
                remaining,
                stage_name=self.stage_name,
                method_name=selector.method_name,
            )
            context.step_index += 1
        return remaining, decisions

    @staticmethod
    def _validate_psi(context: StageContext) -> None:
        settings = context.config.statistics.psi
        mode = settings.mode
        if mode == "month_over_month" and context.schema.time is None:
            msg = (
                "statistics.psi.mode='month_over_month' requires FeatureSchema.time. "
                "Set time or remove psi from statistics.order."
            )
            raise ConfigError(
                msg,
            )
        if mode == "train_valid":
            has_valid = "valid" in context.datasets
            has_split = context.schema.split is not None
            if not has_valid and not has_split:
                msg = (
                    "statistics.psi.mode='train_valid' requires a valid split "
                    "(datasets['valid'] or FeatureSchema.split). "
                    "Remove psi from statistics.order or provide valid data."
                )
                raise ConfigError(
                    msg,
                )
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

    @staticmethod
    def _validate_iv(context: StageContext) -> None:
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

    @staticmethod
    def _validate_stability(context: StageContext) -> None:
        n_sources = len(context.datasets)
        if context.schema.split is not None and n_sources < 2:
            # single DF with split can still encode multiple sources; skeleton allows it
            return
        if n_sources < 2 and context.schema.split is None:
            msg = (
                "stability_classifier requires at least two data sources "
                "(train/valid[/test] mapping or a split column). "
                "Remove stability_classifier from statistics.order or provide "
                "additional splits."
            )
            raise SchemaError(
                msg,
            )
