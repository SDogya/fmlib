"""Statistical selection stage orchestration."""

from __future__ import annotations

from typing import Sequence

from fmlib.feature_selection.base import (
    FeatureDecision,
    Selector,
    StageContext,
    apply_drop_decisions,
)
from fmlib.feature_selection.exceptions import ConfigError, SchemaError
from fmlib.feature_selection.statistics.constants import ConstantsSelector
from fmlib.feature_selection.statistics.correlation import CorrelationSelector
from fmlib.feature_selection.statistics.low_variance import LowVarianceSelector
from fmlib.feature_selection.statistics.null_rate import NullRateSelector
from fmlib.feature_selection.statistics.psi import PsiSelector
from fmlib.feature_selection.statistics.stability_classifier import StabilityClassifierSelector


class StatisticsStage:
    """Run enabled statistical filters sequentially.

    Args:
        selectors: Ordered enabled selectors. When omitted, built from config.
    """

    stage_name = "statistics"

    def __init__(self: StatisticsStage, selectors: Sequence[Selector] | None = None) -> None:
        self._selectors = list(selectors) if selectors is not None else None

    def build_selectors(self: StatisticsStage, context: StageContext) -> list[Selector]:
        """Build enabled selectors from configuration and validate prerequisites.

        Args:
            context: Shared stage context.

        Returns:
            Ordered list of enabled selectors.
        """
        if self._selectors is not None:
            return list(self._selectors)

        stats = context.config.statistics
        selectors: list[Selector] = []

        if stats.null_rate.enabled:
            selectors.append(NullRateSelector(stats.null_rate))
        if stats.constants.enabled:
            selectors.append(ConstantsSelector(stats.constants))
        if stats.low_variance.enabled:
            selectors.append(LowVarianceSelector(stats.low_variance))
        if stats.correlation.enabled:
            selectors.append(CorrelationSelector(stats.correlation))
        if stats.psi.enabled:
            self._validate_psi(context)
            selectors.append(PsiSelector(stats.psi))
        if stats.stability_classifier.enabled:
            self._validate_stability(context)
            selectors.append(StabilityClassifierSelector(stats.stability_classifier))

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
            step_decisions = selector.select(context, remaining)
            decisions.extend(step_decisions)
            remaining = apply_drop_decisions(remaining, step_decisions)
            context.candidates = remaining
        return remaining, decisions

    @staticmethod
    def _validate_psi(context: StageContext) -> None:
        mode = context.config.statistics.psi.mode
        if mode == "month_over_month" and context.schema.time is None:
            msg = "statistics.psi.mode='month_over_month' requires FeatureSchema.time. Set time or disable psi."
            raise ConfigError(
                msg,
            )
        if mode == "train_valid":
            has_valid = "valid" in context.datasets
            has_split = context.schema.split is not None
            if not has_valid and not has_split:
                msg = (
                    "statistics.psi.mode='train_valid' requires a valid split "
                    "(datasets['valid'] or FeatureSchema.split). Disable psi or provide valid data."
                )
                raise ConfigError(
                    msg,
                )

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
                "Disable stability_classifier or provide additional splits."
            )
            raise SchemaError(
                msg,
            )
