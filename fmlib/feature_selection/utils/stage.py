"""Preprocessing stage orchestration."""

from __future__ import annotations

from typing import Any, Sequence

from fmlib.feature_selection.base import (
    FeatureDecision,
    StageContext,
    persist_step_artifact,
)
from fmlib.feature_selection.utils.steps import (
    FeatureDropStep,
    RandomFeatureDropStep,
    RowSampleStep,
)


class UtilsStage:
    """Run enabled preprocessing helpers before statistical filters.

    Args:
        steps: Ordered preprocessing steps. When omitted, built from config.
    """

    stage_name = "preprocessing"

    def __init__(self: UtilsStage, steps: Sequence[Any] | None = None) -> None:
        self._steps = list(steps) if steps is not None else None

    def build_steps(self: UtilsStage, context: StageContext) -> list[Any]:
        """Build enabled utils in fixed order.

        Args:
            context: Shared stage context.

        Returns:
            Ordered list of enabled preprocessing steps.
        """
        if self._steps is not None:
            return list(self._steps)

        preprocessing = context.config.preprocessing
        steps: list[Any] = []
        if preprocessing.feature_drop.enabled:
            steps.append(FeatureDropStep(preprocessing.feature_drop))
        if preprocessing.random_feature_drop.enabled:
            steps.append(RandomFeatureDropStep(preprocessing.random_feature_drop))
        if preprocessing.row_sample.enabled:
            steps.append(RowSampleStep(preprocessing.row_sample))
        return steps

    def run(
        self: UtilsStage,
        context: StageContext,
        candidates: Sequence[str],
    ) -> tuple[list[str], list[FeatureDecision]]:
        """Execute enabled preprocessing steps.

        Args:
            context: Shared stage context.
            candidates: Current candidate features.

        Returns:
            Remaining candidates and accumulated drop decisions.
        """
        remaining = list(candidates)
        decisions: list[FeatureDecision] = []
        for step in self.build_steps(context):
            before = len(context.decisions)
            remaining = step.run(context, remaining)
            decisions.extend(context.decisions[before:])
            persist_step_artifact(
                context,
                remaining,
                stage_name=self.stage_name,
                method_name=step.method_name,
            )
            context.step_index += 1
        context.candidates = remaining
        return remaining, decisions
