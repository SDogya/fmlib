"""Model-based selection stage orchestration."""

from __future__ import annotations

from typing import Sequence

from fmlib.feature_selection.base import (
    FeatureDecision,
    Selector,
    StageContext,
    apply_drop_decisions,
    persist_step_artifact,
)
from fmlib.feature_selection.exceptions import ConfigError
from fmlib.feature_selection.model_based.catboost_rfe import CatBoostRfeSelector
from fmlib.feature_selection.model_based.lasso import LassoSelector
from fmlib.feature_selection.model_based.lightgbm import LightGbmSelector
from fmlib.feature_selection.model_based.random_forest import RandomForestSelector
from fmlib.feature_selection.utils.verbose import run_selector_logged

_MODEL_REGISTRY: dict[str, type] = {
    "lasso": LassoSelector,
    "random_forest": RandomForestSelector,
    "catboost_rfe": CatBoostRfeSelector,
    "lightgbm": LightGbmSelector,
}

STUB_METHODS = frozenset({"lasso", "random_forest"})


class ModelBasedStage:
    """Run exactly one model-based selector chosen by config."""

    stage_name = "model"

    def __init__(self: ModelBasedStage, selector: Selector | None = None) -> None:
        self._selector = selector

    def build_selector(self: ModelBasedStage, context: StageContext) -> Selector:
        """Resolve the configured model selector.

        Args:
            context: Shared stage context.

        Returns:
            Selector instance for ``config.model.method``.
        """
        if self._selector is not None:
            return self._selector
        method = context.config.model.method
        try:
            selector_cls = _MODEL_REGISTRY[method]
        except KeyError as exc:
            msg = f"Unsupported model.method={method!r}. Expected one of: {sorted(_MODEL_REGISTRY)}."
            raise ConfigError(
                msg,
            ) from exc
        return selector_cls(context.config.model)

    def run(
        self: ModelBasedStage,
        context: StageContext,
        candidates: Sequence[str],
    ) -> tuple[list[str], list[FeatureDecision]]:
        """Execute the configured model-based selector.

        Args:
            context: Shared stage context.
            candidates: Current candidate features.

        Returns:
            Remaining candidates and drop decisions.
        """
        if not context.config.model.enabled:
            return list(candidates), []
        if context.schema.target is None:
            msg = "model stage requires FeatureSchema.target."
            raise ConfigError(msg)
        selector = self.build_selector(context)
        decisions = run_selector_logged(selector, context, candidates)
        context.decisions.extend(decisions)
        remaining = apply_drop_decisions(candidates, decisions)
        context.candidates = remaining
        persist_step_artifact(
            context,
            remaining,
            stage_name=self.stage_name,
            method_name=selector.method_name,
        )
        context.step_index += 1
        return remaining, decisions
