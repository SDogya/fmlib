"""Precise / final selection stage orchestration."""

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
from fmlib.feature_selection.precise.boruta_shap import BorutaShapSelector
from fmlib.feature_selection.utils.verbose import run_selector_logged


class PreciseStage:
    """Optional final selection stage (``boruta_shap`` or ``none``)."""

    stage_name = "precise"

    def __init__(self: PreciseStage, selector: Selector | None = None) -> None:
        self._selector = selector

    def build_selector(self: PreciseStage, context: StageContext) -> Selector | None:
        """Resolve the precise selector or skip when method is ``none``.

        Args:
            context: Shared stage context.

        Returns:
            Selector instance or ``None`` when the stage is disabled.
        """
        if self._selector is not None:
            return self._selector
        method = context.config.precise.method
        if method is None or method == "none":
            return None
        if method == "boruta_shap":
            return BorutaShapSelector(context.config.precise)
        msg = f"Unsupported precise.method={method!r}. Expected 'boruta_shap' or 'none'."
        raise ConfigError(
            msg,
        )

    def run(
        self: PreciseStage,
        context: StageContext,
        candidates: Sequence[str],
    ) -> tuple[list[str], list[FeatureDecision]]:
        """Execute the precise selector when configured.

        Args:
            context: Shared stage context.
            candidates: Current candidate features.

        Returns:
            Remaining candidates and drop decisions (empty when method is ``none``).
        """
        if not context.config.precise.enabled:
            return list(candidates), []
        selector = self.build_selector(context)
        if selector is None:
            return list(candidates), []
        if not context.schema.target:
            msg = f"{selector.method_name} requires FeatureSchema.target."
            raise ConfigError(msg)
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
