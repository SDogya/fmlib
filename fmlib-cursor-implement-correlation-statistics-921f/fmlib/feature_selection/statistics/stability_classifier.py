"""Stability classifier statistical filter (stub)."""

from __future__ import annotations

from typing import Sequence

from fmlib.feature_selection.base import FeatureDecision, StageContext, stub_drop_features
from fmlib.feature_selection.config import StabilityClassifierConfig


class StabilityClassifierSelector:
    """Exclude features that discriminate train/valid/test (skeleton stub).

    Args:
        config: Stability classifier settings.
    """

    method_name = "stability_classifier"
    stage_name = "statistics"

    def __init__(self: StabilityClassifierSelector, config: StabilityClassifierConfig) -> None:
        self.config = config

    def select(
        self: StabilityClassifierSelector,
        context: StageContext,
        candidates: Sequence[str],
    ) -> list[FeatureDecision]:
        """Drop a deterministic random subset of candidates.

        Args:
            context: Shared stage context.
            candidates: Current candidate features.

        Returns:
            Stub drop decisions.
        """
        return stub_drop_features(
            candidates,
            seed=context.seed,
            stage=self.stage_name,
            method=self.method_name,
        )
