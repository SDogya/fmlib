"""Random Forest model-based selector (stub)."""

from __future__ import annotations

from typing import Sequence

from fmlib.feature_selection.base import FeatureDecision, StageContext, stub_drop_features
from fmlib.feature_selection.config import ModelConfig


class RandomForestSelector:
    """Random Forest importance-based selection (skeleton stub).

    Args:
        config: Model-based stage settings.
    """

    method_name = "random_forest"
    stage_name = "model"

    def __init__(self: RandomForestSelector, config: ModelConfig) -> None:
        self.config = config

    def select(
        self: RandomForestSelector,
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
