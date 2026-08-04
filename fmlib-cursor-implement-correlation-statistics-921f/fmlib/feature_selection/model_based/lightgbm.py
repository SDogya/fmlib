"""LightGBM model-based selector (stub)."""

from __future__ import annotations

from typing import Sequence

from fmlib.feature_selection.base import FeatureDecision, StageContext, stub_drop_features
from fmlib.feature_selection.config import ModelConfig


class LightGbmSelector:
    """LightGBM importance-based selection (skeleton stub).

    Args:
        config: Model-based stage settings.
    """

    method_name = "lightgbm"
    stage_name = "model"

    def __init__(self: LightGbmSelector, config: ModelConfig) -> None:
        self.config = config

    def select(
        self: LightGbmSelector,
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
