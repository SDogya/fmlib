"""BorutaShap precise selector (stub)."""

from __future__ import annotations

from typing import Sequence

from fmlib.feature_selection.base import FeatureDecision, StageContext, stub_drop_features
from fmlib.feature_selection.config import PreciseConfig


class BorutaShapSelector:
    """BorutaShap final feature selection (skeleton stub).

    Args:
        config: Precise stage settings.
    """

    method_name = "boruta_shap"
    stage_name = "precise"

    def __init__(self: BorutaShapSelector, config: PreciseConfig) -> None:
        self.config = config

    def select(
        self: BorutaShapSelector,
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
