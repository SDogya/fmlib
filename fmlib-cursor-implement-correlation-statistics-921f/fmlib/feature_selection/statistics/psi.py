"""Population Stability Index filter (stub)."""

from __future__ import annotations

from typing import Sequence

from fmlib.feature_selection.base import FeatureDecision, StageContext, stub_drop_features
from fmlib.feature_selection.config import PsiConfig


class PsiSelector:
    """Exclude unstable features via PSI (skeleton stub).

    Args:
        config: PSI filter settings.
    """

    method_name = "psi"
    stage_name = "statistics"

    def __init__(self: PsiSelector, config: PsiConfig) -> None:
        self.config = config

    def select(
        self: PsiSelector,
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
