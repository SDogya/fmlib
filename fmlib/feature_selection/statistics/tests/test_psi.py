"""Tests for PSI selector with stratified sampling functionality."""

from __future__ import annotations

from types import SimpleNamespace

import random
from typing import Any

import pandas as pd
import pytest

from fmlib.feature_selection.base import StageContext
from fmlib.feature_selection.config import FeatureSelectionConfig, PsiConfig, StatisticsConfig
from fmlib.feature_selection.utils.conftest import require_spark_session
from fmlib.feature_selection.schema import FeatureSchema
from fmlib.feature_selection.statistics.psi import PsiSelector


def _make_psi_selector(
    subsample_rows: int | None = None,
    *,
    threshold: float = 0.25,
) -> PsiSelector:
    """Create a PSI selector with optional subsampling."""
    config = FeatureSelectionConfig(
        statistics=StatisticsConfig(
            psi=PsiConfig(
                mode="train_valid",
                threshold=threshold,
                subsample_rows=subsample_rows,
            ),
        ),
    )
    return PsiSelector(config.statistics.psi)


def _make_context(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "target",
    seed: int = 42,
) -> StageContext:
    """Create a StageContext for testing."""
    schema = FeatureSchema(
        categorical=(),
        continuous=tuple([c for c in train_df.columns if c != target_col]),
        target=target_col,
        task_type="binary_classification",
    )
    return StageContext(
        spark=require_spark_session(),
        datasets={"train": train_df, "valid": test_df},
        schema=schema,
        config=None,  # Will be set by selector
        seed=seed,
        candidates=list(train_df.columns),
    )


class _SamplingContext:
    """Minimal context for the bounded-sample helper."""

    def __init__(self, target: str = "target", seed: int = 42) -> None:
        self.schema = SimpleNamespace(target=target, task_type="binary_classification")
        self.seed = seed
        self.run_seed = seed


class TestPsiStratifiedSampling:
    """PSI bounds both populations through the shared sampler."""

    @staticmethod
    def _imbalanced(n_zero: int, n_one: int, seed: int = 42) -> pd.DataFrame:
        rng = random.Random(seed)
        total = n_zero + n_one
        return pd.DataFrame(
            {
                "target": [0] * n_zero + [1] * n_one,
                "feature1": [rng.gauss(0, 1) for _ in range(total)],
                "feature2": [rng.gauss(0, 1) for _ in range(total)],
            },
        )

    def test_subsample_bounds_both_populations(self) -> None:
        train_df = self._imbalanced(160, 40)
        test_df = self._imbalanced(160, 40, seed=7)
        selector = _make_psi_selector(subsample_rows=100)

        bounded_train, bounded_test = selector._apply_subsample_if_needed(
            _SamplingContext(), train_df, test_df,
        )

        assert 0 < len(bounded_train) <= 100
        assert 0 < len(bounded_test) <= 100

    def test_no_subsample_when_not_configured(self) -> None:
        train_df = self._imbalanced(3, 2)
        test_df = self._imbalanced(3, 2, seed=7)
        selector = _make_psi_selector(subsample_rows=None)

        bounded_train, bounded_test = selector._apply_subsample_if_needed(
            _SamplingContext(), train_df, test_df,
        )

        assert bounded_train is train_df
        assert bounded_test is test_df

    def test_no_subsample_when_below_limit(self) -> None:
        train_df = self._imbalanced(3, 2)
        test_df = self._imbalanced(3, 2, seed=7)
        selector = _make_psi_selector(subsample_rows=1000)

        bounded_train, bounded_test = selector._apply_subsample_if_needed(
            _SamplingContext(), train_df, test_df,
        )

        assert len(bounded_train) == len(train_df)
        assert len(bounded_test) == len(test_df)

    def test_stratification_preserves_class_distribution(self) -> None:
        train_df = self._imbalanced(80, 20)
        selector = _make_psi_selector(subsample_rows=50)

        bounded, _ = selector._apply_subsample_if_needed(
            _SamplingContext(), train_df, train_df,
        )

        n_zero = int((bounded["target"] == 0).sum())
        n_one = int((bounded["target"] == 1).sum())
        assert n_zero + n_one == 50
        assert n_zero >= 30
        assert n_one >= 10

    def test_psi_scores_every_candidate_and_drops_none_when_stable(self) -> None:
        train_df = self._imbalanced(160, 40)
        test_df = self._imbalanced(160, 40, seed=7)
        selector = _make_psi_selector(subsample_rows=100)
        context = _make_context(train_df, test_df)
        context.config = selector.config

        decisions = selector.select(context, ["feature1", "feature2"])

        # Only drops are returned now; every candidate is still scored.
        assert all(decision.keep is False for decision in decisions)
        scored = context.scores["psi"]["values"]
        assert set(scored) == {"feature1", "feature2"}

    def test_psi_with_very_small_subsample_still_scores(self) -> None:
        train_df = self._imbalanced(80, 20)
        test_df = self._imbalanced(80, 20, seed=7)
        selector = _make_psi_selector(subsample_rows=30)
        context = _make_context(train_df, test_df)
        context.config = selector.config

        selector.select(context, ["feature1"])

        assert set(context.scores["psi"]["values"]) == {"feature1"}

    def test_shifted_feature_is_dropped(self) -> None:
        rng = random.Random(0)
        train_df = pd.DataFrame(
            {
                "target": [0, 1] * 100,
                "stable": [rng.gauss(0, 1) for _ in range(200)],
                "shifted": [rng.gauss(0, 1) for _ in range(200)],
            },
        )
        test_df = pd.DataFrame(
            {
                "target": [0, 1] * 100,
                "stable": [rng.gauss(0, 1) for _ in range(200)],
                # A five-sigma shift moves every row out of the baseline bins.
                "shifted": [rng.gauss(5, 1) for _ in range(200)],
            },
        )
        selector = _make_psi_selector(threshold=0.25)
        context = _make_context(train_df, test_df)
        context.config = selector.config

        decisions = selector.select(context, ["stable", "shifted"])

        assert [decision.feature for decision in decisions] == ["shifted"]


class TestPsiCategorical:
    """Categorical candidates are compared level by level, not quantile-binned."""

    @staticmethod
    def _context(train_df: pd.DataFrame, test_df: pd.DataFrame) -> StageContext:
        schema = FeatureSchema(
            categorical=("city",),
            continuous=("amount",),
            target="target",
            task_type="binary_classification",
        )
        return StageContext(
            spark=None,
            datasets={"train": train_df, "valid": test_df},
            schema=schema,
            config=FeatureSelectionConfig(),
            seed=42,
            candidates=["city", "amount"],
        )

    def test_string_column_does_not_raise_and_is_scored(self) -> None:
        train_df = pd.DataFrame(
            {
                "city": ["msk"] * 60 + ["spb"] * 30 + ["nsk"] * 10,
                "amount": [float(index % 17) for index in range(100)],
                "target": [0, 1] * 50,
            },
        )
        test_df = pd.DataFrame(
            {
                "city": ["msk"] * 58 + ["spb"] * 32 + ["nsk"] * 10,
                "amount": [float(index % 17) for index in range(100)],
                "target": [0, 1] * 50,
            },
        )
        selector = _make_psi_selector(threshold=0.25)
        context = self._context(train_df, test_df)
        context.config = selector.config

        decisions = selector.select(context, ["city", "amount"])

        assert set(context.scores["psi"]["values"]) == {"city", "amount"}
        assert decisions == []

    def test_level_mix_shift_is_detected(self) -> None:
        train_df = pd.DataFrame(
            {
                "city": ["msk"] * 90 + ["spb"] * 10,
                "amount": [float(index % 17) for index in range(100)],
                "target": [0, 1] * 50,
            },
        )
        test_df = pd.DataFrame(
            {
                "city": ["msk"] * 10 + ["spb"] * 90,
                "amount": [float(index % 17) for index in range(100)],
                "target": [0, 1] * 50,
            },
        )
        selector = _make_psi_selector(threshold=0.25)
        context = self._context(train_df, test_df)
        context.config = selector.config

        decisions = selector.select(context, ["city", "amount"])

        assert [decision.feature for decision in decisions] == ["city"]
        assert context.scores["psi"]["values"]["city"] > 0.25

    def test_unseen_level_lands_in_the_other_bin(self) -> None:
        train_df = pd.DataFrame(
            {
                "city": ["msk"] * 50 + ["spb"] * 50,
                "amount": [float(index % 17) for index in range(100)],
                "target": [0, 1] * 50,
            },
        )
        test_df = pd.DataFrame(
            {
                "city": ["msk"] * 50 + ["ekb"] * 50,
                "amount": [float(index % 17) for index in range(100)],
                "target": [0, 1] * 50,
            },
        )
        selector = _make_psi_selector(threshold=0.25)
        context = self._context(train_df, test_df)
        context.config = selector.config

        decisions = selector.select(context, ["city"])

        assert [decision.feature for decision in decisions] == ["city"]


def test_spark_psi_runs_on_real_dataframes(spark: Any) -> None:
    train = spark.createDataFrame(
        [(float(index), index % 2) for index in range(40)],
        ["feature1", "target"],
    )
    # mode='train_valid' compares train against valid; 'test' is never read.
    valid = spark.createDataFrame(
        [(float(index) + 0.25, index % 2) for index in range(40)],
        ["feature1", "target"],
    )
    selector = _make_psi_selector(subsample_rows=16)
    schema = FeatureSchema(
        categorical=(),
        continuous=("feature1",),
        target="target",
        task_type="binary_classification",
    )
    context = StageContext(
        spark=spark,
        datasets={"train": train, "valid": valid},
        schema=schema,
        config=FeatureSelectionConfig(),
        seed=42,
        candidates=["feature1"],
    )
    selector.select(context, ["feature1"])
    scored = context.scores["psi"]["values"]
    assert set(scored) == {"feature1"}
    assert scored["feature1"] >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
