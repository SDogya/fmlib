"""Tests for PSI selector with stratified sampling functionality."""

from __future__ import annotations

import random
from typing import Any

import pandas as pd
import pytest

from fmlib.feature_selection.base import StageContext
from fmlib.feature_selection.config import FeatureSelectionConfig, PsiConfig, StatisticsConfig
from fmlib.feature_selection.utils.conftest import require_spark_session
from fmlib.feature_selection.schema import FeatureSchema
from fmlib.feature_selection.statistics.psi import PsiSelector


def _make_psi_selector(subsample_rows: int | None = None) -> PsiSelector:
    """Create a PSI selector with optional subsampling."""
    config = FeatureSelectionConfig(
        statistics=StatisticsConfig(
            psi=PsiConfig(
                mode="train_valid",
                threshold=0.25,
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


class TestPsiStratifiedSampling:
    """Tests for stratified sampling in PSI selector."""

    def test_pandas_subsample_applied_when_configured(self) -> None:
        """Test that stratified sampling reduces row count when configured."""
        # Create a larger dataset with imbalance
        random.seed(42)
        n_rows = 200

        # Create imbalanced target (80% class 0, 20% class 1)
        data = {
            "target": [0] * 160 + [1] * 40,
            "feature1": [random.gauss(0, 1) for _ in range(n_rows)],
            "feature2": [random.gauss(0, 1) for _ in range(n_rows)],
        }
        train_df = pd.DataFrame(data)

        # Test df with similar distribution
        test_data = {
            "target": [0] * 160 + [1] * 40,
            "feature1": [random.gauss(0, 1) for _ in range(n_rows)],
            "feature2": [random.gauss(0, 1) for _ in range(n_rows)],
        }
        test_df = pd.DataFrame(test_data)

        # Set subsample to 100 rows
        selector = _make_psi_selector(subsample_rows=100)

        # Apply subsampling directly (pandas)
        result_train = selector._apply_stratified_sampling_pandas(
            train_df, "target", 100, seed=42
        )

        # Verify row count is reduced
        assert len(result_train) <= 100
        assert len(result_train) > 0

    def test_no_subsample_when_not_configured(self) -> None:
        """Test that no subsampling occurs when subsample_rows is None."""
        train_df = pd.DataFrame({
            "target": [0, 1, 0, 1, 0],
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        test_df = pd.DataFrame({
            "target": [0, 1, 0, 1, 0],
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        selector = _make_psi_selector(subsample_rows=None)

        # Mock context
        class MockContext:
            schema = type("Obj", (), {"target": "target"})()
            seed = 42

        result_train, result_test = selector._apply_subsample_if_needed(
            MockContext(), train_df, test_df
        )

        # No sampling should occur
        assert len(result_train) == len(train_df)
        assert len(result_test) == len(test_df)

    def test_no_subsample_when_below_limit(self) -> None:
        """Test that no subsampling occurs when data is already below limit."""
        train_df = pd.DataFrame({
            "target": [0, 1, 0, 1, 0],
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
        test_df = pd.DataFrame({
            "target": [0, 1, 0, 1, 0],
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
        })

        selector = _make_psi_selector(subsample_rows=1000)  # Higher than data size

        # Mock context
        class MockContext:
            schema = type("Obj", (), {"target": "target"})()
            seed = 42

        result_train, result_test = selector._apply_subsample_if_needed(
            MockContext(), train_df, test_df
        )

        # No sampling should occur (data below limit)
        assert len(result_train) == len(train_df)
        assert len(result_test) == len(test_df)

    def test_stratification_preserves_class_distribution(self) -> None:
        """Test that stratified sampling preserves class distribution."""
        random.seed(42)

        # Create imbalanced dataset (80/20 split)
        n_rows = 100
        data = {
            "target": [0] * 80 + [1] * 20,
            "feature1": [random.gauss(0, 1) for _ in range(n_rows)],
        }
        train_df = pd.DataFrame(data)

        selector = _make_psi_selector(subsample_rows=50)

        # Apply sampling
        result = selector._apply_stratified_sampling_pandas(train_df, "target", 50, seed=42)

        # Check class distribution is roughly preserved
        result_class_0 = (result["target"] == 0).sum()
        result_class_1 = (result["target"] == 1).sum()

        # Should have roughly 80/20 split in sample
        assert result_class_0 + result_class_1 == 50
        assert result_class_0 >= 30  # At least ~40% class 0
        assert result_class_1 >= 10  # At least ~20% class 1

    def test_psi_with_subsample(self) -> None:
        """Test that PSI selector works correctly with subsampling enabled."""
        random.seed(42)

        # Create datasets with stable features (low PSI expected)
        n_rows = 200

        # Similar distributions for train and test
        train_data = {
            "target": [0] * 160 + [1] * 40,
            "feature1": [random.gauss(0, 1) for _ in range(n_rows)],
            "feature2": [random.gauss(0, 1) for _ in range(n_rows)],
        }
        train_df = pd.DataFrame(train_data)

        test_data = {
            "target": [0] * 160 + [1] * 40,
            "feature1": [random.gauss(0, 1) for _ in range(n_rows)],
            "feature2": [random.gauss(0, 1) for _ in range(n_rows)],
        }
        test_df = pd.DataFrame(test_data)

        selector = _make_psi_selector(subsample_rows=100)
        context = _make_context(train_df, test_df)
        context.config = selector.config

        # Run feature selection
        decisions = selector.select(context, ["feature1", "feature2"])

        # Should have decisions for both features
        assert len(decisions) == 2
        assert all(d.feature in ["feature1", "feature2"] for d in decisions)

    def test_psi_with_subsample_very_small(self) -> None:
        """Test PSI selector with very small subsample size."""
        random.seed(42)

        n_rows = 100

        train_data = {
            "target": [0] * 80 + [1] * 20,
            "feature1": [random.gauss(0, 1) for _ in range(n_rows)],
        }
        train_df = pd.DataFrame(train_data)

        test_data = {
            "target": [0] * 80 + [1] * 20,
            "feature1": [random.gauss(0, 1) for _ in range(n_rows)],
        }
        test_df = pd.DataFrame(test_data)

        selector = _make_psi_selector(subsample_rows=30)
        context = _make_context(train_df, test_df)
        context.config = selector.config

        decisions = selector.select(context, ["feature1"])

        # Should have decision
        assert len(decisions) == 1
        assert decisions[0].feature == "feature1"


def test_spark_psi_runs_on_real_dataframes(spark: Any) -> None:
    train = spark.createDataFrame(
        [(float(index), index % 2) for index in range(40)],
        ["feature1", "target"],
    )
    test = spark.createDataFrame(
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
        datasets={"train": train, "test": test},
        schema=schema,
        config=FeatureSelectionConfig(),
        seed=42,
        candidates=["feature1"],
    )
    decisions = selector.select(context, ["feature1"])
    assert len(decisions) == 1
    assert decisions[0].feature == "feature1"
    assert decisions[0].method == "psi"
    assert decisions[0].value >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
