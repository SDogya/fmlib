"""Tests for LowVarianceSelector."""

from __future__ import annotations

import pandas as pd
import pytest

from fmlib.feature_selection.base import StageContext
from fmlib.feature_selection.config import FeatureSelectionConfig, LowVarianceConfig
from fmlib.feature_selection.schema import FeatureSchema
from fmlib.feature_selection.statistics.low_variance import LowVarianceSelector
from fmlib.feature_selection.conftest import FakeSparkSession


def _pyspark_available() -> bool:
    try:
        import pyspark  # noqa: F401
    except ImportError:
        return False
    return True


def _context(
    frame: object,
    continuous: list[str],
    *,
    categorical: list[str] | None = None,
    config: FeatureSelectionConfig | None = None,
) -> StageContext:
    categorical = categorical or []
    config = config or FeatureSelectionConfig()
    schema = FeatureSchema(
        categorical=tuple(categorical),
        continuous=tuple(continuous),
        target="response",
        task_type="binary_classification",
    )
    candidates = categorical + continuous
    return StageContext(
        spark=FakeSparkSession(),
        datasets={"train": frame},
        schema=schema,
        config=config,
        seed=config.execution.seed,
        candidates=candidates,
    )


def test_standard_scaling_drops_only_degenerate_variance() -> None:
    frame = pd.DataFrame(
        {
            "constant": [1.0, 1.0, 1.0, 1.0, 1.0],
            "single_value": [None, None, 2.0, None, None],
            "varying": [0.0, 1.0, 2.0, 3.0, 4.0],
        },
    )
    candidates = list(frame.columns)

    decisions = LowVarianceSelector(LowVarianceConfig(min_variance=0.01, scale_method="standard")).select(
        _context(frame, candidates),
        candidates,
    )

    assert {decision.feature for decision in decisions} == {"constant", "single_value"}
    assert all(decision.reason == "low_variance" for decision in decisions)
    assert all(decision.value == 0.0 for decision in decisions)


def test_minmax_uses_scaled_variance_and_strict_threshold() -> None:
    frame = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0, 4.0]})
    context = _context(frame, ["feature"])

    dropped = LowVarianceSelector(LowVarianceConfig(min_variance=0.2, scale_method="minmax")).select(
        context,
        ["feature"],
    )
    kept = LowVarianceSelector(LowVarianceConfig(min_variance=0.15625, scale_method="minmax")).select(
        context,
        ["feature"],
    )

    assert len(dropped) == 1
    assert dropped[0].value == pytest.approx(0.15625)
    assert kept == []


def test_robust_drops_degenerate_iqr_and_ignores_categorical() -> None:
    frame = pd.DataFrame(
        {
            "category": ["same"] * 5,
            "degenerate_iqr": [0.0, 0.0, 0.0, 0.0, 10.0],
            "varying": [0.0, 1.0, 2.0, 3.0, 4.0],
        },
    )
    context = _context(frame, ["degenerate_iqr", "varying"], categorical=["category"])

    decisions = LowVarianceSelector(LowVarianceConfig(min_variance=0.01, scale_method="robust")).select(
        context,
        ["category", "degenerate_iqr", "varying"],
    )

    assert [decision.feature for decision in decisions] == ["degenerate_iqr"]


def test_order_keeps_low_variance_before_correlation() -> None:
    config = FeatureSelectionConfig.from_dict(
        {
            "statistics": {"order": ["low_variance", "correlation"]},
        },
    )
    assert config.statistics.order == ("low_variance", "correlation")


@pytest.mark.skipif(not _pyspark_available(), reason="pyspark not installed")
def test_spark_minmax_supports_dotted_column_names() -> None:
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[1]")
        .appName("test-low-variance-spark")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        frame = spark.createDataFrame([(0.0,), (1.0,), (2.0,), (3.0,), (4.0,)], ["foo.bar"])
        context = StageContext(
            spark=spark,
            datasets={"train": frame},
            schema=FeatureSchema(
                categorical=(),
                continuous=("foo.bar",),
                target="response",
                task_type="binary_classification",
            ),
            config=FeatureSelectionConfig(),
            seed=0,
            candidates=["foo.bar"],
        )

        decisions = LowVarianceSelector(LowVarianceConfig(min_variance=0.2, scale_method="minmax")).select(
            context,
            ["foo.bar"],
        )

        assert len(decisions) == 1
        assert decisions[0].value == pytest.approx(0.15625)
    finally:
        spark.stop()
