"""Tests for NullRateSelector."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from fmlib.feature_selection.base import StageContext
from fmlib.feature_selection.config import FeatureSelectionConfig, NullRateConfig
from fmlib.feature_selection.exceptions import ExecutionError
from fmlib.feature_selection.schema import FeatureSchema
from fmlib.feature_selection.statistics.null_rate import NullRateSelector
from fmlib.feature_selection.tests.conftest import FakeSparkSession


def _pyspark_available() -> bool:
    try:
        import pyspark  # noqa: F401
    except ImportError:
        return False
    return True


def _context(frame: object, candidates: list[str]) -> StageContext:
    schema = FeatureSchema(
        categorical=(),
        continuous=tuple(candidates),
        target="response",
        task_type="binary_classification",
    )
    config = FeatureSelectionConfig()
    return StageContext(
        spark=FakeSparkSession(),
        datasets={"train": frame},
        schema=schema,
        config=config,
        seed=config.execution.seed,
        candidates=candidates,
    )


def test_drops_only_rates_strictly_above_threshold() -> None:
    frame = pd.DataFrame(
        {
            "high_null": [1.0, None, math.nan, None],
            "at_threshold": [1.0, 2.0, None, math.nan],
            "complete": [1.0, 2.0, 3.0, 4.0],
        },
    )
    candidates = list(frame.columns)

    decisions = NullRateSelector(NullRateConfig(threshold=0.5)).select(_context(frame, candidates), candidates)

    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.feature == "high_null"
    assert decision.reason == "high_null_rate"
    assert decision.value == 0.75
    assert decision.threshold == 0.5
    assert decision.keep is False


def test_empty_frame_and_empty_candidates_produce_no_decisions() -> None:
    selector = NullRateSelector(NullRateConfig(threshold=0.0))
    frame = pd.DataFrame({"feature": pd.Series(dtype="float64")})

    assert selector.select(_context(frame, ["feature"]), ["feature"]) == []
    assert selector.select(_context(frame, ["feature"]), []) == []


def test_missing_candidate_has_actionable_error() -> None:
    selector = NullRateSelector(NullRateConfig())

    with pytest.raises(ExecutionError, match=r"columns missing.*missing"):
        selector.select(_context(pd.DataFrame({"feature": [1]}), ["feature"]), ["missing"])


@pytest.mark.skipif(not _pyspark_available(), reason="pyspark not installed")
def test_spark_counts_null_and_nan_without_materialising_rows() -> None:
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[1]")
        .appName("test-null-rate-spark")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        frame = spark.createDataFrame(
            [
                (1.0, "a", 1),
                (math.nan, None, 2),
                (None, None, 3),
            ],
            ["foo.bar", "text feature", "complete"],
        )
        candidates = list(frame.columns)
        context = StageContext(
            spark=spark,
            datasets={"train": frame},
            schema=FeatureSchema(
                categorical=("text feature",),
                continuous=("foo.bar", "complete"),
                target="response",
                task_type="binary_classification",
            ),
            config=FeatureSelectionConfig(),
            seed=0,
            candidates=candidates,
        )

        decisions = NullRateSelector(NullRateConfig(threshold=0.5)).select(context, candidates)

        assert {decision.feature for decision in decisions} == {"foo.bar", "text feature"}
        assert all(decision.value == pytest.approx(2 / 3) for decision in decisions)
    finally:
        spark.stop()
