"""Tests for CorrelationSelector."""

from __future__ import annotations

import pandas as pd
import pytest

from fmlib.feature_selection.base import StageContext
from fmlib.feature_selection.config import CorrelationConfig, FeatureSelectionConfig
from fmlib.feature_selection.exceptions import ExecutionError
from fmlib.feature_selection.schema import FeatureSchema
from fmlib.feature_selection.statistics.correlation import CorrelationSelector
from fmlib.feature_selection.conftest import FakeSparkSession


def _context(
    frame: object,
    *,
    continuous: tuple[str, ...] = ("first", "second", "independent"),
    categorical: tuple[str, ...] = ("category",),
    max_local_rows: int = 1_000_000,
) -> StageContext:
    config = FeatureSelectionConfig.from_dict({"execution": {"max_local_rows": max_local_rows}})
    schema = FeatureSchema(
        categorical=categorical,
        continuous=continuous,
        target="response",
        task_type="binary_classification",
    )
    return StageContext(
        spark=FakeSparkSession(),
        datasets={"train": frame},
        schema=schema,
        config=config,
        seed=0,
        candidates=schema.candidate_features(),
    )


def _pyspark_available() -> bool:
    try:
        import pyspark  # noqa: F401
    except ImportError:
        return False
    return True


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category": ["a", "b", "a", "b", "a"],
            "first": [1.0, 2.0, 3.0, 4.0, 5.0],
            "second": [1.0, 2.0, None, 4.0, 5.0],
            "independent": [2.0, 5.0, 1.0, 4.0, 3.0],
            "response": [0, 1, 0, 1, 0],
        },
    )


def test_uses_schema_continuous_not_all_numeric_columns() -> None:
    frame = _frame()
    # ``independent`` is numeric in the frame but absent from schema.continuous.
    context = _context(frame, continuous=("first", "second"))
    selector = CorrelationSelector(CorrelationConfig(threshold=0.9, tie_break="original_order"))

    decisions = selector.select(context, ["category", "first", "second", "independent"])

    assert [decision.feature for decision in decisions] == ["second"]
    assert "independent" not in {decision.feature for decision in decisions}


def test_null_rate_tie_break_drops_more_incomplete_feature() -> None:
    frame = _frame()
    selector = CorrelationSelector(CorrelationConfig(threshold=0.9, tie_break="null_rate"))

    decisions = selector.select(_context(frame), ["category", "first", "second", "independent"])

    assert len(decisions) == 1
    assert decisions[0].feature == "second"
    assert decisions[0].reason == "high_correlation"
    assert decisions[0].value == 1.0


def test_original_order_drops_later_feature() -> None:
    frame = _frame()
    selector = CorrelationSelector(CorrelationConfig(threshold=0.9, tie_break="original_order"))

    decisions = selector.select(_context(frame), ["second", "first"])

    assert [decision.feature for decision in decisions] == ["first"]


def test_threshold_is_strict() -> None:
    selector = CorrelationSelector(CorrelationConfig(threshold=1.0))

    decisions = selector.select(_context(_frame()), ["first", "second"])

    assert decisions == []


def test_skips_all_null_columns_before_correlation() -> None:
    frame = pd.DataFrame(
        {
            "first": [1.0, 2.0, 3.0],
            "second": [1.0, 2.0, 3.0],
            "all_null": [None, None, None],
            "response": [0, 1, 0],
        },
    )
    context = _context(frame, continuous=("first", "second", "all_null"), categorical=())
    selector = CorrelationSelector(CorrelationConfig(threshold=0.9, tie_break="original_order"))

    decisions = selector.select(context, ["first", "second", "all_null"])

    assert [decision.feature for decision in decisions] == ["second"]
    assert "all_null" not in {decision.feature for decision in decisions}


def test_row_limit_changes_which_pairs_are_correlated() -> None:
    """Only the first ``max_rows`` rows participate in the correlation matrix."""
    frame = pd.DataFrame(
        {
            "first": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "second": [2.0, 4.0, 6.0, 0.0, 50.0, -10.0],
            "response": [0, 1, 0, 1, 0, 1],
        },
    )
    context = _context(frame, continuous=("first", "second"), categorical=())
    limited = CorrelationSelector(CorrelationConfig(threshold=0.95, max_rows=3))
    full = CorrelationSelector(CorrelationConfig(threshold=0.95, max_rows=100))

    assert [decision.feature for decision in limited.select(context, ["first", "second"])] == ["second"]
    assert full.select(context, ["first", "second"]) == []


def test_execution_max_local_rows_caps_correlation_max_rows() -> None:
    frame = pd.DataFrame(
        {
            "first": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "second": [2.0, 4.0, 6.0, 0.0, 50.0, -10.0],
            "response": [0, 1, 0, 1, 0, 1],
        },
    )
    # correlation.max_rows is large, but execution.max_local_rows=3 must win.
    context = _context(
        frame,
        continuous=("first", "second"),
        categorical=(),
        max_local_rows=3,
    )
    selector = CorrelationSelector(CorrelationConfig(threshold=0.95, max_rows=100_000))

    assert [decision.feature for decision in selector.select(context, ["first", "second"])] == ["second"]


@pytest.mark.skipif(not _pyspark_available(), reason="pyspark not installed")
def test_spark_uses_ml_correlation_on_limited_rows_without_to_pandas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyspark.sql import DataFrame as SparkDataFrame
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[1]")
        .appName("test-correlation-row-limit")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    try:
        frame = spark.createDataFrame(
            [
                (
                    str(index % 2),
                    float(index),
                    float(index) if index < 4 else float((index * 7) % 11),
                    float(index % 3),
                    index % 2,
                )
                for index in range(20)
            ],
            ["category", "first", "second", "independent", "response"],
        )
        selector = CorrelationSelector(CorrelationConfig(max_rows=4, threshold=0.9))
        monkeypatch.setattr(
            SparkDataFrame,
            "toPandas",
            lambda *_args, **_kwargs: pytest.fail("Spark correlation must not call toPandas"),
        )

        decisions = selector.select(_context(frame), ["first", "second", "independent"])

        assert [decision.feature for decision in decisions] == ["second"]
        assert decisions[0].value == 1.0

        bad_schema = FeatureSchema(
            categorical=(),
            continuous=("category", "first"),
            target="response",
            task_type="binary_classification",
        )
        bad_context = StageContext(
            spark=spark,
            datasets={"train": frame},
            schema=bad_schema,
            config=FeatureSelectionConfig(),
            seed=0,
            candidates=["category", "first"],
        )
        with pytest.raises(ExecutionError, match=r"FeatureSchema\.continuous.*category:StringType"):
            selector.select(bad_context, ["category", "first"])
    finally:
        spark.stop()
