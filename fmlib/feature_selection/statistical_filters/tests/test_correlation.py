"""Тесты CorrelationSelector."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from fmlib.feature_selection.base import StageContext
from fmlib.feature_selection.config import CorrelationConfig, FeatureSelectionConfig
from fmlib.feature_selection.exceptions import ExecutionError
from fmlib.feature_selection.schema import FeatureSchema
from fmlib.feature_selection.statistical_filters.correlation import (
    CorrelationSelector,
)
from fmlib.feature_selection.utils.conftest import require_spark_session
from fmlib.feature_selection.utils.local_data import sample_frame_rows


def _schema(
    *,
    continuous: tuple[str, ...] = ("first", "second", "independent"),
    categorical: tuple[str, ...] = ("category",),
    target: str | None = "response",
    task_type: str = "binary_classification",
) -> FeatureSchema:
    return FeatureSchema(
        categorical=categorical,
        continuous=continuous,
        target=target,  # type: ignore[arg-type]
        task_type=task_type,
    )


def _pandas_context(
    frame: pd.DataFrame,
    *,
    continuous: tuple[str, ...] = ("first", "second", "independent"),
    categorical: tuple[str, ...] = ("category",),
    max_local_rows: int = 1_000_000,
    target: str | None = "response",
    task_type: str = "binary_classification",
    seed: int = 0,
) -> StageContext:
    schema = _schema(
        continuous=continuous,
        categorical=categorical,
        target=target,
        task_type=task_type,
    )
    return StageContext(
        spark=None,
        datasets={"train": frame},
        schema=schema,
        config=FeatureSelectionConfig.from_dict(
            {"execution": {"max_local_rows": max_local_rows}},
        ),
        seed=seed,
        candidates=schema.candidate_features(),
    )


def _context(
    frame: object,
    *,
    continuous: tuple[str, ...] = ("first", "second", "independent"),
    categorical: tuple[str, ...] = ("category",),
    max_local_rows: int = 1_000_000,
) -> StageContext:
    schema = _schema(continuous=continuous, categorical=categorical)
    return StageContext(
        spark=require_spark_session(),
        datasets={"train": frame},
        schema=schema,
        config=FeatureSelectionConfig.from_dict(
            {"execution": {"max_local_rows": max_local_rows}},
        ),
        seed=0,
        candidates=schema.candidate_features(),
    )


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
    context = _pandas_context(frame, continuous=("first", "second"))
    selector = CorrelationSelector(CorrelationConfig(threshold=0.9, tie_break="original_order"))

    decisions = selector.select(context, ["category", "first", "second", "independent"])

    assert [decision.feature for decision in decisions] == ["second"]
    assert "independent" not in {decision.feature for decision in decisions}


def test_null_rate_tie_break_drops_more_incomplete_feature() -> None:
    frame = _frame()
    selector = CorrelationSelector(CorrelationConfig(threshold=0.9, tie_break="null_rate"))

    decisions = selector.select(
        _pandas_context(frame),
        ["category", "first", "second", "independent"],
    )

    assert len(decisions) == 1
    assert decisions[0].feature == "second"
    assert decisions[0].reason == "high_correlation"
    assert decisions[0].value == 1.0


def test_original_order_drops_later_feature() -> None:
    frame = _frame()
    selector = CorrelationSelector(CorrelationConfig(threshold=0.9, tie_break="original_order"))

    decisions = selector.select(_pandas_context(frame), ["second", "first"])

    assert [decision.feature for decision in decisions] == ["first"]


def test_threshold_is_strict() -> None:
    selector = CorrelationSelector(CorrelationConfig(threshold=1.0))

    decisions = selector.select(_pandas_context(_frame()), ["first", "second"])

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
    context = _pandas_context(frame, continuous=("first", "second", "all_null"), categorical=())
    selector = CorrelationSelector(CorrelationConfig(threshold=0.9, tie_break="original_order"))

    decisions = selector.select(context, ["first", "second", "all_null"])

    assert [decision.feature for decision in decisions] == ["second"]
    assert "all_null" not in {decision.feature for decision in decisions}


def test_requires_target_to_bound_rows() -> None:
    frame = pd.DataFrame({"first": [1.0, 2.0], "second": [1.0, 2.0]})
    context = _pandas_context(
        frame,
        continuous=("first", "second"),
        categorical=(),
        target=None,
    )
    selector = CorrelationSelector(CorrelationConfig(threshold=0.9))
    with pytest.raises(ExecutionError, match="FeatureSchema.target"):
        selector.select(context, ["first", "second"])


def test_stratified_sample_is_not_a_row_prefix() -> None:
    """В начальных строках одного класса признаки идеально коррелируют; в другом классе — шум.

    Выбор первых ``max_rows`` строк привёл бы к исключению ``second``. Стратифицированная выборка
    смешивает классы, поэтому корреляция пары остаётся ниже порога.
    """
    n_class = 20
    frame = pd.DataFrame(
        {
            "first": list(range(n_class)) + list(range(100, 100 + n_class)),
            "second": [2 * value for value in range(n_class)] + list(range(n_class)),
            "response": [0] * n_class + [1] * n_class,
        },
    )
    context = _pandas_context(frame, continuous=("first", "second"), categorical=())
    selector = CorrelationSelector(CorrelationConfig(threshold=0.95, max_rows=10))
    head_corr = frame.head(10)[["first", "second"]].corr().iloc[0, 1]
    assert abs(float(head_corr)) == pytest.approx(1.0)

    decisions = selector.select(context, ["first", "second"])
    assert decisions == []


def test_execution_max_local_rows_caps_correlation_max_rows() -> None:
    frame = pd.DataFrame(
        {
            "first": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "second": [2.0, 4.0, 6.0, 0.0, 50.0, -10.0],
            "response": [0, 0, 0, 1, 1, 1],
        },
    )
    context = _pandas_context(
        frame,
        continuous=("first", "second"),
        categorical=(),
        max_local_rows=3,
    )
    selector = CorrelationSelector(CorrelationConfig(threshold=0.95, max_rows=100_000))
    captured: dict[str, Any] = {}
    original = sample_frame_rows

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        captured["max_rows"] = kwargs["max_rows"]
        captured["stratified"] = kwargs["stratified"]
        return original(*args, **kwargs)

    with patch(
        "fmlib.feature_selection.statistical_filters.correlation.sample_frame_rows",
        wrapped,
    ):
        selector.select(context, ["first", "second"])

    assert captured["max_rows"] == 3
    assert captured["stratified"] is True


def test_spark_uses_ml_correlation_on_limited_rows_without_to_pandas(
    spark: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyspark.sql import DataFrame as SparkDataFrame

    frame = spark.createDataFrame(
        [
            (
                str(index % 2),
                float(index),
                float(index),
                0.0,
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
