"""Тесты локальных числовых выборок ограниченного размера, общих для LightGBM и Boruta."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from fmlib.feature_selection.exceptions import ExecutionError
from fmlib.feature_selection.utils.local_data import (
    _sample_spark,
    _spark_row_order,
    prepare_numeric_frame,
    sample_frame_rows,
)


def _numeric_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [5.0, 6.0, 7.0, 8.0],
            "y": [0, 1, 0, 1],
        },
    )


def test_prepare_reuses_compatible_column_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    context = SimpleNamespace(local_numeric_sample=None)
    calls = {"n": 0}
    original = prepare_numeric_frame.__globals__["_prepare_pandas_frame"]

    def counting_prepare(*args: object, **kwargs: object) -> pd.DataFrame:
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "fmlib.feature_selection.utils.local_data._prepare_pandas_frame",
        counting_prepare,
    )

    first = prepare_numeric_frame(
        _numeric_frame(),
        target_col="y",
        feature_cols=["a", "b"],
        max_rows=10,
        sample_fraction=None,
        seed=7,
        method_name="lightgbm",
        context=context,
    )
    second = prepare_numeric_frame(
        _numeric_frame(),
        target_col="y",
        feature_cols=["a"],
        max_rows=10,
        sample_fraction=None,
        seed=7,
        method_name="boruta_shap",
        context=context,
    )

    assert calls["n"] == 1
    assert list(first.columns) == ["a", "b", "y"]
    assert list(second.columns) == ["a", "y"]
    pd.testing.assert_series_equal(second["a"], first["a"], check_names=True)


def test_prepare_does_not_reuse_when_max_rows_differ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = SimpleNamespace(local_numeric_sample=None)
    calls = {"n": 0}
    original = prepare_numeric_frame.__globals__["_prepare_pandas_frame"]

    def counting_prepare(*args: object, **kwargs: object) -> pd.DataFrame:
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "fmlib.feature_selection.utils.local_data._prepare_pandas_frame",
        counting_prepare,
    )

    prepare_numeric_frame(
        _numeric_frame(),
        target_col="y",
        feature_cols=["a", "b"],
        max_rows=10,
        sample_fraction=None,
        seed=7,
        method_name="lightgbm",
        context=context,
    )
    prepare_numeric_frame(
        _numeric_frame(),
        target_col="y",
        feature_cols=["a"],
        max_rows=2,
        sample_fraction=None,
        seed=7,
        method_name="boruta_shap",
        context=context,
    )

    assert calls["n"] == 2


def test_prepare_keeps_numeric_nulls() -> None:
    frame = _numeric_frame()
    frame.loc[0, "a"] = float("nan")

    prepared = prepare_numeric_frame(
        frame,
        target_col="y",
        feature_cols=["a", "b"],
        max_rows=10,
        sample_fraction=None,
        seed=7,
        method_name="lightgbm",
    )

    assert int(prepared["a"].isna().sum()) == 1
    assert not prepared["b"].isna().any()


def test_prepare_does_not_stratify_regression() -> None:
    frame = pd.DataFrame(
        {
            "a": list(range(20)),
            "b": list(range(20, 40)),
            "y": [float(index) for index in range(20)],
        },
    )
    context = SimpleNamespace(
        local_numeric_sample=None,
        schema=SimpleNamespace(task_type="regression"),
    )
    prepared = prepare_numeric_frame(
        frame,
        target_col="y",
        feature_cols=["a", "b"],
        max_rows=6,
        sample_fraction=None,
        seed=7,
        method_name="lightgbm",
        context=context,
    )
    assert len(prepared) == 6
    # Random sample, not one-row-per-distinct-target.
    assert prepared["y"].nunique() == 6


@pytest.mark.parametrize("stratified", [True, False])
@pytest.mark.parametrize("sample_fraction", [None, 0.13])
def test_spark_sample_membership_is_partition_independent(
    spark: Any, stratified: bool, sample_fraction: float | None,
) -> None:
    frame = spark.createDataFrame(
        [(index, index % 3) for index in range(200)], ["x", "y"],
    )

    def sample(data: Any, seed: int = 7) -> set[int]:
        return {
            row.x for row in _sample_spark(
                data, target_col="y", max_rows=37,
                sample_fraction=sample_fraction, seed=seed,
                method_name="test", stratified=stratified,
            ).collect()
        }

    expected = sample(frame.coalesce(1))
    assert len(expected) == (37 if sample_fraction is None else 26)
    assert sample(frame.repartition(5)) == expected
    assert sample(frame.orderBy("x", ascending=False).repartition(3)) == expected
    assert sample(frame.repartition(5), seed=19) != expected


@pytest.mark.parametrize(
    ("counts", "max_rows", "sample_fraction", "expected"),
    [
        ([90, 9, 1], 10, None, {"a": 8, "b": 1, "c": 1}),
        ([90, 9, 1], 80, 0.25, {"a": 22, "b": 2, "c": 1}),
        ([10, 10, 10], 7, None, {"a": 3, "b": 2, "c": 2}),
        ([90, 9, 1], 3, None, {"a": 1, "b": 1, "c": 1}),
    ],
)
def test_spark_sample_fills_exact_class_quotas(
    spark: Any, counts: list[int], max_rows: int,
    sample_fraction: float | None, expected: dict[str, int],
) -> None:
    frame = spark.createDataFrame(
        [(index, label) for label, size in zip("abc", counts) for index in range(size)],
        ["x", "y"],
    )
    for partitions in (1, 4):
        sampled = _sample_spark(
            frame.repartition(partitions), target_col="y", max_rows=max_rows,
            sample_fraction=sample_fraction, seed=7, method_name="test",
        )
        assert Counter(row.y for row in sampled.collect()) == expected


@pytest.mark.parametrize("stratified", [True, False])
def test_spark_row_sample_entrypoint_is_partition_independent(
    spark: Any, stratified: bool,
) -> None:
    frame = spark.createDataFrame([(index, index % 2) for index in range(100)], ["x", "y"])
    samples = []
    for partitions in (1, 4):
        sampled, original_rows, sampled_rows = sample_frame_rows(
            frame.repartition(partitions), target_col="y", max_rows=17,
            stratified=stratified, seed=7, method_name="row_sample",
        )
        assert (original_rows, sampled_rows) == (100, 17)
        samples.append(set(sampled.collect()))
    assert samples[0] == samples[1]


def test_spark_sample_preserves_duplicates_nulls_and_reserved_columns(spark: Any) -> None:
    columns = ["a.b", "y` label", "__fmlib_stratum__", "__FMLIB_RANK__"]
    rows = [(None, "a", "original", 1)] * 12 + [(2.0, "b", "original", 2)] * 8
    frame = spark.createDataFrame(rows, columns)
    for partitions in (1, 4):
        sampled = _sample_spark(
            frame.repartition(partitions), target_col="y` label", max_rows=10,
            sample_fraction=None, seed=7, method_name="test",
        )
        assert sampled.columns == columns
        assert Counter(tuple(row) for row in sampled.collect()) == {
            (None, "a", "original", 1): 6, (2.0, "b", "original", 2): 4,
        }


@pytest.mark.parametrize("stratified", [True, False])
def test_spark_sample_resolves_hash_collisions(
    spark: Any, monkeypatch: pytest.MonkeyPatch, stratified: bool,
) -> None:
    from pyspark.sql import functions

    monkeypatch.setattr(
        functions, "xxhash64", lambda *args: functions.lit(0).cast("long"),
    )
    frame = spark.createDataFrame([(index, index % 2) for index in range(40)], ["x", "y"])
    samples = []
    for partitions in (1, 4):
        sampled = _sample_spark(
            frame.repartition(partitions), target_col="y", max_rows=9,
            sample_fraction=None, seed=7, method_name="test", stratified=stratified,
        )
        samples.append(set(sampled.collect()))
    assert len(samples[0]) == 9
    assert samples[0] == samples[1]


def test_spark_row_order_preserves_timestamp_microseconds(spark: Any) -> None:
    frame = spark.createDataFrame(
        [(datetime(2026, 1, 1, microsecond=value),) for value in (1, 2)], ["time"],
    )
    assert frame.select(_spark_row_order(frame.columns, 7)[1]).distinct().count() == 2


@pytest.mark.parametrize("task_type", ["binary_classification", "regression"])
def test_spark_numeric_materialization_is_partition_independent(
    spark: Any, task_type: str,
) -> None:
    frame = spark.createDataFrame(
        [(float(index), index % 2) for index in range(100)], ["a.b", "y"],
    )
    samples = [
        prepare_numeric_frame(
            frame.repartition(partitions), target_col="y", feature_cols=["a.b"],
            max_rows=17, sample_fraction=None, seed=7, method_name="lightgbm",
            context=SimpleNamespace(schema=SimpleNamespace(task_type=task_type)),
        )
        for partitions in (1, 4)
    ]
    assert len(samples[0]) == 17
    pd.testing.assert_frame_equal(samples[0], samples[1])


def test_spark_sample_rejects_limit_below_class_count(spark: Any) -> None:
    frame = spark.createDataFrame([(1, "a"), (2, "b"), (3, "c")], ["x", "y"])
    with pytest.raises(ExecutionError, match="retain every target class"):
        _sample_spark(
            frame, target_col="y", max_rows=2,
            sample_fraction=None, seed=7, method_name="test",
        )


@pytest.mark.parametrize("rows", [[], [(1, 0), (2, 1)]])
@pytest.mark.parametrize("stratified", [True, False])
def test_spark_sample_preserves_input_below_limit(
    spark: Any, rows: list[tuple[int, int]], stratified: bool,
) -> None:
    frame = spark.createDataFrame(rows, "x long, y long")
    sampled = _sample_spark(
        frame, target_col="y", max_rows=10,
        sample_fraction=None, seed=7, method_name="test", stratified=stratified,
    )
    assert sampled.schema == frame.schema
    assert Counter(tuple(row) for row in sampled.collect()) == Counter(rows)
