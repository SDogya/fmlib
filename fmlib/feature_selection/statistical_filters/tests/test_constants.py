"""Tests for ConstantsSelector."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from fmlib.feature_selection.base import StageContext
from fmlib.feature_selection.config import ConstantsConfig, FeatureSelectionConfig
from fmlib.feature_selection.utils.conftest import require_spark_session
from fmlib.feature_selection.schema import FeatureSchema
from fmlib.feature_selection.statistical_filters.constants import ConstantsSelector


def _schema() -> FeatureSchema:
    return FeatureSchema(
        categorical=("segment",),
        continuous=("age", "const_col", "quasi_col", "good_col"),
        target="response",
        task_type="binary_classification",
        split="dataset_split",
        id_columns=("client_id",),
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
            "age": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            "const_col": [1] * 10,
            "quasi_col": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            "good_col": list(range(10)),
            "response": [0, 1] * 5,
            "dataset_split": ["train"] * 10,
            "client_id": list(range(10)),
        },
    )


def _context(df: pd.DataFrame, config: FeatureSelectionConfig) -> StageContext:
    return StageContext(
        spark=require_spark_session(),
        datasets={"train": df},
        schema=_schema(),
        config=config,
        seed=config.execution.seed,
        candidates=list(_schema().candidate_features()),
    )


def test_drops_constant_and_quasi_constant() -> None:
    fs_config = FeatureSelectionConfig.from_dict(
        {
            "statistics": {
                "constants": {"max_frequency": 0.9, "min_unique": 2},
            },
            "model": {"method": "lasso"},
            "precise": {"method": "none"},
        },
    )
    selector = ConstantsSelector(fs_config.statistics.constants)
    decisions = selector.select(
        _context(_frame(), fs_config),
        _schema().candidate_features(),
    )
    by_feature = {item.feature: item for item in decisions}
    assert set(by_feature) == {"const_col", "quasi_col"}
    assert by_feature["const_col"].reason == "constant"
    assert by_feature["quasi_col"].reason == "quasi_constant"
    assert by_feature["quasi_col"].value == 0.9
    assert "good_col" not in by_feature
    assert "age" not in by_feature
    assert "segment" not in by_feature


def test_min_unique_rule() -> None:
    config = ConstantsConfig(max_frequency=1.0, min_unique=5)
    selector = ConstantsSelector(config)
    fs_config = FeatureSelectionConfig.from_dict(
        {
            "statistics": {
                "constants": {"max_frequency": 1.0, "min_unique": 5},
            },
        },
    )
    # segment has 2 unique values → too_few_unique when min_unique=5
    df = _frame()
    decisions = selector.select(_context(df, fs_config), ["segment", "good_col"])
    by_feature = {item.feature: item for item in decisions}
    assert by_feature["segment"].reason == "too_few_unique"
    assert by_feature["segment"].value == 2.0
    assert "good_col" not in by_feature


def test_keeps_all_when_thresholds_relaxed() -> None:
    config = ConstantsConfig(max_frequency=1.0, min_unique=None)
    selector = ConstantsSelector(config)
    fs_config = FeatureSelectionConfig()
    # max_frequency=1.0 only drops when a single value covers 100% → const_col still drops
    decisions = selector.select(_context(_frame(), fs_config), ["good_col", "age"])
    assert decisions == []


def test_all_null_column_is_left_to_null_rate_selector() -> None:
    frame = _frame()
    frame["all_null"] = None
    config = ConstantsConfig(max_frequency=0.5)
    fs_config = FeatureSelectionConfig()

    decisions = ConstantsSelector(config).select(_context(frame, fs_config), ["all_null"])

    assert decisions == []


def test_spark_rejects_map_type_before_collect(spark: Any) -> None:
    from fmlib.feature_selection.exceptions import ExecutionError

    df = spark.createDataFrame([({"a": 1}, 0), ({"a": 1}, 1)], ["m", "response"])
    schema = FeatureSchema(
        categorical=(),
        continuous=("m",),
        target="response",
        task_type="binary_classification",
    )
    config = FeatureSelectionConfig.from_dict(
        {
            "statistics": {
                "constants": {"max_frequency": 0.99},
            },
        },
    )
    selector = ConstantsSelector(config.statistics.constants)
    context = StageContext(
        spark=spark,
        datasets={"train": df},
        schema=schema,
        config=config,
        seed=0,
        candidates=["m"],
    )
    with pytest.raises(ExecutionError, match="MapType"):
        selector.select(context, ["m"])


def test_spark_default_path_skips_count_distinct_and_chunks_columns(
    spark: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyspark.sql import functions as F  # noqa: N812

    df = spark.createDataFrame(
        [(1, 0, 1, 0), (1, 0, 2, 1), (1, 0, 3, 0), (1, 0, 4, 1), (1, 1, 5, 0)],
        ["foo.bar", "quasi", "good", "response"],
    )
    schema = FeatureSchema(
        categorical=(),
        continuous=("foo.bar", "quasi", "good"),
        target="response",
        task_type="binary_classification",
    )
    config = FeatureSelectionConfig.from_dict(
        {
            "statistics": {
                "constants": {
                    "max_frequency": 0.8,
                    "chunk_size": 1,
                },
            },
        },
    )
    selector = ConstantsSelector(config.statistics.constants)
    context = StageContext(
        spark=spark,
        datasets={"train": df},
        schema=schema,
        config=config,
        seed=0,
        candidates=["foo.bar", "quasi", "good"],
    )
    with monkeypatch.context() as patch:
        patch.setattr(
            F,
            "countDistinct",
            lambda *_args, **_kwargs: pytest.fail("countDistinct must not run without min_unique"),
        )
        decisions = selector.select(context, ["foo.bar", "quasi", "good"])
    by_feature = {decision.feature: decision for decision in decisions}
    assert set(by_feature) == {"foo.bar", "quasi"}
    assert by_feature["foo.bar"].reason == "constant"
    assert by_feature["quasi"].reason == "quasi_constant"
    assert by_feature["quasi"].value == 0.8

    min_unique_selector = ConstantsSelector(ConstantsConfig(max_frequency=1.0, min_unique=4, chunk_size=1))
    min_unique_decisions = min_unique_selector.select(context, ["quasi", "good"])
    assert [decision.feature for decision in min_unique_decisions] == ["quasi"]
    assert min_unique_decisions[0].reason == "too_few_unique"
