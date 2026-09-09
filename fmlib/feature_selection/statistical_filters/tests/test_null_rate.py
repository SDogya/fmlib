"""Exhaustive tests for NullRateSelector.

Spark cases use a real SparkSession (session fixture ``spark``). There is no
FakeSparkDataFrame and no skip if pyspark is missing: the fixture fails the run.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from fmlib.feature_selection.base import FeatureDecision, StageContext
from fmlib.feature_selection.config import FeatureSelectionConfig, NullRateConfig, VerboseConfig
from fmlib.feature_selection.exceptions import ExecutionError
from fmlib.feature_selection.schema import FeatureSchema
from fmlib.feature_selection.statistical_filters import null_rate as null_rate_module
from fmlib.feature_selection.statistical_filters.null_rate import (
    NullRateSelector,
    _is_spark_dataframe,
    _quoted_col,
    _root_cause,
)
from fmlib.feature_selection.utils.verbose import VerboseRecorder

THRESHOLD = 0.5


class ForbiddenDatasets(dict):
    """Mapping that fails if the train split is read."""

    def __getitem__(self: ForbiddenDatasets, key: Any) -> Any:
        message = f"datasets[{key!r}] must not be read for empty candidates"
        raise AssertionError(message)


class JavaLikeError(RuntimeError):
    """Py4J-style error with a multiline java_exception payload."""

    def __init__(self: JavaLikeError, java_exception: str, message: str = "py4j wrapper") -> None:
        super().__init__(message)
        self.java_exception = java_exception


def _drop_decision(
    feature: str,
    value: float,
    threshold: float = THRESHOLD,
) -> FeatureDecision:
    return FeatureDecision(
        feature=feature,
        stage="statistics",
        method="null_rate",
        reason="high_null_rate",
        value=value,
        threshold=threshold,
        keep=False,
    )


def _spark_context(
    spark: Any,
    train: Any,
    candidates: Sequence[str],
    *,
    categorical: tuple[str, ...] = (),
    continuous: tuple[str, ...] | None = None,
    debug_enabled: bool = False,
) -> StageContext:
    candidate_list = list(candidates)
    if continuous is None:
        continuous = tuple(name for name in candidate_list if name not in categorical)
    return StageContext(
        spark=spark,
        datasets={"train": train},
        schema=FeatureSchema(
            categorical=categorical,
            continuous=continuous,
            target="response",
            task_type="binary_classification",
        ),
        config=FeatureSelectionConfig(),
        seed=0,
        candidates=candidate_list,
        verbose_log=VerboseRecorder(verbose=VerboseConfig(null_rate=debug_enabled)),
    )


@pytest.fixture
def stage_context_factory(spark: Any) -> Callable[..., StageContext]:
    def factory(
        train: Any = None,
        *,
        candidates: Sequence[str] | None = None,
        datasets: dict[str, Any] | None = None,
        debug_enabled: bool = False,
        spark_session: Any | None = None,
        schema: FeatureSchema | None = None,
        config: FeatureSelectionConfig | None = None,
    ) -> StageContext:
        if datasets is None:
            datasets = {} if train is None else {"train": train}
        if candidates is None:
            train_obj = datasets.get("train")
            candidates = list(train_obj.columns) if isinstance(train_obj, pd.DataFrame) else []
        config = config or FeatureSelectionConfig()
        candidate_list = list(candidates)
        if schema is None:
            schema = FeatureSchema(
                categorical=(),
                continuous=tuple(candidate_list),
                target="response",
                task_type="binary_classification",
            )
        return StageContext(
            spark=spark if spark_session is None else spark_session,
            datasets=datasets,
            schema=schema,
            config=config,
            seed=config.execution.seed,
            candidates=candidate_list,
            verbose_log=VerboseRecorder(verbose=VerboseConfig(null_rate=debug_enabled)),
        )

    return factory


@pytest.fixture
def null_rate_config_factory() -> Callable[..., NullRateConfig]:
    def factory(threshold: float = THRESHOLD) -> NullRateConfig:
        return NullRateConfig(threshold=threshold)

    return factory


@pytest.fixture
def pandas_sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "float_all_nan": [np.nan, np.nan, np.nan, np.nan],
            "float_none_missing": [1.0, 2.0, 3.0, 4.0],
            "float_half_nan": [1.0, np.nan, 2.0, math.nan],
            "object_half_none": [None, "a", None, "b"],
            "int_half_na": pd.Series([pd.NA, 1, pd.NA, 2], dtype="Int64"),
            "category_half_null": pd.Series([None, "x", None, "y"], dtype="category"),
            "datetime_half_nat": [
                pd.NaT,
                pd.Timestamp("2020-01-01"),
                pd.NaT,
                pd.Timestamp("2020-01-02"),
            ],
        },
    )


@pytest.fixture
def debug_emit_spy(monkeypatch: pytest.MonkeyPatch) -> Mock:
    spy = Mock(name="verbose_emit")
    monkeypatch.setattr(null_rate_module, "verbose_emit", spy)
    return spy


def test_select_empty_candidates_returns_empty_list(
    stage_context_factory: Callable[..., StageContext],
    null_rate_config_factory: Callable[..., NullRateConfig],
) -> None:
    context = stage_context_factory(datasets=ForbiddenDatasets())
    selector = NullRateSelector(null_rate_config_factory())

    assert selector.select(context, []) == []


@pytest.mark.parametrize("train", [{"a": 1}, [1, 2, 3], "not-a-frame"])
def test_select_unsupported_train_type_raises_execution_error(
    stage_context_factory: Callable[..., StageContext],
    null_rate_config_factory: Callable[..., NullRateConfig],
    train: Any,
) -> None:
    context = stage_context_factory(train=train, candidates=["feature"])
    selector = NullRateSelector(null_rate_config_factory())

    with pytest.raises(ExecutionError, match="null_rate: unsupported train split type") as exc_info:
        selector.select(context, ["feature"])

    message = str(exc_info.value)
    assert repr(type(train)) in message
    assert "Expected a Spark DataFrame or pandas DataFrame." in message


@pytest.mark.parametrize(
    ("rate", "should_drop"),
    [
        (0.49, False),
        (0.499, False),
        (0.5, False),
        (0.501, True),
        (0.51, True),
    ],
)
def test_select_threshold_boundary_filtering(
    monkeypatch: pytest.MonkeyPatch,
    stage_context_factory: Callable[..., StageContext],
    null_rate_config_factory: Callable[..., NullRateConfig],
    rate: float,
    should_drop: bool,
) -> None:
    frame = pd.DataFrame({"feature": [1.0, 2.0]})
    context = stage_context_factory(train=frame, candidates=["feature"])
    selector = NullRateSelector(null_rate_config_factory(threshold=THRESHOLD))
    monkeypatch.setattr(
        NullRateSelector,
        "_compute_null_rates_pandas",
        lambda self, _frame, columns: {columns[0]: rate},
    )

    decisions = selector.select(context, ["feature"])

    if should_drop:
        assert decisions == [_drop_decision("feature", rate, THRESHOLD)]
    else:
        assert decisions == []


def test_select_pandas_drops_only_rates_strictly_above_threshold(
    stage_context_factory: Callable[..., StageContext],
    null_rate_config_factory: Callable[..., NullRateConfig],
) -> None:
    frame = pd.DataFrame(
        {
            "high_null": [1.0, None, math.nan, None],
            "at_threshold": [1.0, 2.0, None, math.nan],
            "complete": [1.0, 2.0, 3.0, 4.0],
        },
    )
    candidates = list(frame.columns)
    context = stage_context_factory(train=frame, candidates=candidates)
    selector = NullRateSelector(null_rate_config_factory(threshold=THRESHOLD))

    decisions = selector.select(context, candidates)

    assert decisions == [_drop_decision("high_null", 0.75, THRESHOLD)]


def test_select_empty_pandas_frame_with_columns_produces_no_decisions(
    stage_context_factory: Callable[..., StageContext],
    null_rate_config_factory: Callable[..., NullRateConfig],
) -> None:
    frame = pd.DataFrame({"feature": pd.Series(dtype="float64")})
    context = stage_context_factory(train=frame, candidates=["feature"])
    selector = NullRateSelector(null_rate_config_factory(threshold=0.0))

    assert selector.select(context, ["feature"]) == []
    assert selector.select(context, []) == []


def test_select_debug_telemetry_pandas(
    monkeypatch: pytest.MonkeyPatch,
    stage_context_factory: Callable[..., StageContext],
    null_rate_config_factory: Callable[..., NullRateConfig],
    debug_emit_spy: Mock,
) -> None:
    frame = pd.DataFrame(
        {
            "low": [None, *list(range(6))],
            "mid": [None, None, *list(range(5))],
            "high": [None, None, None, *list(range(4))],
        },
    )
    candidates = list(frame.columns)
    context = stage_context_factory(train=frame, candidates=candidates)
    threshold = 0.3
    selector = NullRateSelector(null_rate_config_factory(threshold=threshold))
    monkeypatch.setattr(null_rate_module, "verbose_enabled", lambda _context, method: method == "null_rate")

    selector.select(context, candidates)

    rates = [1 / 7, 2 / 7, 3 / 7]
    debug_emit_spy.assert_called_once_with(
        context,
        "null_rate",
        "stats",
        backend="pandas",
        threshold=threshold,
        n_evaluated=3,
        n_above_threshold=1,
        null_rate_min=1 / 7,
        null_rate_max=3 / 7,
        null_rate_mean=round(sum(rates) / len(rates), 6),
        n_rows=len(frame),
    )


def test_select_debug_disabled_does_not_emit(
    monkeypatch: pytest.MonkeyPatch,
    stage_context_factory: Callable[..., StageContext],
    null_rate_config_factory: Callable[..., NullRateConfig],
    debug_emit_spy: Mock,
    pandas_sample_df: pd.DataFrame,
) -> None:
    candidates = ["float_half_nan", "float_none_missing"]
    context = stage_context_factory(train=pandas_sample_df, candidates=candidates, debug_enabled=False)
    selector = NullRateSelector(null_rate_config_factory())
    monkeypatch.setattr(null_rate_module, "verbose_enabled", lambda _context, _method: False)

    selector.select(context, candidates)

    debug_emit_spy.assert_not_called()


def test_select_debug_empty_null_rates_does_not_emit(
    monkeypatch: pytest.MonkeyPatch,
    stage_context_factory: Callable[..., StageContext],
    null_rate_config_factory: Callable[..., NullRateConfig],
    debug_emit_spy: Mock,
) -> None:
    frame = pd.DataFrame({"feature": [1.0]})
    context = stage_context_factory(train=frame, candidates=["feature"])
    selector = NullRateSelector(null_rate_config_factory())
    monkeypatch.setattr(null_rate_module, "verbose_enabled", lambda _context, _method: True)
    monkeypatch.setattr(NullRateSelector, "_compute_null_rates_pandas", lambda self, _frame, _columns: {})

    assert selector.select(context, ["feature"]) == []
    debug_emit_spy.assert_not_called()


def test_select_routes_pandas_dataframe_to_pandas_backend(
    monkeypatch: pytest.MonkeyPatch,
    stage_context_factory: Callable[..., StageContext],
    null_rate_config_factory: Callable[..., NullRateConfig],
) -> None:
    frame = pd.DataFrame({"feature": [1.0]})
    context = stage_context_factory(train=frame, candidates=["feature"])
    selector = NullRateSelector(null_rate_config_factory())
    spark_compute = Mock(side_effect=AssertionError("spark backend must not run for pandas input"))
    monkeypatch.setattr(NullRateSelector, "_compute_null_rates_spark", spark_compute)

    assert selector.select(context, ["feature"]) == []
    spark_compute.assert_not_called()


def test_compute_null_rates_pandas_missing_columns_raises_execution_error(
    pandas_sample_df: pd.DataFrame,
    null_rate_config_factory: Callable[..., NullRateConfig],
) -> None:
    selector = NullRateSelector(null_rate_config_factory())

    with pytest.raises(ExecutionError, match="columns missing from train DataFrame") as exc_info:
        selector._compute_null_rates_pandas(pandas_sample_df, ["float_half_nan", "absent_a", "absent_b"])

    message = str(exc_info.value)
    assert "absent_a" in message
    assert "absent_b" in message
    assert "float_half_nan" not in message


def test_compute_null_rates_pandas_empty_dataframe(
    null_rate_config_factory: Callable[..., NullRateConfig],
) -> None:
    frame = pd.DataFrame(
        {
            "a": pd.Series(dtype="float64"),
            "b": pd.Series(dtype="object"),
        },
    )
    selector = NullRateSelector(null_rate_config_factory())

    assert selector._compute_null_rates_pandas(frame, ["a", "b"]) == {"a": 0.0, "b": 0.0}


def test_compute_null_rates_pandas_empty_dataframe_missing_columns_still_raises(
    null_rate_config_factory: Callable[..., NullRateConfig],
) -> None:
    frame = pd.DataFrame({"present": pd.Series(dtype="float64")})
    selector = NullRateSelector(null_rate_config_factory())

    with pytest.raises(ExecutionError, match="columns missing from train DataFrame"):
        selector._compute_null_rates_pandas(frame, ["present", "missing"])


def test_compute_null_rates_pandas_mixed_types_and_nulls(
    pandas_sample_df: pd.DataFrame,
    null_rate_config_factory: Callable[..., NullRateConfig],
) -> None:
    selector = NullRateSelector(null_rate_config_factory())

    rates = selector._compute_null_rates_pandas(pandas_sample_df, list(pandas_sample_df.columns))

    assert rates["float_all_nan"] == 1.0
    assert rates["float_none_missing"] == 0.0
    assert rates["float_half_nan"] == 0.5
    assert rates["object_half_none"] == 0.5
    assert rates["int_half_na"] == 0.5
    assert rates["category_half_null"] == 0.5
    assert rates["datetime_half_nat"] == 0.5
    assert all(isinstance(value, float) for value in rates.values())


def test_select_pandas_missing_candidate_has_actionable_error(
    stage_context_factory: Callable[..., StageContext],
    null_rate_config_factory: Callable[..., NullRateConfig],
) -> None:
    context = stage_context_factory(train=pd.DataFrame({"feature": [1]}), candidates=["missing"])
    selector = NullRateSelector(null_rate_config_factory())

    with pytest.raises(ExecutionError, match=r"columns missing.*missing"):
        selector.select(context, ["missing"])


def test_root_cause_extraction() -> None:
    java_error = JavaLikeError("org.apache.spark.SparkException: job aborted\nCaused by: boom")
    assert _root_cause(java_error) == "org.apache.spark.SparkException: job aborted"

    try:
        try:
            root_message = "root line\nsecond line"
            raise ValueError(root_message)
        except ValueError as inner:
            wrapper_message = "wrapper\nignored"
            raise RuntimeError(wrapper_message) from inner
    except RuntimeError as wrapped:
        assert _root_cause(wrapped) == "root line"

    assert _root_cause(Exception("single line")) == "single line"
    assert _root_cause(Exception("first\nsecond")) == "first"


def test_root_cause_prefers_java_exception_over_cause() -> None:
    try:
        try:
            cause_message = "cause first\ncause second"
            raise ValueError(cause_message)
        except ValueError as inner:
            java_message = "java first\njava second"
            raise JavaLikeError(java_message) from inner
    except JavaLikeError as error:
        assert _root_cause(error) == "java first"


def test_is_spark_dataframe(spark: Any) -> None:
    frame = spark.createDataFrame([(1,)], ["a"])
    assert _is_spark_dataframe(frame) is True
    assert _is_spark_dataframe(pd.DataFrame({"a": [1]})) is False
    assert _is_spark_dataframe("pyspark") is False
    assert _is_spark_dataframe({"select": True, "agg": True}) is False

    class ThirdPartyWithSparkApi:
        def select(self: ThirdPartyWithSparkApi) -> ThirdPartyWithSparkApi:
            return self

        def agg(self: ThirdPartyWithSparkApi) -> ThirdPartyWithSparkApi:
            return self

    ThirdPartyWithSparkApi.__module__ = "custom.backend"
    assert _is_spark_dataframe(ThirdPartyWithSparkApi()) is False


def test_quoted_col_quotes_dots_spaces_and_strips_backticks(spark: Any) -> None:
    del spark
    from pyspark.sql import Column
    from pyspark.sql import functions as F  # noqa: N812

    dotted = _quoted_col("foo.bar")
    spaced = _quoted_col("text feature")
    stripped = _quoted_col("`already_quoted`")
    assert isinstance(dotted, Column)
    assert isinstance(spaced, Column)
    assert isinstance(stripped, Column)
    assert str(dotted) == str(F.col("`foo.bar`"))
    assert str(spaced) == str(F.col("`text feature`"))
    assert str(stripped) == str(F.col("`already_quoted`"))


def test_select_spark_counts_null_and_nan_on_dotted_and_spaced_names(
    spark: Any,
    null_rate_config_factory: Callable[..., NullRateConfig],
) -> None:
    frame = spark.createDataFrame(
        [
            (1.0, "a", 1),
            (math.nan, None, 2),
            (None, None, 3),
        ],
        ["foo.bar", "text feature", "complete"],
    )
    candidates = list(frame.columns)
    context = _spark_context(
        spark,
        frame,
        candidates,
        categorical=("text feature",),
        continuous=("foo.bar", "complete"),
    )

    decisions = NullRateSelector(null_rate_config_factory(threshold=THRESHOLD)).select(context, candidates)

    by_feature = {decision.feature: decision for decision in decisions}
    assert set(by_feature) == {"foo.bar", "text feature"}
    assert by_feature["foo.bar"] == _drop_decision("foo.bar", pytest.approx(2 / 3), THRESHOLD)
    assert by_feature["text feature"] == _drop_decision("text feature", pytest.approx(2 / 3), THRESHOLD)
    assert "complete" not in by_feature


def test_select_spark_empty_dataframe_keeps_all_candidates(
    spark: Any,
    null_rate_config_factory: Callable[..., NullRateConfig],
) -> None:
    from pyspark.sql.types import DoubleType as SparkDoubleType
    from pyspark.sql.types import StringType as SparkStringType
    from pyspark.sql.types import StructField, StructType

    schema = StructType(
        [
            StructField("dbl", SparkDoubleType(), True),
            StructField("txt", SparkStringType(), True),
        ],
    )
    frame = spark.createDataFrame([], schema)
    candidates = ["dbl", "txt"]
    context = _spark_context(
        spark,
        frame,
        candidates,
        categorical=("txt",),
        continuous=("dbl",),
    )
    selector = NullRateSelector(null_rate_config_factory(threshold=0.0))

    assert selector.select(context, candidates) == []
    assert selector._compute_null_rates_spark(frame, candidates) == {"dbl": 0.0, "txt": 0.0}


def test_select_spark_typed_columns_count_null_and_nan(
    spark: Any,
    null_rate_config_factory: Callable[..., NullRateConfig],
) -> None:
    from pyspark.sql.types import BooleanType as SparkBooleanType
    from pyspark.sql.types import DoubleType as SparkDoubleType
    from pyspark.sql.types import FloatType as SparkFloatType
    from pyspark.sql.types import IntegerType as SparkIntegerType
    from pyspark.sql.types import LongType as SparkLongType
    from pyspark.sql.types import StringType as SparkStringType
    from pyspark.sql.types import StructField, StructType
    from pyspark.sql.types import TimestampType as SparkTimestampType

    schema = StructType(
        [
            StructField("dbl", SparkDoubleType(), True),
            StructField("flt", SparkFloatType(), True),
            StructField("txt", SparkStringType(), True),
            StructField("num", SparkIntegerType(), True),
            StructField("lng", SparkLongType(), True),
            StructField("flag", SparkBooleanType(), True),
            StructField("ts", SparkTimestampType(), True),
        ],
    )
    stamp = datetime(2020, 1, 1)
    frame = spark.createDataFrame(
        [
            (1.0, float("1.0"), "a", 1, 1, True, stamp),
            (math.nan, float("nan"), None, None, None, None, None),
            (None, None, None, 3, 3, False, stamp),
        ],
        schema,
    )
    candidates = [field.name for field in schema.fields]
    context = _spark_context(
        spark,
        frame,
        candidates,
        categorical=("txt", "flag"),
        continuous=("dbl", "flt", "num", "lng"),
    )

    rates = NullRateSelector(null_rate_config_factory())._compute_null_rates_spark(frame, candidates)
    decisions = NullRateSelector(null_rate_config_factory(threshold=THRESHOLD)).select(context, candidates)
    by_feature = {decision.feature: decision.value for decision in decisions}

    assert rates["dbl"] == pytest.approx(2 / 3)
    assert rates["flt"] == pytest.approx(2 / 3)
    assert rates["txt"] == pytest.approx(2 / 3)
    assert rates["num"] == pytest.approx(1 / 3)
    assert rates["lng"] == pytest.approx(1 / 3)
    assert rates["flag"] == pytest.approx(1 / 3)
    assert rates["ts"] == pytest.approx(1 / 3)
    assert by_feature["dbl"] == pytest.approx(2 / 3)
    assert by_feature["flt"] == pytest.approx(2 / 3)
    assert by_feature["txt"] == pytest.approx(2 / 3)
    assert "num" not in by_feature
    assert "lng" not in by_feature
    assert "flag" not in by_feature
    assert "ts" not in by_feature


def test_select_spark_rates_match_pandas_on_the_same_rows(
    spark: Any,
    null_rate_config_factory: Callable[..., NullRateConfig],
) -> None:
    pandas_frame = pd.DataFrame(
        {
            "dbl": [1.0, math.nan, None],
            "txt": ["a", None, None],
            "num": [1, 2, None],
        },
    )
    spark_frame = spark.createDataFrame(pandas_frame)
    selector = NullRateSelector(null_rate_config_factory())
    columns = list(pandas_frame.columns)

    pandas_rates = selector._compute_null_rates_pandas(pandas_frame, columns)
    spark_rates = selector._compute_null_rates_spark(spark_frame, columns)

    assert spark_rates.keys() == pandas_rates.keys()
    for name in columns:
        assert spark_rates[name] == pytest.approx(pandas_rates[name])


def test_select_routes_spark_dataframe_to_spark_backend(
    spark: Any,
    monkeypatch: pytest.MonkeyPatch,
    null_rate_config_factory: Callable[..., NullRateConfig],
) -> None:
    frame = spark.createDataFrame([(1.0,), (None,)], ["feature"])
    context = _spark_context(spark, frame, ["feature"], continuous=("feature",))
    selector = NullRateSelector(null_rate_config_factory(threshold=0.2))
    pandas_compute = Mock(side_effect=AssertionError("pandas backend must not run for Spark input"))
    monkeypatch.setattr(NullRateSelector, "_compute_null_rates_pandas", pandas_compute)

    decisions = selector.select(context, ["feature"])

    pandas_compute.assert_not_called()
    assert decisions == [_drop_decision("feature", 0.5, 0.2)]


def test_compute_null_rates_spark_missing_columns_raises_execution_error(
    spark: Any,
    null_rate_config_factory: Callable[..., NullRateConfig],
) -> None:
    frame = spark.createDataFrame([("a",)], ["present"])
    selector = NullRateSelector(null_rate_config_factory())

    with pytest.raises(ExecutionError, match="columns missing from train schema") as exc_info:
        selector._compute_null_rates_spark(frame, ["present", "absent_a", "absent_b"])

    message = str(exc_info.value)
    assert "absent_a" in message
    assert "absent_b" in message


def test_select_debug_telemetry_spark(
    spark: Any,
    monkeypatch: pytest.MonkeyPatch,
    null_rate_config_factory: Callable[..., NullRateConfig],
    debug_emit_spy: Mock,
) -> None:
    frame = spark.createDataFrame(
        [
            ("a", 1.0),
            ("b", None),
            ("c", None),
            ("d", None),
            ("e", None),
            ("f", None),
            ("g", None),
            ("h", None),
            ("i", None),
            (None, None),
        ],
        ["kept", "dropped"],
    )
    candidates = ["kept", "dropped"]
    context = _spark_context(
        spark,
        frame,
        candidates,
        categorical=("kept",),
        continuous=("dropped",),
    )
    threshold = THRESHOLD
    selector = NullRateSelector(null_rate_config_factory(threshold=threshold))
    monkeypatch.setattr(null_rate_module, "verbose_enabled", lambda _context, method: method == "null_rate")

    decisions = selector.select(context, candidates)

    assert decisions == [_drop_decision("dropped", 0.9, threshold)]
    debug_emit_spy.assert_called_once_with(
        context,
        "null_rate",
        "stats",
        backend="spark",
        threshold=threshold,
        n_evaluated=2,
        n_above_threshold=1,
        null_rate_min=0.1,
        null_rate_max=0.9,
        null_rate_mean=round((0.1 + 0.9) / 2, 6),
        n_rows=None,
    )


def test_compute_null_rates_spark_aggregation_failure_formats_root_cause(
    spark: Any,
    monkeypatch: pytest.MonkeyPatch,
    null_rate_config_factory: Callable[..., NullRateConfig],
) -> None:
    from pyspark.sql import DataFrame as SparkDataFrame

    java_error = JavaLikeError("org.apache.spark.SparkException: job aborted\nCaused by: boom")
    frame = spark.createDataFrame([("a",)], ["feature"])

    def _raise_java(*_args: Any, **_kwargs: Any) -> Any:
        raise java_error

    monkeypatch.setattr(SparkDataFrame, "collect", _raise_java)
    selector = NullRateSelector(null_rate_config_factory())

    with pytest.raises(ExecutionError, match=r"Spark aggregation failed.*Root cause:") as exc_info:
        selector._compute_null_rates_spark(frame, ["feature"])

    message = str(exc_info.value)
    assert "org.apache.spark.SparkException: job aborted" in message
    assert "Caused by: boom" not in message
    assert exc_info.value.__cause__ is java_error


def test_compute_null_rates_spark_aggregation_failure_uses_cause_when_no_java_exception(
    spark: Any,
    monkeypatch: pytest.MonkeyPatch,
    null_rate_config_factory: Callable[..., NullRateConfig],
) -> None:
    from pyspark.sql import DataFrame as SparkDataFrame

    try:
        try:
            root_message = "executor died\nstack"
            raise ValueError(root_message)
        except ValueError as inner:
            wrapper_message = "agg wrapper"
            raise RuntimeError(wrapper_message) from inner
    except RuntimeError as error:
        collect_error = error

    frame = spark.createDataFrame([(1,)], ["feature"])

    def _raise_wrapped(*_args: Any, **_kwargs: Any) -> Any:
        raise collect_error

    monkeypatch.setattr(SparkDataFrame, "collect", _raise_wrapped)
    selector = NullRateSelector(null_rate_config_factory())

    with pytest.raises(ExecutionError, match="Root cause: executor died") as exc_info:
        selector._compute_null_rates_spark(frame, ["feature"])

    assert "stack" not in str(exc_info.value)
