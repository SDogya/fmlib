"""Null-rate statistical filter."""

from __future__ import annotations

from typing import Any, Sequence

import pandas as pd

from fmlib.feature_selection.base import FeatureDecision, StageContext
from fmlib.feature_selection.config import NullRateConfig
from fmlib.feature_selection.exceptions import BackendError, ExecutionError
from fmlib.feature_selection.utils.verbose import emit as verbose_emit
from fmlib.feature_selection.utils.verbose import enabled as verbose_enabled


class NullRateSelector:
    """Exclude features whose missing-value share exceeds the configured threshold.

    Spark inputs are processed with aggregate expressions only. In addition to
    nulls, NaN values are treated as missing for float and double columns.

    Args:
        config: Null-rate filter settings.
    """

    method_name = "null_rate"
    stage_name = "statistics"

    def __init__(self: NullRateSelector, config: NullRateConfig) -> None:
        self.config = config

    def select(
        self: NullRateSelector,
        context: StageContext,
        candidates: Sequence[str],
    ) -> list[FeatureDecision]:
        """Compute null rates on the train split and drop high-null candidates.

        Args:
            context: Shared stage context.
            candidates: Current candidate features.

        Returns:
            Drop decisions for features above the configured threshold.

        Raises:
            BackendError: When Spark APIs are required but pyspark is missing.
            ExecutionError: When statistics cannot be computed for the train split.
        """
        if not candidates:
            return []

        columns = list(candidates)
        train = context.datasets["train"]
        if _is_spark_dataframe(train):
            null_rates = self._compute_null_rates_spark(train, columns)
            backend = "spark"
        elif isinstance(train, pd.DataFrame):
            null_rates = self._compute_null_rates_pandas(train, columns)
            backend = "pandas"
        else:
            msg = (
                f"null_rate: unsupported train split type {type(train)!r}. "
                "Expected a Spark DataFrame or pandas DataFrame."
            )
            raise ExecutionError(msg)

        if verbose_enabled(context, self.method_name) and null_rates:
            rates = list(null_rates.values())
            verbose_emit(
                context,
                self.method_name,
                "stats",
                backend=backend,
                threshold=self.config.threshold,
                n_evaluated=len(rates),
                n_above_threshold=sum(
                    1 for rate in rates if rate > self.config.threshold
                ),
                null_rate_min=min(rates),
                null_rate_max=max(rates),
                null_rate_mean=round(sum(rates) / len(rates), 6),
                n_rows=len(train) if backend == "pandas" else None,
            )

        return [
            FeatureDecision(
                feature=feature,
                stage=self.stage_name,
                method=self.method_name,
                reason="high_null_rate",
                value=null_rate,
                threshold=self.config.threshold,
                keep=False,
            )
            for feature, null_rate in null_rates.items()
            if null_rate > self.config.threshold
        ]

    def _compute_null_rates_spark(
        self: NullRateSelector,
        train: Any,
        columns: list[str],
    ) -> dict[str, float]:
        """Compute null rates with a single Spark aggregation."""
        try:
            from pyspark.sql import functions as F  # noqa: N812
        except ImportError as exc:
            msg = "null_rate: pyspark is required for Spark DataFrames. Install the spark optional dependency group."
            raise BackendError(msg) from exc

        fields = {field.name: field.dataType for field in train.schema.fields}
        missing = [column for column in columns if column not in fields]
        if missing:
            msg = f"null_rate: columns missing from train schema: {missing}."
            raise ExecutionError(msg)

        alias_by_column = {column: f"c{index}" for index, column in enumerate(columns)}
        projected = train.select(*[_quoted_col(column).alias(alias_by_column[column]) for column in columns])

        aggregations = []
        for column in columns:
            alias = alias_by_column[column]
            value = F.col(alias)
            if type(fields[column]).__name__ in {"DoubleType", "FloatType"}:
                missing_value = value.isNull() | F.isnan(value)
                aggregations.append(F.sum(missing_value.cast("long")).alias(f"{alias}__missing"))
            else:
                aggregations.append(F.count(value).alias(f"{alias}__non_null"))
        aggregations.append(F.count("*").alias("__total__"))

        try:
            metrics = projected.agg(*aggregations).collect()[0]
        except Exception as exc:
            msg = f"null_rate: Spark aggregation failed while computing missing-value shares. Root cause: {_root_cause(exc)}."
            raise ExecutionError(msg) from exc

        total_rows = int(metrics["__total__"])
        if total_rows == 0:
            return dict.fromkeys(columns, 0.0)

        null_rates: dict[str, float] = {}
        for column in columns:
            alias = alias_by_column[column]
            if type(fields[column]).__name__ in {"DoubleType", "FloatType"}:
                missing_count = int(metrics[f"{alias}__missing"] or 0)
            else:
                non_null_count = int(metrics[f"{alias}__non_null"] or 0)
                missing_count = total_rows - non_null_count
            null_rates[column] = missing_count / total_rows
        return null_rates

    def _compute_null_rates_pandas(
        self: NullRateSelector,
        frame: pd.DataFrame,
        columns: list[str],
    ) -> dict[str, float]:
        """Compute null rates for an already-local pandas DataFrame."""
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            msg = f"null_rate: columns missing from train DataFrame: {missing}."
            raise ExecutionError(msg)
        if frame.empty:
            return dict.fromkeys(columns, 0.0)
        return {column: float(frame[column].isna().mean()) for column in columns}


def _quoted_col(name: str) -> Any:
    """Build a Spark column reference that tolerates dots and spaces in names."""
    from pyspark.sql import functions as F  # noqa: N812

    escaped = name.replace("`", "")
    return F.col(f"`{escaped}`")


def _root_cause(exc: BaseException) -> str:
    """Extract a concise root cause from Spark/Py4J exceptions."""
    java_exc = getattr(exc, "java_exception", None)
    if java_exc is not None:
        return str(java_exc).splitlines()[0]
    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        return str(cause).splitlines()[0]
    return str(exc).splitlines()[0]


def _is_spark_dataframe(data: Any) -> bool:
    """Return whether data looks like a pyspark DataFrame."""
    module_name = type(data).__module__
    return module_name.startswith("pyspark") and hasattr(data, "select") and hasattr(data, "agg")
