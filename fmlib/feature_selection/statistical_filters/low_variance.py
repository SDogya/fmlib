"""Low-variance statistical filter for continuous features."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd
from pandas.api.types import is_numeric_dtype

from fmlib.feature_selection.base import FeatureDecision, StageContext
from fmlib.feature_selection.config import LowVarianceConfig
from fmlib.feature_selection.exceptions import BackendError, ExecutionError
from fmlib.feature_selection.utils.verbose import emit as verbose_emit
from fmlib.feature_selection.utils.verbose import enabled as verbose_enabled

_NUMERIC_SPARK_TYPE_NAMES = frozenset(
    {
        "ByteType",
        "ShortType",
        "IntegerType",
        "LongType",
        "FloatType",
        "DoubleType",
        "DecimalType",
    },
)


class LowVarianceSelector:
    """Exclude continuous features with low variance after scaling.

    Scaling:

    - ``standard`` gives every non-constant feature a scaled variance of 1;
    - ``minmax`` divides variance by the squared value range;
    - ``robust`` divides variance by the squared interquartile range.

    Features with undefined or zero variance, or a degenerate scaling range, are
    always dropped.

    Args:
        config: Low-variance filter settings.
    """

    method_name = "low_variance"
    stage_name = "statistics"

    def __init__(self: LowVarianceSelector, config: LowVarianceConfig) -> None:
        self.config = config

    def select(
        self: LowVarianceSelector,
        context: StageContext,
        candidates: Sequence[str],
    ) -> list[FeatureDecision]:
        """Compute scaled variances and return decisions for low-variance features."""
        continuous = set(context.schema.continuous)
        columns = [column for column in candidates if column in continuous]
        if not columns:
            return []
        metrics = self.compute(context, list(candidates))
        return self.apply(metrics, candidates, context)

    def compute(
        self: LowVarianceSelector,
        context: StageContext,
        candidates: Sequence[str],
    ) -> dict[str, Any]:
        """Return scaled variances keyed by continuous feature name."""
        continuous = set(context.schema.continuous)
        columns = [column for column in candidates if column in continuous]
        if not columns:
            return {"values": {}}

        train = context.datasets["train"]
        if _is_spark_dataframe(train):
            scaled_variances = self._compute_stats_spark(train, columns)
            backend = "spark"
        elif isinstance(train, pd.DataFrame):
            scaled_variances = self._compute_stats_pandas(train, columns)
            backend = "pandas"
        else:
            msg = (
                f"low_variance: unsupported train split type {type(train)!r}. "
                "Expected a Spark DataFrame or pandas DataFrame."
            )
            raise ExecutionError(msg)

        if verbose_enabled(context, self.method_name) and scaled_variances:
            defined = [
                variance
                for variance in scaled_variances.values()
                if variance is not None
            ]
            verbose_emit(
                context,
                self.method_name,
                "stats",
                backend=backend,
                scale_method=self.config.scale_method,
                min_variance=self.config.min_variance,
                n_continuous_in=len(columns),
                n_evaluated=len(scaled_variances),
                n_undefined_variance=sum(
                    1
                    for variance in scaled_variances.values()
                    if variance is None
                ),
                n_below_threshold=sum(
                    1
                    for variance in defined
                    if variance < self.config.min_variance
                ),
                scaled_variance_min=min(defined) if defined else None,
                scaled_variance_max=max(defined) if defined else None,
            )
        return {"values": scaled_variances}

    def apply(
        self: LowVarianceSelector,
        metrics: Mapping[str, Any],
        candidates: Sequence[str],
        context: StageContext,
    ) -> list[FeatureDecision]:
        """Drop remaining continuous features below the variance threshold."""
        del context
        values = metrics.get("values", metrics)
        if not isinstance(values, Mapping):
            return []
        remaining = set(candidates)
        decisions: list[FeatureDecision] = []
        for feature, scaled_variance in values.items():
            if feature not in remaining:
                continue
            if scaled_variance is not None and scaled_variance >= self.config.min_variance:
                continue
            decisions.append(
                FeatureDecision(
                    feature=feature,
                    stage=self.stage_name,
                    method=self.method_name,
                    reason="low_variance",
                    value=0.0 if scaled_variance is None else scaled_variance,
                    threshold=self.config.min_variance,
                    keep=False,
                ),
            )
        return decisions

    def _compute_stats_spark(
        self: LowVarianceSelector,
        train: Any,
        columns: list[str],
    ) -> dict[str, float | None]:
        """Compute scaled variance statistics in one Spark aggregation."""
        try:
            from pyspark.sql import functions as F  # noqa: N812
        except ImportError as exc:
            msg = "low_variance: pyspark is required for Spark DataFrames. Install the spark optional dependency group."
            raise BackendError(msg) from exc

        fields = {field.name: field.dataType for field in train.schema.fields}
        self._validate_spark_columns(fields, columns)

        aliases = {column: f"c{index}" for index, column in enumerate(columns)}
        projected = train.select(*[_quoted_col(column).alias(aliases[column]) for column in columns])
        aggregations = []
        for column in columns:
            alias = aliases[column]
            value = F.col(alias)
            aggregations.append(F.variance(value).alias(f"{alias}__variance"))
            if self.config.scale_method == "minmax":
                aggregations.append(F.min(value).alias(f"{alias}__min"))
                aggregations.append(F.max(value).alias(f"{alias}__max"))
            elif self.config.scale_method == "robust":
                aggregations.append(F.percentile_approx(value, 0.25).alias(f"{alias}__q25"))
                aggregations.append(F.percentile_approx(value, 0.75).alias(f"{alias}__q75"))

        try:
            summary = projected.agg(*aggregations).collect()[0]
        except Exception as exc:
            msg = (
                "low_variance: Spark aggregation failed while computing variance statistics. "
                f"Root cause: {_root_cause(exc)}."
            )
            raise ExecutionError(msg) from exc

        result: dict[str, float | None] = {}
        for column in columns:
            alias = aliases[column]
            result[column] = self._scaled_variance(
                variance=summary[f"{alias}__variance"],
                minimum=summary[f"{alias}__min"] if self.config.scale_method == "minmax" else None,
                maximum=summary[f"{alias}__max"] if self.config.scale_method == "minmax" else None,
                q25=summary[f"{alias}__q25"] if self.config.scale_method == "robust" else None,
                q75=summary[f"{alias}__q75"] if self.config.scale_method == "robust" else None,
            )
        return result

    def _compute_stats_pandas(
        self: LowVarianceSelector,
        frame: pd.DataFrame,
        columns: list[str],
    ) -> dict[str, float | None]:
        """Compute scaled variance statistics for an already-local pandas frame."""
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            msg = f"low_variance: columns missing from train DataFrame: {missing}."
            raise ExecutionError(msg)
        non_numeric = [column for column in columns if not is_numeric_dtype(frame[column])]
        if non_numeric:
            msg = f"low_variance: continuous candidates must be numeric: {non_numeric}."
            raise ExecutionError(msg)

        result: dict[str, float | None] = {}
        for column in columns:
            series = frame[column].dropna()
            result[column] = self._scaled_variance(
                variance=series.var(),
                minimum=series.min() if self.config.scale_method == "minmax" else None,
                maximum=series.max() if self.config.scale_method == "minmax" else None,
                q25=series.quantile(0.25) if self.config.scale_method == "robust" else None,
                q75=series.quantile(0.75) if self.config.scale_method == "robust" else None,
            )
        return result

    def _scaled_variance(
        self: LowVarianceSelector,
        *,
        variance: Any,
        minimum: Any,
        maximum: Any,
        q25: Any,
        q75: Any,
    ) -> float | None:
        """Apply the configured scaling formula to one feature variance."""
        if variance is None or pd.isna(variance) or float(variance) == 0.0:
            return None
        variance_value = float(variance)

        if self.config.scale_method == "standard":
            return 1.0
        if self.config.scale_method == "minmax":
            if minimum is None or maximum is None or minimum == maximum:
                return None
            scale = float(maximum - minimum)
        else:
            if q25 is None or q75 is None or q25 == q75:
                return None
            scale = float(q75 - q25)
        return variance_value / (scale**2)

    @staticmethod
    def _validate_spark_columns(fields: dict[str, Any], columns: list[str]) -> None:
        """Validate that requested Spark columns exist and are numeric."""
        missing = [column for column in columns if column not in fields]
        if missing:
            msg = f"low_variance: columns missing from train schema: {missing}."
            raise ExecutionError(msg)
        non_numeric = [
            f"{column}:{type(fields[column]).__name__}"
            for column in columns
            if type(fields[column]).__name__ not in _NUMERIC_SPARK_TYPE_NAMES
        ]
        if non_numeric:
            msg = f"low_variance: continuous candidates must be numeric: {non_numeric}."
            raise ExecutionError(msg)


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
