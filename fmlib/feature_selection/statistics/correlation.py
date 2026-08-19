"""Pairwise correlation filter for continuous features."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from fmlib.feature_selection.base import FeatureDecision, StageContext
from fmlib.feature_selection.config import CorrelationConfig
from fmlib.feature_selection.debug import emit as debug_emit, enabled as debug_enabled
from fmlib.feature_selection.exceptions import BackendError, ExecutionError

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


class CorrelationSelector:
    """Exclude one feature from each highly correlated continuous pair.

    Pairwise correlation on continuous candidates:

    1. take ``candidates ∩ schema.continuous`` (not Spark dtype discovery);
    2. bound the train split to ``min(config.max_rows, execution.max_local_rows)``;
    3. drop all-null columns (Spark ``Imputer`` cannot fit them);
    4. fill remaining nulls with the column median;
    5. compute the pairwise correlation matrix;
    6. for every pair with ``abs(corr) > threshold``, drop exactly one feature.

    Spark inputs use ``Imputer`` → ``VectorAssembler`` → ``pyspark.ml.stat.Correlation``
    and never call ``toPandas``. Already-local pandas inputs use the equivalent
    median-fill + ``DataFrame.corr`` path for unit tests and small local runs.

    Which feature is dropped from a correlated pair is controlled by
    ``config.tie_break``:

    - ``original_order`` (default): drop the later candidate;
    - ``null_rate``: drop the feature with the higher null rate, falling back to
      ``original_order`` on ties.

    Args:
        config: Correlation filter settings.
    """

    method_name = "correlation"
    stage_name = "statistics"

    def __init__(self: CorrelationSelector, config: CorrelationConfig) -> None:
        self.config = config

    def select(
        self: CorrelationSelector,
        context: StageContext,
        candidates: Sequence[str],
    ) -> list[FeatureDecision]:
        """Compute pairwise correlations and drop one member of each correlated pair.

        Args:
            context: Shared stage context containing datasets, schema and execution
                config.
            candidates: Feature names still under consideration.

        Returns:
            Drop decisions with ``reason="high_correlation"`` and the measured
            absolute correlation as ``value``.

        Raises:
            BackendError: When Spark APIs are required but pyspark is missing.
            ExecutionError: When statistics cannot be computed for the train split.
        """
        continuous = set(context.schema.continuous)
        columns = [column for column in candidates if column in continuous]
        if len(columns) < 2:
            return []

        max_rows = min(self.config.max_rows, context.config.execution.max_local_rows)
        train = context.datasets["train"]
        if _is_spark_dataframe(train):
            corr_matrix, null_rates, evaluated = self._compute_stats_spark(train, columns, max_rows)
        elif isinstance(train, pd.DataFrame):
            corr_matrix, null_rates, evaluated = self._compute_stats_pandas(train, columns, max_rows)
        else:
            msg = f"correlation: unsupported train split type {type(train)!r}. Expected a Spark DataFrame or pandas DataFrame."
            raise ExecutionError(msg)

        if len(evaluated) < 2:
            return []

        if debug_enabled(context, self.method_name):
            off_diag = np.abs(corr_matrix.copy())
            np.fill_diagonal(off_diag, 0.0)
            n_pairs = int(np.sum(off_diag > self.config.threshold) // 2)
            finite = off_diag[np.isfinite(off_diag)]
            debug_emit(
                context,
                self.method_name,
                "matrix",
                backend="spark" if _is_spark_dataframe(train) else "pandas",
                correlation_method=self.config.method,
                tie_break=self.config.tie_break,
                threshold=self.config.threshold,
                max_rows=max_rows,
                n_continuous_in=len(columns),
                n_evaluated=len(evaluated),
                n_skipped=len(columns) - len(evaluated),
                matrix_rows=int(corr_matrix.shape[0]),
                matrix_cols=int(corr_matrix.shape[1]),
                n_pairs_above_threshold=n_pairs,
                max_abs_corr=float(np.max(finite)) if finite.size else None,
            )

        to_drop = self._resolve_drops(corr_matrix, null_rates, evaluated)
        return [
            FeatureDecision(
                feature=feature,
                stage=self.stage_name,
                method=self.method_name,
                reason="high_correlation",
                value=round(corr_value, 6),
                threshold=self.config.threshold,
                keep=False,
            )
            for feature, corr_value in to_drop.items()
        ]

    def _compute_stats_spark(
        self: CorrelationSelector,
        train: Any,
        columns: list[str],
        max_rows: int,
    ) -> tuple[np.ndarray, dict[str, float], list[str]]:
        """Compute correlation stats with Spark ML on a bounded row projection.

        Filter all-null columns, median-impute, assemble a feature vector,
        then call ``Correlation.corr``.
        """
        try:
            from pyspark.ml.feature import Imputer, VectorAssembler
            from pyspark.ml.stat import Correlation as SparkCorrelation
            from pyspark.sql import functions as F  # noqa: N812
        except ImportError as exc:
            msg = "correlation: pyspark is required for Spark DataFrames. Install the spark optional dependency group."
            raise BackendError(msg) from exc

        fields = {field.name: field.dataType for field in train.schema.fields}
        self._validate_spark_columns(fields, columns)

        aliases = {column: f"c{index}" for index, column in enumerate(columns)}
        projected = train.select(
            *[_quoted_col(column).alias(aliases[column]) for column in columns]).limit(max_rows).cache()

        try:
            aggregations = [F.count("*").alias("__total__")]
            for column in columns:
                alias = aliases[column]
                value = F.col(alias)
                # Match null_rate / Imputer semantics: NaN counts as missing for floats.
                if type(fields[column]).__name__ in {"DoubleType", "FloatType"}:
                    non_null = F.sum((~(value.isNull() | F.isnan(value))).cast("long"))
                else:
                    non_null = F.count(value)
                aggregations.append(non_null.alias(f"{alias}__non_null"))

            summary = projected.agg(*aggregations).collect()[0]
            total_rows = int(summary["__total__"])
            if total_rows == 0:
                return np.empty((0, 0)), dict.fromkeys(columns, 0.0), []

            evaluated: list[str] = []
            null_rates: dict[str, float] = {}
            for column in columns:
                non_null = int(summary[f"{aliases[column]}__non_null"] or 0)
                if non_null == 0:
                    continue
                evaluated.append(column)
                null_rates[column] = (total_rows - non_null) / total_rows

            if len(evaluated) < 2:
                return np.empty((0, 0)), null_rates, evaluated

            evaluated_aliases = [aliases[column] for column in evaluated]
            working = projected.select(*evaluated_aliases)
            filled = (
                Imputer(
                    strategy="median",
                    inputCols=evaluated_aliases,
                    outputCols=evaluated_aliases,
                )
                .fit(working)
                .transform(working)
            )
            vectorized = VectorAssembler(
                inputCols=evaluated_aliases,
                outputCol="__features__",
            ).transform(filled)

            try:
                matrix_row = SparkCorrelation.corr(
                    vectorized,
                    "__features__",
                    method=self.config.method,
                ).head()
            except Exception as exc:
                msg = (
                    f"correlation: Spark ML {self.config.method!r} correlation failed "
                    f"on at most {max_rows} rows. Root cause: {_root_cause(exc)}."
                )
                raise ExecutionError(msg) from exc

            matrix = np.asarray(matrix_row[0].toArray(), dtype=float)
            return matrix, null_rates, evaluated
        except ExecutionError:
            raise
        except Exception as exc:
            msg = f"correlation: Spark aggregation failed while preparing correlation inputs. Root cause: {_root_cause(exc)}."
            raise ExecutionError(msg) from exc
        finally:
            projected.unpersist()

    def _compute_stats_pandas(
        self: CorrelationSelector,
        frame: pd.DataFrame,
        columns: list[str],
        max_rows: int,
    ) -> tuple[np.ndarray, dict[str, float], list[str]]:
        """Compute correlation stats for an already-local pandas DataFrame."""
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            msg = f"correlation: columns missing from train DataFrame: {missing}."
            raise ExecutionError(msg)

        bounded = frame.loc[:, columns].head(max_rows)
        if bounded.empty:
            return np.empty((0, 0)), dict.fromkeys(columns, 0.0), []

        numeric = bounded.apply(pd.to_numeric, errors="coerce")
        non_numeric = [column for column in columns if not is_numeric_dtype(numeric[column])]
        if non_numeric:
            msg = f"correlation: continuous candidates must be numeric: {non_numeric}."
            raise ExecutionError(msg)

        evaluated = [column for column in columns if numeric[column].notna().any()]
        null_rates = {column: float(numeric[column].isna().mean()) for column in evaluated}
        if len(evaluated) < 2:
            return np.empty((0, 0)), null_rates, evaluated

        filled = numeric[evaluated].fillna(numeric[evaluated].median())
        try:
            matrix = filled.corr(method=self.config.method).to_numpy(dtype=float)
        except Exception as exc:
            msg = f"correlation: failed to compute {self.config.method!r} correlation matrix: {exc}"
            raise ExecutionError(msg) from exc
        return matrix, null_rates, evaluated

    def _resolve_drops(
        self: CorrelationSelector,
        corr_matrix: np.ndarray,
        null_rates: dict[str, float],
        candidates: list[str],
    ) -> dict[str, float]:
        """Drop one feature from each upper-triangle pair above the threshold.

        Already-dropped features are skipped so each feature yields at most
        one drop decision.
        """
        upper = np.triu(np.abs(corr_matrix), k=1)
        dropped: dict[str, float] = {}

        for i, feat_a in enumerate(candidates):
            if feat_a in dropped:
                continue
            for j in range(i + 1, len(candidates)):
                feat_b = candidates[j]
                if feat_b in dropped:
                    continue
                corr_value = float(upper[i, j])
                if np.isnan(corr_value) or corr_value <= self.config.threshold:
                    continue
                victim = self._pick_victim(feat_a, feat_b, null_rates, candidates)
                dropped[victim] = corr_value
        return dropped

    def _pick_victim(
        self: CorrelationSelector,
        feat_a: str,
        feat_b: str,
        null_rates: dict[str, float],
        candidates: list[str],
    ) -> str:
        """Choose which of two correlated features to drop."""
        if self.config.tie_break == "null_rate":
            rate_a = null_rates.get(feat_a, 0.0)
            rate_b = null_rates.get(feat_b, 0.0)
            if rate_a != rate_b:
                return feat_a if rate_a > rate_b else feat_b

        # Draft / original_order: drop the later feature in candidate order.
        return feat_b if candidates.index(feat_a) < candidates.index(feat_b) else feat_a

    @staticmethod
    def _validate_spark_columns(fields: dict[str, Any], columns: list[str]) -> None:
        """Validate that requested Spark columns exist and are numeric."""
        missing = [column for column in columns if column not in fields]
        if missing:
            msg = f"correlation: columns missing from train schema: {missing}."
            raise ExecutionError(msg)
        non_numeric = [
            f"{column}:{type(fields[column]).__name__}"
            for column in columns
            if type(fields[column]).__name__ not in _NUMERIC_SPARK_TYPE_NAMES
        ]
        if non_numeric:
            msg = (
                f"correlation: FeatureSchema.continuous columns must have numeric Spark types. Invalid columns: {non_numeric}."
            )
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
