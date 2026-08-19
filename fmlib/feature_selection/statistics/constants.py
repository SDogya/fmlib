"""Constant / quasi-constant statistical filter (Spark-native)."""

from __future__ import annotations

import math
from typing import Any, Sequence

import pandas as pd

from fmlib.feature_selection.base import FeatureDecision, StageContext
from fmlib.feature_selection.config import ConstantsConfig
from fmlib.feature_selection.exceptions import BackendError, ExecutionError
from fmlib.feature_selection.utils.verbose import emit as verbose_emit
from fmlib.feature_selection.utils.verbose import enabled as verbose_enabled

# Spark types that cannot be used as countDistinct / groupBy keys for this filter.
_UNSUPPORTED_SPARK_TYPE_NAMES = frozenset({"MapType", "VariantType"})


class ConstantsSelector:
    """Exclude constant and quasi-constant features.

    For each candidate on the train split the selector computes:

    - ``max_frequency`` — share of the most frequent non-null value among non-null
      rows.

    ``n_unique`` is computed only when the optional ``min_unique`` rule is
    configured. Otherwise, constant columns are identified by
    ``max_frequency == 1`` without an expensive ``countDistinct`` aggregation.

    A feature is dropped when any enabled rule fires:

    - ``n_unique <= 1`` → reason ``constant``;
    - ``min_unique`` is set and ``n_unique < min_unique`` → reason ``too_few_unique``;
    - ``max_frequency >= config.max_frequency`` → reason ``quasi_constant``.

    On Spark, candidate columns are aggregated in chunks to avoid oversized
    Catalyst plans and code-generation limits on wide datasets. The selector
    never materialises the train split to the driver; only compact mode and
    count summaries are collected. A pandas path exists solely for already-local
    DataFrames (unit tests / small local runs).

    Args:
        config: Constants filter settings.
    """

    method_name = "constants"
    stage_name = "statistics"

    def __init__(self: ConstantsSelector, config: ConstantsConfig) -> None:
        self.config = config

    def select(
        self: ConstantsSelector,
        context: StageContext,
        candidates: Sequence[str],
    ) -> list[FeatureDecision]:
        """Drop constant and quasi-constant candidates.

        Args:
            context: Shared stage context containing datasets and execution config.
            candidates: Feature names still under consideration.

        Returns:
            Drop decisions with measured ``value`` and the threshold that fired.

        Raises:
            BackendError: When Spark APIs are required but pyspark is missing.
            ExecutionError: When stats cannot be computed for the train split type.
        """
        if not candidates:
            return []

        columns = list(candidates)
        train = context.datasets["train"]
        if _is_spark_dataframe(train):
            stats = self._compute_stats_spark(train, columns)
            backend = "spark"
        elif isinstance(train, pd.DataFrame):
            stats = self._compute_stats_pandas(train, columns)
            backend = "pandas"
        else:
            msg = (
                f"constants: unsupported train split type {type(train)!r}. "
                "Expected a Spark DataFrame or pandas DataFrame. "
                "Spark inputs are aggregated in-cluster; full toPandas materialisation is not used."
            )
            raise ExecutionError(msg)
        if verbose_enabled(context, self.method_name) and stats:
            frequencies = [frequency for _, frequency in stats.values()]
            uniques = [n_unique for n_unique, _ in stats.values()]
            n_chunks = (
                (len(columns) + self.config.chunk_size - 1)
                // self.config.chunk_size
            )
            verbose_emit(
                context,
                self.method_name,
                "stats",
                backend=backend,
                n_evaluated=len(stats),
                n_chunks=n_chunks if backend == "spark" else 1,
                chunk_size=self.config.chunk_size,
                max_frequency_threshold=self.config.max_frequency,
                min_unique=self.config.min_unique,
                n_all_null=sum(1 for n_unique, _ in stats.values() if n_unique == 0),
                n_unique_min=min(uniques),
                n_unique_max=max(uniques),
                max_frequency_min=min(frequencies),
                max_frequency_max=max(frequencies),
            )
        return self._build_decisions(stats)

    def _compute_stats_spark(
        self: ConstantsSelector,
        train: Any,
        columns: list[str],
    ) -> dict[str, tuple[int, float]]:
        """Compute ``(n_unique, max_frequency)`` with Spark aggregations only.

        Args:
            train: Spark DataFrame.
            columns: Candidate column names.

        Returns:
            Mapping of feature name to ``(n_unique, max_frequency)``.

        Raises:
            BackendError: If ``pyspark`` is not installed.
            ExecutionError: If Spark aggregation fails or column types are unsupported.
        """
        try:
            from pyspark.sql import functions as F  # noqa: N812
        except ImportError as exc:
            msg = (
                "constants: pyspark is required for Spark DataFrames. "
                "Install the spark optional dependency group."
            )
            raise BackendError(msg) from exc

        self._validate_spark_column_types(train, columns)

        chunks = [
            columns[start : start + self.config.chunk_size]
            for start in range(0, len(columns), self.config.chunk_size)
        ]
        result: dict[str, tuple[int, float]] = {}
        try:
            modes: dict[str, Any] = {}
            for chunk in chunks:
                mode_aggs = [F.mode(F.col(col)).alias(col) for col in chunk]
                modes_row = train.agg(*mode_aggs).first()
                if modes_row:
                    modes.update(modes_row.asDict())

            for chunk in chunks:
                summary_aggs = []
                for col in chunk:
                    mode = modes.get(col)
                    if self.config.min_unique is not None:
                        summary_aggs.append(F.countDistinct(F.col(col)).alias(f"{col}__nunique"))
                    summary_aggs.append(F.count(F.col(col)).alias(f"{col}__non_null"))
                    if mode is None:
                        summary_aggs.append(F.lit(0).alias(f"{col}__mode_count"))
                    elif _is_nan(mode):
                        summary_aggs.append(F.sum(F.isnan(F.col(col)).cast("long")).alias(f"{col}__mode_count"))
                    else:
                        summary_aggs.append(
                            F.sum((F.col(col) == F.lit(mode)).cast("long")).alias(f"{col}__mode_count"),
                        )
                stats_row = train.agg(*summary_aggs).first()
                if not stats_row:
                    continue
                stats = stats_row.asDict()
                for col in chunk:
                    non_null = int(stats.get(f"{col}__non_null") or 0)
                    if non_null == 0:
                        result[col] = (0, 0.0)
                        continue
                    mode_count = int(stats.get(f"{col}__mode_count") or 0)
                    if self.config.min_unique is None:
                        n_unique = 1 if mode_count == non_null else 2
                    else:
                        n_unique = int(stats.get(f"{col}__nunique") or 0)
                    result[col] = (n_unique, float(mode_count / non_null))
        except Exception as exc:
            root = _spark_root_cause(exc)
            msg = (
                "constants: Spark aggregation failed while computing mode frequency. "
                f"Root cause: {root}. "
                "Check that candidates are scalar columns (not MapType) and names are valid."
            )
            raise ExecutionError(msg) from exc

        return result

    def _validate_spark_column_types(
        self: ConstantsSelector,
        train: Any,
        columns: list[str],
    ) -> None:
        """Reject Spark column types that break countDistinct/groupBy.

        Args:
            train: Spark DataFrame.
            columns: Candidate column names.

        Raises:
            ExecutionError: When one or more columns have unsupported types.
        """
        fields = {field.name: field.dataType for field in train.schema.fields}
        missing = [col for col in columns if col not in fields]
        if missing:
            msg = f"constants: columns missing from train schema: {missing}."
            raise ExecutionError(msg)

        unsupported: list[str] = []
        for col in columns:
            type_name = type(fields[col]).__name__
            if type_name in _UNSUPPORTED_SPARK_TYPE_NAMES:
                unsupported.append(f"{col}:{type_name}")
        if unsupported:
            msg = (
                "constants: Spark countDistinct/groupBy cannot run on map/complex keys. "
                f"Unsupported candidates: {unsupported}. "
                "Remove them from FeatureSchema categorical/continuous or cast to a scalar type."
            )
            raise ExecutionError(msg)

    def _compute_stats_pandas(
        self: ConstantsSelector,
        df: pd.DataFrame,
        columns: list[str],
    ) -> dict[str, tuple[int, float]]:
        """Compute ``(n_unique, max_frequency)`` for an already-local pandas frame.

        Args:
            df: Local pandas DataFrame (not produced via Spark ``toPandas`` here).
            columns: Candidate column names.

        Returns:
            Mapping of feature name to ``(n_unique, max_frequency)``.
        """
        result: dict[str, tuple[int, float]] = {}
        for col in columns:
            series = df[col]
            non_null = series.dropna()
            n_unique = int(non_null.nunique(dropna=True))
            if len(non_null) == 0:
                result[col] = (0, 0.0)
                continue
            max_count = int(non_null.value_counts(dropna=True).iloc[0])
            max_frequency = float(max_count / len(non_null))
            result[col] = (n_unique, max_frequency)
        return result

    def _build_decisions(
        self: ConstantsSelector,
        stats: dict[str, tuple[int, float]],
    ) -> list[FeatureDecision]:
        """Turn per-feature stats into drop decisions.

        Args:
            stats: Mapping of feature → ``(n_unique, max_frequency)``.

        Returns:
            Drop decisions for features that violate config thresholds.
        """
        decisions: list[FeatureDecision] = []
        min_unique = self.config.min_unique
        max_frequency_threshold = self.config.max_frequency

        for feature, (n_unique, max_frequency) in stats.items():
            decision = self._decide_feature(
                feature=feature,
                n_unique=n_unique,
                max_frequency=max_frequency,
                min_unique=min_unique,
                max_frequency_threshold=max_frequency_threshold,
            )
            if decision is not None:
                decisions.append(decision)
        return decisions

    def _decide_feature(
        self: ConstantsSelector,
        *,
        feature: str,
        n_unique: int,
        max_frequency: float,
        min_unique: int | None,
        max_frequency_threshold: float,
    ) -> FeatureDecision | None:
        """Return a drop decision for one feature, or ``None`` to keep it."""
        if n_unique == 0:
            # All-null columns are left to null-rate filtering.
            return None
        if n_unique <= 1:
            return FeatureDecision(
                feature=feature,
                stage=self.stage_name,
                method=self.method_name,
                reason="constant",
                value=float(n_unique),
                threshold=float(min_unique) if min_unique is not None else 1.0,
                keep=False,
            )
        if min_unique is not None and n_unique < min_unique:
            return FeatureDecision(
                feature=feature,
                stage=self.stage_name,
                method=self.method_name,
                reason="too_few_unique",
                value=float(n_unique),
                threshold=float(min_unique),
                keep=False,
            )
        if max_frequency >= max_frequency_threshold:
            return FeatureDecision(
                feature=feature,
                stage=self.stage_name,
                method=self.method_name,
                reason="quasi_constant",
                value=round(max_frequency, 6),
                threshold=max_frequency_threshold,
                keep=False,
            )
        return None


def _spark_root_cause(exc: BaseException) -> str:
    """Extract a short root-cause string from Py4J / Spark errors."""
    java_exc = getattr(exc, "java_exception", None)
    if java_exc is not None:
        return str(java_exc).splitlines()[0]
    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        return str(cause).splitlines()[0]
    return str(exc).splitlines()[0]


def _is_nan(value: Any) -> bool:
    """Return whether a Spark mode value is NaN."""
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def _is_spark_dataframe(data: Any) -> bool:
    """Return True when ``data`` looks like a pyspark.sql.DataFrame."""
    module_name = type(data).__module__
    return module_name.startswith("pyspark") and hasattr(data, "groupBy") and hasattr(data, "agg")
