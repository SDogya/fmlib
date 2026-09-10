"""Фильтр попарной корреляции для непрерывных признаков."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from fmlib.feature_selection.base import FeatureDecision, StageContext, step_seed
from fmlib.feature_selection.config import CorrelationConfig
from fmlib.feature_selection.exceptions import BackendError, ExecutionError
from fmlib.feature_selection.utils.local_data import sample_frame_rows
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


class CorrelationSelector:
    """Исключает один признак из каждой пары непрерывных признаков с высокой корреляцией.

    Попарная корреляция непрерывных кандидатов:

    1. выбирает ``candidates ∩ schema.continuous`` (без определения типов через Spark);
    2. ограничивает train до ``min(config.max_rows, execution.max_local_rows)`` строк
       с помощью стратификации по целевой переменной (случайной выборки, если ``task_type`` равен
       ``regression``), а не выбирает первые N строк;
    3. удаляет столбцы, состоящие только из пропусков (Spark ``Imputer`` не может на них обучиться);
    4. заполняет оставшиеся пропуски медианой столбца;
    5. вычисляет матрицу попарной корреляции;
    6. для каждой пары с ``abs(corr) > threshold`` исключает ровно один признак.

    Для входных данных Spark используется ``Imputer`` → ``VectorAssembler`` → ``pyspark.ml.stat.Correlation``
    без вызова ``toPandas``. Для уже локальных данных pandas используется аналогичная
    схема заполнения медианой + ``DataFrame.corr`` для модульных тестов и небольших локальных запусков.

    Выбор исключаемого признака из коррелирующей пары регулируется
    ``config.tie_break``:

    - ``original_order`` (по умолчанию): исключается кандидат, стоящий позже;
    - ``null_rate``: исключается признак с большей долей пропусков; при равенстве используется
      ``original_order``.

    Args:
        config: Настройки фильтра корреляции.
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
        """Вычисляет попарные корреляции и исключает один признак из каждой коррелирующей пары.

        Args:
            context: Общий контекст этапа с наборами данных, схемой и конфигурацией
                выполнения.
            candidates: Имена признаков, которые ещё рассматриваются для отбора.

        Returns:
            Решения об исключении с ``reason="high_correlation"`` и измеренным
            модулем корреляции в качестве ``value``.

        Raises:
            BackendError: Если требуется API Spark, но pyspark отсутствует.
            ExecutionError: Если статистики невозможно вычислить для train.
        """
        continuous = set(context.schema.continuous)
        columns = [column for column in candidates if column in continuous]
        if len(columns) < 2:
            return []
        metrics = self.compute(context, list(candidates))
        return self.apply(metrics, candidates, context)

    def compute(
        self: CorrelationSelector,
        context: StageContext,
        candidates: Sequence[str],
    ) -> dict[str, Any]:
        """Возвращает матрицу корреляции, доли пропусков и порядок оценённых признаков."""
        continuous = set(context.schema.continuous)
        columns = [column for column in candidates if column in continuous]
        if len(columns) < 2:
            return {"features": [], "matrix": [], "null_rates": {}}

        max_rows = min(self.config.max_rows, context.config.execution.max_local_rows)
        train, n_rows_in, n_rows_sampled, stratified = self._bound_train(context, max_rows)
        if _is_spark_dataframe(train):
            corr_matrix, null_rates, evaluated = self._compute_stats_spark(train, columns, max_rows)
        elif isinstance(train, pd.DataFrame):
            corr_matrix, null_rates, evaluated = self._compute_stats_pandas(train, columns)
        else:
            msg = f"correlation: unsupported train split type {type(train)!r}. Expected a Spark DataFrame or pandas DataFrame."
            raise ExecutionError(msg)

        if verbose_enabled(context, self.method_name) and len(evaluated) >= 2:
            off_diag = np.abs(corr_matrix.copy())
            np.fill_diagonal(off_diag, 0.0)
            n_pairs = int(np.sum(off_diag > self.config.threshold) // 2)
            finite = off_diag[np.isfinite(off_diag)]
            verbose_emit(
                context,
                self.method_name,
                "matrix",
                backend="spark" if _is_spark_dataframe(train) else "pandas",
                correlation_method=self.config.method,
                tie_break=self.config.tie_break,
                threshold=self.config.threshold,
                max_rows=max_rows,
                stratified=stratified,
                n_rows_in=n_rows_in,
                n_rows_sampled=n_rows_sampled,
                n_continuous_in=len(columns),
                n_evaluated=len(evaluated),
                n_skipped=len(columns) - len(evaluated),
                matrix_rows=int(corr_matrix.shape[0]),
                matrix_cols=int(corr_matrix.shape[1]),
                n_pairs_above_threshold=n_pairs,
                max_abs_corr=float(np.max(finite)) if finite.size else None,
            )
        return {
            "features": list(evaluated),
            "matrix": corr_matrix.tolist() if len(evaluated) >= 2 else [],
            "null_rates": dict(null_rates),
        }

    def apply(
        self: CorrelationSelector,
        metrics: Mapping[str, Any],
        candidates: Sequence[str],
        context: StageContext,
    ) -> list[FeatureDecision]:
        """Жадно исключает признаки из коррелирующих пар среди оставшихся признаков."""
        del context
        features = [str(name) for name in metrics.get("features", [])]
        raw_matrix = metrics.get("matrix", [])
        null_rates_raw = metrics.get("null_rates", {})
        if not isinstance(null_rates_raw, Mapping):
            null_rates_raw = {}
        remaining = [name for name in candidates if name in set(features)]
        if len(remaining) < 2 or len(features) < 2:
            return []
        matrix = np.asarray(raw_matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != len(features) or matrix.shape[1] != len(features):
            msg = (
                "correlation: cached matrix shape does not match cached features. "
                "Delete the statistics cache or set force_recompute: true."
            )
            raise ExecutionError(msg)
        index = {name: position for position, name in enumerate(features)}
        positions = [index[name] for name in remaining]
        submatrix = matrix[np.ix_(positions, positions)]
        null_rates = {
            name: float(null_rates_raw.get(name, 0.0)) for name in remaining
        }
        to_drop = self._resolve_drops(submatrix, null_rates, remaining)
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

    def _bound_train(
        self: CorrelationSelector,
        context: StageContext,
        max_rows: int,
    ) -> tuple[Any, int, int, bool]:
        """Ограничивает число строк train стратифицированной по целевой переменной или случайной выборкой."""
        target = context.schema.target
        if not target:
            msg = (
                "correlation: FeatureSchema.target is required to bound rows "
                "with stratified sampling."
            )
            raise ExecutionError(msg)
        stratified = context.schema.task_type != "regression"
        sampled, n_in, n_out = sample_frame_rows(
            context.datasets["train"],
            target_col=target,
            max_rows=max_rows,
            stratified=stratified,
            seed=step_seed(context),
            method_name=self.method_name,
        )
        return sampled, n_in, n_out, stratified

    def _compute_stats_spark(
        self: CorrelationSelector,
        train: Any,
        columns: list[str],
        max_rows: int,
    ) -> tuple[np.ndarray, dict[str, float], list[str]]:
        """Вычисляет статистики корреляции средствами Spark ML на проекции выбранных строк.

        Удаляет столбцы только с пропусками, заполняет остальные пропуски медианой, собирает вектор признаков,
        затем вызывает ``Correlation.corr``.
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
            *[_quoted_col(column).alias(aliases[column]) for column in columns],
        ).cache()

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
    ) -> tuple[np.ndarray, dict[str, float], list[str]]:
        """Вычисляет статистики корреляции для уже локального pandas DataFrame."""
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            msg = f"correlation: columns missing from train DataFrame: {missing}."
            raise ExecutionError(msg)

        bounded = frame.loc[:, columns]
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
        """Исключает один признак из каждой пары верхнего треугольника матрицы, превышающей порог.

        Уже исключённые признаки пропускаются, поэтому для каждого признака формируется не более
        одного решения об исключении.
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
        """Выбирает, какой из двух коррелирующих признаков исключить."""
        if self.config.tie_break == "null_rate":
            rate_a = null_rates.get(feat_a, 0.0)
            rate_b = null_rates.get(feat_b, 0.0)
            if rate_a != rate_b:
                return feat_a if rate_a > rate_b else feat_b

        # Draft / original_order: drop the later feature in candidate order.
        return feat_b if candidates.index(feat_a) < candidates.index(feat_b) else feat_a

    @staticmethod
    def _validate_spark_columns(fields: dict[str, Any], columns: list[str]) -> None:
        """Проверяет, что запрошенные столбцы Spark существуют и имеют числовой тип."""
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
    """Создаёт ссылку на столбец Spark с поддержкой точек и пробелов в имени."""
    from pyspark.sql import functions as F  # noqa: N812

    escaped = name.replace("`", "")
    return F.col(f"`{escaped}`")


def _root_cause(exc: BaseException) -> str:
    """Извлекает краткое описание первопричины из исключений Spark/Py4J."""
    java_exc = getattr(exc, "java_exception", None)
    if java_exc is not None:
        return str(java_exc).splitlines()[0]
    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        return str(cause).splitlines()[0]
    return str(exc).splitlines()[0]


def _is_spark_dataframe(data: Any) -> bool:
    """Возвращает, соответствуют ли данные интерфейсу pyspark DataFrame."""
    module_name = type(data).__module__
    return module_name.startswith("pyspark") and hasattr(data, "select") and hasattr(data, "agg")
