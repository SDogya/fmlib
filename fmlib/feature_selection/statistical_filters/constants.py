"""Статистический фильтр константных и квазиконстантных признаков средствами Spark."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import pandas as pd

from fmlib.feature_selection.base import FeatureDecision, StageContext
from fmlib.feature_selection.config import ConstantsConfig
from fmlib.feature_selection.exceptions import BackendError, ExecutionError
from fmlib.feature_selection.utils.verbose import emit as verbose_emit
from fmlib.feature_selection.utils.verbose import enabled as verbose_enabled

# Spark types that cannot be used as countDistinct / groupBy keys for this filter.
_UNSUPPORTED_SPARK_TYPE_NAMES = frozenset({"MapType", "VariantType"})


class ConstantsSelector:
    """Исключает константные и квазиконстантные признаки.

    Для каждого кандидата в train метод отбора вычисляет:

    - ``max_frequency`` — долю наиболее частого непустого значения среди строк
      без пропусков.

    ``n_unique`` вычисляется только при заданном необязательном правиле ``min_unique``
    в конфигурации. В остальных случаях константные столбцы определяются по
    ``max_frequency == 1`` без затратной агрегации ``countDistinct``.

    Признак исключается при срабатывании любого включённого правила:

    - ``n_unique <= 1`` → причина ``constant``;
    - задано ``min_unique`` и ``n_unique < min_unique`` → причина ``too_few_unique``;
    - ``max_frequency >= config.max_frequency`` → причина ``quasi_constant``.

    В Spark столбцы-кандидаты агрегируются пакетами, чтобы избежать чрезмерно больших
    планов Catalyst и ограничений генерации кода на широких наборах данных. Метод отбора
    никогда не загружает train целиком в память драйвера; собираются только компактные сводки по модам и
    числу значений. Вариант для pandas предназначен исключительно для уже локальных
    DataFrame (модульные тесты и небольшие локальные запуски).

    Args:
        config: Настройки фильтра константных признаков.
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
        """Исключает константные и квазиконстантные признаки-кандидаты.

        Args:
            context: Общий контекст этапа с наборами данных и конфигурацией выполнения.
            candidates: Имена признаков, которые ещё рассматриваются для отбора.

        Returns:
            Решения об исключении с измеренным ``value`` и сработавшим порогом.

        Raises:
            BackendError: Если требуется API Spark, но pyspark отсутствует.
            ExecutionError: Если статистики невозможно вычислить для типа выборки train.
        """
        if not candidates:
            return []
        metrics = self.compute(context, list(candidates))
        return self.apply(metrics, candidates, context)

    def compute(
        self: ConstantsSelector,
        context: StageContext,
        candidates: Sequence[str],
    ) -> dict[str, Any]:
        """Возвращает ``{feature: {n_unique, max_frequency}}`` для ``candidates``."""
        columns = list(candidates)
        if not columns:
            return {"values": {}}
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
        return {
            "values": {
                feature: {"n_unique": n_unique, "max_frequency": max_frequency}
                for feature, (n_unique, max_frequency) in stats.items()
            },
        }

    def apply(
        self: ConstantsSelector,
        metrics: Mapping[str, Any],
        candidates: Sequence[str],
        context: StageContext,
    ) -> list[FeatureDecision]:
        """Исключает оставшиеся признаки, нарушающие правила уникальности или частоты."""
        del context
        raw = metrics.get("values", metrics)
        if not isinstance(raw, Mapping):
            return []
        remaining = set(candidates)
        stats: dict[str, tuple[int, float]] = {}
        for feature, payload in raw.items():
            if feature not in remaining:
                continue
            if isinstance(payload, Mapping):
                stats[str(feature)] = (
                    int(payload.get("n_unique", 0)),
                    float(payload.get("max_frequency", 0.0)),
                )
            elif isinstance(payload, (list, tuple)) and len(payload) == 2:
                stats[str(feature)] = (int(payload[0]), float(payload[1]))
        return self._build_decisions(stats)

    def _compute_stats_spark(
        self: ConstantsSelector,
        train: Any,
        columns: list[str],
    ) -> dict[str, tuple[int, float]]:
        """Вычисляет ``(n_unique, max_frequency)`` только с помощью агрегаций Spark.

        Args:
            train: Spark DataFrame.
            columns: Имена столбцов-кандидатов.

        Returns:
            Словарь имён признаков и значений ``(n_unique, max_frequency)``.

        Raises:
            BackendError: Если ``pyspark`` не установлен.
            ExecutionError: При ошибке агрегации Spark или неподдерживаемых типах столбцов.
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
                aliases = {col: f"m{index}" for index, col in enumerate(chunk)}
                mode_aggs = [
                    F.mode(_quoted_col(col)).alias(aliases[col]) for col in chunk
                ]
                modes_row = train.agg(*mode_aggs).first()
                if modes_row:
                    as_dict = modes_row.asDict()
                    for col in chunk:
                        modes[col] = as_dict.get(aliases[col])

            for chunk in chunks:
                summary_aggs = []
                aliases = {col: f"s{index}" for index, col in enumerate(chunk)}
                for col in chunk:
                    alias = aliases[col]
                    quoted = _quoted_col(col)
                    mode = modes.get(col)
                    if self.config.min_unique is not None:
                        summary_aggs.append(
                            F.countDistinct(quoted).alias(f"{alias}__nunique"),
                        )
                    summary_aggs.append(F.count(quoted).alias(f"{alias}__non_null"))
                    if mode is None:
                        summary_aggs.append(F.lit(0).alias(f"{alias}__mode_count"))
                    elif _is_nan(mode):
                        summary_aggs.append(
                            F.sum(F.isnan(quoted).cast("long")).alias(f"{alias}__mode_count"),
                        )
                    else:
                        summary_aggs.append(
                            F.sum((quoted == F.lit(mode)).cast("long")).alias(
                                f"{alias}__mode_count",
                            ),
                        )
                stats_row = train.agg(*summary_aggs).first()
                if not stats_row:
                    continue
                stats = stats_row.asDict()
                for col in chunk:
                    alias = aliases[col]
                    non_null = int(stats.get(f"{alias}__non_null") or 0)
                    if non_null == 0:
                        result[col] = (0, 0.0)
                        continue
                    mode_count = int(stats.get(f"{alias}__mode_count") or 0)
                    if self.config.min_unique is None:
                        n_unique = 1 if mode_count == non_null else 2
                    else:
                        n_unique = int(stats.get(f"{alias}__nunique") or 0)
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
        """Отклоняет типы столбцов Spark, несовместимые с countDistinct/groupBy.

        Args:
            train: Spark DataFrame.
            columns: Имена столбцов-кандидатов.

        Raises:
            ExecutionError: Если один или несколько столбцов имеют неподдерживаемые типы.
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
        """Вычисляет ``(n_unique, max_frequency)`` для уже локального pandas DataFrame.

        Args:
            df: Локальный pandas DataFrame (здесь не создаётся через Spark ``toPandas``).
            columns: Имена столбцов-кандидатов.

        Returns:
            Словарь имён признаков и значений ``(n_unique, max_frequency)``.
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
        """Преобразует статистики отдельных признаков в решения об исключении.

        Args:
            stats: Словарь признак → ``(n_unique, max_frequency)``.

        Returns:
            Решения об исключении признаков, нарушающих заданные пороги.
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
        """Возвращает решение об исключении одного признака или ``None``, если его нужно сохранить."""
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


def _quoted_col(name: str) -> Any:
    """Создаёт ссылку на столбец Spark с поддержкой точек и пробелов в имени."""
    from pyspark.sql import functions as F  # noqa: N812

    escaped = name.replace("`", "")
    return F.col(f"`{escaped}`")


def _spark_root_cause(exc: BaseException) -> str:
    """Извлекает краткое описание первопричины из ошибок Py4J / Spark."""
    java_exc = getattr(exc, "java_exception", None)
    if java_exc is not None:
        return str(java_exc).splitlines()[0]
    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        return str(cause).splitlines()[0]
    return str(exc).splitlines()[0]


def _is_nan(value: Any) -> bool:
    """Возвращает, является ли значение моды Spark значением NaN."""
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def _is_spark_dataframe(data: Any) -> bool:
    """Возвращает True, если ``data`` соответствует интерфейсу pyspark.sql.DataFrame."""
    module_name = type(data).__module__
    return module_name.startswith("pyspark") and hasattr(data, "groupBy") and hasattr(data, "agg")
