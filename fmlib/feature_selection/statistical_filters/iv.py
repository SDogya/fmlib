"""Фильтр информационной ценности (IV) для бинарной классификации."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from fmlib.feature_selection.base import FeatureDecision, StageContext
from fmlib.feature_selection.config import IvConfig
from fmlib.feature_selection.exceptions import BackendError, ConfigError, ExecutionError
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
_NULL_LEVEL = "__iv_null__"
_OTHER_LEVEL = "__iv_other__"


class IvSelector:
    """Исключает признаки со слишком низкой или подозрительно высокой информационной ценностью (IV).

    Непрерывные кандидаты разбиваются на квантильные интервалы; для категориальных используются
    отдельные значения (редкие категории могут объединяться). Пропуски образуют отдельную группу.
    Входные данные Spark остаются распределёнными: вычисляются квантили и пакетные агрегации, затем
    собирается компактная таблица частот. Для уже локальных pandas DataFrame используется
    аналогичный вариант на numpy.
    После исключения пропусков таргет должен содержать ровно два класса;
    иначе расчёт завершается с ``ExecutionError`` без удаления признаков.

    Args:
        config: Настройки фильтра IV.
    """

    method_name = "iv"
    stage_name = "statistics"

    def __init__(self: IvSelector, config: IvConfig) -> None:
        self.config = config

    def select(
        self: IvSelector,
        context: StageContext,
        candidates: Sequence[str],
    ) -> list[FeatureDecision]:
        """Вычисляет IV на train и исключает слабые кандидаты или кандидаты с утечкой целевой переменной.

        Args:
            context: Общий контекст этапа.
            candidates: Текущие признаки-кандидаты.

        Returns:
            Решения об исключении признаков за пределами заданного диапазона IV.

        Raises:
            ConfigError: Если задача не является бинарной классификацией.
            BackendError: Если требуется API Spark, но pyspark отсутствует.
            ExecutionError: Если статистики невозможно вычислить.
        """
        if not candidates:
            return []
        schema = context.schema
        if not schema.target:
            msg = "iv requires FeatureSchema.target."
            raise ConfigError(msg)
        if schema.task_type != "binary_classification":
            msg = (
                "iv requires FeatureSchema.task_type='binary_classification'. "
                f"Got {schema.task_type!r}."
            )
            raise ConfigError(msg)
        metrics = self.compute(context, list(candidates))
        return self.apply(metrics, candidates, context)

    def compute(
        self: IvSelector,
        context: StageContext,
        candidates: Sequence[str],
    ) -> dict[str, Any]:
        """Возвращает ``{feature: iv}`` для ``candidates``."""
        schema = context.schema
        if not schema.target:
            msg = "iv requires FeatureSchema.target."
            raise ConfigError(msg)
        if schema.task_type != "binary_classification":
            msg = (
                "iv requires FeatureSchema.task_type='binary_classification'. "
                f"Got {schema.task_type!r}."
            )
            raise ConfigError(msg)
        columns = list(candidates)
        if not columns:
            return {"values": {}}
        train = context.datasets["train"]
        categorical = set(schema.categorical)
        continuous = set(schema.continuous)
        if _is_spark_dataframe(train):
            iv_scores = self._compute_iv_spark(
                train,
                columns,
                target=schema.target,
                categorical=categorical,
                continuous=continuous,
            )
            backend = "spark"
        elif isinstance(train, pd.DataFrame):
            iv_scores = self._compute_iv_pandas(
                train,
                columns,
                target=schema.target,
                categorical=categorical,
                continuous=continuous,
            )
            backend = "pandas"
        else:
            msg = (
                f"iv: unsupported train split type {type(train)!r}. "
                "Expected a Spark DataFrame or pandas DataFrame."
            )
            raise ExecutionError(msg)

        if verbose_enabled(context, self.method_name) and iv_scores:
            values = list(iv_scores.values())
            n_low = sum(1 for value in values if value < self.config.threshold)
            n_high = 0
            if self.config.max_threshold is not None:
                n_high = sum(1 for value in values if value > self.config.max_threshold)
            verbose_emit(
                context,
                self.method_name,
                "stats",
                backend=backend,
                threshold=self.config.threshold,
                max_threshold=self.config.max_threshold,
                num_bins=self.config.num_bins,
                n_evaluated=len(values),
                n_below_threshold=n_low,
                n_above_max_threshold=n_high,
                iv_min=min(values),
                iv_max=max(values),
                iv_mean=round(sum(values) / len(values), 6),
            )
        return {"values": iv_scores}

    def apply(
        self: IvSelector,
        metrics: Mapping[str, Any],
        candidates: Sequence[str],
        context: StageContext,
    ) -> list[FeatureDecision]:
        """Исключает оставшиеся признаки, у которых IV выходит за заданный диапазон."""
        values = metrics.get("values", metrics)
        if not isinstance(values, Mapping):
            values = {}
        remaining = set(candidates)
        scored = {
            feature: float(value)
            for feature, value in values.items()
            if feature in remaining
        }
        context.scores[self.method_name] = {
            "threshold": self.config.threshold,
            "max_threshold": self.config.max_threshold,
            "num_bins": self.config.num_bins,
            "values": dict(scored),
        }
        decisions: list[FeatureDecision] = []
        for feature, value in scored.items():
            if value < self.config.threshold:
                decisions.append(
                    FeatureDecision(
                        feature=feature,
                        stage=self.stage_name,
                        method=self.method_name,
                        reason="low_iv",
                        value=value,
                        threshold=self.config.threshold,
                        keep=False,
                    ),
                )
            elif (
                self.config.max_threshold is not None
                and value > self.config.max_threshold
            ):
                decisions.append(
                    FeatureDecision(
                        feature=feature,
                        stage=self.stage_name,
                        method=self.method_name,
                        reason="high_iv",
                        value=value,
                        threshold=self.config.max_threshold,
                        keep=False,
                    ),
                )
        return decisions

    def _compute_iv_pandas(
        self: IvSelector,
        frame: pd.DataFrame,
        columns: list[str],
        *,
        target: str,
        categorical: set[str],
        continuous: set[str],
    ) -> dict[str, float]:
        """Вычисляет IV для уже локального pandas DataFrame."""
        missing = [name for name in [*columns, target] if name not in frame.columns]
        if missing:
            msg = f"iv: columns missing from train DataFrame: {missing}."
            raise ExecutionError(msg)
        working = frame[[*columns, target]].dropna(subset=[target])
        y = _as_binary_numpy(working[target], method="iv")
        scores: dict[str, float] = {}
        for name in columns:
            if name in continuous:
                bins = _quantile_bins_pandas(
                    working[name],
                    num_bins=self.config.num_bins,
                )
            else:
                bins = _categorical_bins_pandas(
                    working[name],
                    min_bin_share=self.config.min_bin_share,
                    max_levels=self.config.max_levels,
                )
            scores[name] = information_value_from_bins(
                y,
                bins,
                eps=self.config.eps,
            )
        return scores

    def _compute_iv_spark(
        self: IvSelector,
        train: Any,
        columns: list[str],
        *,
        target: str,
        categorical: set[str],
        continuous: set[str],
    ) -> dict[str, float]:
        """Вычисляет IV агрегациями Spark; собирает только численности групп."""
        try:
            from pyspark.sql import functions as F  # noqa: F401, N812
        except ImportError as exc:
            msg = (
                "iv: pyspark is required for Spark DataFrames. "
                "Install the spark optional dependency group."
            )
            raise BackendError(msg) from exc

        fields = {field.name: field.dataType for field in train.schema.fields}
        missing = [name for name in [*columns, target] if name not in fields]
        if missing:
            msg = f"iv: columns missing from train schema: {missing}."
            raise ExecutionError(msg)

        y_col = _quoted_col(target)
        filtered = train.where(y_col.isNotNull())
        y_expr = _spark_binary_target(filtered, target, y_col)
        prepared = filtered.select(
            *[
                _quoted_col(name).alias(f"c{index}")
                for index, name in enumerate(columns)
            ],
            y_expr.alias("__y__"),
        )

        numeric_idx = []
        categorical_idx = []
        for index, name in enumerate(columns):
            if name in continuous:
                type_name = type(fields[name]).__name__
                if type_name not in _NUMERIC_SPARK_TYPE_NAMES:
                    msg = (
                        f"iv: continuous column {name!r} has Spark type "
                        f"{type_name}; expected a numeric type."
                    )
                    raise ExecutionError(msg)
                numeric_idx.append(index)
            else:
                categorical_idx.append(index)

        is_cached = getattr(prepared, "is_cached", False)
        should_unpersist = False
        if not is_cached:
            try:
                prepared = prepared.persist()
                should_unpersist = True
            except Exception:
                should_unpersist = False
        try:
            scores: dict[str, float] = {}
            if numeric_idx:
                scores.update(
                    self._spark_numeric_iv(prepared, columns, numeric_idx),
                )
            if categorical_idx:
                scores.update(
                    self._spark_categorical_iv(prepared, columns, categorical_idx),
                )
            return {name: scores.get(name, 0.0) for name in columns}
        finally:
            if should_unpersist:
                try:
                    prepared.unpersist()
                except Exception:
                    pass

    def _spark_numeric_iv(
        self: IvSelector,
        frame: Any,
        columns: list[str],
        indexes: list[int],
    ) -> dict[str, float]:
        """Разбивает непрерывные столбцы на квантильные интервалы и агрегирует численности good/bad."""
        from pyspark.sql import functions as F  # noqa: N812

        aliases = [f"c{index}" for index in indexes]
        probabilities = [
            i / self.config.num_bins for i in range(1, self.config.num_bins)
        ]
        try:
            all_quantiles = frame.stat.approxQuantile(
                aliases,
                probabilities,
                self.config.relative_error,
            )
        except Exception as exc:
            msg = (
                "iv: Spark approxQuantile failed. "
                f"Root cause: {_root_cause(exc)}."
            )
            raise ExecutionError(msg) from exc

        scores: dict[str, float] = {}
        batch_size = self.config.batch_size
        y_col = F.col("__y__")
        for start in range(0, len(indexes), batch_size):
            batch_idx = indexes[start : start + batch_size]
            batch_q = all_quantiles[start : start + batch_size]
            expressions = []
            bin_counts: dict[str, int] = {}
            for index, quantiles in zip(batch_idx, batch_q):
                alias = f"c{index}"
                name = columns[index]
                edges = sorted({-float("inf"), *quantiles, float("inf")})
                n_bins = len(edges) - 1
                bin_counts[name] = n_bins
                value = F.col(alias)
                null_cond = value.isNull() | F.isnan(value.cast("double"))
                expressions.extend(
                    _count_exprs(null_cond, y_col, f"{alias}__null"),
                )
                for bin_idx in range(n_bins):
                    low = edges[bin_idx]
                    high = edges[bin_idx + 1]
                    valid = ~null_cond
                    if bin_idx == 0:
                        cond = valid & (value <= high)
                    else:
                        cond = valid & (value > low) & (value <= high)
                    expressions.extend(
                        _count_exprs(cond, y_col, f"{alias}__b{bin_idx}"),
                    )
            try:
                row = frame.agg(*expressions).head().asDict()
            except Exception as exc:
                msg = (
                    "iv: Spark aggregation failed while computing IV. "
                    f"Root cause: {_root_cause(exc)}."
                )
                raise ExecutionError(msg) from exc
            for index in batch_idx:
                alias = f"c{index}"
                name = columns[index]
                goods: list[float] = []
                bads: list[float] = []
                for bin_idx in range(bin_counts[name]):
                    n_bad, n_total = _counts_from_row(row, f"{alias}__b{bin_idx}")
                    goods.append(n_total - n_bad)
                    bads.append(n_bad)
                n_bad, n_total = _counts_from_row(row, f"{alias}__null")
                goods.append(n_total - n_bad)
                bads.append(n_bad)
                scores[name] = information_value(goods, bads, self.config.eps)
        return scores

    def _spark_categorical_iv(
        self: IvSelector,
        frame: Any,
        columns: list[str],
        indexes: list[int],
    ) -> dict[str, float]:
        """Группирует категории и агрегирует численности good/bad."""
        from pyspark.sql import functions as F  # noqa: N812

        scores: dict[str, float] = {}
        batch_size = self.config.batch_size
        y_col = F.col("__y__")
        for start in range(0, len(indexes), batch_size):
            batch_idx = indexes[start : start + batch_size]
            pieces = []
            for index in batch_idx:
                alias = f"c{index}"
                pieces.append(
                    frame.select(
                        F.lit(columns[index]).alias("feature"),
                        F.col(alias).cast("string").alias("level"),
                        y_col.alias("y"),
                    ),
                )
            stacked = pieces[0]
            for piece in pieces[1:]:
                stacked = stacked.unionByName(piece)
            try:
                rows = (
                    stacked.groupBy("feature", "level")
                    .agg(
                        F.sum("y").alias("n_bad"),
                        F.count("*").alias("n"),
                    )
                    .collect()
                )
            except Exception as exc:
                msg = (
                    "iv: Spark groupBy failed while computing categorical IV. "
                    f"Root cause: {_root_cause(exc)}."
                )
                raise ExecutionError(msg) from exc
            grouped: dict[str, list[tuple[str, float, float]]] = {
                columns[index]: [] for index in batch_idx
            }
            for row in rows:
                n_total = float(row["n"] or 0)
                n_bad = float(row["n_bad"] or 0)
                level = row["level"]
                key = _NULL_LEVEL if level is None else str(level)
                grouped[row["feature"]].append((key, n_total - n_bad, n_bad))
            for name, levels in grouped.items():
                scores[name] = information_value(
                    *_merge_categorical_counts(
                        levels,
                        min_bin_share=self.config.min_bin_share,
                        max_levels=self.config.max_levels,
                    ),
                    self.config.eps,
                )
        return scores


def information_value(
    goods: Sequence[float],
    bads: Sequence[float],
    eps: float,
) -> float:
    """Возвращает IV по числу good/bad в каждой группе."""
    total_good = float(sum(goods))
    total_bad = float(sum(bads))
    if total_good <= 0.0 or total_bad <= 0.0:
        return 0.0
    iv = 0.0
    for good, bad in zip(goods, bads):
        if good <= 0.0 and bad <= 0.0:
            continue
        dist_good = good / total_good
        dist_bad = bad / total_bad
        iv += (dist_good - dist_bad) * math.log((dist_good + eps) / (dist_bad + eps))
    return float(iv)


def information_value_from_bins(
    y: np.ndarray,
    bins: np.ndarray,
    *,
    eps: float,
) -> float:
    """Возвращает IV по согласованным целевым меткам и идентификаторам групп."""
    goods: list[float] = []
    bads: list[float] = []
    for level in pd.unique(bins):
        mask = bins == level
        n_bad = float(y[mask].sum())
        n_total = float(mask.sum())
        goods.append(n_total - n_bad)
        bads.append(n_bad)
    return information_value(goods, bads, eps)


def _quantile_bins_pandas(series: pd.Series, *, num_bins: int) -> np.ndarray:
    """Присваивает метки квантильных интервалов; пропуски образуют отдельную группу."""
    values = pd.to_numeric(series, errors="coerce")
    labels = np.empty(len(values), dtype=object)
    null_mask = values.isna()
    labels[null_mask] = _NULL_LEVEL
    valid = values[~null_mask]
    if valid.empty:
        return labels
    if valid.nunique(dropna=True) <= 1:
        labels[~null_mask] = "c0"
        return labels
    try:
        coded, _edges = pd.qcut(
            valid,
            q=num_bins,
            labels=False,
            duplicates="drop",
            retbins=True,
        )
    except ValueError:
        labels[~null_mask] = "c0"
        return labels
    labels[~null_mask] = [f"c{int(item)}" for item in coded.to_numpy()]
    return labels


def _categorical_bins_pandas(
    series: pd.Series,
    *,
    min_bin_share: float,
    max_levels: int | None,
) -> np.ndarray:
    """Присваивает метки категориальных групп, объединяя редкие или избыточные категории."""
    as_str = series.astype("string")
    labels = np.where(as_str.isna(), _NULL_LEVEL, as_str.to_numpy(dtype=object))
    counts: dict[str, int] = {}
    for item in labels:
        if item == _NULL_LEVEL:
            continue
        counts[str(item)] = counts.get(str(item), 0) + 1
    n_rows = len(labels)
    keep = _levels_to_keep(counts, n_rows, min_bin_share, max_levels)
    return np.array(
        [
            item if item == _NULL_LEVEL or str(item) in keep else _OTHER_LEVEL
            for item in labels
        ],
        dtype=object,
    )


def _levels_to_keep(
    counts: dict[str, int],
    n_rows: int,
    min_bin_share: float,
    max_levels: int | None,
) -> set[str]:
    """Возвращает категории, которые не следует объединять."""
    items = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if min_bin_share > 0.0 and n_rows > 0:
        items = [
            item
            for item in items
            if (item[1] / n_rows) >= min_bin_share or item[0] == _NULL_LEVEL
        ]
    if max_levels is not None and len(items) > max_levels:
        items = items[:max_levels]
    return {name for name, _count in items}


def _merge_categorical_counts(
    levels: list[tuple[str, float, float]],
    *,
    min_bin_share: float,
    max_levels: int | None,
) -> tuple[list[float], list[float]]:
    """Объединяет редкие категории и возвращает списки численностей good/bad."""
    n_rows = sum(good + bad for _level, good, bad in levels)
    counts = {
        level: int(good + bad)
        for level, good, bad in levels
        if level != _NULL_LEVEL
    }
    keep = _levels_to_keep(counts, int(n_rows), min_bin_share, max_levels)
    goods: dict[str, float] = {}
    bads: dict[str, float] = {}
    for level, good, bad in levels:
        key = level if level == _NULL_LEVEL or level in keep else _OTHER_LEVEL
        goods[key] = goods.get(key, 0.0) + good
        bads[key] = bads.get(key, 0.0) + bad
    keys = list(goods)
    return [goods[key] for key in keys], [bads[key] for key in keys]


def _as_binary_numpy(series: pd.Series, *, method: str) -> np.ndarray:
    """Преобразует бинарную целевую Series в массив чисел с плавающей точкой 0/1."""
    mapping = _binary_mapping(pd.unique(series.dropna()), method=method)
    return series.map(mapping).to_numpy(dtype=np.float64)


def _binary_mapping(unique: Any, *, method: str) -> dict[Any, float]:
    """Отображает ровно две целевые метки в {0.0, 1.0}, иначе вызывает ExecutionError."""
    raw = unique.tolist() if hasattr(unique, "tolist") else list(unique)
    labels = list(dict.fromkeys(item for item in raw if not _is_null(item)))
    if len(labels) != 2:
        msg = (
            f"{method}: target must be binary with exactly two classes. "
            f"Found {len(labels)} distinct non-null values."
        )
        raise ExecutionError(msg)
    as_set = set(labels)
    if as_set <= {0, 1, 0.0, 1.0, False, True}:
        return {
            label: float(label) if not isinstance(label, bool) else float(label)
            for label in labels
        }
    try:
        ordered = sorted(labels)
    except TypeError:
        ordered = sorted(labels, key=str)
    return {ordered[0]: 0.0, ordered[1]: 1.0}


def _spark_binary_target(frame: Any, target: str, y_col: Any) -> Any:
    """Возвращает столбец Spark со значениями 0/1 для ``target``."""
    from pyspark.sql import functions as F  # noqa: N812

    try:
        distinct = [
            row[0]
            for row in frame.select(y_col).distinct().limit(4).collect()
        ]
    except Exception as exc:
        msg = (
            f"iv: failed to inspect target column {target!r}. "
            f"Root cause: {_root_cause(exc)}."
        )
        raise ExecutionError(msg) from exc
    labels = [item for item in distinct if item is not None]
    mapping = _binary_mapping(labels, method="iv")
    (label_a, bit_a), (label_b, bit_b) = list(mapping.items())
    return F.when(y_col == F.lit(label_a), float(bit_a)).otherwise(float(bit_b))


def _count_exprs(cond: Any, y_col: Any, prefix: str) -> list[Any]:
    """Агрегации Spark: число событий и строк при условии ``cond``."""
    from pyspark.sql import functions as F  # noqa: N812

    return [
        F.sum(F.when(cond, y_col).otherwise(0.0)).alias(f"{prefix}__bad"),
        F.count(F.when(cond, 1)).alias(f"{prefix}__n"),
    ]


def _counts_from_row(row: dict[str, Any], prefix: str) -> tuple[float, float]:
    """Читает ``(n_bad, n_total)``, полученные через ``_count_exprs``."""
    n_bad = float(row.get(f"{prefix}__bad") or 0.0)
    n_total = float(row.get(f"{prefix}__n") or 0.0)
    return n_bad, n_total


def _quoted_col(name: str) -> Any:
    """Создаёт ссылку на столбец Spark с поддержкой точек и пробелов."""
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
    return (
        module_name.startswith("pyspark")
        and hasattr(data, "select")
        and hasattr(data, "agg")
    )


def _is_null(value: Any) -> bool:
    """Возвращает, является ли ``value`` скалярным пропуском."""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return value is None
