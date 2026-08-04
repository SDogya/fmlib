from typing import List, Optional

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def get_low_variance_columns(
    df: DataFrame,
    min_variance: float = 1e-5,
    scale_method: str = "minmax",
    exclude_cols: Optional[List[str]] = None,
    num_feature_cols: Optional[List[str]] = None,
    chunk_size: int = 1000,
    target_col: Optional[str] = None,
) -> List[str]:
    """
    Возвращает список числовых колонок с низкой дисперсией после масштабирования.

    :param df: Исходный DataFrame.
    :param min_variance: Минимально допустимая дисперсия отскейлированных данных.
    :param scale_method: Метод масштабирования ("standard", "minmax", "robust").
    :param exclude_cols: Список колонок, которые исключаются из анализа.
    :param num_feature_cols: Список числовых колонок (определяется автоматически, если None).
    :param chunk_size: Количество колонок в одном батче агрегации.
    :param target_col: Целевая колонка, которая должна быть сохранена.
    :return: Список имен колонок подлежащих удалению.
    """
    valid_methods = {"standard", "minmax", "robust"}
    if scale_method not in valid_methods:
        raise ValueError(f"scale_method must be one of {valid_methods}")

    exclude = set(exclude_cols or [])

    # Гарантированно добавляем target_col в exclude, если он есть в данных
    if target_col and target_col in df.columns and target_col not in exclude:
        exclude.add(target_col)

    if num_feature_cols is None:
        numeric_types = (
            "byte",
            "short",
            "int",
            "bigint",
            "float",
            "double",
            "decimal",
            "long",
        )
        dtypes = dict(df.dtypes)
        num_feature_cols = [
            c
            for c in df.columns
            if c not in exclude
            and any(dtypes[c].startswith(t) for t in numeric_types)
        ]
    else:
        num_feature_cols = [c for c in num_feature_cols if c not in exclude]

    if not num_feature_cols:
        return []

    # Разбиваем список колонок на чанки
    col_chunks = [
        num_feature_cols[i : i + chunk_size]
        for i in range(0, len(num_feature_cols), chunk_size)
    ]

    # Собираем статистику по чанкам
    stats: dict = {}
    
    for chunk in col_chunks:
        # Формируем нативные Catalyst-выражения для агрегации в 1 проход
        agg_exprs = []
        for c in chunk:
            col_ref = F.col(c)
            agg_exprs.append(F.variance(col_ref).alias(f"{c}__var"))

            if scale_method == "minmax":
                agg_exprs.append(F.min(col_ref).alias(f"{c}__min"))
                agg_exprs.append(F.max(col_ref).alias(f"{c}__max"))

            elif scale_method == "robust":
                # Вычисление Q25 и Q75 через аппроксимированный процентиль
                agg_exprs.append(
                    F.percentile_approx(col_ref, 0.25).alias(f"{c}__q25")
                )
                agg_exprs.append(
                    F.percentile_approx(col_ref, 0.75).alias(f"{c}__q75")
                )

        # Выполняем агрегацию для текущего чанка
        chunk_stats = df.agg(*agg_exprs).first().asDict()
        stats.update(chunk_stats)
    
    cols_to_drop = []

    for c in num_feature_cols:
        var = stats.get(f"{c}__var")

        # Если дисперсия None (например, 0 или 1 non-null значение) или 0
        if var is None or var == 0.0:
            cols_to_drop.append(c)
            continue

        if scale_method == "standard":
            # При Z-score скейлинге (X / std) дисперсия всегда равна 1.0 (если var > 0).
            # Если разброса нет вовсе, проверка выше отсечет ее.
            scaled_var = 1.0

        elif scale_method == "minmax":
            mn = stats.get(f"{c}__min")
            mx = stats.get(f"{c}__max")
            if mn is None or mx is None or mn == mx:
                cols_to_drop.append(c)
                continue
            range_val = mx - mn
            scaled_var = var / (range_val ** 2)

        elif scale_method == "robust":
            q25 = stats.get(f"{c}__q25")
            q75 = stats.get(f"{c}__q75")
            if q25 is None or q75 is None or q25 == q75:
                cols_to_drop.append(c)
                continue
            iqr = q75 - q25
            scaled_var = var / (iqr ** 2)

        if scaled_var < min_variance:
            cols_to_drop.append(c)

    return cols_to_drop


def drop_low_variance_features(
    df: DataFrame,
    min_variance: float = 0.01,
    scale_method: str = "standard",
    exclude_cols: Optional[List[str]] = None,
    num_feature_cols: Optional[List[str]] = None,
    chunk_size: int = 1000,
    target_col: Optional[str] = None,
) -> DataFrame:
    """
    Удаляет числовые признаки с низкой дисперсией с батчингом колонок.

    :param df: Исходный DataFrame.
    :param min_variance: Минимально допустимая дисперсия после масштабирования.
    :param scale_method: Метод масштабирования ("standard", "minmax", "robust").
    :param exclude_cols: Колонки, которые нельзя удалять.
    :param num_feature_cols: Список числовых колонок (определяется автоматически если None).
    :param chunk_size: Количество колонок в одном батче агрегации.
    :param target_col: Целевая колонка, которая должна быть сохранена.
    :return: DataFrame без низкодисперсных колонок.
    """
    cols_to_drop = get_low_variance_columns(
        df=df,
        min_variance=min_variance,
        scale_method=scale_method,
        exclude_cols=exclude_cols,
        num_feature_cols=num_feature_cols,
        chunk_size=chunk_size,
        target_col=target_col,
    )

    if not cols_to_drop:
        return df

    cols_to_keep = [c for c in df.columns if c not in set(cols_to_drop)]

    # Гарантированно возвращаем target_col в результат, если он был в исходном df
    if target_col and target_col in df.columns and target_col not in cols_to_keep:
        cols_to_keep.append(target_col)

    return df.select(*cols_to_keep)