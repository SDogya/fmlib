 
import math
from typing import List, Sequence, Tuple
from pyspark.sql import DataFrame
import pyspark.sql.functions as F

def drop_high_psi_columns(
    train_df: DataFrame,
    test_df: DataFrame,
    exclude_cols: Sequence[str],
    target_col: str,
    threshold: float = 0.25,
    num_bins: int = 10,
    fast: bool = False,
    eps: float = 1e-4,
) -> List[str]:
    """Анализирует PSI признаков между train_df и test_df и возвращает список
    колонок, значение PSI которых превышает threshold.
    Args:
        train_df: Эталонный DataFrame (baseline).
        test_df: Текущий DataFrame (actual).
        exclude_cols: Список колонок, которые не нужно анализировать.
        target_col: Имя целевой переменной (исключается из анализа).
        threshold: Порог PSI (по умолчанию 0.25 — существенный сдвиг).
        num_bins: Количество бакетов для разбиения.
        fast: Режим работы.
              False — точный расчёт на основе квантилей (equi-frequency).
              True  — ультрабыстрый расчёт на основе интервалов (equi-width)
                      без вычисления квантилей за 1 проход.
        eps: Сглаживающая константа от деления на 0 и log(0).
    Returns:
        List[str]: Список имен колонок, превысивших порог threshold (подлежат
        удалению).
    """
    # 1. Формируем список кандидатов на проверку
    ignore_set = set(exclude_cols) | {target_col}
    feature_cols = [c for c in train_df.columns if c not in ignore_set]
    dropped_cols: List[str] = []
    for col_name in feature_cols:
        if fast:
            psi_value = _calc_psi_fast(
                train_df, test_df, col_name, num_bins, eps
            )
        else:
            psi_value = _calc_psi_exact(
                train_df, test_df, col_name, num_bins, eps
            )
        if psi_value > threshold:
            dropped_cols.append(col_name)
    return dropped_cols

def _calc_psi_exact(
    train_df: DataFrame,
    test_df: DataFrame,
    col_name: str,
    num_bins: int,
    eps: float,
) -> float:
    """Точный расчёт PSI на основе квантилей (Equi-frequency)."""
    # Вычисляем границы квантилей по train_df
    probabilities = [i / num_bins for i in range(1, num_bins)]
    quantiles = train_df.stat.approxQuantile(col_name, probabilities, 0.0001)
    # Удаляем дубликаты совпавших квантилей
    splits = sorted(list(set([-float("inf")] + quantiles + [float("inf")])))
    actual_bins = len(splits) - 1
    # Векторизованный CASE WHEN по готовым точкам квантилей
    bin_expr = F.when(F.col(col_name) <= splits[1], 0)
    for i in range(1, len(splits) - 2):
        bin_expr = bin_expr.when(
            (F.col(col_name) > splits[i]) & (F.col(col_name) <= splits[i + 1]),
            i,
        )
    bin_expr = bin_expr.otherwise(len(splits) - 2).cast("int")
    # Сбор распределений с агрегацией на стороне Executor (Map-side combiner)
    exp_counts, total_exp = _get_counts(train_df, col_name, bin_expr)
    act_counts, total_act = _get_counts(test_df, col_name, bin_expr)
    return _compute_psi_from_counts(
        exp_counts, total_exp, act_counts, total_act, actual_bins, eps
    )

def _calc_psi_fast(
    train_df: DataFrame,
    test_df: DataFrame,
    col_name: str,
    num_bins: int,
    eps: float,
) -> float:
    """Сверхбыстрый расчёт PSI на основе интервалов (Equi-width)."""
    stats = train_df.agg(
        F.min(col_name).alias("min_v"), F.max(col_name).alias("max_v")
    ).head()
    min_v, max_v = stats["min_v"], stats["max_v"]
    if min_v is None or max_v is None:
        return 0.0
    min_val, max_val = float(min_v), float(max_v)
    range_val = max_val - min_val if max_val != min_val else 1e-9
    # Прямой расчёт номера бакета без CASE WHEN за O(1)
    raw_bin = F.floor(
        ((F.col(col_name) - min_val) / F.lit(range_val)) * F.lit(num_bins)
    )
    bin_expr = (
        F.when(raw_bin < 0, 0)
        .when(raw_bin >= num_bins, num_bins - 1)
        .otherwise(raw_bin)
        .cast("int")
    )
    exp_counts, total_exp = _get_counts(train_df, col_name, bin_expr)
    act_counts, total_act = _get_counts(test_df, col_name, bin_expr)
    return _compute_psi_from_counts(
        exp_counts, total_exp, act_counts, total_act, num_bins, eps
    )

def _get_counts(
    df: DataFrame, col_name: str, bin_expr: F.Column
) -> Tuple[dict, int]:
    """Собирает частоты по бакетам за один проход с фильтрацией NULL/NaN."""
    rows = (
        df.filter(F.col(col_name).isNotNull() & ~F.isnan(F.col(col_name)))
        .groupBy(bin_expr.alias("bin_id"))
        .agg(F.count("*").alias("cnt"))
        .collect()
    )
    counts = {r["bin_id"]: r["cnt"] for r in rows}
    total = sum(counts.values())
    return counts, total

def _compute_psi_from_counts(
    exp_counts: dict,
    total_exp: int,
    act_counts: dict,
    total_act: int,
    num_bins: int,
    eps: float,
) -> float:
    """Вычисляет финальный PSI на Driver."""
    if total_exp == 0 or total_act == 0:
        return 0.0
    psi_total = 0.0
    for b in range(num_bins):
        e_prop = exp_counts.get(b, 0) / total_exp
        a_prop = act_counts.get(b, 0) / total_act
        e_adj = max(e_prop, eps)
        a_adj = max(a_prop, eps)
        psi_total += (a_adj - e_adj) * math.log(a_adj / e_adj)
    return float(psi_total)
