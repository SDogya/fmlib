"""Фильтр индекса стабильности популяции (масштабируемый и готовый к промышленному использованию)."""

from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from fmlib.feature_selection.base import FeatureDecision, StageContext, step_seed
from fmlib.feature_selection.config import PsiConfig
from fmlib.feature_selection.exceptions import ExecutionError
from fmlib.feature_selection.utils.verbose import emit as verbose_emit
from fmlib.feature_selection.utils.verbose import enabled as verbose_enabled

logger = logging.getLogger(__name__)


class PsiSelector:
    """Исключает нестабильные признаки по индексу стабильности популяции (PSI).

    Оптимизирован для больших PySpark DataFrame (пакетная обработка признаков, управляемое сохранение в памяти,
    приближённые квантили) и Pandas DataFrame (векторизация через C-API).

    Args:
        config: Настройки фильтра PSI.
    """

    method_name = "psi"
    stage_name = "statistics"

    def __init__(self: PsiSelector, config: PsiConfig) -> None:
        self.config = config

    def _apply_stratified_sampling_pyspark(
        self: PsiSelector,
        df: Any,
        target_col: str,
        max_rows: int,
        seed: int,
    ) -> Any:
        """Формирует стратифицированную выборку по целевому столбцу для PySpark.

        Args:
            df: Исходный PySpark DataFrame.
            target_col: Целевой столбец для стратификации.
            max_rows: Максимальное количество строк после выборки.
            seed: Seed для воспроизводимости.

        Returns:
            DataFrame с выбранными строками.
        """
        import pyspark.sql.functions as F

        # Add helper column for stratification
        strat_df = df.withColumn("_strat_col", F.col(target_col).cast("string"))

        # Calculate class proportions
        strat_count = strat_df.groupBy("_strat_col").count().collect()
        data_len = strat_df.count()

        if data_len <= max_rows:
            logger.info(
                f"PSISelector: Data has {data_len:,} rows, below subsample limit {max_rows:,}. "
                "Skipping stratified sampling."
            )
            result = strat_df.drop("_strat_col")
            return result

        # Calculate fraction to reach max_rows
        fraction = max_rows / data_len

        # Calculate class-wise counts and proportions
        strat_count_dict = {row["_strat_col"]: row["count"] for row in strat_count}
        fractions = {str(key): fraction for key in strat_count_dict.keys()}

        logger.info(
            f"PSISelector: Stratified sampling: fraction={fraction:.4f} "
            f"(max_rows={max_rows}, data_len={data_len:,})"
        )
        logger.info(f"PSISelector: Stratification stats: {strat_count_dict}")

        # Apply sampleBy for stratified sampling
        sampled_df = strat_df.sampleBy(
            "_strat_col", fractions=fractions, seed=seed
        ).drop("_strat_col")

        data_len_after = sampled_df.count()
        logger.info(
            f"PSISelector: Stratified: {data_len:,} -> {data_len_after:,} rows "
            f"(fraction={data_len_after / data_len:.4f})"
        )

        return sampled_df

    def _apply_stratified_sampling_pandas(
        self: PsiSelector,
        df: Any,
        target_col: str,
        max_rows: int,
        seed: int,
    ) -> Any:
        """Формирует стратифицированную выборку по целевому столбцу для Pandas.

        Args:
            df: Исходный Pandas DataFrame.
            target_col: Целевой столбец для стратификации.
            max_rows: Максимальное количество строк после выборки.
            seed: Seed для воспроизводимости.

        Returns:
            DataFrame с выбранными строками.
        """
        import pandas as pd

        data_len = len(df)

        if data_len <= max_rows:
            logger.info(
                f"PSISelector: Data has {data_len:,} rows, below subsample limit {max_rows:,}. "
                "Skipping stratified sampling."
            )
            return df

        # Calculate class proportions
        strat_values = df[target_col].value_counts()
        data_len = len(df)
        strat_count_dict = strat_values.to_dict()

        # Calculate fraction to reach max_rows
        fraction = max_rows / data_len

        logger.info(
            f"PSISelector: Stratified sampling (pandas): fraction={fraction:.4f} "
            f"(max_rows={max_rows}, data_len={data_len:,})"
        )
        logger.info(f"PSISelector: Stratification stats: {strat_count_dict}")

        # Sample each class proportionally
        rng = np.random.RandomState(seed)
        
        sampled_dfs = []
        for class_val, count in strat_count_dict.items():
            class_df = df[df[target_col] == class_val]
            sample_size = max(1, int(count * fraction))
            sample_size = min(sample_size, len(class_df))
            sampled = class_df.sample(n=sample_size, random_state=rng)
            sampled_dfs.append(sampled)

        result = pd.concat(sampled_dfs, ignore_index=True)
        
        data_len_after = len(result)
        logger.info(
            f"PSISelector: Stratified (pandas): {data_len:,} -> {data_len_after:,} rows "
            f"(fraction={data_len_after / data_len:.4f})"
        )

        return result

    def _apply_stratified_sampling(
        self: PsiSelector,
        df: Any,
        target_col: str,
        max_rows: int,
        seed: int,
    ) -> Any:
        """Формирует стратифицированную выборку по целевому столбцу.

        Args:
            df: Исходный DataFrame (PySpark или Pandas).
            target_col: Целевой столбец для стратификации.
            max_rows: Максимальное количество строк после выборки.
            seed: Seed для воспроизводимости.

        Returns:
            DataFrame с выбранными строками.
        """
        # Check if PySpark DataFrame
        if hasattr(df, "stat") and hasattr(df, "agg"):
            return self._apply_stratified_sampling_pyspark(df, target_col, max_rows, seed)
        else:
            return self._apply_stratified_sampling_pandas(df, target_col, max_rows, seed)

    def _apply_random_sampling_pyspark(
        self: PsiSelector,
        df: Any,
        max_rows: int,
        seed: int,
    ) -> Any:
        """Формирует равномерную случайную подвыборку Spark."""
        import pyspark.sql.functions as F  # noqa: N812

        data_len = df.count()
        if data_len <= max_rows:
            return df
        return df.orderBy(F.rand(seed)).limit(max_rows)

    def _apply_random_sampling_pandas(
        self: PsiSelector,
        df: Any,
        max_rows: int,
        seed: int,
    ) -> Any:
        """Формирует равномерную случайную подвыборку pandas."""
        data_len = len(df)
        if data_len <= max_rows:
            return df
        return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    def _apply_sampling(
        self: PsiSelector,
        df: Any,
        target_col: str,
        max_rows: int,
        seed: int,
        *,
        stratified: bool,
    ) -> Any:
        """Формирует подвыборку ``df`` со стратификацией по целевой переменной, если ``stratified`` не равно false."""
        if not stratified:
            if hasattr(df, "stat") and hasattr(df, "agg"):
                return self._apply_random_sampling_pyspark(df, max_rows, seed)
            return self._apply_random_sampling_pandas(df, max_rows, seed)
        return self._apply_stratified_sampling(df, target_col, max_rows, seed)

    def select(
        self: PsiSelector,
        context: StageContext,
        candidates: Sequence[str],
    ) -> list[FeatureDecision]:
        feature_cols = list(candidates)
        if not feature_cols:
            return []
        metrics = self.compute(context, feature_cols)
        return self.apply(metrics, feature_cols, context)

    def compute(
        self: PsiSelector,
        context: StageContext,
        candidates: Sequence[str],
    ) -> dict[str, Any]:
        """Возвращает ``{feature: psi}`` для ``candidates``."""
        feature_cols = list(candidates)
        if not feature_cols:
            return {"values": {}}

        threshold = self.config.threshold
        num_bins = self.config.num_bins
        eps = self.config.eps

        train_df, test_df = self._resolve_population_pair(context)
        train_df, test_df = self._apply_subsample_if_needed(context, train_df, test_df)
        is_pyspark = hasattr(train_df, "stat") and hasattr(train_df, "agg")

        if verbose_enabled(context, self.method_name):
            verbose_emit(
                context,
                self.method_name,
                "inputs",
                mode=self.config.mode,
                threshold=threshold,
                num_bins=num_bins,
                n_features=len(feature_cols),
                backend="spark" if is_pyspark else "pandas",
                subsample_rows=self.config.subsample_rows,
                baseline=context.verbose_log.snapshot_frame(train_df),
                actual=context.verbose_log.snapshot_frame(test_df),
            )

        if is_pyspark:
            psi_scores = self._compute_pyspark_psi(train_df, test_df, feature_cols, num_bins)
        else:
            psi_scores = self._compute_pandas_psi(train_df, test_df, feature_cols, num_bins, eps)

        if verbose_enabled(context, self.method_name) and psi_scores:
            values = [float(score) for score in psi_scores.values()]
            verbose_emit(
                context,
                self.method_name,
                "scores",
                n_scored=len(values),
                n_above_threshold=sum(1 for score in values if score > threshold),
                psi_min=min(values),
                psi_max=max(values),
                psi_mean=round(sum(values) / len(values), 6),
                threshold=threshold,
            )
        return {"values": psi_scores}

    def apply(
        self: PsiSelector,
        metrics: Mapping[str, Any],
        candidates: Sequence[str],
        context: StageContext,
    ) -> list[FeatureDecision]:
        """Сохраняет или исключает оставшиеся признаки по текущему порогу PSI."""
        del context
        values = metrics.get("values", metrics)
        if not isinstance(values, Mapping):
            values = {}
        threshold = self.config.threshold
        decisions: list[FeatureDecision] = []
        for col in candidates:
            score = float(values.get(col, 0.0))
            keep = score <= threshold
            decisions.append(self._make_decision(col, keep=keep, score=score, threshold=threshold))
        return decisions

    def _resolve_population_pair(self, context: StageContext) -> Tuple[Any, Any]:
        """Возвращает пару (baseline, actual), выбранную по ``config.mode``.

        ``datasets['test']`` никогда не читается: отложенная выборка должна оставаться
        нетронутой, чтобы итоговый набор признаков можно было оценить на данных, которые
        не участвовали в отборе. При отсутствии сравниваемой совокупности возникает ошибка вместо пропуска
        фильтра, поскольку незаметный пропуск в отчёте неотличим
        от «измерено и стабильно».

        Args:
            context: Контекст этапа с наборами данных и схемой.

        Returns:
            Кортеж DataFrame (baseline, actual).

        Raises:
            ExecutionError: Если требуемая для режима совокупность данных недоступна.
        """
        baseline = context.datasets.get("train")
        if baseline is None:
            msg = "psi: datasets['train'] is required."
            raise ExecutionError(msg)

        if self.config.mode == "month_over_month":
            return self._split_by_month(context, baseline)

        actual = context.datasets.get("valid")
        if actual is not None:
            return baseline, actual

        split_column = getattr(context.schema, "split", None)
        if split_column is not None:
            return self._split_by_column(baseline, split_column)

        msg = (
            "psi mode='train_valid' requires datasets['valid'] or "
            "FeatureSchema.split. Provide one, or use mode='month_over_month' "
            "to compare the latest periods of train against the earlier ones."
        )
        raise ExecutionError(msg)

    @staticmethod
    def _split_by_column(frame: Any, split_column: str) -> Tuple[Any, Any]:
        """Разделяет один DataFrame на строки train/valid по столбцу разбиения.

        Args:
            frame: DataFrame со столбцом разбиения.
            split_column: Столбец с метками ``train`` / ``valid`` / ``test``.

        Returns:
            Кортеж (строки train, строки valid).

        Raises:
            ExecutionError: Если любая из частей разбиения пуста.
        """
        is_spark = hasattr(frame, "stat") and hasattr(frame, "agg")
        if is_spark:
            import pyspark.sql.functions as F  # noqa: N812

            baseline = frame.filter(F.col(split_column) == "train")
            actual = frame.filter(F.col(split_column) == "valid")
            empty = baseline.limit(1).count() == 0 or actual.limit(1).count() == 0
        else:
            baseline = frame[frame[split_column] == "train"]
            actual = frame[frame[split_column] == "valid"]
            empty = baseline.empty or actual.empty
        if empty:
            msg = (
                f"psi: split column {split_column!r} yielded an empty train or "
                "valid population; expected both labels to be present."
            )
            raise ExecutionError(msg)
        return baseline, actual

    def _apply_subsample_if_needed(
        self: PsiSelector,
        context: StageContext,
        train_df: Any,
        test_df: Any,
    ) -> Tuple[Any, Any]:
        """Формирует стратифицированные подвыборки train и test, если это задано в конфигурации.

        Args:
            context: Контекст этапа со схемой и seed.
            train_df: Обучающий DataFrame.
            test_df: Тестовый или валидационный DataFrame.

        Returns:
            Кортеж DataFrame (subsampled_train, subsampled_test).
        """
        subsample_rows = self.config.subsample_rows
        
        if subsample_rows is None:
            return train_df, test_df
        
        # Get target column from schema
        target_col = context.schema.target
        if target_col is None:
            logger.warning(
                "PSISelector: subsample_rows configured but context.schema.target is None. "
                "Skipping stratified sampling."
            )
            return train_df, test_df
        
        # Prefer PsiConfig.seed, then the runner-assigned step seed.
        seed = (
            self.config.seed
            if self.config.seed is not None
            else step_seed(context)
        )
        
        logger.info(
            f"PSISelector: Applying subsampling to train set (max_rows={subsample_rows})"
        )
        train_sampled = self._apply_sampling(
            train_df,
            target_col,
            subsample_rows,
            seed,
            stratified=context.schema.task_type != "regression",
        )

        logger.info(
            f"PSISelector: Applying subsampling to test set (max_rows={subsample_rows})"
        )
        test_sampled = self._apply_sampling(
            test_df,
            target_col,
            subsample_rows,
            seed,
            stratified=context.schema.task_type != "regression",
        )
        
        return train_sampled, test_sampled

    def _split_by_month(self, context: StageContext, train_df: Any) -> Tuple[Any, Any]:
        """Разделяет train по ``month_column``: последние периоды образуют актуальную выборку.

        Работает с данными Spark и pandas. Ошибки вызывают исключения, а не
        подавляются: прежний резервный вариант возвращал пустую актуальную совокупность,
        из-за чего каждый признак получал psi=0.0 и считался «стабильным».

        Args:
            context: Контекст этапа, сохранённый для совместимости сигнатуры.
            train_df: Входной обучающий DataFrame.

        Returns:
            Кортеж (baseline = более ранние периоды, actual = последние периоды).

        Raises:
            ExecutionError: Если столбец месяца отсутствует или содержит слишком мало
                уникальных периодов для разбиения.
        """
        del context
        month_col = self.config.month_column
        test_months = self.config.test_months
        is_spark = hasattr(train_df, "stat") and hasattr(train_df, "agg")

        if is_spark:
            import pyspark.sql.functions as F  # noqa: N812

            if month_col not in train_df.columns:
                msg = f"psi: month_column={month_col!r} is missing from the train split."
                raise ExecutionError(msg)
            month_df = train_df.select(F.col(month_col).alias("month")).distinct()
            months = [row["month"] for row in month_df.collect() if row["month"] is not None]
        else:
            if month_col not in train_df.columns:
                msg = f"psi: month_column={month_col!r} is missing from the train split."
                raise ExecutionError(msg)
            months = [value for value in train_df[month_col].dropna().unique()]

        if len(months) <= test_months:
            msg = (
                f"psi mode='month_over_month' needs more than test_months="
                f"{test_months} distinct periods in {month_col!r}; found {len(months)}. "
                "Provide a longer history or lower test_months."
            )
            raise ExecutionError(msg)

        cutoff_month = sorted(months, reverse=True)[test_months - 1]
        if is_spark:
            import pyspark.sql.functions as F  # noqa: N812

            baseline = train_df.filter(F.col(month_col) < cutoff_month)
            actual = train_df.filter(F.col(month_col) >= cutoff_month)
        else:
            baseline = train_df[train_df[month_col] < cutoff_month]
            actual = train_df[train_df[month_col] >= cutoff_month]

        logger.info(
            "PSISelector: split %r at cutoff %r (%d periods, last %d are the actual set).",
            month_col,
            cutoff_month,
            len(months),
            test_months,
        )
        return baseline, actual

    def _compute_pyspark_psi(
        self, train_df: Any, test_df: Any, feature_cols: List[str], num_bins: int
    ) -> Dict[str, float]:
        """Вычисляет PSI в PySpark с пакетной обработкой и безопасным кэшированием."""
        import pyspark.sql.functions as F

        relative_error = self.config.relative_error
        batch_size = self.config.batch_size

        # 1. Защита от двойного вычисления тяжелого lineage (Persist)
        is_train_cached = getattr(train_df, "is_cached", False)
        should_unpersist = False

        if not is_train_cached:
            try:
                train_df = train_df.persist()
                should_unpersist = True
            except Exception as e:
                logger.debug(f"Не удалось закешировать train_df: {e}")

        try:
            probabilities = [i / num_bins for i in range(1, num_bins)]

            # 2. Быстрое квантование с настраиваемым relative_error (дефолт 0.001)
            all_quantiles = train_df.stat.approxQuantile(feature_cols, probabilities, relative_error)

            psi_scores: Dict[str, float] = {}

            # 3. Батчевание колонок для предотвращения раздувания плана Catalyst и WholeStageCodegen OOM
            for i in range(0, len(feature_cols), batch_size):
                batch_cols = feature_cols[i : i + batch_size]
                batch_quantiles = all_quantiles[i : i + batch_size]

                exp_agg_exprs = []
                act_agg_exprs = []
                bin_structure_map = {}

                for col_name, quantiles in zip(batch_cols, batch_quantiles):
                    unique_edges = sorted(list(set([-float("inf")] + quantiles + [float("inf")])))
                    num_numeric_bins = len(unique_edges) - 1
                    bin_structure_map[col_name] = num_numeric_bins

                    null_cond = F.col(col_name).isNull() | F.isnan(F.col(col_name))
                    exp_agg_exprs.append(F.count(F.when(null_cond, 1)).alias(f"{col_name}__bin_null"))
                    act_agg_exprs.append(F.count(F.when(null_cond, 1)).alias(f"{col_name}__bin_null"))

                    for b in range(num_numeric_bins):
                        high = unique_edges[b + 1]
                        low = unique_edges[b]
                        valid_cond = ~null_cond

                        cond = valid_cond & (F.col(col_name) <= high) if b == 0 else valid_cond & (F.col(col_name) > low) & (F.col(col_name) <= high)

                        alias = f"{col_name}__bin_{b}"
                        exp_agg_exprs.append(F.count(F.when(cond, 1)).alias(alias))
                        act_agg_exprs.append(F.count(F.when(cond, 1)).alias(alias))

                exp_results = train_df.agg(*exp_agg_exprs).head().asDict()
                act_results = test_df.agg(*act_agg_exprs).head().asDict()

                # 4. Расчёт математического PSI для батча
                for col_name in batch_cols:
                    n_bins = bin_structure_map[col_name]

                    exp_counts = [exp_results.get(f"{col_name}__bin_{b}", 0) for b in range(n_bins)]
                    exp_counts.append(exp_results.get(f"{col_name}__bin_null", 0))

                    act_counts = [act_results.get(f"{col_name}__bin_{b}", 0) for b in range(n_bins)]
                    act_counts.append(act_results.get(f"{col_name}__bin_null", 0))

                    total_exp = sum(exp_counts)
                    total_act = sum(act_counts)

                    if total_exp == 0 or total_act == 0:
                        psi_scores[col_name] = 0.0
                        continue

                    eps_exp = 1.0 / (2.0 * total_exp)
                    eps_act = 1.0 / (2.0 * total_act)

                    psi_val = 0.0
                    for e_cnt, a_cnt in zip(exp_counts, act_counts):
                        e_prop = e_cnt / total_exp
                        a_prop = a_cnt / total_act

                        e_adj = e_prop if e_prop > 0 else eps_exp
                        a_adj = a_prop if a_prop > 0 else eps_act

                        psi_val += (a_adj - e_adj) * math.log(a_adj / e_adj)

                    psi_scores[col_name] = float(psi_val)

            return psi_scores

        finally:
            if should_unpersist:
                try:
                    train_df.unpersist()
                except Exception:
                    pass

    def _compute_pandas_psi(
        self, train_df: Any, test_df: Any, feature_cols: List[str], num_bins: int, eps: float
    ) -> Dict[str, float]:
        """Вычисляет PSI в Pandas векторизованно через C-API np.histogram."""
        from joblib import Parallel, delayed

        def _calc_single(col: str) -> float:
            exp_vals = train_df[col].to_numpy() if hasattr(train_df[col], "to_numpy") else np.asarray(train_df[col])
            act_vals = test_df[col].to_numpy() if hasattr(test_df[col], "to_numpy") else np.asarray(test_df[col])

            exp_null_cnt = np.isnan(exp_vals).sum()
            act_null_cnt = np.isnan(act_vals).sum()

            exp_clean = exp_vals[~np.isnan(exp_vals)]
            act_clean = act_vals[~np.isnan(act_vals)]

            if len(exp_clean) == 0 or len(act_clean) == 0:
                return 0.0

            quantiles = np.linspace(0, 100, num_bins + 1)
            bins = np.percentile(exp_clean, quantiles)
            bins = np.unique(bins)

            if len(bins) < 2:
                exp_counts = np.array([len(exp_clean)])
                act_counts = np.array([len(act_clean)])
            else:
                bins[0] = -np.inf
                bins[-1] = np.inf
                exp_counts, _ = np.histogram(exp_clean, bins=bins)
                act_counts, _ = np.histogram(act_clean, bins=bins)

            exp_counts = np.append(exp_counts, exp_null_cnt)
            act_counts = np.append(act_counts, act_null_cnt)

            total_exp = len(exp_vals)
            total_act = len(act_vals)

            if total_exp == 0 or total_act == 0:
                return 0.0

            exp_pct = exp_counts / total_exp
            act_pct = act_counts / total_act

            eps_exp = 1.0 / (2.0 * total_exp)
            eps_act = 1.0 / (2.0 * total_act)

            exp_pct = np.where(exp_pct > 0, exp_pct, eps_exp)
            act_pct = np.where(act_pct > 0, act_pct, eps_act)

            return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))

        n_jobs = self.config.n_jobs
        results = Parallel(n_jobs=n_jobs)(delayed(_calc_single)(col) for col in feature_cols)
        return dict(zip(feature_cols, results))

    def _make_decision(
        self, feature: str, keep: bool, score: float, threshold: float
    ) -> FeatureDecision:
        reason = (
            f"PSI ({score:.4f}) <= threshold ({threshold})"
            if keep
            else f"PSI ({score:.4f}) > threshold ({threshold})"
        )

        return FeatureDecision(
            feature=feature,
            keep=keep,
            value=score,
            threshold=threshold,
            stage=self.stage_name,
            method=self.method_name,
            reason=reason,
        )
