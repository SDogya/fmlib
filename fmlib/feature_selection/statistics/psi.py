"""Population Stability Index filter (Production Ready & Scalable)."""

from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from fmlib.feature_selection.base import FeatureDecision, StageContext, step_seed
from fmlib.feature_selection.config import PsiConfig
from fmlib.feature_selection.utils.verbose import emit as verbose_emit
from fmlib.feature_selection.utils.verbose import enabled as verbose_enabled

logger = logging.getLogger(__name__)


class PsiSelector:
    """Exclude unstable features via Population Stability Index (PSI).

    Optimized for large-scale PySpark DataFrames (Feature Batching, Managed Persist, 
    Approximate Quantiles) and Pandas DataFrames (C-API Vectorized).

    Args:
        config: PSI filter settings.
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
        """Применяет стратифицированную выборку на основе целевой колонки для PySpark.

        Args:
            df: Исходный PySpark DataFrame.
            target_col: Целевая колонка для стратификации.
            max_rows: Максимальное количество строк после выборки.
            seed: Сид для репродуцируемости.

        Returns:
            DataFrame с отсемплированными данными.
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
        """Применяет стратифицированную выборку на основе целевой колонки для Pandas.

        Args:
            df: Исходный Pandas DataFrame.
            target_col: Целевая колонка для стратификации.
            max_rows: Максимальное количество строк после выборки.
            seed: Сид для репродуцируемости.

        Returns:
            DataFrame с отсемплированными данными.
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
        """Применяет стратифицированную выборку на основе целевой колонки.

        Args:
            df: Исходный DataFrame (PySpark или Pandas).
            target_col: Целевая колонка для стратификации.
            max_rows: Максимальное количество строк после выборки.
            seed: Сид для репродуцируемости.

        Returns:
            DataFrame с отсемплированными данными.
        """
        # Check if PySpark DataFrame
        if hasattr(df, "stat") and hasattr(df, "agg"):
            return self._apply_stratified_sampling_pyspark(df, target_col, max_rows, seed)
        else:
            return self._apply_stratified_sampling_pandas(df, target_col, max_rows, seed)

    def select(
        self: PsiSelector,
        context: StageContext,
        candidates: Sequence[str],
    ) -> list[FeatureDecision]:
        feature_cols = list(candidates)
        if not feature_cols:
            return []

        threshold = self.config.threshold
        num_bins = self.config.num_bins
        eps = self.config.eps

        train_df, test_df = self._extract_dataframes(context)

        if train_df is None or test_df is None:
            logger.warning("PSISelector: Не найдены baseline/actual датасеты. Пропуск фильтрации.")
            return [self._make_decision(col, keep=True, score=0.0, threshold=threshold) for col in feature_cols]

        # Apply stratified sampling if configured
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

        decisions: list[FeatureDecision] = []
        for col in feature_cols:
            score = psi_scores.get(col, 0.0)
            keep = score <= threshold
            decisions.append(self._make_decision(col, keep=keep, score=score, threshold=threshold))

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

        return decisions

    def _extract_dataframes(self, context: StageContext) -> Tuple[Any, Any]:
        """Безопасное извлечение DataFrames без bool(df) контекста (защита от ValueError в Spark).

        Args:
            context: Stage context for schema and datasets access.

        Returns:
            Tuple of (train_df, test_df) dataframes.
        """
        def _pick_first_not_none(obj: Any, attrs: Sequence[str]) -> Any:
            for attr in attrs:
                val = getattr(obj, attr, None)
                if val is not None:
                    return val
            return None

        train_attrs = ["baseline_df", "train_df", "reference_df", "expected_df", "df_train"]
        test_attrs = ["df", "test_df", "actual_df", "current_df", "df_test"]

        train_df = _pick_first_not_none(context, train_attrs)
        test_df = _pick_first_not_none(context, test_attrs)

        # If not found in context attributes, check context.datasets
        if train_df is None and hasattr(context, "datasets"):
            train_df = context.datasets.get("train")
        if test_df is None and hasattr(context, "datasets"):
            test_df = context.datasets.get("test")

        if (train_df is None or test_df is None) and hasattr(context, "get_data"):
            try:
                data = context.get_data()
                if isinstance(data, tuple) and len(data) == 2:
                    train_df = train_df if train_df is not None else data[0]
                    test_df = test_df if test_df is not None else data[1]
                elif isinstance(data, dict):
                    train_df = train_df if train_df is not None else _pick_first_not_none(data, train_attrs)
                    test_df = test_df if test_df is not None else _pick_first_not_none(data, test_attrs)
            except Exception:
                pass

        # If test_df is not provided and mode is month_over_month, split train by months
        if test_df is None and train_df is not None and self.config.mode == "month_over_month":
            train_df, test_df = self._split_by_month(context, train_df)

        return train_df, test_df

    def _apply_subsample_if_needed(
        self: PsiSelector,
        context: StageContext,
        train_df: Any,
        test_df: Any,
    ) -> Tuple[Any, Any]:
        """Apply stratified subsampling to train and test DataFrames if configured.

        Args:
            context: Stage context with schema and seed.
            train_df: Training DataFrame.
            test_df: Test/Validation DataFrame.

        Returns:
            Tuple of (subsampled_train, subsampled_test) DataFrames.
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
        
        logger.info(f"PSISelector: Applying stratified subsampling to train set (max_rows={subsample_rows})")
        train_sampled = self._apply_stratified_sampling(train_df, target_col, subsample_rows, seed)
        
        logger.info(f"PSISelector: Applying stratified subsampling to test set (max_rows={subsample_rows})")
        test_sampled = self._apply_stratified_sampling(test_df, target_col, subsample_rows, seed)
        
        return train_sampled, test_sampled

    def _split_by_month(self, context: StageContext, train_df: Any) -> Tuple[Any, Any]:
        """Split train dataframe by month_part column for train/test split.

        Args:
            context: Stage context for schema access.
            train_df: Input train dataframe.

        Returns:
            Tuple of (train_split, test_split) dataframes.
        """
        month_col = self.config.month_column
        test_months = self.config.test_months

        import pyspark.sql.functions as F
        from pyspark.sql import Window

        # Get unique months and select latest N for test
        try:
            # Collect all unique months
            month_df = train_df.select(F.col(month_col).alias("month")).distinct()
            months = [row["month"] for row in month_df.collect()]

            if len(months) <= test_months:
                # Not enough months to split, use all for train, empty test
                logger.warning(
                    f"PSISelector: Not enough unique months ({len(months)}) for test split. "
                    f"Need at least {test_months + 1} months. Using all data for train."
                )
                return train_df, train_df.limit(0)

            # Sort months and get the cutoff
            sorted_months = sorted(months, reverse=True)
            cutoff_month = sorted_months[test_months - 1]

            # Split: train = months < cutoff, test = months >= cutoff
            train_split = train_df.filter(F.col(month_col) < cutoff_month)
            test_split = train_df.filter(F.col(month_col) >= cutoff_month)

            # Log split info
            train_count = train_split.count()
            test_count = test_split.count()
            logger.info(
                f"PSISelector: Split by month '{month_col}'. "
                f"Cutoff month: {cutoff_month}. Train: {train_count} rows, Test: {test_count} rows."
            )

            return train_split, test_split
        except Exception as e:
            logger.warning(
                f"PSISelector: Failed to split by month '{month_col}': {e}. "
                f"Using original train as both train and test."
            )
            return train_df, train_df.limit(0)

    def _compute_pyspark_psi(
        self, train_df: Any, test_df: Any, feature_cols: List[str], num_bins: int
    ) -> Dict[str, float]:
        """Оптимизированный расчёт PSI на PySpark с пакетированием и безопасным кэшированием."""
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
        """Векторный расчёт PSI в Pandas через C-API np.histogram."""
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
