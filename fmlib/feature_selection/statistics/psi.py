"""Population Stability Index filter (Production Ready & Scalable)."""

from __future__ import annotations

import math
import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from fmlib.feature_selection.base import FeatureDecision, StageContext, step_seed
from fmlib.feature_selection.config import PsiConfig
from fmlib.feature_selection.backends.spark import persist_unless_cached
from fmlib.feature_selection.exceptions import ExecutionError
from fmlib.feature_selection.utils.local_data import sample_frame_rows
from fmlib.feature_selection.utils.verbose import emit as verbose_emit
from fmlib.feature_selection.utils.verbose import enabled as verbose_enabled

logger = logging.getLogger(__name__)

_NULL_LEVEL = "__psi_null__"
_OTHER_LEVEL = "__psi_other__"


def psi_from_counts(
    expected: Sequence[float],
    actual: Sequence[float],
) -> float:
    """Population Stability Index from aligned per-bin counts.

    An empty bin on either side takes the conventional ``1 / (2N)`` floor
    instead of producing an infinite log ratio.
    """
    total_expected = float(sum(expected))
    total_actual = float(sum(actual))
    if total_expected <= 0.0 or total_actual <= 0.0:
        return 0.0
    floor_expected = 1.0 / (2.0 * total_expected)
    floor_actual = 1.0 / (2.0 * total_actual)
    psi = 0.0
    for expected_count, actual_count in zip(expected, actual):
        expected_share = expected_count / total_expected or floor_expected
        actual_share = actual_count / total_actual or floor_actual
        psi += (actual_share - expected_share) * math.log(actual_share / expected_share)
    return float(psi)


def _psi_from_level_counts(
    expected: Mapping[str, float],
    actual: Mapping[str, float],
    expected_total: float,
    actual_total: float,
) -> float:
    """PSI over a shared level set, with everything unseen folded into ``other``."""
    levels = list(expected)
    expected_counts = [float(expected.get(level, 0.0)) for level in levels]
    actual_counts = [float(actual.get(level, 0.0)) for level in levels]
    expected_counts.append(max(0.0, expected_total - sum(expected_counts)))
    actual_counts.append(max(0.0, actual_total - sum(actual_counts)))
    return psi_from_counts(expected_counts, actual_counts)


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
    return (
        module_name.startswith("pyspark")
        and hasattr(data, "select")
        and hasattr(data, "agg")
    )


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
        """Return ``{feature: psi}`` for ``candidates``."""
        feature_cols = list(candidates)
        if not feature_cols:
            return {"values": {}}

        threshold = self.config.threshold
        num_bins = self.config.num_bins

        continuous = set(context.schema.continuous)
        numeric_cols = [name for name in feature_cols if name in continuous]
        categorical_cols = [name for name in feature_cols if name not in continuous]

        train_df, test_df = self._resolve_population_pair(context)
        train_df, test_df = self._apply_subsample_if_needed(context, train_df, test_df)
        is_pyspark = _is_spark_dataframe(train_df)

        if verbose_enabled(context, self.method_name):
            verbose_emit(
                context,
                self.method_name,
                "inputs",
                mode=self.config.mode,
                threshold=threshold,
                num_bins=num_bins,
                n_features=len(feature_cols),
                n_numeric=len(numeric_cols),
                n_categorical=len(categorical_cols),
                backend="spark" if is_pyspark else "pandas",
                subsample_rows=self.config.subsample_rows,
                baseline=context.verbose_log.snapshot_frame(train_df),
                actual=context.verbose_log.snapshot_frame(test_df),
            )

        psi_scores: Dict[str, float] = {}
        if is_pyspark:
            if numeric_cols:
                psi_scores.update(
                    self._compute_pyspark_psi(train_df, test_df, numeric_cols, num_bins),
                )
            if categorical_cols:
                psi_scores.update(
                    self._compute_pyspark_psi_categorical(train_df, test_df, categorical_cols),
                )
        else:
            if numeric_cols:
                psi_scores.update(
                    self._compute_pandas_psi(train_df, test_df, numeric_cols, num_bins),
                )
            if categorical_cols:
                psi_scores.update(
                    self._compute_pandas_psi_categorical(train_df, test_df, categorical_cols),
                )

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
        """Drop remaining features whose PSI exceeds the threshold.

        Only drops are returned, matching every other selector: a ``keep=True``
        decision per candidate inflates ``context.decisions`` by the full
        feature count on every PSI step and is filtered out again downstream.

        A candidate with no measured PSI is an error rather than a silent
        ``0.0``: unscored and stable look identical in the artifact otherwise.
        """
        values = metrics.get("values", metrics)
        if not isinstance(values, Mapping):
            values = {}
        threshold = self.config.threshold
        missing = [name for name in candidates if name not in values]
        if missing:
            msg = (
                f"psi: no PSI value for {len(missing)} candidate(s), "
                f"first: {missing[:5]}. The cached metrics were computed for a "
                "different candidate set; set statistics.cache.force_recompute."
            )
            raise ExecutionError(msg)
        scored = {name: float(values[name]) for name in candidates}
        context.scores[self.method_name] = {
            "threshold": threshold,
            "mode": self.config.mode,
            "num_bins": self.config.num_bins,
            "values": scored,
        }
        return [
            self._make_decision(name, keep=False, score=score, threshold=threshold)
            for name, score in scored.items()
            if score > threshold
        ]

    def _resolve_population_pair(self, context: StageContext) -> Tuple[Any, Any]:
        """Return the (baseline, actual) pair selected by ``config.mode``.

        ``datasets['test']`` is never read: the held-out split must stay
        untouched so the resulting feature set can be judged on data that took
        no part in selection. A missing population raises instead of skipping
        the filter, because a silent skip is indistinguishable in the report
        from "measured and stable".

        Args:
            context: Stage context holding datasets and schema.

        Returns:
            Tuple of (baseline, actual) DataFrames.

        Raises:
            ExecutionError: If the mode's required population is unavailable.
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
        """Split one frame into train/valid rows using a split column.

        Args:
            frame: Frame carrying the split column.
            split_column: Column holding ``train`` / ``valid`` / ``test`` labels.

        Returns:
            Tuple of (train rows, valid rows).

        Raises:
            ExecutionError: If either side of the split is empty.
        """
        is_spark = _is_spark_dataframe(frame)
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
        """Bound both populations with the shared target-stratified sampler.

        Uses ``utils.local_data.sample_frame_rows`` rather than a private copy:
        one sampling semantics across the module, and one Spark action per
        population instead of a groupBy plus two counts taken only for a log line.

        Args:
            context: Stage context with schema and seed.
            train_df: Baseline population.
            test_df: Actual population.

        Returns:
            Tuple of (bounded baseline, bounded actual).
        """
        subsample_rows = self.config.subsample_rows
        if subsample_rows is None:
            return train_df, test_df

        target_col = context.schema.target
        if not target_col:
            logger.warning(
                "PSISelector: subsample_rows is set but FeatureSchema.target is "
                "empty; skipping the bounded sample.",
            )
            return train_df, test_df

        # Prefer PsiConfig.seed, then the runner-assigned step seed.
        seed = self.config.seed if self.config.seed is not None else step_seed(context)
        stratified = context.schema.task_type != "regression"

        bounded = []
        for name, frame in (("baseline", train_df), ("actual", test_df)):
            sampled, rows_in, rows_out = sample_frame_rows(
                frame,
                target_col=target_col,
                max_rows=subsample_rows,
                stratified=stratified,
                seed=seed,
                method_name=self.method_name,
            )
            logger.info(
                "PSISelector: %s population %d -> %d rows (max_rows=%d).",
                name,
                rows_in,
                rows_out,
                subsample_rows,
            )
            bounded.append(sampled)
        return bounded[0], bounded[1]

    def _split_by_month(self, context: StageContext, train_df: Any) -> Tuple[Any, Any]:
        """Split train by ``month_column``: latest periods become the actual set.

        Works on both Spark and pandas inputs. Failures are raised rather than
        swallowed: the previous fallback returned an empty actual population,
        which scores every feature at psi=0.0 and reads as "stable".

        Args:
            context: Stage context, kept for signature compatibility.
            train_df: Input train dataframe.

        Returns:
            Tuple of (baseline = earlier periods, actual = latest periods).

        Raises:
            ExecutionError: If the month column is missing or holds too few
                distinct periods to split.
        """
        del context
        month_col = self.config.month_column
        test_months = self.config.test_months
        is_spark = _is_spark_dataframe(train_df)

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
        """Quantile-bin continuous columns and count both populations per batch.

        Columns are projected onto positional aliases first: a raw feature name
        with a dot or a space breaks both ``F.col`` and the ``{name}__bin_{i}``
        aggregation aliases.
        """
        import pyspark.sql.functions as F

        relative_error = self.config.relative_error
        batch_size = self.config.batch_size
        aliases = {name: f"c{index}" for index, name in enumerate(feature_cols)}
        alias_list = [aliases[name] for name in feature_cols]

        baseline = train_df.select(
            *[_quoted_col(name).alias(aliases[name]) for name in feature_cols],
        )
        actual = test_df.select(
            *[_quoted_col(name).alias(aliases[name]) for name in feature_cols],
        )
        # Both populations are aggregated once per column batch, so both need
        # the same protection from a repeated scan -- not just the baseline.
        baseline, drop_baseline = persist_unless_cached(baseline)
        actual, drop_actual = persist_unless_cached(actual)

        try:
            probabilities = [i / num_bins for i in range(1, num_bins)]
            all_quantiles = baseline.stat.approxQuantile(
                alias_list,
                probabilities,
                relative_error,
            )

            psi_scores: Dict[str, float] = {}
            for start in range(0, len(feature_cols), batch_size):
                batch_cols = feature_cols[start : start + batch_size]
                batch_quantiles = all_quantiles[start : start + batch_size]

                agg_exprs = []
                bin_structure_map = {}
                for col_name, quantiles in zip(batch_cols, batch_quantiles):
                    alias = aliases[col_name]
                    edges = sorted({-float("inf"), *quantiles, float("inf")})
                    n_bins = len(edges) - 1
                    bin_structure_map[col_name] = n_bins

                    value = F.col(alias)
                    null_cond = value.isNull() | F.isnan(value.cast("double"))
                    agg_exprs.append(
                        F.count(F.when(null_cond, 1)).alias(f"{alias}__bin_null"),
                    )
                    for index in range(n_bins):
                        low = edges[index]
                        high = edges[index + 1]
                        valid = ~null_cond
                        cond = (
                            valid & (value <= high)
                            if index == 0
                            else valid & (value > low) & (value <= high)
                        )
                        agg_exprs.append(
                            F.count(F.when(cond, 1)).alias(f"{alias}__bin_{index}"),
                        )

                exp_results = baseline.agg(*agg_exprs).head().asDict()
                act_results = actual.agg(*agg_exprs).head().asDict()

                for col_name in batch_cols:
                    alias = aliases[col_name]
                    n_bins = bin_structure_map[col_name]
                    keys = [f"{alias}__bin_{index}" for index in range(n_bins)]
                    keys.append(f"{alias}__bin_null")
                    psi_scores[col_name] = psi_from_counts(
                        [float(exp_results.get(key) or 0) for key in keys],
                        [float(act_results.get(key) or 0) for key in keys],
                    )
            return psi_scores
        finally:
            drop_actual()
            drop_baseline()

    def _compute_pyspark_psi_categorical(
        self,
        train_df: Any,
        test_df: Any,
        feature_cols: List[str],
    ) -> Dict[str, float]:
        """Compare categorical level distributions between the two populations.

        The level set comes from the baseline, capped at ``max_levels`` most
        frequent values; everything else collapses into one ``other`` bin on
        both sides, which bounds the driver-side result at
        ``n_features * max_levels`` rows regardless of column cardinality.
        Nulls are a level of their own rather than a dropped row.
        """
        max_levels = self.config.max_levels
        batch_size = self.config.batch_size
        scores: Dict[str, float] = {}

        for start in range(0, len(feature_cols), batch_size):
            batch_cols = feature_cols[start : start + batch_size]
            baseline_counts, baseline_total = self._spark_level_counts(
                train_df,
                batch_cols,
                max_levels=max_levels,
            )
            keep = {
                name: set(levels) for name, levels in baseline_counts.items()
            }
            actual_counts, actual_total = self._spark_level_counts(
                test_df,
                batch_cols,
                max_levels=None,
                keep_levels=keep,
            )
            for name in batch_cols:
                scores[name] = _psi_from_level_counts(
                    baseline_counts.get(name, {}),
                    actual_counts.get(name, {}),
                    baseline_total,
                    actual_total,
                )
        return scores

    def _spark_level_counts(
        self,
        frame: Any,
        feature_cols: List[str],
        *,
        max_levels: Optional[int],
        keep_levels: Optional[Dict[str, set]] = None,
    ) -> Tuple[Dict[str, Dict[str, float]], int]:
        """Return ``{feature: {level: count}}`` plus the population row count."""
        import pyspark.sql.functions as F
        from pyspark.sql import Window

        pieces = []
        for name in feature_cols:
            level = F.coalesce(
                _quoted_col(name).cast("string"),
                F.lit(_NULL_LEVEL),
            )
            pieces.append(
                frame.select(F.lit(name).alias("feature"), level.alias("level")),
            )
        stacked = pieces[0]
        for piece in pieces[1:]:
            stacked = stacked.unionByName(piece)

        grouped = stacked.groupBy("feature", "level").agg(F.count("*").alias("n"))
        if max_levels is not None:
            window = Window.partitionBy("feature").orderBy(
                F.col("n").desc(),
                F.col("level").asc(),
            )
            grouped = (
                grouped.withColumn("__rank__", F.row_number().over(window))
                .where(F.col("__rank__") <= max_levels)
                .drop("__rank__")
            )
        elif keep_levels is not None:
            allowed = [
                (name, level)
                for name, levels in keep_levels.items()
                for level in levels
            ]
            if allowed:
                spark = frame.sparkSession
                allowed_df = F.broadcast(
                    spark.createDataFrame(allowed, ["feature", "level"]),
                )
                grouped = grouped.join(allowed_df, ["feature", "level"], "inner")
            else:
                grouped = grouped.where(F.lit(False))

        try:
            rows = grouped.collect()
            total = int(frame.count())
        except Exception as exc:  # noqa: BLE001 - Spark/Py4J exception hierarchy
            msg = (
                "psi: Spark aggregation failed while counting categorical "
                f"levels. Root cause: {_root_cause(exc)}."
            )
            raise ExecutionError(msg) from exc

        counts: Dict[str, Dict[str, float]] = {name: {} for name in feature_cols}
        for row in rows:
            counts[row["feature"]][row["level"]] = float(row["n"] or 0)
        return counts, total

    def _compute_pandas_psi(
        self, train_df: Any, test_df: Any, feature_cols: List[str], num_bins: int
    ) -> Dict[str, float]:
        """Quantile-bin continuous columns with ``np.histogram``."""
        from joblib import Parallel, delayed

        def _calc_single(col: str) -> float:
            exp_vals = pd.to_numeric(train_df[col], errors="coerce").to_numpy(dtype=float)
            act_vals = pd.to_numeric(test_df[col], errors="coerce").to_numpy(dtype=float)

            exp_null = np.isnan(exp_vals)
            act_null = np.isnan(act_vals)
            exp_clean = exp_vals[~exp_null]
            act_clean = act_vals[~act_null]

            if len(exp_clean) == 0 or len(act_clean) == 0:
                return 0.0

            edges = np.unique(np.percentile(exp_clean, np.linspace(0, 100, num_bins + 1)))
            if len(edges) < 2:
                exp_counts = np.array([float(len(exp_clean))])
                act_counts = np.array([float(len(act_clean))])
            else:
                edges[0] = -np.inf
                edges[-1] = np.inf
                exp_counts = np.histogram(exp_clean, bins=edges)[0].astype(float)
                act_counts = np.histogram(act_clean, bins=edges)[0].astype(float)

            exp_counts = np.append(exp_counts, float(exp_null.sum()))
            act_counts = np.append(act_counts, float(act_null.sum()))
            return psi_from_counts(exp_counts, act_counts)

        results = Parallel(n_jobs=self.config.n_jobs)(
            delayed(_calc_single)(col) for col in feature_cols
        )
        return dict(zip(feature_cols, results))

    def _compute_pandas_psi_categorical(
        self, train_df: Any, test_df: Any, feature_cols: List[str]
    ) -> Dict[str, float]:
        """Compare categorical level distributions on already-local frames."""
        max_levels = self.config.max_levels
        scores: Dict[str, float] = {}
        baseline_total = float(len(train_df))
        actual_total = float(len(test_df))
        for name in feature_cols:
            exp = train_df[name].astype("string").fillna(_NULL_LEVEL)
            act = test_df[name].astype("string").fillna(_NULL_LEVEL)
            exp_counts = exp.value_counts()
            if max_levels is not None and len(exp_counts) > max_levels:
                exp_counts = exp_counts.iloc[:max_levels]
            levels = list(exp_counts.index)
            act_counts = act.value_counts()
            scores[name] = _psi_from_level_counts(
                {level: float(exp_counts[level]) for level in levels},
                {
                    level: float(act_counts.get(level, 0.0))
                    for level in levels
                },
                baseline_total,
                actual_total,
            )
        return scores

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
