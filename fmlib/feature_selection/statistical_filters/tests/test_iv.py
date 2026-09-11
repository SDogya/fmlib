"""Тесты отбора по информационной ценности (IV) для промышленного использования.

Варианты pandas работают с реальными DataFrame. Варианты Spark используют общую реальную SparkSession.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pytest

from fmlib.feature_selection.base import StageContext
from fmlib.feature_selection.config import FeatureSelectionConfig, IvConfig, VerboseConfig
from fmlib.feature_selection.utils.conftest import require_spark_session
from fmlib.feature_selection.exceptions import ConfigError, ExecutionError
from fmlib.feature_selection.pipeline import FeatureSelectionPipeline
from fmlib.feature_selection.schema import FeatureSchema
from fmlib.feature_selection.statistical_filters import iv as iv_mod
from fmlib.feature_selection.statistical_filters.iv import (
    IvSelector,
    _binary_mapping,
    _categorical_bins_pandas,
    _count_exprs,
    _counts_from_row,
    _is_null,
    _is_spark_dataframe,
    _levels_to_keep,
    _merge_categorical_counts,
    _quantile_bins_pandas,
    _quoted_col,
    _root_cause,
    information_value,
    information_value_from_bins,
)
from fmlib.feature_selection.utils.verbose import VerboseRecorder

_REPO_ROOT = Path(__file__).resolve().parents[4]
_NULL = iv_mod._NULL_LEVEL
_OTHER = iv_mod._OTHER_LEVEL


def _frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = np.array([0] * (n // 2) + [1] * (n - n // 2))
    strong = y.astype(float) + rng.normal(0.0, 0.85, n)
    weak = rng.normal(0.0, 1.0, n)
    leak = y.astype(float)
    cat = np.where(y == 1, "yes", "no")
    flip = rng.random(n) < 0.04
    cat = np.where(flip, np.where(cat == "yes", "no", "yes"), cat)
    null_flag = np.array(strong, dtype=object)
    for index in range(0, n, 12):
        if y[index] == 1:
            null_flag[index] = None
    return pd.DataFrame(
        {
            "strong": strong,
            "weak": weak,
            "leak": leak,
            "cat_signal": cat,
            "null_flag": null_flag,
            "response": y,
        },
    )


def _context(
    frame: object,
    *,
    task_type: str = "binary_classification",
    target: str = "response",
    categorical: tuple[str, ...] = ("cat_signal", "null_flag"),
    continuous: tuple[str, ...] = ("strong", "weak", "leak"),
    spark: object | None = None,
    verbose_iv: bool = False,
) -> StageContext:
    schema = FeatureSchema(
        categorical=categorical,
        continuous=continuous,
        target=target,
        task_type=task_type,
    )
    recorder = VerboseRecorder(verbose=VerboseConfig(iv=verbose_iv))
    return StageContext(
        spark=spark if spark is not None else require_spark_session(),
        datasets={"train": frame},
        schema=schema,
        config=FeatureSelectionConfig(),
        seed=0,
        candidates=schema.candidate_features(),
        verbose_log=recorder,
    )


def _manual_iv(goods: Sequence[float], bads: Sequence[float], eps: float) -> float:
    total_good = float(sum(goods))
    total_bad = float(sum(bads))
    if total_good <= 0.0 or total_bad <= 0.0:
        return 0.0
    value = 0.0
    for good, bad in zip(goods, bads, strict=False):
        if good <= 0.0 and bad <= 0.0:
            continue
        dist_good = good / total_good
        dist_bad = bad / total_bad
        value += (dist_good - dist_bad) * math.log((dist_good + eps) / (dist_bad + eps))
    return float(value)


# ---------------------------------------------------------------------------
# Block 1: pure math and helpers (no mocks of IV functions)
# ---------------------------------------------------------------------------


class TestMathAndHelperFunctions:
    def test_information_value_zero_when_all_good_or_all_bad(self) -> None:
        eps = 1e-4
        assert information_value([0.0, 0.0], [10.0, 5.0], eps) == 0.0
        assert information_value([10.0, 5.0], [0.0, 0.0], eps) == 0.0
        assert information_value([], [], eps) == 0.0

    def test_information_value_skips_empty_bins_and_matches_formula(self) -> None:
        eps = 1e-4
        goods = [50.0, 0.0, 50.0]
        bads = [10.0, 0.0, 90.0]
        expected = _manual_iv(goods, bads, eps)
        assert information_value(goods, bads, eps) == pytest.approx(expected)
        assert expected == pytest.approx(
            _manual_iv([50.0, 50.0], [10.0, 90.0], eps),
        )

    def test_information_value_zero_good_nonzero_bad_is_not_skipped(self) -> None:
        eps = 1e-4
        goods = [10.0, 0.0]
        bads = [10.0, 5.0]
        expected = _manual_iv(goods, bads, eps)
        assert information_value(goods, bads, eps) == pytest.approx(expected)
        assert expected != 0.0

    def test_information_value_from_bins_groups_in_pd_unique_order(self) -> None:
        y = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        bins = np.array(["b", "b", "a", "a", "a"], dtype=object)
        unique = list(pd.unique(bins))
        assert unique == ["b", "a"]
        expected_goods = [2.0, 0.0]
        expected_bads = [0.0, 3.0]
        eps = 1e-4
        got = information_value_from_bins(y, bins, eps=eps)
        assert got == pytest.approx(information_value(expected_goods, expected_bads, eps))

    def test_quantile_bins_all_null_become_iv_null(self) -> None:
        series = pd.Series([None, np.nan, None])
        labels = _quantile_bins_pandas(series, num_bins=4)
        assert list(labels) == [_NULL, _NULL, _NULL]

    def test_quantile_bins_constant_column_uses_c0_and_null(self) -> None:
        series = pd.Series([3.0, 3.0, None, 3.0])
        labels = _quantile_bins_pandas(series, num_bins=8)
        assert list(labels) == ["c0", "c0", _NULL, "c0"]

    def test_quantile_bins_regular_qcut_labels(self) -> None:
        series = pd.Series([float(i) for i in range(20)])
        labels = _quantile_bins_pandas(series, num_bins=4)
        assert set(labels) <= {f"c{i}" for i in range(4)}
        assert len(set(labels)) >= 2
        assert labels[0].startswith("c")
        assert labels[-1].startswith("c")

    def test_quantile_bins_qcut_valueerror_falls_back_to_c0(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def boom(*_args: object, **_kwargs: object) -> None:
            msg = "Bin edges must be unique"
            raise ValueError(msg)

        monkeypatch.setattr(iv_mod.pd, "qcut", boom)
        series = pd.Series([1.0, 2.0, 3.0, 4.0])
        labels = _quantile_bins_pandas(series, num_bins=4)
        assert list(labels) == ["c0", "c0", "c0", "c0"]

    def test_categorical_bins_nulls_rare_merge_and_max_levels(self) -> None:
        series = pd.Series(["a", "a", "a", "a", "b", "b", "c", None])
        labels = _categorical_bins_pandas(series, min_bin_share=0.3, max_levels=1)
        assert labels[-1] == _NULL
        assert labels[0] == "a"
        assert "c" not in set(labels)
        assert _OTHER in set(labels)
        assert set(labels) <= {"a", _OTHER, _NULL}

    def test_levels_to_keep_sorts_by_frequency_then_name(self) -> None:
        counts = {"b": 5, "a": 5, "c": 9, "rare": 1}
        kept = _levels_to_keep(counts, n_rows=20, min_bin_share=0.0, max_levels=2)
        assert kept == {"c", "a"}

    def test_levels_to_keep_filters_share_but_keeps_null_level(self) -> None:
        counts = {"common": 80, "rare": 2, _NULL: 1}
        kept = _levels_to_keep(counts, n_rows=83, min_bin_share=0.1, max_levels=None)
        assert "common" in kept
        assert "rare" not in kept
        assert _NULL in kept

    def test_levels_to_keep_skips_share_when_nrows_zero_or_share_zero(self) -> None:
        counts = {"a": 1}
        assert _levels_to_keep(counts, n_rows=0, min_bin_share=0.5, max_levels=None) == {"a"}
        assert _levels_to_keep(counts, n_rows=10, min_bin_share=0.0, max_levels=None) == {"a"}

    def test_levels_to_keep_max_levels_slice(self) -> None:
        counts = {"a": 3, "b": 2, "c": 1}
        assert _levels_to_keep(counts, n_rows=6, min_bin_share=0.0, max_levels=1) == {"a"}
        assert _levels_to_keep(counts, n_rows=6, min_bin_share=0.0, max_levels=10) == {"a", "b", "c"}

    def test_merge_categorical_counts_combines_other_and_keeps_null(self) -> None:
        levels = [
            ("keep", 40.0, 10.0),
            ("rare_a", 1.0, 1.0),
            ("rare_b", 2.0, 0.0),
            (_NULL, 5.0, 7.0),
        ]
        goods, bads = _merge_categorical_counts(
            levels,
            min_bin_share=0.2,
            max_levels=None,
        )
        keep_set = _levels_to_keep(
            {"keep": 50, "rare_a": 2, "rare_b": 2},
            n_rows=66,
            min_bin_share=0.2,
            max_levels=None,
        )
        seen: dict[str, tuple[float, float]] = {}
        keys_in_order: list[str] = []
        for level, good, bad in levels:
            key = level if level == _NULL or level in keep_set else _OTHER
            if key not in seen:
                keys_in_order.append(key)
                seen[key] = (0.0, 0.0)
            seen[key] = (seen[key][0] + good, seen[key][1] + bad)
        assert goods == [seen[key][0] for key in keys_in_order]
        assert bads == [seen[key][1] for key in keys_in_order]
        assert seen[_NULL] == (5.0, 7.0)
        assert seen[_OTHER] == (3.0, 1.0)
        assert sum(goods) == pytest.approx(48.0)
        assert sum(bads) == pytest.approx(18.0)

    def test_binary_mapping_numeric_and_bool(self) -> None:
        assert _binary_mapping([0, 1], method="iv") == {0: 0.0, 1: 1.0}
        assert _binary_mapping([0.0, 1.0], method="iv") == {0.0: 0.0, 1.0: 1.0}
        mapped = _binary_mapping([False, True], method="iv")
        assert mapped[False] == 0.0
        assert mapped[True] == 1.0

    def test_binary_mapping_string_labels_sorted(self) -> None:
        assert _binary_mapping(["yes", "no"], method="iv") == {"no": 0.0, "yes": 1.0}

    def test_binary_mapping_incomparable_types_sort_by_str(self) -> None:
        mapped = _binary_mapping([1, "a"], method="iv")
        ordered = sorted([1, "a"], key=str)
        assert mapped == {ordered[0]: 0.0, ordered[1]: 1.0}

    def test_binary_mapping_numpy(self) -> None:
        assert _binary_mapping(np.array([0, 1]), method="iv") == {0: 0.0, 1: 1.0}

    @pytest.mark.parametrize("labels", [[], [None, np.nan, pd.NA], ["only"], [0], [1, 1], [1, 1, 1]])
    def test_binary_mapping_fewer_than_two_classes_raises(self, labels: list[Any]) -> None:
        with pytest.raises(ExecutionError, match="target must be binary with exactly two classes"):
            _binary_mapping(labels, method="iv")

    def test_binary_mapping_more_than_two_classes_raises(self) -> None:
        with pytest.raises(ExecutionError, match="target must be binary"):
            _binary_mapping([0, 1, 2], method="iv")

    def test_is_null_scalars_and_exception_branch(self) -> None:
        assert _is_null(None) is True
        assert _is_null(np.nan) is True
        assert _is_null(pd.NA) is True
        assert _is_null(0) is False
        assert _is_null("x") is False

        class NoArray:
            def __array__(self: NoArray) -> None:
                msg = "cannot convert"
                raise TypeError(msg)

        assert _is_null(NoArray()) is False
        assert _is_null([1, 2]) is False

    def test_root_cause_java_cause_and_plain(self) -> None:
        class SparkFailureError(Exception):
            java_exception = "org.apache.spark.SparkException: boom\nmore"

        assert _root_cause(SparkFailureError("outer")) == "org.apache.spark.SparkException: boom"
        chained = RuntimeError("outer")
        chained.__cause__ = ValueError("inner\nline2")
        assert _root_cause(chained) == "inner"
        assert _root_cause(ValueError("only\nsecond")) == "only"


# ---------------------------------------------------------------------------
# Block 2: IvSelector.select contracts
# ---------------------------------------------------------------------------


class TestIvSelectorValidationAndDecisions:
    def test_empty_candidates_return_empty_without_scores(self) -> None:
        context = _context(_frame())
        assert IvSelector(IvConfig()).select(context, []) == []
        assert "iv" not in context.scores

    def test_missing_target_raises_config_error(self) -> None:
        context = _context(_frame())
        context.schema = SimpleNamespace(  # type: ignore[assignment]
            target=None,
            task_type="binary_classification",
            categorical=("cat_signal",),
            continuous=("strong",),
        )
        with pytest.raises(ConfigError, match=r"iv requires FeatureSchema\.target"):
            IvSelector(IvConfig()).select(context, ["strong"])

    def test_non_binary_task_raises_config_error(self) -> None:
        frame = _frame()
        for task_type in ("regression", "classification"):
            schema = FeatureSchema(
                categorical=("cat_signal", "null_flag"),
                continuous=("strong", "weak", "leak"),
                target="response",
                task_type=task_type,
            )
            context = StageContext(
                spark=None,
                datasets={"train": frame},
                schema=schema,
                config=FeatureSelectionConfig(),
                seed=0,
                candidates=schema.candidate_features(),
            )
            with pytest.raises(ConfigError, match="binary_classification"):
                IvSelector(IvConfig()).select(context, ["strong"])

    def test_unsupported_train_type_raises_execution_error(self) -> None:
        context = _context(_frame())
        context.datasets["train"] = [1, 2, 3]
        with pytest.raises(ExecutionError, match="unsupported train split type"):
            IvSelector(IvConfig()).select(context, ["strong"])

    def test_low_iv_drops_weak_feature(self) -> None:
        frame = _frame()
        context = _context(frame, categorical=("cat_signal",), continuous=("strong", "weak"))
        decisions = IvSelector(IvConfig(threshold=0.02, num_bins=8)).select(
            context,
            ["strong", "weak", "cat_signal"],
        )
        dropped = {item.feature: item for item in decisions}
        assert dropped["weak"].keep is False
        assert dropped["weak"].reason == "low_iv"
        assert dropped["weak"].value is not None
        assert dropped["weak"].value < 0.02
        assert dropped["weak"].threshold == 0.02
        assert "strong" not in dropped
        assert "cat_signal" not in dropped
        scores = context.scores["iv"]["values"]
        assert scores["strong"] > scores["weak"]
        assert scores["cat_signal"] >= 0.02

    def test_mid_range_iv_is_kept(self) -> None:
        frame = _frame()
        context = _context(frame, categorical=(), continuous=("strong", "weak", "leak"))
        config = IvConfig(threshold=0.02, num_bins=8, max_threshold=8.0)
        decisions = IvSelector(config).select(context, ["strong", "weak", "leak"])
        by_feature = {item.feature: item for item in decisions}
        strong_iv = context.scores["iv"]["values"]["strong"]
        assert 0.02 <= strong_iv <= 8.0
        assert "strong" not in by_feature
        assert by_feature["weak"].reason == "low_iv"
        assert by_feature["leak"].reason == "high_iv"

    def test_high_iv_drops_leaky_feature(self) -> None:
        frame = _frame()
        context = _context(frame, categorical=(), continuous=("leak",))
        decisions = IvSelector(IvConfig(threshold=0.02, max_threshold=1.0, num_bins=8)).select(
            context,
            ["leak"],
        )
        assert len(decisions) == 1
        assert decisions[0].feature == "leak"
        assert decisions[0].keep is False
        assert decisions[0].reason == "high_iv"
        assert decisions[0].value is not None
        assert decisions[0].value > 1.0
        assert decisions[0].threshold == 1.0

    def test_debug_emit_payload_with_max_threshold(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_emit(_context: object, method: str, stage: str, **payload: object) -> None:
            captured["method"] = method
            captured["stage"] = stage
            captured["payload"] = payload

        monkeypatch.setattr(iv_mod, "verbose_emit", fake_emit)
        frame = _frame()
        context = _context(
            frame,
            categorical=(),
            continuous=("strong", "weak", "leak"),
            verbose_iv=True,
        )
        config = IvConfig(threshold=0.02, num_bins=8, max_threshold=8.0)
        IvSelector(config).select(context, ["strong", "weak", "leak"])
        values = list(context.scores["iv"]["values"].values())
        payload = captured["payload"]
        assert captured["method"] == "iv"
        assert captured["stage"] == "stats"
        assert payload["backend"] == "pandas"
        assert payload["threshold"] == 0.02
        assert payload["max_threshold"] == 8.0
        assert payload["num_bins"] == 8
        assert payload["n_evaluated"] == 3
        assert payload["n_below_threshold"] == sum(1 for value in values if value < 0.02)
        assert payload["n_above_max_threshold"] == sum(1 for value in values if value > 8.0)
        assert payload["iv_min"] == min(values)
        assert payload["iv_max"] == max(values)
        assert payload["iv_mean"] == round(sum(values) / len(values), 6)

    def test_debug_emit_without_max_threshold_sets_n_high_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured: dict[str, Any] = {}

        def fake_emit(_context: object, method: str, stage: str, **payload: object) -> None:
            captured["payload"] = payload

        monkeypatch.setattr(iv_mod, "verbose_emit", fake_emit)
        context = _context(
            _frame(),
            categorical=(),
            continuous=("strong",),
            verbose_iv=True,
        )
        IvSelector(IvConfig(threshold=0.02, num_bins=8)).select(context, ["strong"])
        assert captured["payload"]["max_threshold"] is None
        assert captured["payload"]["n_above_max_threshold"] == 0

    def test_debug_skipped_when_scores_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[object] = []
        monkeypatch.setattr(iv_mod, "verbose_emit", lambda *a, **k: calls.append(1))
        monkeypatch.setattr(IvSelector, "_compute_iv_pandas", lambda *_a, **_k: {})
        context = _context(_frame(), categorical=(), continuous=("strong",), verbose_iv=True)
        decisions = IvSelector(IvConfig()).select(context, ["strong"])
        assert decisions == []
        assert calls == []
        assert context.scores["iv"]["values"] == {}

    def test_debug_not_emitted_when_verbose_disabled(self) -> None:
        context = _context(_frame(), categorical=(), continuous=("strong",), verbose_iv=False)
        IvSelector(IvConfig()).select(context, ["strong"])
        assert context.verbose_log.events == []


# ---------------------------------------------------------------------------
# Block 3: pandas path
# ---------------------------------------------------------------------------


class TestIvSelectorPandas:
    def test_missing_columns_raise_execution_error(self) -> None:
        frame = pd.DataFrame({"strong": [1.0, 0.0], "response": [1, 0]})
        with pytest.raises(ExecutionError, match=r"columns missing.*absent"):
            IvSelector(IvConfig()).select(
                _context(frame, categorical=(), continuous=("strong", "absent")),
                ["absent"],
            )

    def test_all_target_nulls_raise(self) -> None:
        frame = pd.DataFrame(
            {
                "num": [1.0, 2.0, 3.0],
                "cat": ["a", "b", "a"],
                "response": [None, np.nan, None],
            },
        )
        context = _context(frame, categorical=("cat",), continuous=("num",))
        with pytest.raises(ExecutionError, match="Found 0 distinct non-null values"):
            IvSelector(IvConfig(threshold=0.02)).select(context, ["num", "cat"])
        assert "iv" not in context.scores

    def test_mixed_types_end_to_end_matches_helpers(self) -> None:
        frame = pd.DataFrame(
            {
                "num": [0.0, 0.2, 0.4, 0.6, 1.0, 1.2, 1.4, np.nan],
                "cat": ["a", "a", "a", "b", "b", "rare", None, "a"],
                "response": [0, 0, 0, 0, 1, 1, 1, 1],
            },
        )
        config = IvConfig(threshold=0.01, num_bins=3, min_bin_share=0.2, max_levels=2, eps=1e-4)
        context = _context(frame, categorical=("cat",), continuous=("num",))
        decisions = IvSelector(config).select(context, ["num", "cat"])
        working = frame[["num", "cat", "response"]].dropna(subset=["response"])
        y = iv_mod._as_binary_numpy(working["response"], method="iv")
        num_bins = _quantile_bins_pandas(working["num"], num_bins=3)
        cat_bins = _categorical_bins_pandas(working["cat"], min_bin_share=0.2, max_levels=2)
        expected_num = information_value_from_bins(y, num_bins, eps=1e-4)
        expected_cat = information_value_from_bins(y, cat_bins, eps=1e-4)
        scores = context.scores["iv"]["values"]
        assert scores["num"] == pytest.approx(expected_num)
        assert scores["cat"] == pytest.approx(expected_cat)
        by_feature = {item.feature: item for item in decisions}
        for name, value in scores.items():
            if value < config.threshold:
                assert by_feature[name].reason == "low_iv"
            else:
                assert name not in by_feature

    def test_string_and_bool_targets_map_to_binary(self) -> None:
        frame = pd.DataFrame(
            {
                "x": [0.0, 0.0, 1.0, 1.0],
                "response": ["no", "no", "yes", "yes"],
            },
        )
        context = _context(frame, categorical=(), continuous=("x",), target="response")
        IvSelector(IvConfig(threshold=0.0, num_bins=2)).select(context, ["x"])
        assert context.scores["iv"]["values"]["x"] > 0.0

        bool_frame = pd.DataFrame(
            {
                "x": [0.0, 0.0, 1.0, 1.0],
                "response": [False, False, True, True],
            },
        )
        bool_context = _context(bool_frame, categorical=(), continuous=("x",))
        IvSelector(IvConfig(threshold=0.0, num_bins=2)).select(bool_context, ["x"])
        assert bool_context.scores["iv"]["values"]["x"] > 0.0

    def test_multiclass_target_raises(self) -> None:
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "response": [0, 1, 2]})
        with pytest.raises(ExecutionError, match="target must be binary"):
            IvSelector(IvConfig()).select(
                _context(frame, categorical=(), continuous=("x",)),
                ["x"],
            )

    @pytest.mark.parametrize("label", [0, 1, False, True, "only"])
    @pytest.mark.parametrize("entrypoint", ["select", "compute"])
    def test_single_class_target_raises(self, label: Any, entrypoint: str) -> None:
        frame = pd.DataFrame({
            "num": [1.0, 2.0, 3.0, 4.0],
            "cat": ["a", "b", "a", "b"],
            "response": [label, label, None, label],
        })
        context = _context(frame, categorical=("cat",), continuous=("num",))
        selector = IvSelector(IvConfig(threshold=0.02, num_bins=2))
        with pytest.raises(ExecutionError, match="Found 1 distinct non-null values"):
            getattr(selector, entrypoint)(context, ["num", "cat"])
        assert "iv" not in context.scores

    def test_empty_train_raises(self) -> None:
        frame = pd.DataFrame(columns=["num", "response"])
        context = _context(frame, categorical=(), continuous=("num",))
        with pytest.raises(ExecutionError, match="Found 0 distinct non-null values"):
            IvSelector(IvConfig()).select(context, ["num"])
        assert "iv" not in context.scores

    def test_nulls_form_a_separate_bin(self) -> None:
        frame = pd.DataFrame(
            {
                "nullable": [1.0, 1.0, 1.0, None, None, None],
                "response": [0, 0, 0, 1, 1, 1],
            },
        )
        context = _context(frame, categorical=(), continuous=("nullable",))
        decisions = IvSelector(IvConfig(threshold=0.5, num_bins=4)).select(context, ["nullable"])
        assert decisions == []
        assert context.scores["iv"]["values"]["nullable"] > 0.5

    def test_counts_from_row_handles_missing_and_none(self) -> None:
        assert _counts_from_row({}, "x") == (0.0, 0.0)
        assert _counts_from_row({"x__bad": None, "x__n": None}, "x") == (0.0, 0.0)
        assert _counts_from_row({"x__bad": 3, "x__n": 10}, "x") == (3.0, 10.0)


# ---------------------------------------------------------------------------
# Block 4: Spark path
# ---------------------------------------------------------------------------


def _spark_from_pandas(spark: Any, frame: pd.DataFrame) -> Any:
    return spark.createDataFrame(frame)


def _iv_on_pandas_and_spark(
    spark: Any,
    pandas_frame: pd.DataFrame,
    *,
    categorical: tuple[str, ...],
    continuous: tuple[str, ...],
    candidates: list[str],
    config: IvConfig,
    target: str = "response",
) -> tuple[dict[str, float], dict[str, float], set[str], set[str]]:
    pandas_ctx = _context(
        pandas_frame,
        categorical=categorical,
        continuous=continuous,
        target=target,
    )
    spark_ctx = _context(
        _spark_from_pandas(spark, pandas_frame),
        categorical=categorical,
        continuous=continuous,
        target=target,
        spark=spark,
    )
    pandas_dropped = {item.feature for item in IvSelector(config).select(pandas_ctx, candidates)}
    spark_dropped = {item.feature for item in IvSelector(config).select(spark_ctx, candidates)}
    return (
        pandas_ctx.scores["iv"]["values"],
        spark_ctx.scores["iv"]["values"],
        pandas_dropped,
        spark_dropped,
    )


class TestIvSelectorSpark:
    def test_is_spark_dataframe_detection(self, spark: Any) -> None:
        pandas_frame = pd.DataFrame({"a": [1]})
        assert _is_spark_dataframe(pandas_frame) is False
        plain = SimpleNamespace(select=lambda *_a: None, agg=lambda *_a: None)
        assert _is_spark_dataframe(plain) is False
        real = spark.createDataFrame([(1,)], ["a"])
        assert _is_spark_dataframe(real) is True

        class Almost:
            def select(self: Almost, *_args: object) -> None:
                return None

        missing_agg = Almost()
        assert _is_spark_dataframe(missing_agg) is False

    def test_missing_spark_columns_raise(self, spark: Any) -> None:
        train = spark.createDataFrame([(1.0, 0)], ["strong", "response"])
        context = _context(train, categorical=(), continuous=("strong", "absent"), spark=spark)
        with pytest.raises(ExecutionError, match="columns missing from train schema"):
            IvSelector(IvConfig()).select(context, ["absent"])

    def test_non_numeric_continuous_raises(self, spark: Any) -> None:
        string_train = spark.createDataFrame(
            [("a", 0), ("b", 1), ("a", 0), ("b", 1)],
            ["cat_as_num", "response"],
        )
        string_ctx = _context(
            string_train,
            categorical=(),
            continuous=("cat_as_num",),
            spark=spark,
        )
        with pytest.raises(ExecutionError, match="expected a numeric type"):
            IvSelector(IvConfig()).select(string_ctx, ["cat_as_num"])

        bool_train = spark.createDataFrame(
            [(True, 0), (False, 1), (True, 0), (False, 1)],
            ["flag", "response"],
        )
        bool_ctx = _context(bool_train, categorical=(), continuous=("flag",), spark=spark)
        with pytest.raises(ExecutionError, match="BooleanType"):
            IvSelector(IvConfig()).select(bool_ctx, ["flag"])

    def test_persist_and_unpersist_when_not_cached(
        self,
        spark: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from pyspark.sql import DataFrame as SparkDataFrame

        pandas_frame = pd.DataFrame(
            {"num": [0.0, 1.0, 0.0, 1.0], "response": [0, 1, 0, 1]},
        )
        train = spark.createDataFrame(pandas_frame)
        calls = {"persist": 0, "unpersist": 0}
        original_persist = SparkDataFrame.persist
        original_unpersist = SparkDataFrame.unpersist

        def persist(self: Any, *args: Any, **kwargs: Any) -> Any:
            calls["persist"] += 1
            return original_persist(self, *args, **kwargs)

        def unpersist(self: Any, *args: Any, **kwargs: Any) -> Any:
            calls["unpersist"] += 1
            return original_unpersist(self, *args, **kwargs)

        monkeypatch.setattr(SparkDataFrame, "persist", persist)
        monkeypatch.setattr(SparkDataFrame, "unpersist", unpersist)
        IvSelector(IvConfig(num_bins=2, threshold=0.0)).select(
            _context(train, categorical=(), continuous=("num",), spark=spark),
            ["num"],
        )
        assert calls["persist"] >= 1
        assert calls["unpersist"] >= 1

    def test_already_cached_skips_persist(
        self,
        spark: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from pyspark.sql import DataFrame as SparkDataFrame

        pandas_frame = pd.DataFrame(
            {"num": [0.0, 1.0, 0.0, 1.0], "response": [0, 1, 0, 1]},
        )
        train = spark.createDataFrame(pandas_frame)
        original_select = SparkDataFrame.select
        original_persist = SparkDataFrame.persist
        original_unpersist = SparkDataFrame.unpersist
        cached_frames = []

        def select(self: Any, *args: Any, **kwargs: Any) -> Any:
            prepared = original_select(self, *args, **kwargs)
            if prepared.columns == ["c0", "__y__"]:
                original_persist(prepared)
                cached_frames.append(prepared)
            return prepared

        monkeypatch.setattr(SparkDataFrame, "select", select)

        def persist(self: Any, *_args: Any, **_kwargs: Any) -> Any:
            del self
            pytest.fail("persist must not run when the prepared frame is cached")

        monkeypatch.setattr(SparkDataFrame, "persist", persist)
        context = _context(train, categorical=(), continuous=("num",), spark=spark)
        try:
            IvSelector(IvConfig(num_bins=2, threshold=0.0)).select(context, ["num"])
            assert "num" in context.scores["iv"]["values"]
            assert len(cached_frames) == 1
            assert cached_frames[0].is_cached
        finally:
            for prepared in cached_frames:
                original_unpersist(prepared)

    def test_persist_failure_still_computes(
        self,
        spark: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from pyspark.sql import DataFrame as SparkDataFrame

        pandas_frame = pd.DataFrame(
            {"num": [0.0, 1.0, 0.0, 1.0], "response": [0, 1, 0, 1]},
        )
        train = spark.createDataFrame(pandas_frame)

        def persist(self: Any, *_args: Any, **_kwargs: Any) -> Any:
            del self
            raise RuntimeError("cannot persist")

        monkeypatch.setattr(SparkDataFrame, "persist", persist)
        context = _context(train, categorical=(), continuous=("num",), spark=spark)
        IvSelector(IvConfig(num_bins=2, threshold=0.0)).select(context, ["num"])
        assert "num" in context.scores["iv"]["values"]

    def test_unpersist_failure_is_swallowed(
        self,
        spark: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from pyspark.sql import DataFrame as SparkDataFrame

        pandas_frame = pd.DataFrame(
            {"num": [0.0, 1.0, 0.0, 1.0], "response": [0, 1, 0, 1]},
        )
        train = spark.createDataFrame(pandas_frame)
        original_persist = SparkDataFrame.persist
        calls = {"unpersist": 0}

        def persist(self: Any, *args: Any, **kwargs: Any) -> Any:
            return original_persist(self, *args, **kwargs)

        def unpersist(self: Any, *_args: Any, **_kwargs: Any) -> Any:
            del self
            calls["unpersist"] += 1
            raise RuntimeError("cannot unpersist")

        monkeypatch.setattr(SparkDataFrame, "persist", persist)
        monkeypatch.setattr(SparkDataFrame, "unpersist", unpersist)
        context = _context(train, categorical=(), continuous=("num",), spark=spark)
        decisions = IvSelector(IvConfig(num_bins=2, threshold=0.0)).select(context, ["num"])
        assert isinstance(decisions, list)
        assert calls["unpersist"] >= 1
        assert "num" in context.scores["iv"]["values"]

    def test_numeric_and_drop_set_match_pandas(self, spark: Any) -> None:
        pandas_frame = _frame(n=240)[["strong", "weak", "response"]]
        config = IvConfig(threshold=0.02, num_bins=6, relative_error=0.0)
        pandas_scores, spark_scores, pandas_dropped, spark_dropped = _iv_on_pandas_and_spark(
            spark,
            pandas_frame,
            categorical=(),
            continuous=("strong", "weak"),
            candidates=["strong", "weak"],
            config=config,
        )
        assert pandas_dropped == spark_dropped
        assert set(spark_scores) == set(pandas_scores)
        for name, value in spark_scores.items():
            assert value == pytest.approx(pandas_scores[name], rel=0.2, abs=1e-6)

    def test_numeric_batch_size_does_not_change_scores(self, spark: Any) -> None:
        pandas_frame = pd.DataFrame(
            {
                "n0": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                "n1": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                "response": [0, 1, 0, 1, 0, 1, 0, 1],
            },
        )
        train = spark.createDataFrame(pandas_frame)
        candidates = ["n0", "n1"]
        one = _context(train, categorical=(), continuous=("n0", "n1"), spark=spark)
        many = _context(train, categorical=(), continuous=("n0", "n1"), spark=spark)
        IvSelector(IvConfig(num_bins=2, batch_size=1, threshold=0.0, relative_error=0.0)).select(
            one,
            candidates,
        )
        IvSelector(IvConfig(num_bins=2, batch_size=50, threshold=0.0, relative_error=0.0)).select(
            many,
            candidates,
        )
        assert one.scores["iv"]["values"] == many.scores["iv"]["values"]

    def test_approx_quantile_failure_uses_root_cause(
        self,
        spark: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from pyspark.sql.dataframe import DataFrameStatFunctions

        train = spark.createDataFrame(
            pd.DataFrame({"num": [0.0, 1.0, 0.0, 1.0], "response": [0, 1, 0, 1]}),
        )

        def boom(self: Any, *_args: Any, **_kwargs: Any) -> Any:
            del self
            raise RuntimeError("quantile exploded\nmore")

        monkeypatch.setattr(DataFrameStatFunctions, "approxQuantile", boom)
        context = _context(train, categorical=(), continuous=("num",), spark=spark)
        with pytest.raises(ExecutionError, match=r"approxQuantile failed.*quantile exploded"):
            IvSelector(IvConfig(num_bins=2)).select(context, ["num"])

    def test_numeric_agg_failure_uses_root_cause(
        self,
        spark: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from pyspark.sql import DataFrame as SparkDataFrame

        train = spark.createDataFrame(
            pd.DataFrame({"num": [0.0, 1.0, 0.0, 1.0], "response": [0, 1, 0, 1]}),
        )

        def boom(self: Any, *_args: Any, **_kwargs: Any) -> Any:
            del self
            raise RuntimeError("agg exploded\nmore")

        monkeypatch.setattr(SparkDataFrame, "agg", boom)
        context = _context(train, categorical=(), continuous=("num",), spark=spark)
        with pytest.raises(ExecutionError, match=r"Spark aggregation failed.*agg exploded"):
            IvSelector(IvConfig(num_bins=2)).select(context, ["num"])

    def test_categorical_iv_matches_pandas_with_null_and_rare(self, spark: Any) -> None:
        pandas_frame = pd.DataFrame(
            {
                "cat": ["keep"] * 50 + ["rare"] * 2 + [None] * 6,
                "response": [0] * 40 + [1] * 10 + [0, 1] + [1] * 6,
            },
        )
        config = IvConfig(min_bin_share=0.2, max_levels=None, eps=1e-4, threshold=0.0)
        pandas_scores, spark_scores, _, _ = _iv_on_pandas_and_spark(
            spark,
            pandas_frame,
            categorical=("cat",),
            continuous=(),
            candidates=["cat"],
            config=config,
        )
        goods, bads = _merge_categorical_counts(
            [
                ("keep", 40.0, 10.0),
                ("rare", 1.0, 1.0),
                (_NULL, 0.0, 6.0),
            ],
            min_bin_share=0.2,
            max_levels=None,
        )
        expected = information_value(goods, bads, config.eps)
        assert pandas_scores["cat"] == pytest.approx(expected)
        assert spark_scores["cat"] == pytest.approx(pandas_scores["cat"])

    def test_categorical_batch_size_does_not_change_scores(self, spark: Any) -> None:
        pandas_frame = pd.DataFrame(
            {
                "cata": ["a"] * 10 + ["z"] * 10,
                "catb": ["b"] * 10 + ["z"] * 10,
                "response": [0, 1] * 10,
            },
        )
        train = spark.createDataFrame(pandas_frame)
        one = _context(train, categorical=("cata", "catb"), continuous=(), spark=spark)
        many = _context(train, categorical=("cata", "catb"), continuous=(), spark=spark)
        IvSelector(IvConfig(batch_size=1, threshold=0.0)).select(one, ["cata", "catb"])
        IvSelector(IvConfig(batch_size=50, threshold=0.0)).select(many, ["cata", "catb"])
        assert one.scores["iv"]["values"] == many.scores["iv"]["values"]
        pandas_ctx = _context(pandas_frame, categorical=("cata", "catb"), continuous=())
        IvSelector(IvConfig(threshold=0.0)).select(pandas_ctx, ["cata", "catb"])
        for name in ("cata", "catb"):
            assert one.scores["iv"]["values"][name] == pytest.approx(
                pandas_ctx.scores["iv"]["values"][name],
            )

    def test_categorical_groupby_failure_uses_root_cause(
        self,
        spark: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from pyspark.sql import DataFrame as SparkDataFrame

        train = spark.createDataFrame(
            pd.DataFrame({"cat": ["a", "b", "a", "b"], "response": [0, 1, 0, 1]}),
        )

        def boom(self: Any, *_args: Any, **_kwargs: Any) -> Any:
            del self
            raise RuntimeError("groupby exploded\nmore")

        monkeypatch.setattr(SparkDataFrame, "groupBy", boom)
        context = _context(train, categorical=("cat",), continuous=(), spark=spark)
        with pytest.raises(ExecutionError, match=r"Spark groupBy failed.*groupby exploded"):
            IvSelector(IvConfig()).select(context, ["cat"])

    def test_inspect_target_failure_uses_root_cause(
        self,
        spark: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from pyspark.sql import DataFrame as SparkDataFrame

        train = spark.createDataFrame(
            pd.DataFrame({"num": [0.0, 1.0, 0.0, 1.0], "response": [0, 1, 0, 1]}),
        )

        def boom(self: Any, *_args: Any, **_kwargs: Any) -> Any:
            del self
            raise RuntimeError("inspect exploded\nmore")

        monkeypatch.setattr(SparkDataFrame, "collect", boom)
        context = _context(train, categorical=(), continuous=("num",), spark=spark)
        with pytest.raises(ExecutionError, match=r"failed to inspect target.*inspect exploded"):
            IvSelector(IvConfig()).select(context, ["num"])

    @pytest.mark.parametrize("labels, target_type, count", [
        ([], "int", 0),
        ([None, None], "int", 0),
        ([0, 0, None], "int", 1),
        ([1, 1, None], "int", 1),
        ([False, False, None], "boolean", 1),
        ([True, True, None], "boolean", 1),
        (["only", "only", None], "string", 1),
    ])
    @pytest.mark.parametrize("entrypoint", ["select", "compute"])
    def test_spark_degenerate_target_raises(
        self, spark: Any, labels: list[Any], target_type: str, count: int, entrypoint: str,
    ) -> None:
        train = spark.createDataFrame(
            [(float(index), str(index % 2), label) for index, label in enumerate(labels)],
            f"num double, cat string, response {target_type}",
        )
        context = _context(train, categorical=("cat",), continuous=("num",), spark=spark)
        selector = IvSelector(IvConfig())
        with pytest.raises(ExecutionError, match=f"Found {count} distinct non-null values"):
            getattr(selector, entrypoint)(context, ["num", "cat"])
        assert "iv" not in context.scores

    def test_spark_binary_target_three_labels_raises(self, spark: Any) -> None:
        three = spark.createDataFrame(
            pd.DataFrame({"num": [1.0, 2.0, 3.0], "response": [0, 1, 2]}),
        )
        three_ctx = _context(three, categorical=(), continuous=("num",), spark=spark)
        with pytest.raises(ExecutionError, match="target must be binary"):
            IvSelector(IvConfig()).select(three_ctx, ["num"])

    def test_spark_string_target_matches_pandas(self, spark: Any) -> None:
        pandas_frame = pd.DataFrame(
            {
                "num": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                "response": ["yes", "no", "yes", "no", "yes", "no"],
            },
        )
        pandas_scores, spark_scores, pandas_dropped, spark_dropped = _iv_on_pandas_and_spark(
            spark,
            pandas_frame,
            categorical=(),
            continuous=("num",),
            candidates=["num"],
            config=IvConfig(num_bins=2, threshold=0.0, relative_error=0.0),
        )
        assert pandas_dropped == spark_dropped
        assert spark_scores["num"] == pytest.approx(pandas_scores["num"], rel=0.2, abs=1e-6)

    def test_mixed_numeric_and_categorical_match_pandas(self, spark: Any) -> None:
        pandas_frame = _frame(n=240)[["strong", "weak", "cat_signal", "response"]]
        config = IvConfig(threshold=0.02, num_bins=6, relative_error=0.0)
        pandas_scores, spark_scores, pandas_dropped, spark_dropped = _iv_on_pandas_and_spark(
            spark,
            pandas_frame,
            categorical=("cat_signal",),
            continuous=("strong", "weak"),
            candidates=["strong", "weak", "cat_signal"],
            config=config,
        )
        assert pandas_dropped == spark_dropped
        assert spark_scores["cat_signal"] == pytest.approx(pandas_scores["cat_signal"])
        for name in ("strong", "weak"):
            assert spark_scores[name] == pytest.approx(pandas_scores[name], rel=0.2, abs=1e-6)

    def test_quoted_col_and_count_exprs(self, spark: Any) -> None:
        from pyspark.sql import functions as F

        assert str(_quoted_col("foo.bar")) == str(F.col("`foo.bar`"))
        assert str(_quoted_col("a`b")) == str(F.col("`ab`"))
        exprs = _count_exprs(F.lit(True), F.col("y"), "c0__null")
        row = spark.createDataFrame([(1.0,)], ["y"]).agg(*exprs).collect()[0].asDict()
        assert "c0__null__bad" in row
        assert "c0__null__n" in row
        assert row["c0__null__n"] == 1

    def test_spark_binary_target_java_root_cause(
        self,
        spark: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from pyspark.sql import DataFrame as SparkDataFrame

        class JavaFailureError(Exception):
            java_exception = "java.lang.RuntimeException: spark died\nstack"

        train = spark.createDataFrame(
            pd.DataFrame({"num": [0.0, 1.0, 0.0, 1.0], "response": [0, 1, 0, 1]}),
        )

        def boom(self: Any, *_args: Any, **_kwargs: Any) -> Any:
            del self
            raise JavaFailureError("wrapper")

        monkeypatch.setattr(SparkDataFrame, "collect", boom)
        context = _context(train, categorical=(), continuous=("num",), spark=spark)
        with pytest.raises(ExecutionError, match=r"java\.lang\.RuntimeException: spark died"):
            IvSelector(IvConfig()).select(context, ["num"])

    def test_spark_iv_matches_pandas_drop_set(self, spark: Any) -> None:
        pandas_frame = _frame(n=240)[["strong", "weak", "cat_signal", "response"]]
        config = IvConfig(threshold=0.02, num_bins=6, relative_error=0.01)
        pandas_scores, spark_scores, pandas_dropped, spark_dropped = _iv_on_pandas_and_spark(
            spark,
            pandas_frame,
            categorical=("cat_signal",),
            continuous=("strong", "weak"),
            candidates=["strong", "weak", "cat_signal"],
            config=config,
        )
        assert pandas_dropped == spark_dropped
        assert set(spark_scores) == {"strong", "weak", "cat_signal"}


class TestIvConfigAndPipeline:
    def test_pipeline_order_iv_writes_artifact(self, tmp_path: Path) -> None:
        frame = _frame()
        config = FeatureSelectionConfig.from_dict(
            {
                "statistics": {"order": ["iv"], "iv": {"threshold": 0.02, "num_bins": 8}},
            },
        )
        schema = FeatureSchema(
            categorical=("cat_signal", "null_flag"),
            continuous=("strong", "weak", "leak"),
            target="response",
            task_type="binary_classification",
        )
        result = FeatureSelectionPipeline(config).fit_select(
            require_spark_session(),
            datasets={"train": frame},
            schema=schema,
            output_dir=tmp_path,
        )
        assert "weak" not in result.selected_features
        assert "strong" in result.selected_features
        assert "cat_signal" in result.selected_features
        assert (tmp_path / "00_statistics_iv_results.json").exists()

    def test_statistics_iv_block_holds_hyperparameters(self) -> None:
        config = FeatureSelectionConfig.from_dict(
            {
                "statistics": {
                    "order": ["iv"],
                    "iv": {
                        "threshold": 0.1,
                        "num_bins": 4,
                        "max_threshold": 3.0,
                    },
                },
            },
        )
        assert config.statistics.iv.threshold == 0.1
        assert config.statistics.iv.num_bins == 4
        assert config.statistics.iv.max_threshold == 3.0
        assert config.statistics.iv.eps == 1e-4

    def test_from_yaml_iv_example_and_main_conf(self) -> None:
        standalone = FeatureSelectionConfig.from_yaml(
            _REPO_ROOT / "examples/configs/feature_selection/iv.yaml",
        )
        assert [step.method for step in standalone.order] == ["iv"]
        assert standalone.order[0].params["threshold"] == 0.02
        assert standalone.statistics.iv.threshold == 0.02
        assert standalone.statistics.iv.num_bins == 10

        main = FeatureSelectionConfig.from_yaml(
            _REPO_ROOT / "examples/big_c/main_conf.yaml",
        )
        assert "iv" not in [step.method for step in main.order]
        assert main.statistics.iv.threshold == 0.02
        assert main.statistics.iv.max_threshold is None
