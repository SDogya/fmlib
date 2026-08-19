"""Production-grade tests for Information Value selection.

Covers ``fmlib.feature_selection.statistics.iv`` statement and branch paths.
Pure helpers are exercised on real data; Spark control-flow uses a local
DataFrame double plus a ``pyspark.sql.functions`` stub when PySpark is absent.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pytest

from fmlib.feature_selection.base import StageContext
from fmlib.feature_selection.config import FeatureSelectionConfig, IvConfig, VerboseConfig
from fmlib.feature_selection.conftest import FakeSparkSession
from fmlib.feature_selection.debug import DebugRecorder
from fmlib.feature_selection.exceptions import BackendError, ConfigError, ExecutionError
from fmlib.feature_selection.pipeline import FeatureSelectionPipeline
from fmlib.feature_selection.schema import FeatureSchema
from fmlib.feature_selection.statistics import iv as iv_mod
from fmlib.feature_selection.statistics.iv import (
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

_REPO_ROOT = Path(__file__).resolve().parents[4]
_NULL = iv_mod._NULL_LEVEL
_OTHER = iv_mod._OTHER_LEVEL


def _pyspark_available() -> bool:
    try:
        import pyspark  # noqa: F401
    except ImportError:
        return False
    return True


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
    debug = DebugRecorder(verbose=VerboseConfig(iv=verbose_iv))
    return StageContext(
        spark=spark if spark is not None else FakeSparkSession(),
        datasets={"train": frame},
        schema=schema,
        config=FeatureSelectionConfig(),
        seed=0,
        candidates=schema.candidate_features(),
        debug=debug,
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


class IntegerType:
    """Spark type double; ``__name__`` must match production type names."""


class StringType:
    """Spark string type double."""


class BooleanType:
    """Spark boolean type double (rejected for continuous IV)."""


class DecimalType:
    """Spark decimal type double; allowed for continuous IV."""


class FakeCol:
    """Minimal Spark Column double for ``pyspark.sql.functions`` stubs."""

    def __init__(self: FakeCol, name: str = "c") -> None:
        self.name = name
        self.source = name

    def alias(self: FakeCol, name: str) -> FakeCol:
        out = FakeCol(name)
        out.source = self.source
        return out

    def isNotNull(self: FakeCol) -> FakeCol:  # noqa: N802 - Spark Column API
        return FakeCol(f"isNotNull({self.name})")

    def isNull(self: FakeCol) -> FakeCol:  # noqa: N802 - Spark Column API
        return FakeCol(f"isNull({self.name})")

    def cast(self: FakeCol, dtype: str) -> FakeCol:
        return FakeCol(f"cast({self.name},{dtype})")

    def otherwise(self: FakeCol, value: object) -> FakeCol:
        return FakeCol(f"otherwise({self.name},{value})")

    def __eq__(self: FakeCol, other: object) -> FakeCol:  # type: ignore[override]
        return FakeCol(f"eq({self.name})")

    def __invert__(self: FakeCol) -> FakeCol:
        return FakeCol(f"not({self.name})")

    def __and__(self: FakeCol, other: object) -> FakeCol:
        return FakeCol(f"and({self.name})")

    def __or__(self: FakeCol, other: object) -> FakeCol:
        return FakeCol(f"or({self.name})")

    def __le__(self: FakeCol, other: object) -> FakeCol:
        return FakeCol(f"le({self.name})")

    def __gt__(self: FakeCol, other: object) -> FakeCol:
        return FakeCol(f"gt({self.name})")


class _InspectChain:
    def __init__(self: _InspectChain, owner: FakeSparkDataFrame) -> None:
        self.owner = owner

    def distinct(self: _InspectChain) -> _InspectChain:
        return self

    def limit(self: _InspectChain, _n: int) -> _InspectChain:
        return self

    def collect(self: _InspectChain) -> list[Any]:
        if self.owner.inspect_error is not None:
            raise self.owner.inspect_error
        return list(self.owner.distinct_rows)


class _AggHead:
    def __init__(self: _AggHead, payload: dict[str, Any]) -> None:
        self._payload = payload

    def head(self: _AggHead) -> _AggHead:
        return self

    def asDict(self: _AggHead) -> dict[str, Any]:  # noqa: N802 - Spark Row API
        return dict(self._payload)


class FakeSparkDataFrame:
    """Spark DataFrame double: module name starts with ``pyspark``."""

    def __init__(self: FakeSparkDataFrame) -> None:
        self.role = "train"
        self.schema = SimpleNamespace(fields=[])
        self.is_cached = False
        self.distinct_rows: list[Any] = [(0,), (1,)]
        self.as_dict: dict[str, Any] = {}
        self.cat_rows: list[Any] = []
        self.quantiles: list[float] = [0.5]
        self.persist_error: BaseException | None = None
        self.unpersist_error: BaseException | None = None
        self.approx_error: BaseException | None = None
        self.agg_error: BaseException | None = None
        self.groupby_error: BaseException | None = None
        self.inspect_error: BaseException | None = None
        self._grouped = False
        self.trace: dict[str, Any] = {
            "persist": 0,
            "unpersist": 0,
            "approx": [],
            "agg": 0,
            "union": 0,
        }

    def _clone(self: FakeSparkDataFrame) -> FakeSparkDataFrame:
        other = FakeSparkDataFrame()
        other.trace = self.trace
        other.schema = self.schema
        other.is_cached = self.is_cached
        other.distinct_rows = self.distinct_rows
        other.as_dict = self.as_dict
        other.cat_rows = self.cat_rows
        other.quantiles = self.quantiles
        other.persist_error = self.persist_error
        other.unpersist_error = self.unpersist_error
        other.approx_error = self.approx_error
        other.agg_error = self.agg_error
        other.groupby_error = self.groupby_error
        other.inspect_error = self.inspect_error
        other.role = self.role
        return other

    def where(self: FakeSparkDataFrame, _cond: object) -> FakeSparkDataFrame:
        filtered = self._clone()
        filtered.role = "filtered"
        return filtered

    def select(self: FakeSparkDataFrame, *cols: object) -> object:
        if self.role in {"train", "filtered"} and len(cols) == 1:
            return _InspectChain(self)
        if self.role in {"train", "filtered"}:
            prepared = self._clone()
            prepared.role = "prepared"
            return prepared
        piece = self._clone()
        piece.role = "piece"
        if cols:
            feature_name = getattr(cols[0], "source", None)
            if feature_name:
                piece.cat_rows = [
                    row for row in self.cat_rows if row["feature"] == feature_name
                ]
        return piece

    def persist(self: FakeSparkDataFrame) -> FakeSparkDataFrame:
        if self.persist_error is not None:
            raise self.persist_error
        self.trace["persist"] += 1
        self.is_cached = True
        return self

    def unpersist(self: FakeSparkDataFrame) -> FakeSparkDataFrame:
        self.trace["unpersist"] += 1
        if self.unpersist_error is not None:
            raise self.unpersist_error
        return self

    @property
    def stat(self: FakeSparkDataFrame) -> FakeSparkDataFrame:
        return self

    def approxQuantile(  # noqa: N802 - Spark DataFrame.stat API
        self: FakeSparkDataFrame,
        aliases: Sequence[str],
        probabilities: Sequence[float],
        relative_error: float,
    ) -> list[list[float]]:
        self.trace["approx"].append(
            {
                "aliases": list(aliases),
                "probabilities": list(probabilities),
                "relative_error": relative_error,
            },
        )
        if self.approx_error is not None:
            raise self.approx_error
        return [list(self.quantiles) for _ in aliases]

    def agg(self: FakeSparkDataFrame, *_exprs: object) -> object:
        if self._grouped:
            if self.groupby_error is not None:
                raise self.groupby_error
            return self
        self.trace["agg"] += 1
        if self.agg_error is not None:
            raise self.agg_error
        return _AggHead(self.as_dict)

    def unionByName(self: FakeSparkDataFrame, other: object) -> FakeSparkDataFrame:  # noqa: N802
        self.trace["union"] += 1
        stacked = self._clone()
        extra = getattr(other, "cat_rows", [])
        stacked.cat_rows = list(self.cat_rows) + list(extra)
        return stacked

    def groupBy(self: FakeSparkDataFrame, *_args: object) -> FakeSparkDataFrame:  # noqa: N802
        grouped = self._clone()
        grouped._grouped = True
        grouped.role = "grouped"
        return grouped

    def collect(self: FakeSparkDataFrame) -> list[Any]:
        return list(self.cat_rows)


FakeSparkDataFrame.__module__ = "pyspark.sql.dataframe"


def _install_pyspark_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    functions = ModuleType("pyspark.sql.functions")
    functions.col = lambda name: FakeCol(str(name))
    functions.lit = lambda value: FakeCol(str(value))
    functions.when = lambda _cond, _value: FakeCol("when")
    functions.isnan = lambda _col: FakeCol("isnan")
    functions.sum = lambda _col: FakeCol("sum")
    functions.count = lambda _col: FakeCol("count")
    sql = ModuleType("pyspark.sql")
    sql.functions = functions
    root = ModuleType("pyspark")
    root.sql = sql
    monkeypatch.setitem(sys.modules, "pyspark", root)
    monkeypatch.setitem(sys.modules, "pyspark.sql", sql)
    monkeypatch.setitem(sys.modules, "pyspark.sql.functions", functions)


def _field(name: str, type_cls: type) -> SimpleNamespace:
    return SimpleNamespace(name=name, dataType=type_cls())


def _spark_train(
    columns: Sequence[str],
    *,
    target: str = "response",
    continuous: Sequence[str] = (),
    types: dict[str, type] | None = None,
) -> FakeSparkDataFrame:
    type_map = dict(types or {})
    fields = []
    for name in [*columns, target]:
        if name in type_map:
            type_cls = type_map[name]
        elif name in continuous or name == target:
            type_cls = IntegerType
        else:
            type_cls = StringType
        fields.append(_field(name, type_cls))
    frame = FakeSparkDataFrame()
    frame.schema = SimpleNamespace(fields=fields)
    return frame


def _numeric_counts(
    alias: str,
    bins: Sequence[tuple[float, float]],
    null: tuple[float, float] = (0.0, 0.0),
) -> dict[str, float]:
    payload = {
        f"{alias}__null__bad": null[0],
        f"{alias}__null__n": null[1],
    }
    for index, (n_bad, n_total) in enumerate(bins):
        payload[f"{alias}__b{index}__bad"] = n_bad
        payload[f"{alias}__b{index}__n"] = n_total
    return payload


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

    def test_binary_mapping_one_label_empty_and_numpy(self) -> None:
        assert _binary_mapping(["only"], method="iv") == {"only": 1.0}
        assert _binary_mapping([], method="iv") == {}
        assert _binary_mapping(np.array([0, 1]), method="iv") == {0: 0.0, 1: 1.0}

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
        with pytest.raises(ConfigError, match="binary_classification"):
            IvSelector(IvConfig()).select(
                _context(_frame(), task_type="regression"),
                ["strong"],
            )

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

        monkeypatch.setattr(iv_mod, "debug_emit", fake_emit)
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

        monkeypatch.setattr(iv_mod, "debug_emit", fake_emit)
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
        monkeypatch.setattr(iv_mod, "debug_emit", lambda *a, **k: calls.append(1))
        monkeypatch.setattr(IvSelector, "_compute_iv_pandas", lambda *_a, **_k: {})
        context = _context(_frame(), categorical=(), continuous=("strong",), verbose_iv=True)
        decisions = IvSelector(IvConfig()).select(context, ["strong"])
        assert decisions == []
        assert calls == []
        assert context.scores["iv"]["values"] == {}

    def test_debug_not_emitted_when_verbose_disabled(self) -> None:
        context = _context(_frame(), categorical=(), continuous=("strong",), verbose_iv=False)
        IvSelector(IvConfig()).select(context, ["strong"])
        assert context.debug.events == []


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

    def test_all_target_nulls_return_zero_iv(self) -> None:
        frame = pd.DataFrame(
            {
                "num": [1.0, 2.0, 3.0],
                "cat": ["a", "b", "a"],
                "response": [None, np.nan, None],
            },
        )
        context = _context(frame, categorical=("cat",), continuous=("num",))
        decisions = IvSelector(IvConfig(threshold=0.02)).select(context, ["num", "cat"])
        assert context.scores["iv"]["values"] == {"num": 0.0, "cat": 0.0}
        assert {item.feature for item in decisions} == {"num", "cat"}
        assert all(item.reason == "low_iv" for item in decisions)

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

    def test_single_class_target_yields_zero_iv(self) -> None:
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "response": [1, 1, 1, 1]})
        context = _context(frame, categorical=(), continuous=("x",))
        IvSelector(IvConfig(threshold=0.02, num_bins=2)).select(context, ["x"])
        assert context.scores["iv"]["values"]["x"] == 0.0

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


class TestIvSelectorSpark:
    def test_is_spark_dataframe_detection(self) -> None:
        pandas_frame = pd.DataFrame({"a": [1]})
        assert _is_spark_dataframe(pandas_frame) is False
        plain = SimpleNamespace(select=lambda *a: None, agg=lambda *a: None)
        assert _is_spark_dataframe(plain) is False
        sparkish = FakeSparkDataFrame()
        assert _is_spark_dataframe(sparkish) is True

        class Almost:
            pass

        Almost.__module__ = "pyspark.sql.dataframe"
        missing_agg = Almost()
        missing_agg.select = lambda *a: None
        assert _is_spark_dataframe(missing_agg) is False

    def test_pyspark_missing_raises_backend_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "pyspark.sql", None)
        monkeypatch.setitem(sys.modules, "pyspark", None)
        train = _spark_train(["strong"], continuous=("strong",))
        context = _context(train, categorical=(), continuous=("strong",))
        with pytest.raises(BackendError, match="pyspark is required"):
            IvSelector(IvConfig()).select(context, ["strong"])

    def test_missing_spark_columns_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_pyspark_stub(monkeypatch)
        train = _spark_train(["strong"], continuous=("strong",))
        context = _context(
            train,
            categorical=(),
            continuous=("strong", "absent"),
        )
        with pytest.raises(ExecutionError, match="columns missing from train schema"):
            IvSelector(IvConfig()).select(context, ["absent"])

    def test_non_numeric_continuous_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_pyspark_stub(monkeypatch)
        train = _spark_train(
            ["cat_as_num"],
            continuous=("cat_as_num",),
            types={"cat_as_num": StringType, "response": IntegerType},
        )
        context = _context(train, categorical=(), continuous=("cat_as_num",))
        with pytest.raises(ExecutionError, match="expected a numeric type"):
            IvSelector(IvConfig()).select(context, ["cat_as_num"])

        bool_train = _spark_train(
            ["flag"],
            continuous=("flag",),
            types={"flag": BooleanType, "response": IntegerType},
        )
        bool_context = _context(bool_train, categorical=(), continuous=("flag",))
        with pytest.raises(ExecutionError, match="BooleanType"):
            IvSelector(IvConfig()).select(bool_context, ["flag"])

    def test_persist_and_unpersist_when_not_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_pyspark_stub(monkeypatch)
        train = _spark_train(["num"], continuous=("num",))
        train.as_dict = _numeric_counts("c0", [(2.0, 10.0), (8.0, 10.0)], null=(1.0, 2.0))
        context = _context(train, categorical=(), continuous=("num",))
        IvSelector(IvConfig(num_bins=2, threshold=0.0)).select(context, ["num"])
        assert train.trace["persist"] == 1
        assert train.trace["unpersist"] == 1

    def test_already_cached_skips_persist(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_pyspark_stub(monkeypatch)
        train = _spark_train(["num"], continuous=("num",))
        train.is_cached = True
        train.as_dict = _numeric_counts("c0", [(2.0, 10.0), (8.0, 10.0)])
        context = _context(train, categorical=(), continuous=("num",))
        IvSelector(IvConfig(num_bins=2, threshold=0.0)).select(context, ["num"])
        assert train.trace["persist"] == 0
        assert train.trace["unpersist"] == 0

    def test_persist_failure_still_computes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_pyspark_stub(monkeypatch)
        train = _spark_train(["num"], continuous=("num",))
        train.persist_error = RuntimeError("cannot persist")
        train.as_dict = _numeric_counts("c0", [(2.0, 10.0), (8.0, 10.0)])
        context = _context(train, categorical=(), continuous=("num",))
        IvSelector(IvConfig(num_bins=2, threshold=0.0)).select(context, ["num"])
        assert "num" in context.scores["iv"]["values"]
        assert train.trace["unpersist"] == 0

    def test_unpersist_failure_is_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_pyspark_stub(monkeypatch)
        train = _spark_train(["num"], continuous=("num",))
        train.unpersist_error = RuntimeError("cannot unpersist")
        train.as_dict = _numeric_counts("c0", [(2.0, 10.0), (8.0, 10.0)])
        context = _context(train, categorical=(), continuous=("num",))
        decisions = IvSelector(IvConfig(num_bins=2, threshold=0.0)).select(context, ["num"])
        assert isinstance(decisions, list)
        assert train.trace["unpersist"] == 1

    def test_numeric_iv_matches_information_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_pyspark_stub(monkeypatch)
        train = _spark_train(["num"], continuous=("num",), types={"num": DecimalType})
        bins = [(2.0, 10.0), (8.0, 10.0)]
        null = (1.0, 2.0)
        train.as_dict = _numeric_counts("c0", bins, null=null)
        config = IvConfig(num_bins=2, eps=1e-4, threshold=0.0, relative_error=0.05)
        context = _context(train, categorical=(), continuous=("num",))
        IvSelector(config).select(context, ["num"])
        goods = [10.0 - 2.0, 10.0 - 8.0, 2.0 - 1.0]
        bads = [2.0, 8.0, 1.0]
        expected = information_value(goods, bads, config.eps)
        assert context.scores["iv"]["values"]["num"] == pytest.approx(expected)
        assert train.trace["approx"][0]["relative_error"] == 0.05
        assert train.trace["approx"][0]["probabilities"] == [0.5]

    def test_numeric_iv_single_bin_when_no_quantiles(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_pyspark_stub(monkeypatch)
        train = _spark_train(["num"], continuous=("num",))
        train.quantiles = []
        train.as_dict = _numeric_counts("c0", [(4.0, 10.0)])
        context = _context(train, categorical=(), continuous=("num",))
        IvSelector(IvConfig(num_bins=2, threshold=0.0)).select(context, ["num"])
        expected = information_value([6.0, 0.0], [4.0, 0.0], 1e-4)
        assert context.scores["iv"]["values"]["num"] == pytest.approx(expected)

    def test_numeric_batch_size_splits_agg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_pyspark_stub(monkeypatch)
        train = _spark_train(["n0", "n1"], continuous=("n0", "n1"))
        train.as_dict = {
            **_numeric_counts("c0", [(1.0, 8.0), (7.0, 8.0)]),
            **_numeric_counts("c1", [(2.0, 8.0), (6.0, 8.0)]),
        }
        context = _context(train, categorical=(), continuous=("n0", "n1"))
        IvSelector(IvConfig(num_bins=2, batch_size=1, threshold=0.0)).select(
            context,
            ["n0", "n1"],
        )
        assert len(train.trace["approx"]) == 1
        assert train.trace["approx"][0]["aliases"] == ["c0", "c1"]
        assert train.trace["agg"] == 2
        assert set(context.scores["iv"]["values"]) == {"n0", "n1"}

    def test_approx_quantile_failure_uses_root_cause(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_pyspark_stub(monkeypatch)
        train = _spark_train(["num"], continuous=("num",))
        train.approx_error = RuntimeError("quantile exploded\nmore")
        context = _context(train, categorical=(), continuous=("num",))
        with pytest.raises(ExecutionError, match=r"approxQuantile failed.*quantile exploded"):
            IvSelector(IvConfig(num_bins=2)).select(context, ["num"])

    def test_numeric_agg_failure_uses_root_cause(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_pyspark_stub(monkeypatch)
        train = _spark_train(["num"], continuous=("num",))
        train.agg_error = RuntimeError("agg exploded\nmore")
        context = _context(train, categorical=(), continuous=("num",))
        with pytest.raises(ExecutionError, match=r"Spark aggregation failed.*agg exploded"):
            IvSelector(IvConfig(num_bins=2)).select(context, ["num"])

    def test_categorical_iv_merges_null_and_other(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_pyspark_stub(monkeypatch)
        train = _spark_train(["cat"], continuous=())
        train.cat_rows = [
            {"feature": "cat", "level": "keep", "n": 50, "n_bad": 10},
            {"feature": "cat", "level": "rare", "n": 2, "n_bad": 1},
            {"feature": "cat", "level": None, "n": 6, "n_bad": 6},
        ]
        config = IvConfig(min_bin_share=0.2, max_levels=None, eps=1e-4, threshold=0.0)
        context = _context(train, categorical=("cat",), continuous=())
        IvSelector(config).select(context, ["cat"])
        goods, bads = _merge_categorical_counts(
            [
                ("keep", 40.0, 10.0),
                ("rare", 1.0, 1.0),
                (_NULL, 0.0, 6.0),
            ],
            min_bin_share=0.2,
            max_levels=None,
        )
        assert context.scores["iv"]["values"]["cat"] == pytest.approx(
            information_value(goods, bads, config.eps),
        )

    def test_categorical_union_and_none_counts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_pyspark_stub(monkeypatch)
        train = _spark_train(["cata", "catb"], continuous=())
        train.cat_rows = [
            {"feature": "cata", "level": "a", "n": 10, "n_bad": 1},
            {"feature": "cata", "level": "z", "n": 10, "n_bad": 9},
            {"feature": "catb", "level": "b", "n": None, "n_bad": None},
        ]
        context = _context(train, categorical=("cata", "catb"), continuous=())
        IvSelector(IvConfig(batch_size=50, threshold=0.0)).select(context, ["cata", "catb"])
        assert train.trace["union"] == 1
        assert context.scores["iv"]["values"]["catb"] == 0.0
        assert context.scores["iv"]["values"]["cata"] > 0.0

    def test_categorical_batch_size_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_pyspark_stub(monkeypatch)
        train = _spark_train(["cata", "catb"], continuous=())
        train.cat_rows = [
            {"feature": "cata", "level": "a", "n": 10, "n_bad": 1},
            {"feature": "cata", "level": "z", "n": 10, "n_bad": 9},
            {"feature": "catb", "level": "b", "n": 10, "n_bad": 1},
            {"feature": "catb", "level": "z", "n": 10, "n_bad": 9},
        ]
        context = _context(train, categorical=("cata", "catb"), continuous=())
        IvSelector(IvConfig(batch_size=1, threshold=0.0)).select(context, ["cata", "catb"])
        assert train.trace["union"] == 0
        assert set(context.scores["iv"]["values"]) == {"cata", "catb"}
        assert context.scores["iv"]["values"]["cata"] > 0.0
        assert context.scores["iv"]["values"]["catb"] > 0.0

    def test_categorical_groupby_failure_uses_root_cause(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_pyspark_stub(monkeypatch)
        train = _spark_train(["cat"], continuous=())
        train.groupby_error = RuntimeError("groupby exploded\nmore")
        context = _context(train, categorical=("cat",), continuous=())
        with pytest.raises(ExecutionError, match=r"Spark groupBy failed.*groupby exploded"):
            IvSelector(IvConfig()).select(context, ["cat"])

    def test_inspect_target_failure_uses_root_cause(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_pyspark_stub(monkeypatch)
        train = _spark_train(["num"], continuous=("num",))
        train.inspect_error = RuntimeError("inspect exploded\nmore")
        context = _context(train, categorical=(), continuous=("num",))
        with pytest.raises(ExecutionError, match=r"failed to inspect target.*inspect exploded"):
            IvSelector(IvConfig()).select(context, ["num"])

    def test_spark_binary_target_empty_one_and_three_labels(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _install_pyspark_stub(monkeypatch)
        empty = _spark_train(["num"], continuous=("num",))
        empty.distinct_rows = []
        empty.as_dict = _numeric_counts("c0", [(5.0, 10.0), (5.0, 10.0)])
        empty_ctx = _context(empty, categorical=(), continuous=("num",))
        IvSelector(IvConfig(num_bins=2, threshold=0.0)).select(empty_ctx, ["num"])
        assert "num" in empty_ctx.scores["iv"]["values"]

        single = _spark_train(["num"], continuous=("num",))
        single.distinct_rows = [(1,)]
        single.as_dict = _numeric_counts("c0", [(0.0, 10.0), (0.0, 10.0)])
        single_ctx = _context(single, categorical=(), continuous=("num",))
        IvSelector(IvConfig(num_bins=2, threshold=0.0)).select(single_ctx, ["num"])
        assert "num" in single_ctx.scores["iv"]["values"]

        three = _spark_train(["num"], continuous=("num",))
        three.distinct_rows = [(0,), (1,), (2,)]
        three_ctx = _context(three, categorical=(), continuous=("num",))
        with pytest.raises(ExecutionError, match="target must be binary"):
            IvSelector(IvConfig()).select(three_ctx, ["num"])

    def test_spark_string_target_and_null_distinct(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_pyspark_stub(monkeypatch)
        train = _spark_train(["num"], continuous=("num",))
        train.distinct_rows = [("yes",), (None,), ("no",)]
        train.as_dict = _numeric_counts("c0", [(2.0, 10.0), (8.0, 10.0)])
        context = _context(train, categorical=(), continuous=("num",))
        IvSelector(IvConfig(num_bins=2, threshold=0.0)).select(context, ["num"])
        assert context.scores["iv"]["values"]["num"] > 0.0

    def test_mixed_numeric_and_categorical_spark(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_pyspark_stub(monkeypatch)
        train = _spark_train(["num", "cat"], continuous=("num",))
        train.as_dict = _numeric_counts("c0", [(1.0, 8.0), (7.0, 8.0)])
        train.cat_rows = [{"feature": "cat", "level": "a", "n": 16, "n_bad": 8}]
        context = _context(train, categorical=("cat",), continuous=("num",))
        IvSelector(IvConfig(num_bins=2, threshold=0.0)).select(context, ["num", "cat"])
        scores = context.scores["iv"]["values"]
        assert set(scores) == {"num", "cat"}
        assert scores["num"] > 0.0

    def test_quoted_col_and_count_exprs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_pyspark_stub(monkeypatch)
        dotted = _quoted_col("foo.bar")
        assert dotted.name == "`foo.bar`"
        stripped = _quoted_col("a`b")
        assert stripped.name == "`ab`"
        exprs = _count_exprs(FakeCol("cond"), FakeCol("y"), "c0__null")
        assert [item.name for item in exprs] == ["c0__null__bad", "c0__null__n"]

    def test_spark_binary_target_java_root_cause(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_pyspark_stub(monkeypatch)

        class JavaFailureError(Exception):
            java_exception = "java.lang.RuntimeException: spark died\nstack"

        train = _spark_train(["num"], continuous=("num",))
        train.inspect_error = JavaFailureError("wrapper")
        context = _context(train, categorical=(), continuous=("num",))
        with pytest.raises(ExecutionError, match=r"java\.lang\.RuntimeException: spark died"):
            IvSelector(IvConfig()).select(context, ["num"])

    @pytest.mark.skipif(not _pyspark_available(), reason="pyspark not installed")
    def test_spark_iv_matches_pandas_drop_set(self) -> None:
        from pyspark.sql import SparkSession

        pandas_frame = _frame(n=240)
        spark = (
            SparkSession.builder.master("local[1]")
            .appName("test-iv-spark")
            .config("spark.ui.enabled", "false")
            .config("spark.driver.host", "127.0.0.1")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")
        try:
            spark_frame = spark.createDataFrame(
                pandas_frame[["strong", "weak", "cat_signal", "response"]],
            )
            candidates = ["strong", "weak", "cat_signal"]
            config = IvConfig(threshold=0.02, num_bins=6, relative_error=0.01)
            pandas_decisions = IvSelector(config).select(
                _context(
                    pandas_frame,
                    categorical=("cat_signal",),
                    continuous=("strong", "weak"),
                ),
                candidates,
            )
            spark_decisions = IvSelector(config).select(
                _context(
                    spark_frame,
                    categorical=("cat_signal",),
                    continuous=("strong", "weak"),
                    spark=spark,
                ),
                candidates,
            )
            assert {item.feature for item in pandas_decisions} == {
                item.feature for item in spark_decisions
            }
            assert {item.reason for item in spark_decisions} <= {"low_iv"}
        finally:
            spark.stop()


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
            FakeSparkSession(),
            datasets={"train": frame},
            schema=schema,
            output_dir=tmp_path,
        )
        assert "weak" not in result.selected_features
        assert "strong" in result.selected_features
        assert "cat_signal" in result.selected_features
        assert (tmp_path / "statistics_iv_results.json").exists()

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
        assert standalone.statistics.order == ("iv",)
        assert standalone.statistics.iv.threshold == 0.02
        assert standalone.statistics.iv.num_bins == 10

        main = FeatureSelectionConfig.from_yaml(
            _REPO_ROOT / "examples/big_c/main_conf.yaml",
        )
        assert "iv" not in main.statistics.order
        assert main.statistics.iv.threshold == 0.02
        assert main.statistics.iv.max_threshold is None
