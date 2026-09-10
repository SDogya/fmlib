"""Регрессии PSI для категориальных признаков и смешанных схем."""

from dataclasses import replace
import math

import numpy as np
import pandas as pd
import pytest

from fmlib.feature_selection.base import StageContext
from fmlib.feature_selection.config import FeatureSelectionConfig, PsiConfig
from fmlib.feature_selection.exceptions import ConfigError
from fmlib.feature_selection.runner import run_order
from fmlib.feature_selection.schema import FeatureSchema
from fmlib.feature_selection.statistical_filters.psi import PsiSelector
from fmlib.feature_selection.utils.statistics_cache import compute_fingerprint


def _context(train, valid, *, continuous=(), spark=None):
    """Создаёт контекст без зависимости pandas-тестов от JVM."""
    categorical = tuple(name for name in train.columns if name not in continuous)
    schema = FeatureSchema(
        categorical=categorical, continuous=continuous, target="target",
        task_type="binary_classification",
    )
    return StageContext(
        spark=spark, datasets={"train": train, "valid": valid, "test": object()},
        schema=schema, config=FeatureSelectionConfig(), seed=42,
        candidates=schema.candidate_features(),
    )


@pytest.mark.parametrize("dtype", ["object", "string", "category"])
def test_mixed_candidates_use_separate_binning(dtype):
    train = pd.DataFrame({
        "category": pd.Series(["a"] * 80 + ["b"] * 20, dtype=dtype),
        "number": np.arange(100, dtype=float),
    })
    valid = pd.DataFrame({
        "category": pd.Series(["a"] * 20 + ["b"] * 80, dtype=dtype),
        "number": np.arange(100, dtype=float),
    })
    context = _context(train, valid, continuous=("number",))
    decisions = PsiSelector(PsiConfig(n_jobs=1)).select(context, ["number", "category"])
    assert [decision.feature for decision in decisions] == ["number", "category"]
    assert decisions[0].keep
    assert decisions[0].value == pytest.approx(0.0)
    assert not decisions[1].keep
    assert decisions[1].value == pytest.approx(1.2 * math.log(4))


@pytest.mark.parametrize("values", [
    ["a", "b", None, np.nan, pd.NA],
    [1, 2, 3, None, np.nan],
    [True, False, None, True, False],
    [None, np.nan, pd.NA, None, None],
    ["__NULL__", "__OTHER__", None, "other", "null"],
])
def test_identical_categorical_distributions_are_stable(values):
    train = pd.DataFrame({"cat": values})
    valid = pd.concat([train, train], ignore_index=True)
    score = PsiSelector(PsiConfig()).compute(_context(train, valid), ["cat"])["values"]["cat"]
    assert score == pytest.approx(0.0)


def test_numeric_codes_are_categorical_when_declared_in_schema(monkeypatch):
    train = pd.DataFrame({"code": [1] * 80 + [2] * 20})
    valid = pd.DataFrame({"code": [1] * 20 + [2] * 80})
    selector = PsiSelector(PsiConfig())

    def fail(*args, **kwargs):
        raise AssertionError("Числовая ветка не должна получать категориальные коды")

    monkeypatch.setattr(selector, "_compute_pandas_psi", fail)
    score = selector.compute(_context(train, valid), ["code"])["values"]["code"]
    assert score == pytest.approx(1.2 * math.log(4))


@pytest.mark.parametrize("config", [
    PsiConfig(min_bin_share=0.1), PsiConfig(max_levels=2),
])
def test_rare_and_new_categories_share_other_from_baseline(config):
    train = pd.DataFrame({"cat": ["a"] * 80 + ["b"] * 15 + ["rare"] * 5})
    valid = pd.DataFrame({"cat": ["a"] * 80 + ["b"] * 15 + ["new"] * 5})
    selector = PsiSelector(config)
    assert selector.compute(_context(train, valid), ["cat"])["values"]["cat"] == pytest.approx(0.0)
    unmerged = PsiSelector(PsiConfig(max_levels=None))
    assert unmerged.compute(_context(train, valid), ["cat"])["values"]["cat"] > 0.0


def test_new_category_is_not_lost():
    train = pd.DataFrame({"cat": ["a"] * 100})
    valid = pd.DataFrame({"cat": ["new"] * 100})
    decision = PsiSelector(PsiConfig()).select(_context(train, valid), ["cat"])[0]
    assert not decision.keep
    assert decision.value == pytest.approx(2 * 0.995 * math.log(200))


def test_missing_values_do_not_collide_with_literal_categories():
    train = pd.DataFrame({"cat": [None] * 100})
    valid = pd.DataFrame({"cat": ["__NULL__"] * 100})
    decision = PsiSelector(PsiConfig()).select(_context(train, valid), ["cat"])[0]
    assert not decision.keep
    assert math.isfinite(decision.value)


def test_level_limit_ties_are_deterministic():
    selector = PsiSelector(PsiConfig(max_levels=2))
    assert selector._categorical_psi_from_counts(
        {"c": 10, "b": 10, "a": 10}, {"a": 30},
    ) == selector._categorical_psi_from_counts(
        {"a": 10, "b": 10, "c": 10}, {"a": 30},
    )


@pytest.mark.parametrize("param,value", [
    ("min_bin_share", -0.1), ("min_bin_share", 1.0),
    ("min_bin_share", True), ("min_bin_share", float("nan")),
    ("max_levels", 1), ("max_levels", True), ("max_levels", 2.5),
])
@pytest.mark.parametrize("in_step", [False, True])
def test_invalid_categorical_options_are_rejected(param, value, in_step):
    raw = {"order": [{"psi": {param: value}}]} if in_step else {
        "statistics": {"psi": {param: value}},
    }
    with pytest.raises(ConfigError, match=param):
        FeatureSelectionConfig.from_dict(raw)


def test_categorical_options_roundtrip_and_cache_identity():
    config = FeatureSelectionConfig.from_dict({
        "statistics": {"psi": {"min_bin_share": 0.05, "max_levels": None}},
    })
    assert FeatureSelectionConfig.from_dict(config.to_dict()).statistics.psi == config.statistics.psi
    settings = PsiConfig()
    key = compute_fingerprint("psi", settings, max_local_rows=100)
    assert key != compute_fingerprint("psi", replace(settings, min_bin_share=0.1), max_local_rows=100)
    assert key != compute_fingerprint("psi", replace(settings, max_levels=None), max_local_rows=100)
    assert key == compute_fingerprint("psi", replace(settings, threshold=0.9), max_local_rows=100)


def test_categorical_psi_cache_matches_uncached_pipeline(tmp_path, monkeypatch):
    train = pd.DataFrame({"cat": ["a"] * 80 + ["b"] * 15 + ["rare"] * 5, "month_part": 1})
    valid = pd.DataFrame({"cat": ["a"] * 80 + ["b"] * 15 + ["new"] * 5, "month_part": 2})
    original = PsiSelector.compute
    calls = []

    def compute(self, context, candidates):
        calls.append(self.config.min_bin_share)
        return original(self, context, candidates)

    monkeypatch.setattr(PsiSelector, "compute", compute)

    def run(*, enabled, min_bin_share):
        """Запускает одинаковый пайплайн с выбранными параметрами кеша и корзин."""
        context = _context(train, valid)
        context.config = FeatureSelectionConfig.from_dict({
            "order": [{"psi": {"min_bin_share": min_bin_share, "threshold": 0.1}}],
            "statistics": {"cache": {
                "enabled": enabled, "path": str(tmp_path / "psi.json"), "dataset_version": "v1",
            }},
        })
        return run_order(context, ["cat"])

    uncached = run(enabled=False, min_bin_share=0.1)
    assert run(enabled=True, min_bin_share=0.1) == uncached
    assert run(enabled=True, min_bin_share=0.1) == uncached
    assert calls == [0.1, 0.1]
    changed = run(enabled=True, min_bin_share=0.0)
    assert calls == [0.1, 0.1, 0.0]
    assert changed[0] == []
    assert uncached[0] == ["cat"]


def test_categorical_psi_month_mode_with_sampling():
    train = pd.DataFrame({
        "cat": ["a"] * 100 + ["b"] * 100,
        "month_part": [1] * 100 + [2] * 100,
    })
    context = _context(train, train)
    context.schema = replace(context.schema, categorical=("cat",), time="month_part", task_type="regression")
    selector = PsiSelector(PsiConfig(mode="month_over_month", subsample_rows=40, seed=7))
    decisions = selector.select(context, ["cat"])
    assert not decisions[0].keep
    assert math.isfinite(decisions[0].value)


def test_spark_categorical_matches_pandas(spark):
    train = pd.DataFrame({
        "cat.name": ["a"] * 60 + ["b"] * 25 + ["rare"] * 5 + [None] * 10,
        "code": [1] * 80 + [2] * 20,
        "flag": [True] * 70 + [False] * 30,
        "float_code": [1.0] * 50 + [float("nan")] * 50,
        "num": np.arange(100, dtype=float),
    })
    valid = train.copy()
    valid["cat.name"] = ["a"] * 20 + ["b"] * 65 + ["new"] * 5 + [None] * 10
    valid["code"] = [1] * 20 + [2] * 80
    valid["float_code"] = [1.0] * 70 + [float("nan")] * 30
    selector = PsiSelector(PsiConfig(n_jobs=1, min_bin_share=0.1))
    expected = selector.compute(_context(train, valid, continuous=("num",)), list(train.columns))
    train_spark = spark.createDataFrame(list(train.itertuples(index=False, name=None)), list(train.columns))
    valid_spark = spark.createDataFrame(list(valid.itertuples(index=False, name=None)), list(valid.columns))
    actual = selector.compute(
        _context(train_spark, valid_spark, continuous=("num",), spark=spark), list(train.columns),
    )
    assert actual["values"] == pytest.approx(expected["values"])
