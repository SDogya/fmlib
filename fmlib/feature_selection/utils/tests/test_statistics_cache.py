"""Spark-free tests for the optional statistics metrics cache."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from fmlib.feature_selection.base import StageContext
from fmlib.feature_selection.config import (
    ConstantsConfig,
    CorrelationConfig,
    FeatureSelectionConfig,
    LowVarianceConfig,
    NullRateConfig,
)
from fmlib.feature_selection.runner import run_order
from fmlib.feature_selection.schema import FeatureSchema
from fmlib.feature_selection.statistics.correlation import CorrelationSelector
from fmlib.feature_selection.statistics.null_rate import NullRateSelector
from fmlib.feature_selection.utils.statistics_cache import (
    StatisticsMetricsCache,
    compute_fingerprint,
    resolve_cache_path,
)


def _schema() -> FeatureSchema:
    return FeatureSchema(
        categorical=("cat",),
        continuous=("keep", "drop_null", "a", "b"),
        target="response",
        task_type="binary_classification",
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cat": ["x", "x", "y", "y", "x"],
            "keep": [1.0, 2.0, 3.0, 4.0, 5.0],
            "drop_null": [None, None, None, None, 1.0],
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [1.0, 2.0, 3.0, 4.0, 5.0],
            "response": [0, 1, 0, 1, 0],
        },
    )


def _context(
    config: FeatureSelectionConfig,
    frame: pd.DataFrame | None = None,
) -> StageContext:
    schema = _schema()
    return StageContext(
        spark=None,
        datasets={"train": frame if frame is not None else _frame()},
        schema=schema,
        config=config,
        seed=config.execution.seed,
        candidates=schema.candidate_features(),
    )


def test_resolve_cache_path_defaults_to_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert resolve_cache_path(None) == tmp_path / "statistics_metrics.json"
    assert resolve_cache_path("custom.json") == tmp_path / "custom.json"


def test_cache_miss_appends_and_hit_reuses(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    config = FeatureSelectionConfig.from_dict(
        {
            "order": [{"null_rate": {"threshold": 0.5}}],
            "statistics": {"cache": {"enabled": True, "path": str(path)}},
        },
    )
    context = _context(config)
    calls = {"n": 0}
    original = NullRateSelector.compute

    def wrapped(
        self: NullRateSelector,
        context: StageContext,
        candidates: list[str],
    ) -> dict:
        calls["n"] += 1
        return original(self, context, candidates)

    with patch.object(NullRateSelector, "compute", wrapped):
        remaining, _ = run_order(context, context.candidates)
        first_kept = list(remaining)

        context2 = _context(config)
        remaining2, _ = run_order(context2, context2.candidates)
        assert calls["n"] == 1
        assert remaining2 == first_kept

    cache = StatisticsMetricsCache.load(path)
    assert cache.lookup("null_rate", {}) is not None
    assert "drop_null" not in remaining2
    assert "keep" in remaining2


def test_different_scale_method_is_a_second_entry(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    frame = pd.DataFrame(
        {
            "keep": [0.0, 1.0, 2.0, 3.0, 4.0],
            "flat": [1.0, 1.0, 1.0, 1.0, 1.0],
            "response": [0, 1, 0, 1, 0],
        },
    )
    schema = FeatureSchema(
        categorical=(),
        continuous=("keep", "flat"),
        target="response",
        task_type="binary_classification",
    )

    def _run(scale_method: str) -> None:
        config = FeatureSelectionConfig.from_dict(
            {
                "order": [
                    {
                        "low_variance": {
                            "min_variance": 0.01,
                            "scale_method": scale_method,
                        },
                    },
                ],
                "statistics": {"cache": {"enabled": True, "path": str(path)}},
            },
        )
        context = StageContext(
            spark=None,
            datasets={"train": frame},
            schema=schema,
            config=config,
            seed=0,
            candidates=list(schema.candidate_features()),
        )
        run_order(context, context.candidates)

    _run("robust")
    _run("minmax")
    payload = path.read_text(encoding="utf-8")
    assert payload.count('"method": "low_variance"') == 2
    assert '"scale_method": "robust"' in payload
    assert '"scale_method": "minmax"' in payload


def test_force_recompute_replaces_entry(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    config = FeatureSelectionConfig.from_dict(
        {
            "order": [{"null_rate": {"threshold": 0.5}}],
            "statistics": {
                "cache": {
                    "enabled": True,
                    "path": str(path),
                    "force_recompute": True,
                },
            },
        },
    )
    calls = {"n": 0}
    original = NullRateSelector.compute

    def wrapped(
        self: NullRateSelector,
        context: StageContext,
        candidates: list[str],
    ) -> dict:
        calls["n"] += 1
        return original(self, context, candidates)

    with patch.object(NullRateSelector, "compute", wrapped):
        run_order(_context(config), _schema().candidate_features())
        run_order(_context(config), _schema().candidate_features())
        assert calls["n"] == 2


def test_disabled_cache_does_not_touch_file(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    config = FeatureSelectionConfig.from_dict(
        {
            "order": [{"null_rate": {"threshold": 0.5}}],
            "statistics": {"cache": {"enabled": False, "path": str(path)}},
        },
    )
    run_order(_context(config), _schema().candidate_features())
    assert not path.exists()


def test_correlation_apply_on_subset_uses_cached_matrix() -> None:
    frame = pd.DataFrame(
        {
            "first": [1.0, 2.0, 3.0, 4.0, 5.0],
            "second": [1.0, 2.0, 3.0, 4.0, 5.0],
            "independent": [2.0, 5.0, 1.0, 4.0, 3.0],
            "response": [0, 1, 0, 1, 0],
        },
    )
    schema = FeatureSchema(
        categorical=(),
        continuous=("first", "second", "independent"),
        target="response",
        task_type="binary_classification",
    )
    config = FeatureSelectionConfig()
    context = StageContext(
        spark=None,
        datasets={"train": frame},
        schema=schema,
        config=config,
        seed=0,
        candidates=list(schema.candidate_features()),
    )
    selector = CorrelationSelector(CorrelationConfig(threshold=0.9, tie_break="original_order"))
    metrics = selector.compute(context, schema.candidate_features())
    decisions = selector.apply(metrics, ["first", "second"], context)
    assert [item.feature for item in decisions] == ["second"]
    empty = selector.apply(metrics, ["first", "independent"], context)
    assert empty == []


def test_fingerprint_omits_thresholds() -> None:
    tight = compute_fingerprint(
        "null_rate",
        NullRateConfig(threshold=0.99),
        max_local_rows=1000,
    )
    wide = compute_fingerprint(
        "null_rate",
        NullRateConfig(threshold=0.5),
        max_local_rows=1000,
    )
    assert tight == {}
    assert tight == wide
    robust = compute_fingerprint(
        "low_variance",
        LowVarianceConfig(min_variance=0.5, scale_method="robust"),
        max_local_rows=1000,
    )
    other_threshold = compute_fingerprint(
        "low_variance",
        LowVarianceConfig(min_variance=0.01, scale_method="robust"),
        max_local_rows=1000,
    )
    minmax = compute_fingerprint(
        "low_variance",
        LowVarianceConfig(min_variance=0.01, scale_method="minmax"),
        max_local_rows=1000,
    )
    assert robust == other_threshold
    assert robust != minmax
    constants_default = compute_fingerprint(
        "constants",
        ConstantsConfig(min_unique=None),
        max_local_rows=1000,
    )
    constants_min = compute_fingerprint(
        "constants",
        ConstantsConfig(min_unique=5),
        max_local_rows=1000,
    )
    assert constants_default == {}
    assert constants_min == {"min_unique": 5}
