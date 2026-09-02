"""Guards against a run that returns a wrong feature list without saying so.

Each case here used to produce a plausible-looking ``SelectionResult``:
a placeholder selector dropped features at random, a cache handed back numbers
measured on other data, and a single dominant feature emptied the model stage.
The artifact recorded all three exactly like a measured outcome.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fmlib.feature_selection import (
    FeatureSchema,
    FeatureSelectionConfig,
    FeatureSelectionPipeline,
)
from fmlib.feature_selection.exceptions import ConfigError
from fmlib.feature_selection.model_based.lightgbm import _cumulative_select

_STUBS = ["lasso", "random_forest", "stability_classifier"]


def _frame(n_rows: int = 400, n_features: int = 12, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {f"f{index}": rng.normal(size=n_rows) for index in range(n_features)},
    )
    frame["response"] = (rng.random(n_rows) < 0.3).astype(int)
    return frame


def _schema(frame: pd.DataFrame) -> FeatureSchema:
    return FeatureSchema(
        categorical=(),
        continuous=tuple(name for name in frame.columns if name != "response"),
        target="response",
        task_type="binary_classification",
    )


@pytest.mark.parametrize("method", _STUBS)
def test_placeholder_selectors_are_refused_before_any_work(method: str) -> None:
    """A stub must fail at validation, not drop 10-20 features at random."""
    config = FeatureSelectionConfig.from_dict({"order": [{method: {}}]})
    frame = _frame()

    with pytest.raises(ConfigError, match="not implemented"):
        FeatureSelectionPipeline(config).fit_select(
            schema=_schema(frame),
            datasets={"train": frame},
        )


def test_pandas_only_run_needs_no_spark_session() -> None:
    """Every selector has a pandas path; requiring a session blocked local runs."""
    frame = _frame()
    config = FeatureSelectionConfig.from_dict(
        {"order": [{"null_rate": {"threshold": 0.5}}]},
    )

    result = FeatureSelectionPipeline(config).fit_select(
        schema=_schema(frame),
        datasets={"train": frame},
    )

    assert result.selected_features == list(_schema(frame).continuous)
    assert result.warnings == []


def _cache_config(cache_path: Path, dataset_id: str) -> FeatureSelectionConfig:
    return FeatureSelectionConfig.from_dict(
        {
            "order": [{"null_rate": {"threshold": 0.5}}],
            "statistics": {
                "cache": {
                    "enabled": True,
                    "path": str(cache_path),
                    "dataset_id": dataset_id,
                },
            },
        },
    )


def test_enabling_the_cache_without_a_dataset_id_is_refused() -> None:
    """Shape and column names cannot tell two tables apart; the caller must say."""
    with pytest.raises(ConfigError, match="dataset_id is required"):
        FeatureSelectionConfig.from_dict(
            {
                "order": [{"null_rate": {}}],
                "statistics": {"cache": {"enabled": True, "path": "m.json"}},
            },
        )


def test_cache_does_not_carry_metrics_across_datasets(tmp_path: Path) -> None:
    """Two datasets of identical shape must not share cached metrics."""
    cache_path = tmp_path / "metrics.json"

    all_null = _frame()
    all_null["f0"] = np.nan
    first = FeatureSelectionPipeline(
        _cache_config(cache_path, "with-nulls"),
    ).fit_select(schema=_schema(all_null), datasets={"train": all_null})
    assert [item.feature for item in first.dropped_features] == ["f0"]

    # Same row count, same columns, different contents.
    clean = _frame(seed=1)
    second = FeatureSelectionPipeline(
        _cache_config(cache_path, "clean"),
    ).fit_select(schema=_schema(clean), datasets={"train": clean})
    assert second.dropped_features == []


def test_cache_still_reuses_metrics_for_the_same_data(tmp_path: Path) -> None:
    """The data key must not defeat the cache on a genuine repeat run."""
    from fmlib.feature_selection.utils.statistics_cache import StatisticsMetricsCache

    cache_path = tmp_path / "metrics.json"
    config = _cache_config(cache_path, "same-data")
    frame = _frame()
    for _ in range(2):
        FeatureSelectionPipeline(config).fit_select(
            schema=_schema(frame),
            datasets={"train": frame},
        )

    cache = StatisticsMetricsCache.load(cache_path)
    assert len(cache._entries) == 1  # noqa: SLF001 - test probe


def test_dominant_feature_does_not_empty_the_lightgbm_cut() -> None:
    """A feature whose own share exceeds the threshold crosses it on row one."""
    selected, _norm, _cumsum = _cumulative_select(
        np.array([90.0, 5.0, 5.0]),
        ["a", "b", "c"],
        0.85,
        empty_total_message="unused",
    )

    assert selected == {"a"}
