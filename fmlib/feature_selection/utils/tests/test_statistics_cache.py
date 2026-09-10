"""Тесты необязательного кэша статистических метрик без Spark."""

from __future__ import annotations

import json
from dataclasses import replace
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
from fmlib.feature_selection.exceptions import ConfigError
from fmlib.feature_selection.schema import FeatureSchema
from fmlib.feature_selection.statistical_filters.correlation import (
    CorrelationSelector,
)
from fmlib.feature_selection.statistical_filters.null_rate import (
    NullRateSelector,
)
from fmlib.feature_selection.utils.statistics_cache import (
    StatisticsMetricsCache,
    compute_data_fingerprint,
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


def test_resolve_cache_path_requires_output_or_explicit_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError, match="output_dir"):
        resolve_cache_path(None)
    assert resolve_cache_path(None, output_dir=tmp_path / "run") == tmp_path / "run" / "statistics_metrics.json"
    assert resolve_cache_path("custom.json") == tmp_path / "custom.json"


def test_cache_miss_appends_and_hit_reuses(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    config = FeatureSelectionConfig.from_dict(
        {
            "order": [{"null_rate": {"threshold": 0.5}}],
            "statistics": {"cache": {"enabled": True, "path": str(path), "dataset_version": "v1"}},
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
    fingerprint = {
        "data": compute_data_fingerprint(context),
        "parameters": {},
        "candidates": context.schema.candidate_features(),
    }
    assert cache.lookup("null_rate", fingerprint) is not None
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
                "statistics": {"cache": {"enabled": True, "path": str(path), "dataset_version": "v1"}},
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
                    "dataset_version": "v1",
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
    pearson = compute_fingerprint(
        "correlation",
        CorrelationConfig(threshold=0.9, method="pearson"),
        max_local_rows=1000,
        seed=0,
        task_type="binary_classification",
    )
    other_seed = compute_fingerprint(
        "correlation",
        CorrelationConfig(threshold=0.5, method="pearson"),
        max_local_rows=1000,
        seed=1,
        task_type="binary_classification",
    )
    regression = compute_fingerprint(
        "correlation",
        CorrelationConfig(threshold=0.9, method="pearson"),
        max_local_rows=1000,
        seed=0,
        task_type="regression",
    )
    assert pearson["seed"] == 0
    assert pearson["stratified"] is True
    assert pearson["max_rows"] == 1000
    assert other_seed["seed"] == 1
    assert regression["stratified"] is False


def _cache_config(path: Path | None, version: str = "v1") -> FeatureSelectionConfig:
    """Создаёт конфигурацию для проверок идентичности данных."""
    return FeatureSelectionConfig.from_dict({
        "order": [{"null_rate": {"threshold": 0.5}}],
        "statistics": {"cache": {
            "enabled": True,
            "path": str(path) if path is not None else None,
            "dataset_version": version,
        }},
    })


def test_changed_values_with_same_shape_do_not_reuse_metrics(tmp_path: Path) -> None:
    config = _cache_config(tmp_path / "metrics.json")
    first = _context(config)
    remaining, _ = run_order(first, first.candidates)
    assert "drop_null" not in remaining

    changed = _frame().fillna(0.0)
    second = _context(config, changed)
    remaining, _ = run_order(second, second.candidates)
    assert "drop_null" in remaining
    entries = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))["entries"]
    assert len(entries) == 2


def test_new_candidate_is_computed(tmp_path: Path) -> None:
    config = _cache_config(tmp_path / "metrics.json")
    first = _context(config)
    run_order(first, first.candidates)
    frame = _frame().assign(new_feature=float("nan"))
    second = _context(config, frame)
    second.schema = replace(second.schema, continuous=(*second.schema.continuous, "new_feature"))
    second.candidates = second.schema.candidate_features()
    remaining, _ = run_order(second, second.candidates)
    assert "new_feature" not in remaining
    assert any(item.feature == "new_feature" for item in second.decisions)


def test_threshold_change_reuses_metrics(tmp_path: Path) -> None:
    config = _cache_config(tmp_path / "metrics.json")
    first = _context(config)
    run_order(first, first.candidates)
    payload = config.to_dict()
    payload["order"] = [{"null_rate": {"threshold": 0.9}}]
    second = _context(FeatureSelectionConfig.from_dict(payload))
    with patch.object(NullRateSelector, "compute", side_effect=AssertionError("cache miss")):
        remaining, _ = run_order(second, second.candidates)
    assert "drop_null" in remaining


@pytest.mark.parametrize("change", ["version", "target", "types", "valid", "row_order"])
def test_data_fingerprint_tracks_input_changes(tmp_path: Path, change: str) -> None:
    first = _context(_cache_config(tmp_path / "metrics.json"))
    second = _context(first.config)
    if change == "version":
        second.config = _cache_config(tmp_path / "metrics.json", "v2")
    elif change == "target":
        second.schema = replace(second.schema, target="other_target")
    elif change == "types":
        second.datasets["train"]["keep"] = second.datasets["train"]["keep"].astype("int64")
    elif change == "valid":
        first.datasets["valid"] = _frame()
        second.datasets["valid"] = _frame().fillna(0.0)
    else:
        second.datasets["train"] = _frame().iloc[::-1]
    assert compute_data_fingerprint(first) != compute_data_fingerprint(second)


def test_row_sample_invalidates_cache_within_run(tmp_path: Path) -> None:
    payload = _cache_config(tmp_path / "metrics.json").to_dict()
    payload["order"] = [
        {"null_rate": {"threshold": 1.0}},
        {"row_sample": {"max_rows": 3, "stratified": False}},
        {"null_rate": {"threshold": 1.0}},
    ]
    context = _context(FeatureSelectionConfig.from_dict(payload))
    run_order(context, context.candidates)
    entries = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))["entries"]
    assert len(entries) == 2
    assert entries[0]["fingerprint"]["data"]["row_transforms"] == []
    assert entries[1]["fingerprint"]["data"]["row_transforms"] == [{
        "method": "row_sample", "max_rows": 3, "stratified": False, "seed": context.seed,
    }]


def test_default_cache_is_written_inside_output_dir(tmp_path: Path) -> None:
    context = _context(_cache_config(None))
    context.output_dir = tmp_path / "run"
    run_order(context, context.candidates)
    assert (context.output_dir / "statistics_metrics.json").is_file()


@pytest.mark.parametrize("version", [None, "", "  ", 5, True])
def test_enabled_cache_requires_valid_version(version: object) -> None:
    with pytest.raises(ConfigError, match="dataset_version"):
        FeatureSelectionConfig.from_dict({"statistics": {"cache": {
            "enabled": True, "dataset_version": version,
        }}})


def test_legacy_cache_is_rejected_unless_recomputed(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({
        "format_version": 1,
        "entries": [{"method": "null_rate", "fingerprint": {}, "metrics": {"values": {"keep": 1.0}}}],
    }), encoding="utf-8")
    with pytest.raises(ConfigError, match="format_version"):
        StatisticsMetricsCache.load(path)
    config = _cache_config(path)
    config = replace(config, statistics=replace(config.statistics, cache=replace(
        config.statistics.cache, force_recompute=True,
    )))
    context = _context(config)
    remaining, _ = run_order(context, context.candidates)
    assert "keep" in remaining
    assert json.loads(path.read_text(encoding="utf-8"))["format_version"] == 2


def test_spark_fingerprint_uses_snapshot_without_reading_rows(tmp_path: Path) -> None:
    """Проверяет только построение ключа по метаданным, без выполнения Spark."""
    class Schema:
        def jsonValue(self) -> dict:
            return {"type": "struct", "fields": [{"name": "keep", "type": "double"}]}

    class MetadataOnlyFrame:
        __module__ = "pyspark.sql.dataframe"
        schema = Schema()

        def count(self) -> int:
            raise AssertionError("Fingerprint must not scan Spark data")

        def collect(self) -> list:
            raise AssertionError("Fingerprint must not collect Spark data")

    context = _context(_cache_config(tmp_path / "metrics.json"))
    context.datasets = {"train": MetadataOnlyFrame(), "test": object()}
    first = compute_data_fingerprint(context)
    assert set(first["splits"]) == {"train"}
    context.config = _cache_config(tmp_path / "metrics.json", "v2")
    assert compute_data_fingerprint(context) != first
    second = compute_data_fingerprint(context)
    context.statistics_row_transforms.append({"method": "row_sample", "seed": 19, "max_rows": 100})
    assert compute_data_fingerprint(context) != second


def test_new_snapshot_version_recomputes_metrics(tmp_path: Path) -> None:
    first = _context(_cache_config(tmp_path / "metrics.json"))
    run_order(first, first.candidates)
    second = _context(_cache_config(tmp_path / "metrics.json", "v2"))
    original = NullRateSelector.compute
    with patch.object(NullRateSelector, "compute", autospec=True, side_effect=original) as compute:
        run_order(second, second.candidates)
    assert compute.call_count == 1


@pytest.mark.parametrize("previous_step", ["null_rate", "feature_drop", "random_feature_drop", "row_sample"])
def test_correlation_cache_matches_uncached_after_previous_steps(tmp_path: Path, previous_step: str) -> None:
    """Обычный запуск, промах и попадание кеша используют одинаковые входы и решения."""
    drop_path = tmp_path / "drop.txt"
    drop_path.write_text("drop_null\n", encoding="utf-8")
    steps = {
        "null_rate": {"threshold": 0.5},
        "feature_drop": {"path": str(drop_path)},
        "random_feature_drop": {"n_features": 1},
        "row_sample": {"max_rows": 4, "stratified": False},
    }
    payload = _cache_config(tmp_path / "metrics.json").to_dict()
    payload["order"] = [
        {previous_step: steps[previous_step]},
        {"correlation": {"threshold": 0.9}},
    ]
    payload["statistics"]["cache"]["enabled"] = False
    uncached = _context(FeatureSelectionConfig.from_dict(payload))
    original = CorrelationSelector.compute
    with patch.object(CorrelationSelector, "compute", autospec=True, side_effect=original) as compute:
        expected = run_order(uncached, uncached.candidates)
    expected_candidates = list(compute.call_args.args[2])
    assert compute.call_count == 1

    payload["statistics"]["cache"]["enabled"] = True
    cached = _context(FeatureSelectionConfig.from_dict(payload))
    with patch.object(CorrelationSelector, "compute", autospec=True, side_effect=original) as compute:
        actual = run_order(cached, cached.candidates)
    assert actual == expected
    assert compute.call_count == 1
    assert list(compute.call_args.args[2]) == expected_candidates
    pd.testing.assert_frame_equal(cached.datasets["train"], uncached.datasets["train"])

    repeated = _context(cached.config)
    with patch.object(CorrelationSelector, "compute", side_effect=AssertionError("cache miss")):
        assert run_order(repeated, repeated.candidates) == expected


@pytest.mark.parametrize("second_candidates", [["a", "b"], ["b", "a", "keep"]])
def test_candidate_subset_and_order_have_separate_entries(tmp_path: Path, second_candidates: list[str]) -> None:
    payload = _cache_config(tmp_path / "metrics.json").to_dict()
    payload["order"] = [{"correlation": {"threshold": 0.9}}]
    config = FeatureSelectionConfig.from_dict(payload)
    first = _context(config)
    run_order(first, ["keep", "a", "b"])

    second = _context(config)
    original = CorrelationSelector.compute
    with patch.object(CorrelationSelector, "compute", autospec=True, side_effect=original) as compute:
        actual = run_order(second, second_candidates)
    assert compute.call_count == 1
    assert list(compute.call_args.args[2]) == second_candidates
    payload["statistics"]["cache"]["enabled"] = False
    uncached = _context(FeatureSelectionConfig.from_dict(payload))
    assert actual == run_order(uncached, second_candidates)
    entries = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))["entries"]
    assert [entry["fingerprint"]["candidates"] for entry in entries] == [["keep", "a", "b"], second_candidates]


def test_repeated_statistics_compute_only_remaining_candidates(tmp_path: Path) -> None:
    payload = _cache_config(tmp_path / "metrics.json").to_dict()
    payload["order"] = [{"null_rate": {"threshold": 0.5}}, {"null_rate": {"threshold": 0.9}}]
    context = _context(FeatureSelectionConfig.from_dict(payload))
    initial = list(context.candidates)
    original = NullRateSelector.compute
    with patch.object(NullRateSelector, "compute", autospec=True, side_effect=original) as compute:
        run_order(context, initial)
    assert [list(call.args[2]) for call in compute.call_args_list] == [
        initial, [feature for feature in initial if feature != "drop_null"],
    ]


def test_empty_candidates_do_not_compute_or_write_metrics(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    context = _context(_cache_config(path))
    with patch.object(NullRateSelector, "compute", side_effect=AssertionError("unexpected compute")):
        assert run_order(context, []) == ([], [])
    assert not path.exists()


def test_old_entry_without_candidates_is_not_reused(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    context = _context(_cache_config(path))
    cache = StatisticsMetricsCache(path)
    cache.upsert("null_rate", {"data": compute_data_fingerprint(context), "parameters": {}}, {
        "values": {"keep": 1.0},
    })
    remaining, _ = run_order(context, context.candidates)
    assert "keep" in remaining
    assert "drop_null" not in remaining
