"""Tests for per-method verbose debug logging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fmlib.feature_selection.config import FeatureSelectionConfig, VerboseConfig
from fmlib.feature_selection.debug import DebugRecorder
from fmlib.feature_selection.pipeline import FeatureSelectionPipeline
from fmlib.feature_selection.schema import FeatureSchema
from fmlib.feature_selection.statistics.correlation import CorrelationSelector
from fmlib.feature_selection.conftest import (
    FakeSparkSession,
    make_pandas_frame,
    make_wide_schema_columns,
)
from fmlib.feature_selection.tests.test_pipeline import _config, _schema


def test_recorder_strips_feature_name_lists_and_respects_flags() -> None:
    recorder = DebugRecorder(
        verbose=VerboseConfig(lightgbm=True, correlation=False),
    )
    recorder.emit(
        "lightgbm",
        "aggregate",
        n_dropped=3,
        dropped=["secret_a", "secret_b"],
        selected=["keep_me"],
        n_rows=100,
    )
    recorder.emit("correlation", "matrix", n_evaluated=10)

    assert len(recorder.events) == 1
    event = recorder.events[0]
    assert event["method"] == "lightgbm"
    assert event["n_dropped"] == 3
    assert event["n_rows"] == 100
    assert "dropped" not in event
    assert "selected" not in event
    blob = json.dumps(event)
    assert "secret_a" not in blob
    assert "keep_me" not in blob


def test_verbose_pipeline_writes_debug_log_without_feature_names(
    tmp_path: Path,
) -> None:
    categorical, continuous, columns = make_wide_schema_columns(24)
    schema = _schema(categorical, continuous, with_split=False)
    config = _config(
        execution={
            "seed": 42,
            "verbose": {
                "pipeline": True,
                "correlation": True,
            },
        },
    )
    result = FeatureSelectionPipeline(config).fit_select(
        FakeSparkSession(),
        datasets={"train": make_pandas_frame(columns)},
        schema=schema,
        output_dir=tmp_path,
    )

    path = tmp_path / "debug_log.json"
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["n_events"] == len(payload["events"])
    methods = {event["method"] for event in payload["events"]}
    assert "pipeline" in methods
    assert "correlation" in methods
    assert "lasso" not in methods
    assert "null_rate" not in methods

    start = next(
        event
        for event in payload["events"]
        if event["method"] == "pipeline" and event["stage"] == "start"
    )
    assert start["datasets"]["train"]["n_rows"] == 200
    assert start["n_candidates"] == len(categorical) + len(continuous)

    correlation_end = next(
        event
        for event in payload["events"]
        if event["method"] == "correlation" and event["stage"] == "end"
    )
    assert "duration_seconds" in correlation_end
    assert correlation_end["n_candidates_in"] == len(categorical) + len(continuous)
    assert "n_dropped" in correlation_end
    assert "drop_reasons" in correlation_end

    names = set(schema.candidate_features()) | set(result.selected_features)
    names.update(item.feature for item in result.dropped_features)
    _assert_no_feature_name_lists(payload["events"], names)


def test_emit_prints_compact_line_for_notebooks(capsys: pytest.CaptureFixture[str]) -> None:
    recorder = DebugRecorder(verbose=VerboseConfig(pipeline=True))
    recorder.emit("pipeline", "start", n_candidates=3)
    captured = capsys.readouterr()
    assert "fs.debug pipeline/start" in captured.out
    assert "n_candidates=3" in captured.out


def test_debug_log_written_when_selector_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    categorical, continuous, columns = make_wide_schema_columns(16)
    config = _config(
        execution={
            "seed": 42,
            "verbose": {
                "pipeline": True,
                "correlation": True,
            },
        },
    )

    def boom(
        self: CorrelationSelector,
        context: object,
        candidates: object,
    ) -> list:
        raise RuntimeError("simulated selector failure")

    monkeypatch.setattr(CorrelationSelector, "select", boom)
    with pytest.raises(RuntimeError, match="simulated selector failure"):
        FeatureSelectionPipeline(config).fit_select(
            FakeSparkSession(),
            datasets={"train": make_pandas_frame(columns)},
            schema=_schema(categorical, continuous, with_split=False),
            output_dir=tmp_path,
        )

    path = tmp_path / "debug_log.json"
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    error_events = [event for event in payload["events"] if event["stage"] == "error"]
    assert error_events
    assert any(event["method"] in {"pipeline", "correlation"} for event in error_events)


def test_spark_snapshot_never_counts_rows() -> None:
    class _SparkFrame:
        columns = ["a", "b"]

        def count(self: _SparkFrame) -> int:
            raise AssertionError("debug must not call Spark count()")

        def collect(self: _SparkFrame) -> list:
            raise AssertionError("debug must not call Spark collect()")

        def toPandas(self: _SparkFrame) -> None:
            raise AssertionError("debug must not call Spark toPandas()")

    _SparkFrame.__module__ = "pyspark.sql.dataframe"
    recorder = DebugRecorder(verbose=VerboseConfig(pipeline=True))
    snapshot = recorder.snapshot_frame(_SparkFrame(), count_rows=True)
    assert snapshot["type"] == "_SparkFrame"
    assert snapshot["n_cols"] == 2
    assert "n_rows" not in snapshot


def test_pandas_snapshot_includes_n_rows() -> None:
    recorder = DebugRecorder(verbose=VerboseConfig(pipeline=True))
    snapshot = recorder.snapshot_frame(make_pandas_frame(["num_0", "response"]))
    assert snapshot["n_rows"] == 200
    assert snapshot["n_cols"] == 2


def test_verbose_off_does_not_write_debug_log(tmp_path: Path) -> None:
    categorical, continuous, columns = make_wide_schema_columns(24)
    FeatureSelectionPipeline(_config()).fit_select(
        FakeSparkSession(),
        datasets={"train": make_pandas_frame(columns)},
        schema=_schema(categorical, continuous, with_split=False),
        output_dir=tmp_path,
    )
    assert not (tmp_path / "debug_log.json").exists()
    assert (tmp_path / "final_results.json").exists()


def test_verbose_true_records_enabled_selectors_only(
    tmp_path: Path,
) -> None:
    categorical, continuous, columns = make_wide_schema_columns(16)
    config = FeatureSelectionConfig.from_dict(
        {
            "statistics": {"order": ["null_rate"]},
            "model": {"enabled": True, "method": "lasso"},
            "precise": {"enabled": False, "method": "none"},
            "execution": {"seed": 42, "verbose": True},
        },
    )
    FeatureSelectionPipeline(config).fit_select(
        FakeSparkSession(),
        datasets={"train": make_pandas_frame(columns)},
        schema=FeatureSchema(
            categorical=tuple(categorical),
            continuous=tuple(continuous),
            target="response",
            task_type="binary_classification",
        ),
        output_dir=tmp_path,
    )
    payload = json.loads((tmp_path / "debug_log.json").read_text(encoding="utf-8"))
    methods = {event["method"] for event in payload["events"]}
    assert "pipeline" in methods
    assert "null_rate" in methods
    assert "lasso" in methods
    assert "correlation" not in methods
    assert "constants" not in methods


def _assert_no_feature_name_lists(
    events: list[dict],
    feature_names: set[str],
) -> None:
    """Debug events may mention counts, but must not dump keep/drop name lists."""
    for event in events:
        _assert_no_feature_name_lists_in_value(event, feature_names)


def _assert_no_feature_name_lists_in_value(value: object, feature_names: set[str]) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _assert_no_feature_name_lists_in_value(item, feature_names)
        return
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value):
        overlap = feature_names.intersection(value)
        assert not overlap, overlap
