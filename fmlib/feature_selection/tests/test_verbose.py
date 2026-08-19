"""Tests for per-method verbose event logging."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fmlib.feature_selection.config import FeatureSelectionConfig, VerboseConfig
from fmlib.feature_selection.utils.conftest import (
    make_pandas_frame,
    make_wide_schema_columns,
    require_spark_session,
)
from fmlib.feature_selection.pipeline import FeatureSelectionPipeline
from fmlib.feature_selection.schema import FeatureSchema
from fmlib.feature_selection.statistics.correlation import CorrelationSelector
from fmlib.feature_selection.tests.test_pipeline import _config, _schema
from fmlib.feature_selection.utils.verbose import VERBOSE_LOG_FILENAME, VerboseRecorder


def test_recorder_strips_feature_name_lists_and_respects_flags() -> None:
    recorder = VerboseRecorder(
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


def test_verbose_pipeline_writes_verbose_log_without_feature_names(
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
        require_spark_session(),
        datasets={"train": make_pandas_frame(columns)},
        schema=schema,
        output_dir=tmp_path,
    )

    path = tmp_path / VERBOSE_LOG_FILENAME
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["n_events"] == len(payload["events"])
    assert result.verbose_log is not None
    assert result.verbose_log["n_events"] == payload["n_events"]
    methods = {event["method"] for event in payload["events"]}
    assert "pipeline" in methods
    assert "correlation" in methods
    assert "lasso" not in methods
    assert "null_rate" not in methods
    assert "verbose_log" not in result.to_dict()

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
    recorder = VerboseRecorder(verbose=VerboseConfig(pipeline=True))
    recorder.emit("pipeline", "start", n_candidates=3)
    captured = capsys.readouterr()
    assert "fs.verbose pipeline/start" in captured.out
    assert "n_candidates=3" in captured.out
    assert "fs.debug" not in captured.out


def test_verbose_without_output_dir_keeps_events_on_result(
    capsys: pytest.CaptureFixture[str],
) -> None:
    categorical, continuous, columns = make_wide_schema_columns(16)
    config = _config(
        execution={
            "seed": 42,
            "verbose": {"pipeline": True, "null_rate": True},
        },
        statistics={"order": ["null_rate"]},
    )
    result = FeatureSelectionPipeline(config).fit_select(
        require_spark_session(),
        datasets={"train": make_pandas_frame(columns)},
        schema=_schema(categorical, continuous, with_split=False),
    )

    assert result.verbose_log is not None
    assert result.verbose_log["n_events"] >= 1
    methods = {event["method"] for event in result.verbose_log["events"]}
    assert "pipeline" in methods
    captured = capsys.readouterr()
    assert "fs.verbose" in captured.out
    assert "result.verbose_log" in captured.out
    assert "output_dir" in captured.out
    assert VERBOSE_LOG_FILENAME in captured.out


def test_verbose_log_written_when_selector_raises(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
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
        message = "simulated selector failure"
        raise RuntimeError(message)

    monkeypatch.setattr(CorrelationSelector, "select", boom)
    with pytest.raises(RuntimeError, match="simulated selector failure"):
        FeatureSelectionPipeline(config).fit_select(
            require_spark_session(),
            datasets={"train": make_pandas_frame(columns)},
            schema=_schema(categorical, continuous, with_split=False),
            output_dir=tmp_path,
        )

    path = tmp_path / VERBOSE_LOG_FILENAME
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    error_events = [event for event in payload["events"] if event["stage"] == "error"]
    assert error_events
    assert any(event["method"] in {"pipeline", "correlation"} for event in error_events)
    captured = capsys.readouterr()
    assert f"wrote {path}" in captured.out or "fs.verbose wrote" in captured.out


def test_spark_snapshot_never_counts_rows() -> None:
    class _SparkFrame:
        columns: tuple[str, ...] = ("a", "b")

        def count(self: _SparkFrame) -> int:
            message = "verbose log must not call Spark count()"
            raise AssertionError(message)

        def collect(self: _SparkFrame) -> list:
            message = "verbose log must not call Spark collect()"
            raise AssertionError(message)

        def toPandas(self: _SparkFrame) -> None:  # noqa: N802 - Spark API
            message = "verbose log must not call Spark toPandas()"
            raise AssertionError(message)

    _SparkFrame.__module__ = "pyspark.sql.dataframe"
    recorder = VerboseRecorder(verbose=VerboseConfig(pipeline=True))
    snapshot = recorder.snapshot_frame(_SparkFrame(), count_rows=True)
    assert snapshot["type"] == "_SparkFrame"
    assert snapshot["n_cols"] == 2
    assert "n_rows" not in snapshot


def test_pandas_snapshot_includes_n_rows() -> None:
    recorder = VerboseRecorder(verbose=VerboseConfig(pipeline=True))
    snapshot = recorder.snapshot_frame(make_pandas_frame(["num_0", "response"]))
    assert snapshot["n_rows"] == 200
    assert snapshot["n_cols"] == 2


def test_verbose_off_does_not_write_verbose_log(tmp_path: Path) -> None:
    categorical, continuous, columns = make_wide_schema_columns(24)
    result = FeatureSelectionPipeline(_config()).fit_select(
        require_spark_session(),
        datasets={"train": make_pandas_frame(columns)},
        schema=_schema(categorical, continuous, with_split=False),
        output_dir=tmp_path,
    )
    assert not (tmp_path / VERBOSE_LOG_FILENAME).exists()
    assert not (tmp_path / "debug_log.json").exists()
    assert (tmp_path / "final_results.json").exists()
    assert result.verbose_log is None


def test_verbose_true_records_enabled_selectors_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
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
    result = FeatureSelectionPipeline(config).fit_select(
        require_spark_session(),
        datasets={"train": make_pandas_frame(columns)},
        schema=FeatureSchema(
            categorical=tuple(categorical),
            continuous=tuple(continuous),
            target="response",
            task_type="binary_classification",
        ),
        output_dir=tmp_path,
    )
    payload = json.loads((tmp_path / VERBOSE_LOG_FILENAME).read_text(encoding="utf-8"))
    methods = {event["method"] for event in payload["events"]}
    assert "pipeline" in methods
    assert "null_rate" in methods
    assert "lasso" in methods
    assert "correlation" not in methods
    assert "constants" not in methods
    assert result.verbose_log is not None
    captured = capsys.readouterr()
    assert "fs.verbose wrote" in captured.out
    assert VERBOSE_LOG_FILENAME in captured.out


def _assert_no_feature_name_lists(
    events: list[dict],
    feature_names: set[str],
) -> None:
    """Verbose events may mention counts, but must not dump keep/drop name lists."""
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
