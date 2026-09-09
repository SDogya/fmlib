"""Tests for FeatureSelectionPipeline skeleton."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import pytest

from fmlib.feature_selection import (
    FeatureSchema,
    FeatureSelectionConfig,
    FeatureSelectionPipeline,
    SelectionResult,
)
from fmlib.feature_selection.base import FeatureDecision, StageContext
from fmlib.feature_selection.utils.conftest import (
    make_pandas_frame,
    make_wide_schema_columns,
    require_spark_session,
)
from fmlib.feature_selection.exceptions import ConfigError, SchemaError
from fmlib.feature_selection.model_based.lightgbm import LightGbmSelector
from fmlib.feature_selection.precise.boruta_shap import BorutaShapSelector


def _schema(categorical: list[str], continuous: list[str], *, with_split: bool = True) -> FeatureSchema:
    return FeatureSchema(
        categorical=tuple(categorical),
        continuous=tuple(continuous),
        target="response",
        task_type="binary_classification",
        time="event_date",
        split="dataset_split" if with_split else None,
        id_columns=("client_id",),
    )


def _config(**overrides: object) -> FeatureSelectionConfig:
    payload: dict = {
        "statistics": {"order": ["correlation"]},
        "model": {"enabled": False, "method": "lightgbm"},
        "precise": {"enabled": False, "method": "none"},
        "execution": {"seed": 42},
    }
    payload.update(overrides)
    return FeatureSelectionConfig.from_dict(payload)


def test_fit_select_single_dataframe() -> None:
    categorical, continuous, columns = make_wide_schema_columns(40)
    spark = require_spark_session()
    data = make_pandas_frame(columns)
    pipeline = FeatureSelectionPipeline(_config())
    result = pipeline.fit_select(
        spark,
        data=data,
        schema=_schema(categorical, continuous, with_split=True),
    )
    assert result.datasets_mode == "single"
    assert len(result.selected_features) <= len(categorical) + len(continuous)
    assert len(result.selected_features) >= 1
    assert all(name in categorical + continuous for name in result.selected_features)
    assert result.warnings == []
    dropped_names = {item.feature for item in result.dropped_features}
    assert set(result.selected_features).isdisjoint(dropped_names)
    # inputs not mutated
    assert list(data.columns) == columns


def test_fit_select_datasets_mapping() -> None:
    categorical, continuous, columns = make_wide_schema_columns(40)
    spark = require_spark_session()
    train = make_pandas_frame(columns)
    pipeline = FeatureSelectionPipeline(_config())
    result = pipeline.fit_select(
        spark,
        datasets={"train": train},
        schema=_schema(categorical, continuous, with_split=False),
    )
    assert result.datasets_mode == "mapping"
    assert result.selected_features
    projected = pipeline.transform(train, result)
    for name in result.selected_features:
        assert name in projected.columns
    assert "response" in projected.columns
    assert "client_id" in projected.columns


def test_feature_drop_file_runs_before_selectors_and_updates_schema(
    tmp_path: Path,
) -> None:
    categorical, continuous, columns = make_wide_schema_columns(24)
    dropped = [categorical[0], continuous[0]]
    drop_path = tmp_path / "drop.txt"
    drop_path.write_text(
        "\n".join([*dropped, "unknown_feature", ""]),
        encoding="utf-8",
    )
    frame = make_pandas_frame(columns)
    config = _config(
        preprocessing={
            "feature_drop": {
                "enabled": True,
                "path": str(drop_path),
                "strict": False,
            },
        },
    )

    result = FeatureSelectionPipeline(config).fit_select(
        require_spark_session(),
        datasets={"train": frame},
        schema=_schema(categorical, continuous, with_split=False),
        output_dir=tmp_path / "artifacts",
    )

    manual_drops = [
        item
        for item in result.dropped_features
        if item.method == "feature_drop"
    ]
    assert [item.feature for item in manual_drops] == dropped
    assert all(item.stage == "preprocessing" for item in manual_drops)
    assert dropped[0] not in result.schema.categorical
    assert dropped[1] not in result.schema.continuous
    assert result.scores["feature_drop#0"]["dropped"] == dropped
    assert result.scores["feature_drop#0"]["unknown"] == ["unknown_feature"]
    assert result.stage_backends["preprocessing"] == "spark"
    assert (
        tmp_path
        / "artifacts"
        / "00_preprocessing_feature_drop_results.json"
    ).exists()
    selector_artifact = SelectionResult.load(
        tmp_path / "artifacts" / "01_statistics_correlation_results.json",
    )
    final_artifact = SelectionResult.load(
        tmp_path / "artifacts" / "final_results.json",
    )
    for artifact in (selector_artifact, final_artifact):
        manual = [
            item.feature
            for item in artifact.dropped_features
            if item.method == "feature_drop"
        ]
        assert manual == dropped

    FeatureSelectionPipeline(config).fit_select(
        require_spark_session(),
        datasets={"train": frame},
        schema=_schema(categorical, continuous, with_split=False),
        output_dir=tmp_path / "artifacts",
    )
    assert (
        tmp_path
        / "artifacts"
        / "00_preprocessing_feature_drop_results_1.json"
    ).exists()
    assert (tmp_path / "artifacts" / "final_results_1.json").exists()
    assert all(feature in frame.columns for feature in dropped)


def test_test_run_preprocessing_samples_rows_and_drops_random_features(
    tmp_path: Path,
) -> None:
    categorical, continuous, columns = make_wide_schema_columns(24)
    frame = make_pandas_frame(columns, n_rows=200)
    config = _config(
        preprocessing={
            "random_feature_drop": {
                "enabled": True,
                "n_features": 2,
            },
            "row_sample": {
                "enabled": True,
                "max_rows": 40,
                "stratified": False,
            },
        },
    )

    result = FeatureSelectionPipeline(config).fit_select(
        require_spark_session(),
        datasets={"train": frame},
        schema=_schema(categorical, continuous, with_split=False),
        output_dir=tmp_path,
    )

    random_drops = [
        item
        for item in result.dropped_features
        if item.method == "random_feature_drop"
    ]
    assert len(random_drops) == 2
    assert all(item.reason == "random_test_drop" for item in random_drops)
    assert all(
        item.feature not in result.schema.candidate_features()
        for item in random_drops
    )
    train_sample = result.scores["row_sample#1"]["splits"]["train"]
    assert train_sample["original_rows"] == 200
    assert train_sample["sampled_rows"] == 40
    assert isinstance(train_sample["seed"], int)
    assert len(frame) == 200
    assert (tmp_path / "00_preprocessing_random_feature_drop_results.json").exists()
    assert (tmp_path / "01_preprocessing_row_sample_results.json").exists()


def test_mutually_exclusive_inputs() -> None:
    categorical, continuous, columns = make_wide_schema_columns(20)
    spark = require_spark_session()
    frame = make_pandas_frame(columns)
    pipeline = FeatureSelectionPipeline(_config())
    with pytest.raises(ConfigError, match="either data"):
        pipeline.fit_select(
            spark,
            data=frame,
            datasets={"train": frame},
            schema=_schema(categorical, continuous),
        )


def test_single_dataframe_requires_split() -> None:
    categorical, continuous, columns = make_wide_schema_columns(20)
    spark = require_spark_session()
    pipeline = FeatureSelectionPipeline(_config())
    with pytest.raises(SchemaError, match="split"):
        pipeline.fit_select(
            spark,
            data=make_pandas_frame(columns),
            schema=_schema(categorical, continuous, with_split=False),
        )


def test_deterministic_across_runs() -> None:
    categorical, continuous, columns = make_wide_schema_columns(40)
    spark = require_spark_session()
    schema = _schema(categorical, continuous)
    config = _config()
    first = FeatureSelectionPipeline(config).fit_select(
        spark,
        data=make_pandas_frame(columns),
        schema=schema,
    )
    second = FeatureSelectionPipeline(config).fit_select(
        spark,
        data=make_pandas_frame(columns),
        schema=schema,
    )
    assert first.selected_features == second.selected_features
    assert [item.feature for item in first.dropped_features] == [item.feature for item in second.dropped_features]


def test_precise_boruta_stage_produces_drops_and_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    categorical, continuous, columns = make_wide_schema_columns(50)
    spark = require_spark_session()
    config = _config(
        precise={
            "enabled": True,
            "method": "boruta_shap",
            "params": {"model_type": "rf"},
        },
    )

    def fake_select(
        _selector: BorutaShapSelector,
        context: StageContext,
        candidates: Sequence[str],
    ) -> list[FeatureDecision]:
        feature = next(name for name in candidates if name in continuous)
        accepted = [
            name for name in candidates if name in continuous and name != feature
        ]
        context.scores["boruta_shap"] = {
            "accepted": accepted,
            "rejected": [feature],
            "tentative": [],
            "model_type": "rf",
            "best_auc": 0.8,
            "best_params": {"n_estimators": 100},
            "boruta_trials": 2,
        }
        return [
            *[
                FeatureDecision(
                    feature=name,
                    stage="precise",
                    method="boruta_shap",
                    reason="boruta_accepted",
                    keep=True,
                )
                for name in accepted
            ],
            FeatureDecision(
                feature=feature,
                stage="precise",
                method="boruta_shap",
                reason="boruta_rejected",
                keep=False,
            ),
        ]

    monkeypatch.setattr(BorutaShapSelector, "select", fake_select)

    result = FeatureSelectionPipeline(config).fit_select(
        spark,
        data=make_pandas_frame(columns),
        schema=_schema(categorical, continuous),
    )

    boruta_drops = [
        item
        for item in result.dropped_features
        if item.method == "boruta_shap"
    ]
    assert len(boruta_drops) == 1
    assert boruta_drops[0].stage == "precise"
    assert result.scores["boruta_shap#2"]["model_type"] == "rf"


def test_correlation_drop_decisions_have_real_values() -> None:
    """Correlation selector should produce numeric value in FeatureDecision."""
    categorical, continuous, columns = make_wide_schema_columns(30)
    spark = require_spark_session()
    config = _config()
    result = FeatureSelectionPipeline(config).fit_select(
        spark,
        data=make_pandas_frame(columns),
        schema=_schema(categorical, continuous),
    )
    corr_drops = [item for item in result.dropped_features if item.method == "correlation"]
    for drop in corr_drops:
        assert drop.value is not None
        assert 0.0 <= drop.value <= 1.0
        assert drop.threshold is not None


def test_low_variance_selector_runs_in_pipeline() -> None:
    frame = pd.DataFrame(
        {
            "constant": [1.0, 1.0, 1.0, 1.0],
            "varying": [1.0, 2.0, 3.0, 4.0],
            "response": [0, 1, 0, 1],
        },
    )
    schema = FeatureSchema(
        categorical=(),
        continuous=("constant", "varying"),
        target="response",
        task_type="binary_classification",
    )
    config = FeatureSelectionConfig.from_dict(
        {
            "statistics": {"order": ["low_variance"]},
            "model": {"enabled": False, "method": "lightgbm"},
            "precise": {"enabled": False, "method": "none"},
        },
    )

    result = FeatureSelectionPipeline(config).fit_select(
        require_spark_session(),
        datasets={"train": frame},
        schema=schema,
    )

    low_variance_drops = [item for item in result.dropped_features if item.method == "low_variance"]
    assert [item.feature for item in low_variance_drops] == ["constant"]
    assert result.selected_features == ["varying"]


def test_lightgbm_stage_produces_real_scores_without_stub_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "category": ["a", "b", "a", "b"],
            "feature": [1.0, 2.0, 3.0, 4.0],
            "response": [0, 1, 0, 1],
        },
    )
    schema = FeatureSchema(
        categorical=("category",),
        continuous=("feature",),
        target="response",
        task_type="binary_classification",
    )
    config = FeatureSelectionConfig.from_dict(
        {
            "model": {"enabled": True, "method": "lightgbm"},
            "precise": {"enabled": False, "method": "none"},
        },
    )

    def fake_select(
        _selector: LightGbmSelector,
        context: StageContext,
        _candidates: Sequence[str],
    ) -> list[FeatureDecision]:
        context.scores["lightgbm"] = {
            "importances": {"feature": 1.0},
            "shap_importances": {"feature": 1.0},
            "lgbm_threshold": 0.85,
            "shap_threshold": 0.85,
        }
        return [
            FeatureDecision(
                feature="feature",
                stage="model",
                method="lightgbm",
                reason="failed_lgbm_shap_selection",
                value=1.1,
                threshold=1.0,
                keep=False,
            ),
        ]

    monkeypatch.setattr(LightGbmSelector, "select", fake_select)

    result = FeatureSelectionPipeline(config).fit_select(
        require_spark_session(),
        datasets={"train": frame},
        schema=schema,
    )

    assert result.selected_features == ["category"]
    assert [item.feature for item in result.dropped_features] == ["feature"]
    assert result.warnings == []
    assert result.scores["lightgbm#0"]["importances"] == {"feature": 1.0}
    assert result.scores["lightgbm#0"]["shap_importances"] == {"feature": 1.0}


def test_model_enabled_false_does_not_run_lightgbm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "category": ["a", "b", "a", "b"],
            "feature": [1.0, 2.0, 3.0, 4.0],
            "response": [0, 1, 0, 1],
        },
    )
    schema = FeatureSchema(
        categorical=("category",),
        continuous=("feature",),
        target="response",
        task_type="binary_classification",
    )
    config = FeatureSelectionConfig.from_dict(
        {
            "model": {"enabled": False, "method": "lightgbm"},
            "precise": {"enabled": False, "method": "none"},
        },
    )
    called = {"lightgbm": False}

    def fake_select(
        _selector: LightGbmSelector,
        context: StageContext,
        _candidates: Sequence[str],
    ) -> list[FeatureDecision]:
        called["lightgbm"] = True
        return []

    monkeypatch.setattr(LightGbmSelector, "select", fake_select)

    result = FeatureSelectionPipeline(config).fit_select(
        require_spark_session(),
        datasets={"train": frame},
        schema=schema,
    )

    assert called["lightgbm"] is False
    assert result.selected_features == ["category", "feature"]
    assert "lightgbm" not in result.scores
    assert result.warnings == []


def test_result_schema_apply_projection() -> None:
    """SelectionResult.apply should keep only selected + service columns."""
    categorical, continuous, columns = make_wide_schema_columns(30)
    spark = require_spark_session()
    config = _config()
    schema = _schema(categorical, continuous)
    result = FeatureSelectionPipeline(config).fit_select(
        spark,
        data=make_pandas_frame(columns),
        schema=schema,
    )
    fresh_data = make_pandas_frame(columns)
    projected = result.apply(fresh_data)
    keep = set(result.columns_to_keep())
    assert set(projected.columns) == keep & set(columns)


def test_no_service_columns_in_selected() -> None:
    """target, split, time, id columns must never appear in selected_features."""
    categorical, continuous, columns = make_wide_schema_columns(30)
    spark = require_spark_session()
    schema = _schema(categorical, continuous)
    result = FeatureSelectionPipeline(_config()).fit_select(
        spark,
        data=make_pandas_frame(columns),
        schema=schema,
    )
    forbidden = set(schema.service_columns())
    assert not (set(result.selected_features) & forbidden)


def test_apply_pandas_dataframe_selects_columns() -> None:
    """SelectionResult.apply keeps selected columns on a pandas frame."""
    categorical, continuous, columns = make_wide_schema_columns(30)
    spark = require_spark_session()
    result = FeatureSelectionPipeline(_config()).fit_select(
        spark,
        data=make_pandas_frame(columns),
        schema=_schema(categorical, continuous),
    )
    projected = result.apply(make_pandas_frame(columns, n_rows=8))
    for name in result.selected_features:
        assert name in projected.columns


def test_apply_spark_dataframe_selects_columns(spark: Any) -> None:
    """SelectionResult.apply keeps selected columns on a real Spark frame."""
    categorical, continuous, columns = make_wide_schema_columns(16)
    result = FeatureSelectionPipeline(_config()).fit_select(
        spark,
        data=make_pandas_frame(columns, n_rows=20),
        schema=_schema(categorical, continuous),
    )
    spark_frame = spark.createDataFrame(make_pandas_frame(columns, n_rows=8))
    projected = result.apply(spark_frame)
    keep = set(projected.columns)
    for name in result.selected_features:
        assert name in keep


def test_empty_order_keeps_all_candidates() -> None:
    categorical, continuous, columns = make_wide_schema_columns(16)
    result = FeatureSelectionPipeline(FeatureSelectionConfig()).fit_select(
        require_spark_session(),
        datasets={"train": make_pandas_frame(columns)},
        schema=_schema(categorical, continuous, with_split=False),
    )
    assert result.selected_features == categorical + continuous
    assert result.dropped_features == []
    assert result.warnings == []
    assert result.stage_backends == {}


def test_custom_statistics_order_runs_correlation_before_null_rate(tmp_path: Path) -> None:
    categorical, continuous, columns = make_wide_schema_columns(24)
    config = _config(
        statistics={"order": ["correlation", "null_rate"]},
        execution={"seed": 42, "verbose": True},
    )
    FeatureSelectionPipeline(config).fit_select(
        require_spark_session(),
        datasets={"train": make_pandas_frame(columns)},
        schema=_schema(categorical, continuous, with_split=False),
        output_dir=tmp_path,
    )
    events = json.loads((tmp_path / "verbose_log.json").read_text(encoding="utf-8"))["events"]
    starts = [
        (event["method"], event["step_index"])
        for event in events
        if event["stage"] == "start" and event["method"] in {"correlation", "null_rate"}
    ]
    assert starts == [("correlation", 0), ("null_rate", 1)]
    assert (tmp_path / "00_statistics_correlation_results.json").exists()
    assert (tmp_path / "01_statistics_null_rate_results.json").exists()


def test_utils_run_before_statistics(tmp_path: Path) -> None:
    categorical, continuous, columns = make_wide_schema_columns(24)
    frame = make_pandas_frame(columns, n_rows=80)
    config = _config(
        statistics={"order": ["null_rate", "correlation"]},
        preprocessing={
            "row_sample": {"enabled": True, "max_rows": 25, "stratified": False},
        },
        execution={"seed": 42, "verbose": True},
    )
    result = FeatureSelectionPipeline(config).fit_select(
        require_spark_session(),
        datasets={"train": frame},
        schema=_schema(categorical, continuous, with_split=False),
        output_dir=tmp_path,
    )
    events = json.loads((tmp_path / "verbose_log.json").read_text(encoding="utf-8"))["events"]
    starts = [
        (event["method"], event["step_index"])
        for event in events
        if event["stage"] == "start"
        and event["method"] in {"row_sample", "null_rate", "correlation"}
    ]
    assert starts == [("row_sample", 0), ("null_rate", 1), ("correlation", 2)]
    assert result.scores["row_sample#0"]["splits"]["train"]["sampled_rows"] == 25
    assert (tmp_path / "01_statistics_null_rate_results.json").exists()
    assert (tmp_path / "00_preprocessing_row_sample_results.json").exists()
    assert (tmp_path / "02_statistics_correlation_results.json").exists()


def test_unknown_statistics_order_method_is_config_error() -> None:
    with pytest.raises(ConfigError, match="Unknown method in statistics.order"):
        FeatureSelectionPipeline(
            FeatureSelectionConfig.from_dict({"statistics": {"order": ["xgboost"]}}),
        )


def test_method_enabled_field_is_unknown() -> None:
    with pytest.raises(ConfigError, match="Unknown fields"):
        FeatureSelectionConfig.from_dict(
            {"statistics": {"null_rate": {"enabled": False}}},
        )


def test_feature_drop_enabled_without_path_is_config_error() -> None:
    with pytest.raises(ConfigError, match=r"feature_drop\.path"):
        FeatureSelectionPipeline(
            _config(preprocessing={"feature_drop": {"enabled": True}}),
        )
