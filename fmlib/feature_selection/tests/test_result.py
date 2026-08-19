"""Tests for SelectionResult serialization and apply."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from fmlib.feature_selection.base import FeatureDecision, StageContext
from fmlib.feature_selection.config import FeatureSelectionConfig
from fmlib.feature_selection.utils.conftest import require_spark_session
from fmlib.feature_selection.exceptions import SchemaError
from fmlib.feature_selection.result import (
    DroppedFeature,
    SelectionResult,
    _save_intermediate_result,
)
from fmlib.feature_selection.schema import FeatureSchema


def _sample_result() -> SelectionResult:
    schema = FeatureSchema(
        categorical=["segment"],
        continuous=["age", "balance"],
        target="response",
        task_type="binary_classification",
        id_columns=["client_id"],
    )
    return SelectionResult(
        selected_features=["age", "segment"],
        dropped_features=[
            DroppedFeature(
                feature="balance",
                stage="statistics",
                method="psi",
                reason="threshold_exceeded",
                value=0.31,
                threshold=0.25,
            ),
        ],
        schema=schema,
        config={"execution": {"seed": 42}},
        seed=42,
        stage_backends={"statistics": "spark"},
        scores={
            "lightgbm": {
                "importances": {"age": 0.7, "balance": 0.3},
                "shap_importances": {"age": 0.8, "balance": 0.2},
            },
        },
    )


def test_save_load_roundtrip(tmp_path: Path) -> None:
    result = _sample_result()
    path = tmp_path / "artifact.json"
    result.save(path)
    loaded = SelectionResult.load(path)
    assert loaded.format_version == 1
    assert loaded.selected_features == ["age", "segment"]
    assert loaded.dropped_features[0].feature == "balance"
    assert loaded.dropped_features[0].value == 0.31
    assert loaded.schema.target == "response"
    assert loaded.scores == result.scores


def test_save_preserves_existing_files_with_numeric_suffixes(
    tmp_path: Path,
) -> None:
    result = _sample_result()
    path = tmp_path / "artifact.json"

    first = result.save(path)
    second = result.save(path)
    third = result.save(path)

    assert first.name == "artifact.json"
    assert second.name == "artifact_1.json"
    assert third.name == "artifact_2.json"
    assert SelectionResult.load(second).selected_features == [
        "age",
        "segment",
    ]


def test_intermediate_result_preserves_context_scores(tmp_path: Path) -> None:
    sample = _sample_result()
    config = FeatureSelectionConfig.from_dict(
        {"model": {"method": "lightgbm"}},
    )
    context = StageContext(
        spark=require_spark_session(),
        datasets={},
        schema=sample.schema,
        config=config,
        seed=42,
        candidates=list(sample.selected_features),
        scores={
            "lightgbm": {
                "importances": {"age": 0.7, "balance": 0.3},
                "shap_importances": {"age": 0.8, "balance": 0.2},
            },
        },
    )

    _save_intermediate_result(
        remaining=list(sample.selected_features),
        decisions=[
            FeatureDecision(
                feature="balance",
                stage="model",
                method="lightgbm",
                reason="failed_lgbm_shap_selection",
                value=1.2,
                threshold=1.0,
                keep=False,
            ),
        ],
        stage_name="model",
        method_name="lightgbm",
        output_dir=tmp_path,
        context=context,
    )

    loaded = SelectionResult.load(tmp_path / "model_lightgbm_results.json")
    assert loaded.scores == context.scores
    assert loaded.selected_features == ["age", "segment"]
    assert [item.feature for item in loaded.dropped_features] == ["balance"]


def test_precise_intermediate_result_preserves_boruta_scores(
    tmp_path: Path,
) -> None:
    sample = _sample_result()
    config = FeatureSelectionConfig.from_dict(
        {
            "precise": {
                "method": "boruta_shap",
                "params": {"model_type": "rf"},
            },
        },
    )
    context = StageContext(
        spark=require_spark_session(),
        datasets={},
        schema=sample.schema,
        config=config,
        seed=42,
        candidates=["segment", "age"],
        scores={
            "boruta_shap": {
                "accepted": ["age"],
                "rejected": ["balance"],
                "tentative": [],
                "model_type": "rf",
            },
        },
    )

    _save_intermediate_result(
        remaining=["segment", "age"],
        decisions=[
            FeatureDecision(
                feature="balance",
                stage="precise",
                method="boruta_shap",
                reason="boruta_rejected",
                keep=False,
            ),
        ],
        stage_name="precise",
        method_name="boruta_shap",
        output_dir=tmp_path,
        context=context,
    )

    loaded = SelectionResult.load(
        tmp_path / "precise_boruta_shap_results.json",
    )
    assert loaded.scores == context.scores
    assert loaded.dropped_features[0].stage == "precise"
    assert loaded.dropped_features[0].method == "boruta_shap"


def test_apply_projects_selected_and_service() -> None:
    result = _sample_result()
    frame = pd.DataFrame(
        {
            "segment": ["a"],
            "age": [20],
            "balance": [100.0],
            "response": [0],
            "client_id": [10],
            "extra": [1],
        },
    )
    projected = result.apply(frame)
    assert projected.columns.tolist() == ["age", "segment", "response", "client_id"]


def test_apply_pandas_preserves_rows_and_data() -> None:
    result = _sample_result()
    frame = pd.DataFrame(
        {
            "segment": ["a", "b"],
            "age": [20, 30],
            "balance": [100.0, 200.0],
            "response": [0, 1],
            "client_id": [10, 11],
        },
    )

    projected = result.apply(frame)

    assert projected.columns.tolist() == [
        "age",
        "segment",
        "response",
        "client_id",
    ]
    assert projected.to_dict(orient="records") == [
        {
            "age": 20,
            "segment": "a",
            "response": 0,
            "client_id": 10,
        },
        {
            "age": 30,
            "segment": "b",
            "response": 1,
            "client_id": 11,
        },
    ]


def test_apply_missing_selected_raises() -> None:
    result = _sample_result()
    frame = pd.DataFrame({"segment": ["a"], "response": [0], "client_id": [10]})
    with pytest.raises(SchemaError, match="selected features missing"):
        result.apply(frame)


def test_apply_spark_projects_selected_and_service(spark: Any) -> None:
    result = _sample_result()
    frame = spark.createDataFrame(
        [("a", 20, 100.0, 0, 10, 1)],
        ["segment", "age", "balance", "response", "client_id", "extra"],
    )
    projected = result.apply(frame)
    assert projected.columns == ["age", "segment", "response", "client_id"]


def test_apply_quotes_literal_spark_column_names(spark: Any) -> None:
    frame = spark.createDataFrame(
        [(1.0, 0), (2.0, 1)],
        ["foo.bar", "response"],
    )
    schema = FeatureSchema(
        categorical=(),
        continuous=("foo.bar",),
        target="response",
        task_type="binary_classification",
    )
    result = SelectionResult(
        selected_features=["foo.bar"],
        dropped_features=[],
        schema=schema,
        config={},
        seed=42,
    )

    projected = result.apply(frame)

    assert projected.columns == ["foo.bar", "response"]
    assert projected.collect()[0]["foo.bar"] == 1.0
