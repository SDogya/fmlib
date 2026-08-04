"""Tests for SelectionResult serialization and apply."""

from pathlib import Path

import pytest

from fmlib.feature_selection.exceptions import SchemaError
from fmlib.feature_selection.result import DroppedFeature, SelectionResult
from fmlib.feature_selection.schema import FeatureSchema
from fmlib.feature_selection.tests.conftest import FakeDataFrame


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


def test_apply_projects_selected_and_service() -> None:
    result = _sample_result()
    frame = FakeDataFrame(columns=["segment", "age", "balance", "response", "client_id", "extra"])
    projected = result.apply(frame)
    assert projected.columns == ["age", "segment", "response", "client_id"]


def test_apply_missing_selected_raises() -> None:
    result = _sample_result()
    frame = FakeDataFrame(columns=["segment", "response", "client_id"])
    with pytest.raises(SchemaError, match="selected features missing"):
        result.apply(frame)
