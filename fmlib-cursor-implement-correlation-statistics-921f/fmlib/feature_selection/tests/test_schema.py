"""Tests for FeatureSchema validation."""

import pytest

from fmlib.feature_selection.exceptions import SchemaError
from fmlib.feature_selection.schema import FeatureSchema


def test_candidate_order_and_service_exclusion() -> None:
    schema = FeatureSchema(
        categorical=["segment", "region"],
        continuous=["age", "balance"],
        target="response",
        task_type="binary_classification",
        time="event_date",
        split="dataset_split",
        id_columns=["client_id"],
    )
    assert schema.candidate_features() == ["segment", "region", "age", "balance"]
    assert "response" not in schema.candidate_features()
    assert "client_id" in schema.service_columns()


def test_categorical_continuous_overlap_raises() -> None:
    with pytest.raises(SchemaError, match="disjoint"):
        FeatureSchema(
            categorical=["age"],
            continuous=["age"],
            target="y",
            task_type="regression",
        )


def test_invalid_task_type_raises() -> None:
    with pytest.raises(SchemaError, match="task_type"):
        FeatureSchema(
            categorical=[],
            continuous=["age"],
            target="y",
            task_type="clustering",
        )


def test_target_in_candidates_raises() -> None:
    with pytest.raises(SchemaError, match="target"):
        FeatureSchema(
            categorical=[],
            continuous=["y"],
            target="y",
            task_type="regression",
        )


def test_validate_against_columns_missing() -> None:
    schema = FeatureSchema(
        categorical=["a"],
        continuous=["b"],
        target="y",
        task_type="regression",
        split="split",
    )
    with pytest.raises(SchemaError, match="missing"):
        schema.validate_against_columns(["a", "y", "split"], require_split=True)


def test_roundtrip_dict() -> None:
    schema = FeatureSchema(
        categorical=["a"],
        continuous=["b"],
        target="y",
        task_type="classification",
        fold="fold_id",
        id_columns=["id"],
    )
    restored = FeatureSchema.from_dict(schema.to_dict())
    assert restored == schema
