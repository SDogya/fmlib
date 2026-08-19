"""Tests for file-driven preprocessing feature exclusions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from fmlib.feature_selection import (
    FeatureSchema,
    apply_feature_drop_file,
    load_feature_names,
)
from fmlib.feature_selection.exceptions import BackendError, ConfigError
from fmlib.feature_selection.result import DroppedFeature, SelectionResult


def _schema() -> FeatureSchema:
    return FeatureSchema(
        categorical=("category", "segment"),
        continuous=("first", "second"),
        target="response",
        task_type="binary_classification",
        time="event_date",
        id_columns=("client_id",),
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category": ["a", "b"],
            "segment": ["x", "y"],
            "first": [1.0, 2.0],
            "second": [3.0, 4.0],
            "response": [0, 1],
            "event_date": ["2026-01-01", "2026-01-02"],
            "client_id": [1, 2],
        },
    )


def test_load_feature_names_ignores_comments_blanks_and_duplicates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "drop.txt"
    path.write_text(
        "\n# generated exclusions\n first \n\ncategory\nfirst\n",
        encoding="utf-8",
    )

    assert load_feature_names(path) == ["first", "category"]


def test_load_feature_names_has_actionable_file_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="failed to read"):
        load_feature_names(tmp_path / "missing.txt")

    empty = tmp_path / "empty.txt"
    empty.write_text("# no features\n\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="contains no feature names"):
        load_feature_names(empty)


def test_load_feature_names_from_selection_result_json(tmp_path: Path) -> None:
    result = SelectionResult(
        selected_features=["first", "segment"],
        dropped_features=[
            DroppedFeature(
                feature="category",
                stage="statistics",
                method="null_rate",
                reason="high_null_rate",
                value=0.99,
                threshold=0.95,
            ),
            DroppedFeature(
                feature="second",
                stage="model",
                method="lightgbm",
                reason="failed_lgbm_shap_selection",
            ),
            DroppedFeature(
                feature="category",
                stage="precise",
                method="boruta_shap",
                reason="boruta_rejected",
            ),
        ],
        schema=_schema(),
        config={"execution": {"seed": 42}},
        seed=42,
    )
    path = tmp_path / "final_results.json"
    result.save(path)

    assert load_feature_names(path) == ["category", "second"]


def test_load_feature_names_from_result_json_rejects_bad_payloads(
    tmp_path: Path,
) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_feature_names(broken)

    array_path = tmp_path / "array.json"
    array_path.write_text('["first", "second"]\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="SelectionResult object"):
        load_feature_names(array_path)

    missing = tmp_path / "missing_dropped.json"
    missing.write_text('{"selected_features": ["first"]}\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="no dropped_features"):
        load_feature_names(missing)

    empty = tmp_path / "empty_dropped.json"
    empty.write_text('{"dropped_features": []}\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="contains no feature names"):
        load_feature_names(empty)


def test_apply_drop_from_selection_result_json(tmp_path: Path) -> None:
    path = tmp_path / "final_results.json"
    SelectionResult(
        selected_features=["first", "segment"],
        dropped_features=[
            DroppedFeature(
                feature="category",
                stage="statistics",
                method="iv",
                reason="low_iv",
            ),
            DroppedFeature(
                feature="second",
                stage="model",
                method="lightgbm",
                reason="failed_lgbm_shap_selection",
            ),
        ],
        schema=_schema(),
        config={},
        seed=0,
    ).save(path)
    train = _frame()

    datasets, schema, report = apply_feature_drop_file(
        {"train": train},
        _schema(),
        path,
    )

    assert report.dropped == ("category", "second")
    assert "category" not in datasets["train"].columns
    assert "second" not in datasets["train"].columns
    assert schema.categorical == ("segment",)
    assert schema.continuous == ("first",)
    assert "category" in train.columns


def test_apply_drop_updates_all_frames_and_schema_without_mutating_inputs(
    tmp_path: Path,
) -> None:
    path = tmp_path / "drop.txt"
    path.write_text("category\nsecond\nunknown_feature\n", encoding="utf-8")
    train = _frame()
    valid = _frame()

    datasets, schema, report = apply_feature_drop_file(
        {"train": train, "valid": valid},
        _schema(),
        path,
    )

    assert "category" not in datasets["train"].columns
    assert "second" not in datasets["valid"].columns
    assert "category" in train.columns
    assert "second" in valid.columns
    assert schema.categorical == ("segment",)
    assert schema.continuous == ("first",)
    assert schema.target == "response"
    assert report.dropped == ("category", "second")
    assert report.unknown == ("unknown_feature",)


def test_strict_mode_rejects_unknown_features(tmp_path: Path) -> None:
    path = tmp_path / "drop.txt"
    path.write_text("missing\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="not declared"):
        apply_feature_drop_file(
            {"train": _frame()},
            _schema(),
            path,
            strict=True,
        )


@pytest.mark.parametrize("service_column", ["response", "event_date", "client_id"])
def test_service_columns_cannot_be_dropped(
    tmp_path: Path,
    service_column: str,
) -> None:
    path = tmp_path / "drop.txt"
    path.write_text(f"{service_column}\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="service columns"):
        apply_feature_drop_file({"train": _frame()}, _schema(), path)


def test_unsupported_frame_type_has_backend_error(tmp_path: Path) -> None:
    path = tmp_path / "drop.txt"
    path.write_text("first\n", encoding="utf-8")

    with pytest.raises(BackendError, match="unsupported 'train' split"):
        apply_feature_drop_file(
            {"train": object()},
            _schema(),
            path,
        )


def test_apply_drop_on_spark_frame(spark: Any, tmp_path: Path) -> None:
    path = tmp_path / "drop.txt"
    path.write_text("category\nsecond\n", encoding="utf-8")
    train = spark.createDataFrame(_frame())
    datasets, schema, report = apply_feature_drop_file(
        {"train": train},
        _schema(),
        path,
    )
    assert "category" not in datasets["train"].columns
    assert "second" not in datasets["train"].columns
    assert "category" in train.columns
    assert schema.categorical == ("segment",)
    assert schema.continuous == ("first",)
    assert report.dropped == ("category", "second")
