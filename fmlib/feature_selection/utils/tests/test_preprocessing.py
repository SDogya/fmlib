"""Тесты предобработки строк и случайных столбцов для тестового запуска."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from fmlib.feature_selection.exceptions import ConfigError
from fmlib.feature_selection.schema import FeatureSchema
from fmlib.feature_selection.utils.local_data import sample_frame_rows
from fmlib.feature_selection.utils.preprocessing import (
    apply_random_feature_drop,
    apply_row_sample,
)


def _schema() -> FeatureSchema:
    return FeatureSchema(
        categorical=("cat_a", "cat_b"),
        continuous=("first", "second", "third", "fourth"),
        target="response",
        task_type="binary_classification",
        time="event_date",
        id_columns=("client_id",),
    )


def _frame(n_rows: int = 100) -> pd.DataFrame:
    target = np.resize(np.array([0, 1]), n_rows)
    return pd.DataFrame(
        {
            "cat_a": np.where(target == 1, "a", "b"),
            "cat_b": np.where(target == 1, "x", "y"),
            "first": np.arange(n_rows, dtype=float),
            "second": np.arange(n_rows, dtype=float) * 2,
            "third": np.arange(n_rows, dtype=float) * 3,
            "fourth": np.arange(n_rows, dtype=float) * 4,
            "response": target,
            "event_date": ["2026-01-01"] * n_rows,
            "client_id": np.arange(n_rows),
        },
    )


def test_random_feature_drop_is_deterministic_and_preserves_service_columns() -> None:
    first_datasets, first_schema, first_report = apply_random_feature_drop(
        {"train": _frame()},
        _schema(),
        n_features=3,
        seed=42,
    )
    second_datasets, second_schema, second_report = apply_random_feature_drop(
        {"train": _frame()},
        _schema(),
        n_features=3,
        seed=42,
    )

    assert first_report.dropped == second_report.dropped
    assert len(first_report.dropped) == 3
    assert first_schema == second_schema
    assert all(
        feature not in first_schema.candidate_features()
        for feature in first_report.dropped
    )
    for service in ("response", "event_date", "client_id"):
        assert service in first_datasets["train"].columns


def test_random_feature_drop_rejects_removing_every_candidate() -> None:
    with pytest.raises(ConfigError, match="leave at least one"):
        apply_random_feature_drop(
            {"train": _frame()},
            _schema(),
            n_features=len(_schema().candidate_features()),
            seed=42,
        )


def test_stratified_row_sample_is_bounded_deterministic_and_balanced() -> None:
    datasets = {"train": _frame(100)}

    first, first_report = apply_row_sample(
        datasets,
        _schema(),
        max_rows=20,
        stratified=True,
        seed=42,
    )
    second, second_report = apply_row_sample(
        datasets,
        _schema(),
        max_rows=20,
        stratified=True,
        seed=42,
    )

    assert len(first["train"]) == 20
    assert first["train"]["response"].value_counts().to_dict() == {0: 10, 1: 10}
    assert first["train"].equals(second["train"])
    assert first_report == second_report
    assert first_report.splits[0].original_rows == 100
    assert first_report.splits[0].sampled_rows == 20


def test_uniform_row_sample_is_exact_and_deterministic() -> None:
    first, _ = apply_row_sample(
        {"train": _frame(100)},
        _schema(),
        max_rows=17,
        stratified=False,
        seed=7,
    )
    second, _ = apply_row_sample(
        {"train": _frame(100)},
        _schema(),
        max_rows=17,
        stratified=False,
        seed=7,
    )

    assert len(first["train"]) == 17
    assert first["train"].equals(second["train"])


def test_row_sample_does_not_discard_rows_below_limit() -> None:
    frame = _frame(20)

    sampled, original_rows, sampled_rows = sample_frame_rows(
        frame,
        target_col="response",
        max_rows=100,
        stratified=True,
        seed=42,
        method_name="row_sample",
    )

    assert original_rows == 20
    assert sampled_rows == 20
    assert sampled.equals(frame)


def test_row_sample_caps_every_dataset_split_independently() -> None:
    sampled, report = apply_row_sample(
        {
            "train": _frame(100),
            "valid": _frame(30),
            "test": _frame(10),
        },
        _schema(),
        max_rows=20,
        stratified=False,
        seed=42,
    )

    assert {split: len(frame) for split, frame in sampled.items()} == {
        "train": 20,
        "valid": 20,
        "test": 10,
    }
    assert {
        item.split: (item.original_rows, item.sampled_rows)
        for item in report.splits
    } == {
        "train": (100, 20),
        "valid": (30, 20),
        "test": (10, 10),
    }


def test_regression_requires_non_stratified_row_sampling() -> None:
    schema = FeatureSchema(
        categorical=(),
        continuous=("first",),
        target="response",
        task_type="regression",
    )
    frame = pd.DataFrame(
        {
            "first": np.arange(20, dtype=float),
            "response": np.linspace(0.0, 1.0, 20),
        },
    )

    with pytest.raises(ConfigError, match="regression"):
        apply_row_sample(
            {"train": frame},
            schema,
            max_rows=10,
            stratified=True,
            seed=42,
        )


def test_spark_row_sample_guard_and_uniform_cap(spark: Any) -> None:
    frame = spark.createDataFrame(
        [(float(index), index % 2) for index in range(20)],
        ["first", "response"],
    )

    unchanged, original_rows, unchanged_rows = sample_frame_rows(
        frame,
        target_col="response",
        max_rows=100,
        stratified=True,
        seed=42,
        method_name="row_sample",
    )
    sampled, _, sampled_rows = sample_frame_rows(
        frame,
        target_col="response",
        max_rows=7,
        stratified=False,
        seed=42,
        method_name="row_sample",
    )

    assert original_rows == 20
    assert unchanged_rows == 20
    assert unchanged.count() == 20
    assert sampled_rows == 7
    assert sampled.count() == 7
