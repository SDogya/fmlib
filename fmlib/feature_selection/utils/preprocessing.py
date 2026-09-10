"""Функции предобработки строк и случайных столбцов-признаков для тестового запуска."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from fmlib.feature_selection.base import derive_seed
from fmlib.feature_selection.exceptions import ConfigError
from fmlib.feature_selection.utils.feature_drop import (
    FeatureDropReport,
    apply_feature_drop_names,
)
from fmlib.feature_selection.utils.local_data import sample_frame_rows
from fmlib.feature_selection.schema import FeatureSchema


@dataclass(frozen=True)
class SplitSampleReport:
    """Число строк до и после обработки одной выборки."""

    split: str
    seed: int
    original_rows: int
    sampled_rows: int


@dataclass(frozen=True)
class RowSampleReport:
    """Сводка формирования выборок строк на этапе предобработки по всем частям данных."""

    max_rows: int
    stratified: bool
    splits: tuple[SplitSampleReport, ...]


def apply_random_feature_drop(
    datasets: Mapping[str, Any],
    schema: FeatureSchema,
    *,
    n_features: int,
    seed: int,
) -> tuple[dict[str, Any], FeatureSchema, FeatureDropReport]:
    """Исключает воспроизводимое случайное подмножество кандидатов из схемы."""
    candidates = schema.candidate_features()
    if n_features >= len(candidates):
        msg = (
            "random_feature_drop: n_features must leave at least one schema "
            f"candidate; got {n_features} for {len(candidates)} candidates."
        )
        raise ConfigError(msg)
    rng = np.random.default_rng(
        derive_seed(seed, "preprocessing", "random_feature_drop"),
    )
    selected_indices = {
        int(index)
        for index in rng.choice(
            len(candidates),
            size=n_features,
            replace=False,
        )
    }
    selected = [
        feature
        for index, feature in enumerate(candidates)
        if index in selected_indices
    ]
    return apply_feature_drop_names(
        datasets,
        schema,
        selected,
        source="<random_feature_drop>",
        strict=True,
    )


def apply_row_sample(
    datasets: Mapping[str, Any],
    schema: FeatureSchema,
    *,
    max_rows: int,
    stratified: bool,
    seed: int,
) -> tuple[dict[str, Any], RowSampleReport]:
    """Ограничивает размер каждой выборки, сохраняя меньшие выборки без изменений."""
    if stratified and schema.task_type == "regression":
        msg = (
            "row_sample: stratified sampling is not supported for regression; "
            "set preprocessing.row_sample.stratified=false."
        )
        raise ConfigError(msg)
    sampled_datasets: dict[str, Any] = {}
    reports: list[SplitSampleReport] = []
    for split, frame in datasets.items():
        split_seed = derive_seed(
            seed,
            "preprocessing",
            "row_sample",
            split,
        )
        sampled, original_rows, sampled_rows = sample_frame_rows(
            frame,
            target_col=schema.target,
            max_rows=max_rows,
            stratified=stratified,
            seed=split_seed,
            method_name="row_sample",
        )
        sampled_datasets[split] = sampled
        reports.append(
            SplitSampleReport(
                split=split,
                seed=split_seed,
                original_rows=original_rows,
                sampled_rows=sampled_rows,
            ),
        )
    return sampled_datasets, RowSampleReport(
        max_rows=max_rows,
        stratified=stratified,
        splits=tuple(reports),
    )
