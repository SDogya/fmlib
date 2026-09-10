"""Ручное исключение признаков перед этапами отбора."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Union

import pandas as pd

from fmlib.feature_selection.exceptions import BackendError, ConfigError
from fmlib.feature_selection.utils.local_data import is_spark_dataframe
from fmlib.feature_selection.schema import FeatureSchema


@dataclass(frozen=True)
class FeatureDropReport:
    """Сводка исключений из текстового файла или JSON с SelectionResult."""

    path: str
    requested: tuple[str, ...]
    dropped: tuple[str, ...]
    unknown: tuple[str, ...]


def load_feature_names(path: Union[str, Path]) -> list[str]:
    """Загружает уникальные имена исключаемых признаков из текстового файла или JSON с результатом.

    Текстовые файлы читаются построчно: пустые строки и комментарии ``#`` игнорируются.

    JSON-файлы используют тот же артефакт, который записывает ``SelectionResult.save``
    (``final_results.json`` или ``{step}_{stage}_{method}_results.json``). Имена
    берутся из ``dropped_features`` в порядке сохранения.
    """
    file_path = Path(path).expanduser()
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"feature_drop: failed to read {str(file_path)!r}: {exc}."
        raise ConfigError(msg) from exc

    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        names = _names_from_result_json(stripped, file_path)
    else:
        names = _names_from_text_lines(text)

    if not names:
        msg = f"feature_drop: {str(file_path)!r} contains no feature names."
        raise ConfigError(msg)
    return names


def _names_from_text_lines(text: str) -> list[str]:
    """Выполняет парсинг построчного списка исключений."""
    names: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        name = line.strip()
        if not name or name.startswith("#") or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _names_from_result_json(text: str, file_path: Path) -> list[str]:
    """Выполняет парсинг ``dropped_features`` из JSON-артефакта SelectionResult."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = (
            f"feature_drop: {str(file_path)!r} is not valid JSON: {exc}."
        )
        raise ConfigError(msg) from exc

    if not isinstance(payload, Mapping):
        msg = (
            f"feature_drop: {str(file_path)!r} must be a SelectionResult object "
            "with dropped_features, not a JSON array."
        )
        raise ConfigError(msg)

    dropped = payload.get("dropped_features")
    if dropped is None:
        msg = (
            f"feature_drop: {str(file_path)!r} has no dropped_features. "
            "Pass final_results.json or a stage *_results.json artifact."
        )
        raise ConfigError(msg)
    if not isinstance(dropped, list):
        msg = (
            f"feature_drop: {str(file_path)!r} dropped_features must be a list."
        )
        raise ConfigError(msg)

    names: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(dropped):
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, Mapping) and item.get("feature") is not None:
            name = str(item["feature"]).strip()
        else:
            msg = (
                f"feature_drop: {str(file_path)!r} dropped_features[{index}] "
                "must be a feature name or an object with a 'feature' field."
            )
            raise ConfigError(msg)
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names



def apply_feature_drop_file(
    datasets: Mapping[str, Any],
    schema: FeatureSchema,
    path: Union[str, Path],
    *,
    strict: bool = False,
) -> tuple[dict[str, Any], FeatureSchema, FeatureDropReport]:
    """Удаляет перечисленных кандидатов из всех наборов данных и перестраивает схему."""
    requested = load_feature_names(path)
    return apply_feature_drop_names(
        datasets,
        schema,
        requested,
        source=str(Path(path).expanduser()),
        strict=strict,
    )


def apply_feature_drop_names(
    datasets: Mapping[str, Any],
    schema: FeatureSchema,
    requested: Sequence[str],
    *,
    source: str,
    strict: bool = False,
) -> tuple[dict[str, Any], FeatureSchema, FeatureDropReport]:
    """Удаляет кандидатов с переданными именами из всех наборов данных и перестраивает схему."""
    requested = list(dict.fromkeys(str(name) for name in requested))
    candidates = set(schema.candidate_features())
    service = set(schema.service_columns())
    forbidden = [name for name in requested if name in service]
    if forbidden:
        msg = (
            "feature_drop: service columns cannot be excluded: "
            f"{forbidden}."
        )
        raise ConfigError(msg)

    dropped = [name for name in requested if name in candidates]
    unknown = [
        name
        for name in requested
        if name not in candidates and name not in service
    ]
    if strict and unknown:
        msg = (
            "feature_drop: features are not declared as schema candidates: "
            f"{unknown}."
        )
        raise ConfigError(msg)

    updated_datasets = {
        split: _drop_frame_columns(frame, dropped, split)
        for split, frame in datasets.items()
    }
    dropped_set = set(dropped)
    updated_schema = FeatureSchema(
        categorical=tuple(
            name for name in schema.categorical if name not in dropped_set
        ),
        continuous=tuple(
            name for name in schema.continuous if name not in dropped_set
        ),
        target=schema.target,
        task_type=schema.task_type,
        time=schema.time,
        split=schema.split,
        fold=schema.fold,
        id_columns=schema.id_columns,
    )
    report = FeatureDropReport(
        path=source,
        requested=tuple(requested),
        dropped=tuple(dropped),
        unknown=tuple(unknown),
    )
    return updated_datasets, updated_schema, report


def _drop_frame_columns(
    frame: Any,
    columns: Sequence[str],
    split: str,
) -> Any:
    """Возвращает новый DataFrame без указанных столбцов."""
    if not columns:
        return frame
    if isinstance(frame, pd.DataFrame):
        return frame.drop(columns=list(columns), errors="ignore")
    if is_spark_dataframe(frame):
        existing = [column for column in columns if column in frame.columns]
        return frame.drop(*existing) if existing else frame
    msg = (
        f"feature_drop: unsupported {split!r} split type {type(frame)!r}. "
        "Expected a Spark DataFrame or pandas DataFrame."
    )
    raise BackendError(msg)
