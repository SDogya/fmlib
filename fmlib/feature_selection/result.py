"""Артефакт результата отбора и его сериализация."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import pandas as pd

from fmlib.feature_selection.base import FeatureDecision
from fmlib.feature_selection.exceptions import SchemaError
from fmlib.feature_selection.schema import FeatureSchema
from fmlib.feature_selection.utils.local_data import is_spark_dataframe

FORMAT_VERSION = 1


@dataclass(frozen=True)
class DroppedFeature:
    """Запись о признаке, исключённом пайплайном.

    Args:
        feature: Имя признака.
        stage: Этап, исключивший признак, включая предобработку.
        method: Метод внутри этапа.
        reason: Машиночитаемая причина.
        value: Измеренная метрика, если есть.
        threshold: Порог, использованный для принятия решения, если есть.
    """

    feature: str
    stage: str
    method: str
    reason: str
    value: Optional[float] = None
    threshold: Optional[float] = None

    def to_dict(self: DroppedFeature) -> dict[str, Any]:
        """Сериализует в обычный словарь."""
        return asdict(self)

    @classmethod
    def from_dict(cls: type[DroppedFeature], payload: dict[str, Any]) -> DroppedFeature:
        """Создаёт объект из словаря."""
        return cls(
            feature=payload["feature"],
            stage=payload["stage"],
            method=payload["method"],
            reason=payload["reason"],
            value=payload.get("value"),
            threshold=payload.get("threshold"),
        )


@dataclass
class SelectionResult:
    """Артефакт, созданный ``FeatureSelectionPipeline.fit_select``.

    Args:
        selected_features: Признаки, сохранённые после всех этапов, в стабильном порядке.
        dropped_features: Исключённые признаки с метаданными об этапе, методе и причине.
        schema: Входная схема признаков.
        config: Снимок нормализованной конфигурации в виде словаря.
        seed: Базовый seed для воспроизводимости.
        stage_backends: Имя бэкенда, использованного на каждом этапе.
        warnings: Диагностика некритичных проблем.
        format_version: Версия формата артефакта.
        datasets_mode: Режим входных данных ``single`` или ``mapping``.
        scores: Необязательные оценки признаков или их важности по методам.
        verbose_log: События подробного журнала в памяти при включённом ``execution.verbose``.
            Не записываются в ``final_results.json``; сохраняются в ``verbose_log.json``,
            если задан ``output_dir``.
    """

    selected_features: list[str]
    dropped_features: list[DroppedFeature]
    schema: FeatureSchema
    config: dict[str, Any]
    seed: int
    stage_backends: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    format_version: int = FORMAT_VERSION
    datasets_mode: str = "single"
    scores: dict[str, Any] = field(default_factory=dict)
    verbose_log: Optional[dict[str, Any]] = None

    def to_dict(self: SelectionResult) -> dict[str, Any]:
        """Сериализует артефакт в словарь, совместимый с JSON.

        Returns:
            Словарь, включающий ``format_version``.
        """
        return {
            "format_version": self.format_version,
            "selected_features": list(self.selected_features),
            "dropped_features": [item.to_dict() for item in self.dropped_features],
            "schema": self.schema.to_dict(),
            "config": self.config,
            "seed": self.seed,
            "stage_backends": dict(self.stage_backends),
            "warnings": list(self.warnings),
            "datasets_mode": self.datasets_mode,
            "scores": dict(self.scores),
        }

    def save(self: SelectionResult, path: Union[str, Path]) -> Path:
        """Сохраняет артефакт в формате JSON.

        Args:
            path: Путь к выходному файлу.

        Returns:
            Фактически использованный путь. Существующие файлы сохраняются за счёт добавления
            ``_1``, ``_2`` и так далее.
        """
        file_path = _next_available_path(Path(path))
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        return file_path

    @classmethod
    def load(cls: type[SelectionResult], path: Union[str, Path]) -> SelectionResult:
        """Загружает артефакт из JSON.

        Args:
            path: Путь к исходному файлу.

        Returns:
            Восстановленный ``SelectionResult``.

        Raises:
            SchemaError: Если ``format_version`` не поддерживается.
        """
        file_path = Path(path)
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls: type[SelectionResult], payload: dict[str, Any]) -> SelectionResult:
        """Создаёт результат из словаря.

        Args:
            payload: Сериализованный артефакт.

        Returns:
            Экземпляр ``SelectionResult``.
        """
        version = payload.get("format_version", FORMAT_VERSION)
        if version != FORMAT_VERSION:
            message = f"Unsupported SelectionResult format_version={version}. Supported version: {FORMAT_VERSION}."
            raise SchemaError(message)
        return cls(
            selected_features=list(payload.get("selected_features", [])),
            dropped_features=[DroppedFeature.from_dict(item) for item in payload.get("dropped_features", [])],
            schema=FeatureSchema.from_dict(payload["schema"]),
            config=dict(payload.get("config", {})),
            seed=int(payload["seed"]),
            stage_backends=dict(payload.get("stage_backends", {})),
            warnings=list(payload.get("warnings", [])),
            format_version=version,
            datasets_mode=str(payload.get("datasets_mode", "single")),
            scores=dict(payload.get("scores", {})),
        )

    def columns_to_keep(self: SelectionResult) -> list[str]:
        """Возвращает отобранные признаки и служебные столбцы схемы.

        Returns:
            Упорядоченные уникальные имена столбцов для проекции при применении.
        """
        seen: set[str] = set()
        ordered: list[str] = []
        for name in list(self.selected_features) + self.schema.service_columns():
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered

    def apply(self: SelectionResult, data: Any) -> Any:
        """Проецирует ``data`` на отобранные и служебные столбцы.

        Поддерживает объекты с интерфейсом Spark и атрибутами ``select`` / ``columns``, а также простые
        тестовые аналоги, предоставляющие ``columns`` как изменяемую последовательность.

        Args:
            data: Входной объект с интерфейсом DataFrame.

        Returns:
            Объект с интерфейсом DataFrame и выбранными столбцами.

        Raises:
            SchemaError: Если отобранный признак отсутствует в ``data``.
        """
        keep = self.columns_to_keep()
        available = _get_columns(data)
        missing = [name for name in self.selected_features if name not in available]
        if missing:
            message = f"Cannot apply SelectionResult: selected features missing from data: {missing}."
            raise SchemaError(message)

        if isinstance(data, pd.DataFrame):
            projected = [name for name in keep if name in available]
            return data.loc[:, projected].copy()
        if is_spark_dataframe(data):
            from pyspark.sql import functions

            expressions = []
            for name in keep:
                if name not in available:
                    continue
                escaped = name.replace("`", "")
                expressions.append(
                    functions.col(f"`{escaped}`").alias(name),
                )
            return data.select(*expressions)
        if hasattr(data, "select") and callable(data.select):
            return data.select(*[name for name in keep if name in available])

        # Testing double / list-columns fallback
        projected = [name for name in keep if name in available]
        if hasattr(data, "columns"):
            clone = type(data)(columns=projected) if _accepts_columns_kwarg(data) else data
            if clone is data and isinstance(getattr(data, "columns", None), list):
                data.columns = projected
                return data
            return clone
        return projected


def stage_backends_for_config(config: Any) -> dict[str, str]:
    """Метка бэкенда Spark для каждого выполняемого этапа."""
    from fmlib.feature_selection.backends.spark import SPARK_CAPABILITIES
    from fmlib.feature_selection.config import METHOD_STAGE

    backends: dict[str, str] = {}
    for step in getattr(config, "order", ()):
        backends[METHOD_STAGE[step.method]] = SPARK_CAPABILITIES.name
    return backends


def _save_intermediate_result(
    remaining: list[str],
    decisions: list[FeatureDecision],
    stage_name: str,
    method_name: str,
    output_dir: Path,
    context: Any,
    step_index: int = 0,
) -> None:
    """Сохраняет частичный результат после метода отбора.

    Args:
        remaining: Текущие признаки-кандидаты после отбора.
        decisions: Все решения, накопленные к этому моменту.
        stage_name: Метка этапа (preprocessing/statistics/model).
        method_name: Имя метода отбора (null_rate/constants/и т. д.).
        output_dir: Каталог для сохранения результатов.
        context: Контекст этапа со схемой и конфигурацией.
        step_index: Позиция этого шага в порядке пайплайна, начиная с 0.
    """
    stage_backends = stage_backends_for_config(context.config)
    result = SelectionResult(
        selected_features=list(remaining),
        dropped_features=[
            DroppedFeature(
                feature=decision.feature,
                stage=decision.stage,
                method=decision.method,
                reason=decision.reason,
                value=decision.value,
                threshold=decision.threshold,
            )
            for decision in decisions
            if not decision.keep
        ],
        schema=context.schema,
        config=context.config.to_dict(),
        seed=context.seed,
        stage_backends=stage_backends,
        warnings=[],
        datasets_mode=context.datasets_mode,
        scores=dict(context.scores),
    )
    path = output_dir / f"{step_index:02d}_{stage_name}_{method_name}_results.json"
    result.save(path)


def _get_columns(data: Any) -> Sequence[str]:
    columns = getattr(data, "columns", None)
    if columns is None:
        msg = "Data object has no columns attribute; cannot apply SelectionResult."
        raise SchemaError(msg)
    return list(columns)


def _accepts_columns_kwarg(data: Any) -> bool:
    try:
        type(data)(columns=["__probe__"])
    except TypeError:
        return False
    except Exception:  # noqa: BLE001 - probe only
        return False
    return True


def _next_available_path(path: Path) -> Path:
    """Возвращает ``path`` или первый свободный путь ``stem_i`` в том же каталоге."""
    if not path.exists():
        return path
    index = 1
    while True:
        candidate = path.with_name(
            f"{path.stem}_{index}{path.suffix}",
        )
        if not candidate.exists():
            return candidate
        index += 1
