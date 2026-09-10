"""Схема признаков, описывающая кандидатов и служебные столбцы."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from fmlib.feature_selection.exceptions import SchemaError

TASK_TYPES = frozenset({"binary_classification", "classification", "regression"})
SPLIT_VALUES = frozenset({"train", "valid", "test"})


@dataclass(frozen=True)
class FeatureSchema:
    """Описание ролей признаков для пайплайна отбора.

    Роли признаков отделены от физической схемы DataFrame. Кандидатами
    могут быть только ``categorical`` и ``continuous``; целевой и служебные столбцы
    никогда не участвуют в отборе как кандидаты.

    Args:
        categorical: Имена столбцов категориальных признаков.
        continuous: Имена столбцов непрерывных признаков.
        target: Имя целевого столбца (обязательно для этапов модели).
        task_type: ``binary_classification``, ``classification`` или ``regression``.
        time: Необязательный временной столбец для помесячного PSI или кросс-валидации по времени.
        split: Необязательный столбец разбиения со значениями ``train`` / ``valid`` / ``test``.
        fold: Необязательный столбец фолда кросс-валидации внутри train.
        id_columns: Столбцы-идентификаторы, не участвующие в отборе.
    """

    categorical: tuple[str, ...]
    continuous: tuple[str, ...]
    target: str
    task_type: str
    time: Optional[str] = None
    split: Optional[str] = None
    fold: Optional[str] = None
    id_columns: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self: FeatureSchema) -> None:
        object.__setattr__(self, "categorical", tuple(self.categorical))
        object.__setattr__(self, "continuous", tuple(self.continuous))
        object.__setattr__(self, "id_columns", tuple(self.id_columns))
        self._validate_internal()

    def _validate_internal(self: FeatureSchema) -> None:
        if self.task_type not in TASK_TYPES:
            message = f"Unsupported task_type={self.task_type!r}. Expected one of: {sorted(TASK_TYPES)}."
            raise SchemaError(message)

        categorical_set = set(self.categorical)
        continuous_set = set(self.continuous)
        overlap = categorical_set & continuous_set
        if overlap:
            message = f"categorical and continuous must be disjoint; overlap={sorted(overlap)}."
            raise SchemaError(message)

        if len(self.categorical) != len(categorical_set):
            msg = "categorical contains duplicate column names."
            raise SchemaError(msg)
        if len(self.continuous) != len(continuous_set):
            msg = "continuous contains duplicate column names."
            raise SchemaError(msg)
        if len(self.id_columns) != len(set(self.id_columns)):
            msg = "id_columns contains duplicate column names."
            raise SchemaError(msg)

        candidates = set(self.candidate_features())
        if self.target in candidates:
            msg = "target must not appear in categorical or continuous."
            raise SchemaError(msg)

        service = set(self.service_columns()) - {self.target}
        leaked = candidates & service
        if leaked:
            message = (
                f"Service columns must not appear among candidates: {sorted(leaked)}. Remove them from categorical/continuous."
            )
            raise SchemaError(message)

    def candidate_features(self: FeatureSchema) -> list[str]:
        """Возвращает кандидатов для отбора в стабильном порядке: сначала категориальные, затем непрерывные.

        Returns:
            Упорядоченный список имён признаков-кандидатов.
        """
        return list(self.categorical) + list(self.continuous)

    def service_columns(self: FeatureSchema) -> list[str]:
        """Возвращает служебные столбцы вне списка кандидатов, которые нужно сохранить при применении.

        Returns:
            Упорядоченные уникальные имена служебных столбцов (target, time, split, fold, ids).
        """
        columns: list[str] = [self.target]
        for name in (self.time, self.split, self.fold):
            if name is not None:
                columns.append(name)
        columns.extend(self.id_columns)
        seen: set[str] = set()
        ordered: list[str] = []
        for name in columns:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered

    def all_declared_columns(self: FeatureSchema) -> list[str]:
        """Возвращает все столбцы, объявленные в схеме.

        Returns:
            Упорядоченные уникальные имена столбцов из кандидатов и служебных полей.
        """
        seen: set[str] = set()
        ordered: list[str] = []
        for name in self.candidate_features() + self.service_columns():
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered

    def validate_against_columns(
        self: FeatureSchema,
        columns: Sequence[str],
        *,
        require_split: bool = False,
    ) -> None:
        """Проверяет наличие объявленных столбцов в списке столбцов DataFrame.

        Args:
            columns: Физические имена столбцов, доступных во входных данных.
            require_split: При True ``split`` должен быть задан и присутствовать в данных.

        Raises:
            SchemaError: Если обязательные столбцы отсутствуют или требуется split, но он отсутствует.
        """
        available = set(columns)
        missing = [name for name in self.all_declared_columns() if name not in available]

        if missing:
            message = f"Declared columns missing from DataFrame: {missing}. Align FeatureSchema with the input schema."
            raise SchemaError(message)
            
        if require_split:
            if self.split is None:
                msg = (
                    "FeatureSchema.split is required when a single DataFrame is passed. "
                    "Set split or pass datasets={'train': ...} instead."
                )
                raise SchemaError(
                    msg,
                )
            if self.split not in available:
                msg = f"split column {self.split!r} is missing from DataFrame."
                raise SchemaError(msg)

    def to_dict(self: FeatureSchema) -> dict:
        """Сериализует схему в обычный словарь.

        Returns:
            Словарь, совместимый с JSON.
        """
        return {
            "categorical": list(self.categorical),
            "continuous": list(self.continuous),
            "target": self.target,
            "task_type": self.task_type,
            "time": self.time,
            "split": self.split,
            "fold": self.fold,
            "id_columns": list(self.id_columns),
        }

    @classmethod
    def from_dict(cls: type[FeatureSchema], payload: dict) -> FeatureSchema:
        """Создаёт схему из словаря.

        Args:
            payload: Словарь с полями схемы.

        Returns:
            Проверенный экземпляр ``FeatureSchema``.
        """
        return cls(
            categorical=tuple(payload.get("categorical", ())),
            continuous=tuple(payload.get("continuous", ())),
            target=payload["target"],
            task_type=payload["task_type"],
            time=payload.get("time"),
            split=payload.get("split"),
            fold=payload.get("fold"),
            id_columns=tuple(payload.get("id_columns", ())),
        )


def ensure_no_feature_leak(
    candidates: Iterable[str],
    schema: FeatureSchema,
) -> None:
    """Проверяет, что кандидаты не включают целевой и служебные столбцы.

    Args:
        candidates: Имена признаков-кандидатов.
        schema: Схема признаков.

    Raises:
        SchemaError: Если служебный или целевой столбец попал в кандидаты.
    """
    forbidden = set(schema.service_columns())
    leaked = sorted(set(candidates) & forbidden)
    if leaked:
        message = f"Candidates include forbidden service columns: {leaked}."
        raise SchemaError(message)
