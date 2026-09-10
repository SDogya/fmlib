"""Контракты возможностей бэкендов."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class BackendCapabilities:
    """Декларативное описание возможностей бэкенда для работы с DataFrame.

    Args:
        name: Идентификатор бэкенда (``spark``, ``pandas``, ``polars``).
        supports_distributed: Поддерживает ли бэкенд распределённые вычисления без сбора данных.
        requires_local_materialization: Требуется ли собирать данные локально.
    """

    name: str
    supports_distributed: bool
    requires_local_materialization: bool


class BackendAdapter(Protocol):
    """Минимальный протокол адаптеров DataFrame, используемых на этапах отбора."""

    capabilities: BackendCapabilities

    def get_columns(self: BackendAdapter, data: object) -> list[str]:
        """Возвращает имена столбцов ``data``."""
        ...

    def project_columns(self: BackendAdapter, data: object, columns: list[str]) -> object:
        """Возвращает проекцию ``data`` на столбцы ``columns``."""
        ...
