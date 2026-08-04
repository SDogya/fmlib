from copy import deepcopy
from typing import Any, Dict, Self

import torch


def _check_positive_value(value: int, name: str):
    if not isinstance(value, int):
        msg: str = f"{name} must be an integer, got {type(value).__name__}"
        raise TypeError(msg)
    if value <= 0:
        msg: str = f"{name} must be a positive integer, got {value}"
        raise ValueError(msg)


class BaseFeatureEmbedding(torch.nn.Module):
    """
    Инициализирует базовый слой эмбеддинга для табличных признаков.

    Args:
        embedding_dim (int): Размерность эмбеддингов.
    """

    def __init__(self: Self, embedding_dim: int):
        super().__init__()
        _check_positive_value(embedding_dim, "embedding_dim")
        self.embedding_dim = embedding_dim

    def forward(
        self: Self,
        cat_features: torch.LongTensor,
        num_features: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        """
        Метод прямого прохода через слой эмбеддингов.

        Должен быть переопределен в подклассах для реализации специфической логики
        обработки категориальных, числовых и скрытых признаков.

        Args:
            cat_features (torch.LongTensor): Тензор категориальных признаков.
            num_features (torch.Tensor): Тензор числовых признаков.
            hidden_states (torch.Tensor): Скрытые состояния модели.

        Raises:
            NotImplementedError: Метод не реализован в базовом классе.
        """
        msg: str = "Forward method must be implemented by child classes"
        raise NotImplementedError(msg)


class BaseEventEmbedding(torch.nn.Module):
    """
    Базовый класс для слоев эмбеддингов последовательностей событий.

    Этот класс представляет собой общий интерфейс для создания эмбеддингов, обрабатывающих
    данные в виде последовательностей событий. Он проверяет корректность метаданных
    столбцов и предоставляет доступ к информации о колонках и их свойствах.

    Args:
        embedding_dim (int): Размерность эмбеддингов.
        columns_meta (Dict[str, Dict[str, Any]]): Словарь, сопоставляющий имена колонок
            с их метаданными. Каждая запись должна содержать:
                - 'type': тип данных ('numeric' или 'categorical').
                - 'cardinality': количество уникальных категорий для категориальных признаков.

    Raises:
        ValueError: Если структура `columns_meta` некорректна или содержит недопустимые значения.
    """

    def __init__(self: Self, embedding_dim: int, columns_meta: Dict[str, Dict[str, Any]]):
        """
        Инициализирует базовый слой эмбеддинга для последовательностей событий.

        Args:
            embedding_dim (int): Размерность эмбеддингов.
            columns_meta (Dict[str, Dict[str, Any]]): Метаданные для каждой колонки.
        """
        super().__init__()
        _check_positive_value(embedding_dim, "embedding_dim")
        self.embedding_dim = embedding_dim

        reason: str | None = BaseEventEmbedding._check_columns_meta(columns_meta)
        if reason is not None:
            msg: str = f"columns_meta must be a dictionary with valid structure, got {columns_meta}. Reason: {reason}."
            raise ValueError(msg)

        self._columns_meta = deepcopy(columns_meta)
        self._columns_keys = tuple((self._columns_meta).keys())

    @staticmethod
    def _check_columns_meta(columns_meta) -> str | None:
        """
        Проверяет корректность структуры и значений метаданных колонок.

        Args:
            columns_meta (Dict[str, Dict[str, Any]]): Метаданные колонок.

        Returns:
            bool: True, если метаданные корректны, иначе False.
        """
        for col, col_info in columns_meta.items():
            # Validate 'type'
            if "type" not in col_info:
                msg: str = f"Type is required for the column {col}."
                return msg
            typ = col_info["type"]
            if not isinstance(typ, str) or typ not in ("numerical", "categorical"):
                msg: str = f"Type for the column {col} must be 'numerical' or 'categorical', got {typ}."
                return msg
            # Нет необходимости проверять cardinality для numerical
            if typ == "numerical":
                continue
            # Validate 'cardinality'
            if "cardinality" not in col_info:
                msg: str = f"Cardinality is required for the column {col}."
                return msg
            cardinality = col_info["cardinality"]
            if not isinstance(cardinality, int) or cardinality < 1:
                msg: str = f"Cardinality for the categorical column {col} must be a positive integer, got {cardinality}."
                return msg
        return None

    @property
    def columns(self) -> list[str]:
        """
        Возвращает список имен колонок, используемых в данном слое эмбеддинга.

        Returns:
            list[str]: Список имен колонок.
        """
        return list(self._columns_meta.keys())

    @property
    def columns_meta(self) -> Dict[str, Any]:
        """
        Возвращает копию метаданных колонок.

        Returns:
            Dict[str, Any]: Словарь, сопоставляющий имена колонок с их метаданными.
        """
        return deepcopy(self._columns_meta)

    def forward(self: Self, events: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Метод прямого прохода через слой эмбеддинга.

        Должен быть переопределен в подклассах для реализации специфической логики
        обработки событийных последовательностей.

        Args:
            events (Dict[str, torch.Tensor]): Словарь тензоров событий по колонкам.

        Raises:
            NotImplementedError: Метод не реализован в базовом классе.
        """
        msg: str = "Forward method must be implemented by child classes"
        raise NotImplementedError(msg)
