from typing import Any, Dict, Self

import torch

from .base import BaseEventEmbedding
from .linear import LinearEmbedding


class EventEmbedding(BaseEventEmbedding):
    """
    Слой эмбеддингов для последовательностей событий, поддерживающий как категориальные,
    так и числовые значения.

    Каждая колонка входных данных получает соответствующий тип эмбеддинга на основе её метаданных.
    Категориальные значения обрабатываются с использованием `torch.nn.Embedding`, а числовые —
    проецируются линейным слоем. Все эмбеддинги объединяются в единый тензор по оси признаков.

    Args:
        embedding_dim (int): Размерность выходного эмбеддинга.
        columns_meta (Dict[str, Dict[str, Any]]): Словарь, сопоставляющий имена колонок
            с их метаданными. Каждая запись должна содержать:
            - 'type': Тип данных ("categorical" или "numeric").
            - 'cardinality': Для категориальных — количество уникальных значений (включая специальные токены).
                             Для числовых можно не указывать.
            - 'clip' (optional): Максимальное значение для категориальных признаков (исключающая верхняя граница).
            - 'event_id' (optional): Идентификатор(ы) типа события.

    Input Shape:
        - events (Dict[str, torch.Tensor]): Словарь тензоров по колонкам.
          Формат каждого тензора: (batch_size, sequence_length).

    Output Shape:
        - torch.Tensor: Результирующий тензор размерности (batch_size, sequence_length, n_columns, embedding_dim),
          где `n_columns` — количество колонок в `columns_meta`.

    Example:
        >>> columns_meta = {
        ...     "txn_mcc": {"type": "categorical", "cardinality": 10, "clip": 8},
        ...     "price": {"type": "numeric", "cardinality": 1}
        ... }
        >>> embedding = EventEmbedding(embedding_dim=32, columns_meta=columns_meta)
        >>> batch = {
        ...     "txn_mcc": torch.randint(0, 8, (32, 10)),
        ...     "price": torch.rand(32, 10)
        ... }
        >>> output = embedding(batch)  # shape: (32, 10, 2, 32)

    Attributes:
        embedding_layer (torch.nn.ModuleDict): Словарь эмбеддинговых слоев по колонкам.
        column_types (tuple): Типы колонок ("categorical"/"numeric").
        column_clips (tuple): Значения ограничения для категориальных колонок.
        column_cardinality (tuple): Количество классов для каждой колонки.
    """

    def __init__(self: Self, embedding_dim: int, columns_meta: Dict[str, Dict[str, Any]]):
        """
        Инициализирует слой эмбеддингов для последовательностей событий.

        Args:
            embedding_dim (int): Размерность эмбеддингов.
            columns_meta (Dict[str, Dict[str, Any]]): Метаданные для колонок.
        """
        super().__init__(embedding_dim=embedding_dim, columns_meta=columns_meta)

        # Pre-process column types and parameters
        self.column_types = []
        self.column_clips = []

        embedding_dict = {}
        for key, item in self._columns_meta.items():
            self.column_types.append(item["type"])
            self.column_clips.append(item.get("clip", None))

            if item["type"] == "categorical":
                embedding_dict[key] = torch.nn.Embedding(
                    num_embeddings=item.get("clip", item["cardinality"]),
                    embedding_dim=self.embedding_dim,
                    padding_idx=0,
                )
            else:
                embedding_dict[key] = LinearEmbedding(
                    embedding_dim=self.embedding_dim,
                    feature_count=1,
                )

        self.embedding_layer = torch.nn.ModuleDict(embedding_dict)
        # Convert to tuples/lists for better torch compilation
        self.column_types = tuple(self.column_types)
        self.column_clips = tuple(self.column_clips)

    def forward(self: Self, events: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Выполняет прямой проход через слой эмбеддингов.

        Args:
            events (Dict[str, torch.Tensor]): Входные данные в виде словаря тензоров по колонкам.

        Returns:
            torch.Tensor: Объединённые эмбеддинги размерности (batch_size, seq_len, n_features, embedding_dim).
        """
        sequence_embedding = []

        for idx, column in enumerate(self._columns_keys):
            col_type = self.column_types[idx]
            col_features = events[column]

            if col_type == "categorical":
                clip_val = self.column_clips[idx]
                if clip_val is not None:
                    col_features = col_features.clamp(max=clip_val - 1)
                embedding = self.embedding_layer[column](col_features).unsqueeze(2)
            else:
                embedding = self.embedding_layer[column](col_features.unsqueeze(-1))

            sequence_embedding.append(embedding)

        return torch.cat(sequence_embedding, dim=2)
