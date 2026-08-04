from typing import Self

import torch
from torch.nn.parameter import Parameter

from .base import _check_positive_value


class LinearEmbedding(torch.nn.Module):
    """
    Линейные эмбеддинги для непрерывных признаков.

    Этот класс реализует линейное преобразование, которое проецирует каждый
    из `feature_count` числовых признаков в вектор размерности `embedding_dim`.

    Форматы:

    - Вход: `(*, feature_count)` — тензор произвольной размерности, последняя размерность которого содержит
      `feature_count` числовых признаков.
    - Выход: `(*, feature_count, embedding_dim)` — к размерности входного тензора добавлена размерность `embedding_dim`.

    Пример:

    >>> batch_size = 2
    >>> n_cont_features = 3
    >>> x = torch.randn(batch_size, n_cont_features)
    >>> d_embedding = 4
    >>> m = LinearEmbedding(d_embedding, n_cont_features)
    >>> m(x).shape
    torch.Size([2, 3, 4])
    """

    def __init__(self: Self, embedding_dim: int, feature_count: int) -> None:
        """
        Инициализирует слой линейного эмбеддинга.

        Args:
            embedding_dim (int): Размерность эмбеддингов.
            feature_count (int): Количество числовых признаков.

        Raises:
            ValueError: Если `feature_count` или `embedding_dim` <= 0.
        """
        super().__init__()
        _check_positive_value(feature_count, "feature_count")
        _check_positive_value(embedding_dim, "embedding_dim")

        self.weight = Parameter(torch.empty(feature_count, embedding_dim))
        self.bias = Parameter(torch.empty(feature_count, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Инициализирует веса и смещения равномерным распределением.

        Используется коэффициент масштабирования `1/sqrt(embedding_dim)`.
        """
        d_rqsrt = self.weight.shape[1] ** -0.5
        torch.nn.init.uniform_(self.weight, -d_rqsrt, d_rqsrt)
        torch.nn.init.uniform_(self.bias, -d_rqsrt, d_rqsrt)

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        """
        Выполняет прямой проход через слой.

        Args:
            x (torch.Tensor): Входной тензор размерности `(*, feature_count)`.

        Returns:
            torch.Tensor: Результирующий тензор размерности `(*, feature_count, embedding_dim)`,
                где каждый признак был линейно преобразован в вектор размерности `embedding_dim`.

        Raises:
            ValueError: Если форма входного тензора не соответствует ожидаемому количеству признаков.
        """
        if x.ndim < 1:
            msg: str = f"The input must have at least one dimension, however: {x.ndim=}"
            raise ValueError(
                msg,
            )
        if x.shape[-1] != self.weight.shape[0]:
            msg: str = f"The last dimension of the input was expected to be {self.weight.shape[0]}, however, {x.shape[-1]=}"
            raise ValueError(
                msg,
            )
        return torch.addcmul(self.bias, self.weight, x[..., None])
