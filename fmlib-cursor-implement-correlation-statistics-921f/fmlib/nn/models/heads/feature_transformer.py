from typing import Self

import torch

from .base import BaseHead


class FeatureTransformerClassificationHead(BaseHead):
    """
    Голова модели для задач классификации, использующая трансформерную архитектуру.

    Эта голова применяется к выходным скрытым состояниям трансформерной модели и состоит из
    последовательности слоёв: Dropout, Linear, SELU, BatchNorm1d и второй Linear,
    который сужает размерность до количества классов.

    Args:
        hidden_dim (int): Размерность входного тензора (размерность скрытого пространства модели).
        num_classes (int): Количество классов для классификации.
        dropout (float): Вероятность dropout, применяемая к входным данным перед линейными преобразованиями.

    Attributes:
        out_head (torch.nn.Sequential): Последовательность слоёв, выполняющих классификацию.
    """

    def __init__(
        self: Self,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ):
        """
        Инициализирует голову классификации.

        Args:
            hidden_dim (int): Размерность входного тензора.
            num_classes (int): Количество выходных классов.
            dropout (float): Вероятность dropout.
        """
        super().__init__(hidden_dim=hidden_dim, num_classes=num_classes)
        self.out_head = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SELU(),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через голову классификации.

        Args:
            x (torch.Tensor): Входной тензор размерности (*, hidden_dim).

        Returns:
            torch.Tensor: Тензор предсказаний размерности (*, num_classes).
        """
        return self.out_head(x)


class FFNHead(BaseHead):
    """
    Голова модели для задач классификации, использующая полносвязный головной модуль (Feed-Forward Network Head).
    Модуль состоит из последовательности слоёв: Dropout, Linear, SELU, BatchNorm1d и Linear.

    Attributes:
        out_head (torch.nn.Sequential): Последовательность слоёв, реализующая преобразование входных данных.
    """

    def __init__(
        self,
        hidden_dim: int,
        linear_dim: int,
        num_classes: int,
        dropout: float,
    ):
        """
        Инициализация FFNHead.

        Args:
            hidden_dim (int): Размерность скрытого слоя на входе.
            linear_dim (int): Размерность промежуточного линейного слоя.
            num_classes (int): Количество выходных классов.
            dropout (float): Вероятность dropout для регуляризации.
        """
        super().__init__(hidden_dim=hidden_dim, num_classes=num_classes)
        self.out_head = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, linear_dim),
            torch.nn.SELU(),
            torch.nn.BatchNorm1d(linear_dim),
            torch.nn.Linear(linear_dim, num_classes),
        )

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через FFNHead.

        Args:
            x (torch.Tensor): Входной тензор размерности (*, hidden_dim).

        Returns:
            torch.Tensor: Выходной тензор размерности (*, num_classes).
        """
        return self.out_head(x)
