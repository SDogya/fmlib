from typing import Self

import torch

from .base import BaseHead


class TrivanClassificationHead(BaseHead):
    """
    Модуль головы классификации, используемый для получения итогового распределения классов.

    Этот модуль принимает выходные данные из последнего слоя трансформерной модели и
    преобразует их в вектор классов с использованием сигмоидной активации и линейного слоя.

    Args:
        hidden_dim (int): Размерность скрытого состояния входных данных.
        num_classes (int): Количество классов для классификации.

    Attributes:
        input_gate (torch.nn.Parameter): Тренируемый вектор-гейт размером [1, 3], инициализированный единицами.
        pad_tensor (torch.nn.Parameter): Тренируемый тензор заполнителя размером [hidden_dim], инициализированный нулями.
        out_layer (torch.nn.Linear): Линейный слой, сопоставляющий скрытое состояние с числом классов.

    Shape:
        - Input: (batch_size, seq_len, hidden_dim)
        - Output: (batch_size, num_classes)

    Пример:
        >>> head = ClassificationHead(hidden_dim=512, num_classes=10)
        >>> x = torch.randn(32, 10, 512)  # batch_size=32, seq_len=10
        >>> output = head(x)  # shape (32, 10)
    """

    def __init__(
        self: Self,
        hidden_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__(hidden_dim=hidden_dim, num_classes=num_classes)
        self.input_gate = torch.nn.Parameter(torch.ones(1, 3))
        self.pad_tensor = torch.nn.Parameter(torch.zeros(hidden_dim))
        self.out_layer = torch.nn.Linear(hidden_dim, num_classes)

        torch.nn.init.normal_(self.out_layer.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.out_layer.bias)

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        """
        Функция прямого прохода через голову классификации.

        Args:
            x (torch.Tensor): Входной тензор формы (batch_size, seq_len, hidden_dim).

        Returns:
            torch.Tensor: Выходной тензор формы (batch_size, num_classes).
        """
        x = (torch.nn.functional.sigmoid(self.input_gate) @ x)[:, 0, :]
        x = x + self.pad_tensor
        x = self.out_layer(x)
        return x
