from typing import Self

import torch


class BaseHead(torch.nn.Module):
    """
    Базовый класс для головы модели.

    Этот класс служит родительским классом для всех типов голов моделей, предназначенных
    для выполнения конкретных задач (например, классификации). Реализует интерфейс и хранит
    основные параметры, такие как размер скрытого слоя и количество классов.

    Args:
        hidden_dim (int): Размерность входного тензора, подаваемого на голову.
        num_classes (int): Количество выходных классов.
    """

    def __init__(
        self: Self,
        hidden_dim: int,
        num_classes: int,
    ) -> None:
        """
        Инициализирует базовую голову модели.

        Args:
            hidden_dim (int): Размерность скрытого слоя.
            num_classes (int): Количество классов.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        """
        Выполняет прямой проход через голову модели.

        Эта метод должен быть переопределён в подклассах для реализации
        конкретной логики головы модели.

        Args:
            x (torch.Tensor): Входной тензор размерности (*, hidden_dim).

        Raises:
            NotImplementedError: Метод не реализован в базовом классе.

        Returns:
            torch.Tensor: Результирующий тензор размерности (*, num_classes).
        """
        raise NotImplementedError()
