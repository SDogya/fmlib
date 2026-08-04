from typing import Self

import torch


class BaseHiddenStateAggregator(torch.nn.Module):
    """
    Базовый класс для агрегации скрытых состояний из другой модели с эмбеддингами.

    Этот класс служит родительским для модулей, объединяющих информацию из внешнего
    скрытого состояния (например, от RNN или трансформера) и локальных эмбеддингов признаков.
    Подклассы должны реализовать метод `forward` для определения способа агрегации.

    Args:
        hidden_state_dim (int): Размерность внешнего скрытого состояния.
        embedding_dim (int): Размерность входных эмбеддингов.

    Attributes:
        hidden_state_dim (int): Размерность внешнего скрытого состояния.
        embedding_dim (int): Размерность эмбеддингов признаков.
    """

    def __init__(
        self: Self,
        hidden_state_dim: int,
        embedding_dim: int,
    ):
        """
        Инициализирует базовый агрегатор скрытых состояний.

        Args:
            hidden_state_dim (int): Размерность внешнего скрытого состояния.
            embedding_dim (int): Размерность эмбеддингов признаков.
        """
        super().__init__()
        self.hidden_state_dim = hidden_state_dim
        self.embedding_dim = embedding_dim

    def forward(self: Self, hidden_states: torch.Tensor, embeddings: torch.Tensor):
        """
        Метод прямого прохода. Объединяет внешнее скрытое состояние и эмбеддинги.

        Args:
            hidden_states (torch.Tensor): Тензор внешних скрытых состояний.
                Формат: (batch_size, hidden_state_dim).
            embeddings (torch.Tensor): Тензор эмбеддингов признаков.
                Формат: (batch_size, n_features, embedding_dim).

        Raises:
            NotImplementedError: Метод должен быть переопределен в подклассах.
        """
        msg: str = "This must be implemented in the subclass."
        raise NotImplementedError(msg)


class LayerNormConcatenate(BaseHiddenStateAggregator):
    """
    Агрегатор, применяющий нормализацию слоя и конкатенацию с эмбеддингами.

    Этот класс реализует агрегацию внешнего скрытого состояния с эмбеддингами признаков,
    используя нормализацию слоя (`LayerNorm`) и необязательную проекцию для согласования размерностей.
    Результирующий тензор получается путем конкатенации нормализованного скрытого состояния и эмбеддингов.

    Args:
        hidden_state_dim (int): Размерность внешнего скрытого состояния.
        embedding_dim (int): Размерность эмбеддингов признаков.

    Attributes:
        layer_norm (torch.nn.LayerNorm): Слой нормализации для скрытого состояния.
        proj (Optional[torch.nn.Linear]): Линейный слой для проекции скрытого состояния
            в пространство эмбеддингов, если их размерности различаются.
    """

    def __init__(
        self: Self,
        hidden_state_dim: int,
        embedding_dim: int,
    ):
        """
        Инициализирует агрегатор с нормализацией слоя и конкатенацией.

        Args:
            hidden_state_dim (int): Размерность внешнего скрытого состояния.
            embedding_dim (int): Размерность эмбеддингов признаков.
        """
        super().__init__(
            hidden_state_dim=hidden_state_dim,
            embedding_dim=embedding_dim,
        )
        self.layer_norm = torch.nn.LayerNorm(hidden_state_dim)
        if hidden_state_dim != embedding_dim:
            self.proj = torch.nn.Linear(hidden_state_dim, embedding_dim)
        else:
            self.proj = None

    def forward(
        self: Self,
        hidden_states: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Объединяет внешнее скрытое состояние и эмбеддинги признаков.

        Args:
            hidden_states (torch.Tensor): Тензор внешних скрытых состояний.
                Размерность: (batch_size, hidden_state_dim).
            embeddings (torch.Tensor): Тензор эмбеддингов признаков.
                Размерность: (batch_size, n_features, embedding_dim).

        Returns:
            torch.Tensor: Конкатенированный тензор размерности
                (batch_size, n_features + 1, embedding_dim),
                где первый элемент — это внешнее скрытое состояние.
        """
        hidden_states = self.layer_norm(hidden_states)
        if self.proj is not None:
            hidden_states = self.proj(hidden_states)

        return torch.cat((hidden_states.unsqueeze(1), embeddings), dim=1)


class LayerNormSum(BaseHiddenStateAggregator):
    """
    Агрегатор, применяющий нормализацию слоя и суммирование с эмбеддингами.

    Этот класс реализует агрегацию внешнего скрытого состояния с эмбеддингами признаков,
    используя нормализацию слоя (`LayerNorm`) и необязательную проекцию для согласования размерностей.
    Результирующий тензор получается путем суммирования нормализованного скрытого состояния с эмбеддингами.

    Args:
        hidden_state_dim (int): Размерность внешнего скрытого состояния.
        embedding_dim (int): Размерность эмбеддингов признаков.

    Attributes:
        layer_norm (torch.nn.LayerNorm): Слой нормализации для скрытого состояния.
        proj (Optional[torch.nn.Linear]): Линейный слой для проекции скрытого состояния
            в пространство эмбеддингов, если их размерности различаются.
    """

    def __init__(
        self: Self,
        hidden_state_dim: int,
        embedding_dim: int,
    ):
        """
        Инициализирует агрегатор с нормализацией слоя и суммированием.

        Args:
            hidden_state_dim (int): Размерность внешнего скрытого состояния.
            embedding_dim (int): Размерность эмбеддингов признаков.
        """
        super().__init__(
            hidden_state_dim=hidden_state_dim,
            embedding_dim=embedding_dim,
        )
        self.layer_norm = torch.nn.LayerNorm(hidden_state_dim)
        if hidden_state_dim != embedding_dim:
            self.proj = torch.nn.Linear(hidden_state_dim, embedding_dim)
        else:
            self.proj = None

    def forward(
        self: Self,
        hidden_states: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Объединяет внешнее скрытое состояние и эмбеддинги признаков методом суммирования.

        Args:
            hidden_states (torch.Tensor): Тензор внешних скрытых состояний.
                Формат: (batch_size, hidden_state_dim).
            embeddings (torch.Tensor): Тензор эмбеддингов признаков.
                Формат: (batch_size, n_features, embedding_dim).

        Returns:
            torch.Tensor: Результирующий тензор после суммирования.
                Формат: (batch_size, n_features, embedding_dim),
                где скрытое состояние было добавлено к каждому эмбеддингу.
        """
        hidden_states = self.layer_norm(hidden_states)
        if self.proj is not None:
            hidden_states = self.proj(hidden_states)

        return hidden_states.unsqueeze(1) + embeddings
