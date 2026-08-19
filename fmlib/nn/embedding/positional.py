import math
from typing import Self

import torch


class BasePositionalEmbedding(torch.nn.Module):
    """
    Базовый класс для временных (позиционных) эмбеддингов.

    Этот класс служит родительским для всех слоев, предназначенных для работы с временными метками.
    Он определяет общую структуру метода `forward`, который должен быть реализован в подклассах.

    Args:
        embedding_dim (int): Размерность эмбеддингов.

    Attributes:
        embedding_dim (int): Размерность скрытого пространства эмбеддингов.

    Example:
        >>> class CustomTemporalEmbedding(BasePositionalEmbedding):
        ...     def forward(self, timestamps):
        ...         # Реализация конкретного способа обработки временных меток
        ...         return embedded_timestamps
    """

    def __init__(self: Self, embedding_dim: int):
        """
        Инициализирует базовый слой временных эмбеддингов.

        Args:
            embedding_dim (int): Размерность эмбеддингов.
        """
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self: Self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Метод прямого прохода через слой временных эмбеддингов.

        Должен быть переопределен в подклассах для реализации специфической логики
        обработки временных меток.

        Args:
            timestamps (torch.Tensor): Тензор временных меток.

        Raises:
            NotImplementedError: Метод не реализован в базовом классе.
        """
        msg: str = "Forward method must be implemented by child classes"
        raise NotImplementedError(msg)


class PositionalEmbedding(BasePositionalEmbedding):
    """
    Модуль для создания временных позиционных эмбеддингов.

    Реализует метод, аналогичный синусоидальному кодированию из оригинальной статьи о трансформерах,
    но адаптированный под временные метки. Каждый временной момент преобразуется в вектор размерности
    `embedding_dim`, где чётные и нечётные позиции заполняются значениями синуса и косинуса соответственно.

    Args:
        embedding_dim (int): Размерность выходного эмбеддинга.
        max_timepoint (float): Максимальное значение времени, используемое для инициализации частотного пространства.
        scale_factor (float): Коэффициент масштабирования для результирующих эмбеддингов.

    Attributes:
        sin_div_term (torch.nn.Parameter): Тензор коэффициентов для синуса.
        cos_div_term (torch.nn.Parameter): Тензор коэффициентов для косинуса.
        scale_factor (float): Фактор масштабирования.
    """

    def __init__(
        self: Self,
        embedding_dim: int,
        max_timepoint: float = 10000.0,
        scale_factor: float = 0.05,
    ):
        """
        Инициализирует модуль временных позиционных эмбеддингов.

        Args:
            embedding_dim (int): Размерность эмбеддинга.
            max_timepoint (float): Максимальное значение времени.
            scale_factor (float): Множитель масштабирования.
        """
        super().__init__(embedding_dim=embedding_dim)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(max_timepoint) / embedding_dim),
        )

        if self.embedding_dim % 2 == 0:
            self.sin_div_term = torch.nn.Parameter(div_term, requires_grad=False)
            self.cos_div_term = torch.nn.Parameter(div_term, requires_grad=False)
        else:
            self.sin_div_term = torch.nn.Parameter(div_term, requires_grad=False)
            self.cos_div_term = torch.nn.Parameter(div_term[:-1], requires_grad=False)

        self.scale_factor = scale_factor

    def relative_time(self: Self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет относительное время от первого элемента последовательности.

        Args:
            timestamps (torch.Tensor): Тензор временных меток формы (batch_size, seq_len).

        Returns:
            torch.Tensor: Относительные временные метки.

        Raises:
            ValueError: Если форма тензора отличается от (batch_size, seq_len).
        """
        if timestamps.ndim != 2:
            msg: str = "Timestamp tensor must have shape (batch_size, seq_len)"
            raise ValueError(
                msg,
            )
        return timestamps - timestamps[:, 0].unsqueeze(1)

    def forward(self: Self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через слой позиционного эмбеддинга.

        Args:
            timestamps (torch.Tensor): Входной тензор временных меток формы (batch_size, seq_len).

        Returns:
            torch.Tensor: Результирующий тензор временных эмбеддингов формы (batch_size, seq_len, embedding_dim),
                умноженный на `scale_factor`.
        """
        bsz, seq_len = timestamps.shape
        device = timestamps.device
        t = timestamps.unsqueeze(-1)

        temporal_embeddings = torch.zeros(
            bsz,
            seq_len,
            self.embedding_dim,
            device=device,
        )

        temporal_embeddings[:, :, 0::2] = torch.sin(
            t * self.sin_div_term.unsqueeze(0).unsqueeze(0),
        )
        temporal_embeddings[:, :, 1::2] = torch.cos(
            t * self.cos_div_term.unsqueeze(0).unsqueeze(0),
        )

        return temporal_embeddings * self.scale_factor
