from typing import Self

import torch


class BaseFFN(torch.nn.Module):
    """
    Базовый класс для реализации Feed-Forward Network.

    Этот класс служит абстрактным родительским классом для всех фид-форвардных сетей.
    Он определяет общие параметры: размерность модели, размерность линейного слоя и вероятность dropout.

    Args:
        model_dim (int): Размерность модели.
        linear_dim (int): Размерность скрытого слоя.
        dropout (float): Вероятность dropout.

    Note:
        Метод forward не реализован, так как он должен быть переопределен в наследуемых классах.
    """

    def __init__(self: Self, model_dim: int, linear_dim: int, dropout: float) -> None:
        super().__init__()
        self.base_model_dim = model_dim
        self.base_linear_dim = linear_dim
        self.base_dropout_p = dropout

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        """Функция прямого прохода. Должна быть реализована в наследуемом классе."""
        raise NotImplementedError()


class PositionWiseFFN(BaseFFN):
    """
    Позиционно-независимая фид-форвардная сеть (Position-wise Feed-Forward Network).

    Реализует типичную трансформерную архитектуру с двумя линейными слоями, разделенными
    GELU-активацией и dropout. Сеть применяется к каждому элементу последовательности
    независимо от его позиции.

    Args:
        model_dim (int): Размерность входных и выходных данных.
        linear_dim (int): Размерность скрытого линейного слоя.
        dropout (float): Вероятность dropout для регуляризации.

    Attributes:
        w_1 (torch.nn.Linear): Первый линейный слой, увеличивающий размерность.
        w_2 (torch.nn.Linear): Второй линейный слой, возвращающий исходную размерность.
        activation (torch.nn.GELU): Функция активации GELU.
        dropout (torch.nn.Dropout): Слой dropout для предотвращения переобучения.

    Размерности:
        - Input: (..., model_dim)
        - Output: (..., model_dim)

    Пример:
        >>> ff = PositionWiseFFN(model_dim=512, linear_dim=2048, dropout=0.1)
        >>> x = torch.randn(32, 10, 512)  # batch_size=32, seq_len=10
        >>> out = ff(x)  # shape (32, 10, 512)
    """

    def __init__(self: Self, model_dim: int, linear_dim: int, dropout: float) -> None:
        super().__init__(
            model_dim=model_dim,
            linear_dim=linear_dim,
            dropout=dropout,
        )
        self.w_1 = torch.nn.Linear(self.base_model_dim, self.base_linear_dim)
        self.w_2 = torch.nn.Linear(self.base_linear_dim, self.base_model_dim)
        self.dropout = torch.nn.Dropout(self.base_dropout_p)
        self.activation = torch.nn.GELU()

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        """
        Выполняет прямой проход через сеть.

        Args:
            x (torch.Tensor): Входной тензор размерности (..., model_dim).

        Returns:
            torch.Tensor: Выходной тензор той же формы, что и вход.
        """
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class DropoutLastPositionWiseFFN(BaseFFN):
    """
    Позиционно-независимая фид-форвардная сеть (Position-wise Feed-Forward Network).

    Реализует типичную трансформерную архитектуру с двумя линейными слоями, разделенными
    GELU-активацией и dropout. Сеть применяется к каждому элементу последовательности
    независимо от его позиции.

    Args:
        model_dim (int): Размерность входных и выходных данных.
        linear_dim (int): Размерность скрытого линейного слоя.
        dropout (float): Вероятность dropout для регуляризации.

    Attributes:
        up_proj (torch.nn.Linear): Первый линейный слой, увеличивающий размерность.
        down_proj (torch.nn.Linear): Второй линейный слой, возвращающий исходную размерность.
        activation (torch.nn.GELU): Функция активации GELU.
        dropout (torch.nn.Dropout): Слой dropout для предотвращения переобучения.

    Размерности:
        - Input: (..., model_dim)
        - Output: (..., model_dim)

    Пример:
        >>> ff = DropoutLastPositionWiseFFN(model_dim=512, linear_dim=2048, dropout=0.1)
        >>> x = torch.randn(32, 10, 512)  # batch_size=32, seq_len=10
        >>> out = ff(x)  # shape (32, 10, 512)
    """

    def __init__(self: Self, model_dim: int, linear_dim: int, dropout: float) -> None:
        super().__init__(
            model_dim=model_dim,
            linear_dim=linear_dim,
            dropout=dropout,
        )
        self.up_proj = torch.nn.Linear(self.base_model_dim, self.base_linear_dim)
        self.down_proj = torch.nn.Linear(self.base_linear_dim, self.base_model_dim)
        self.dropout = torch.nn.Dropout(self.base_dropout_p)
        self.activation = torch.nn.GELU()

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        """
        Выполняет прямой проход через сеть.

        Args:
            x (torch.Tensor): Входной тензор размерности (..., model_dim).

        Returns:
            torch.Tensor: Выходной тензор той же формы, что и вход.
        """
        activated: torch.Tensor = self.activation(self.up_proj(x))
        return self.dropout(self.down_proj(activated))


class STEv2FFN(BaseFFN):
    """
    Фид-форвардная сеть на основе SELU и нормализации (STEv2FFN).

    Реализует типичную архитектуру трансформеров, включающую:
    1. Линейное расширение размерности входных данных.
    2. Нормализацию слоя (LayerNorm).
    3. Активацию SELU.
    4. Dropout для регуляризации.
    5. Линейное проецирование обратно к исходной размерности.

    Args:
        model_dim (int): Размерность входных и выходных данных.
        linear_dim (int): Размерность скрытого линейного слоя.
        dropout (float): Вероятность dropout.

    Attributes:
        w1 (torch.nn.Linear): Первый линейный слой, увеличивающий размерность.
        act (torch.nn.SELU): Функция активации SELU.
        norm (torch.nn.LayerNorm): Слой нормализации.
        dropout (torch.nn.Dropout): Слой dropout.
        w2 (torch.nn.Linear): Второй линейный слой, возвращающий исходную размерность.

    Размерности:
        - Input: (..., model_dim)
        - Output: (..., model_dim)

    Пример:
        >>> ff = STEv2FFN(model_dim=512, linear_dim=2048, dropout=0.1)
        >>> x = torch.randn(32, 10, 512)  # batch_size=32, seq_len=10
        >>> out = ff(x)  # shape (32, 10, 512)
    """

    def __init__(self: Self, model_dim: int, linear_dim: int, dropout: float) -> None:
        super().__init__(
            model_dim=model_dim,
            linear_dim=linear_dim,
            dropout=dropout,
        )
        self.w1 = torch.nn.Linear(self.base_model_dim, self.base_linear_dim)
        self.act = torch.nn.SELU()
        self.norm = torch.nn.LayerNorm(self.base_linear_dim)
        self.dropout = torch.nn.Dropout(p=self.base_dropout_p)
        self.w2 = torch.nn.Linear(self.base_linear_dim, self.base_model_dim)

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        """
        Функция прямого прохода через сеть.

        Args:
            x (torch.Tensor): Входной тензор размерности (..., d_model).

        Returns:
            torch.Tensor: Выходной тензор той же размерности, что и вход.
        """
        return self.w2(self.dropout(self.act(self.norm(self.w1(x)))))


class FeatureFFN(BaseFFN):
    """
    Фид-форвардная сеть (FFN) с поддержкой многомерных признаков.

    Эта реализация FFN расширяет базовый класс `BaseFFN`, добавляя возможность работы
    с тензорами, содержащими информацию по нескольким признакам. Сеть включает:
    - Линейный слой `w1` для проецирования входа.
    - Активацию SELU.
    - Нормализацию слоя по двум измерениям: по признакам и скрытому размеру.
    - Dropout для регуляризации.
    - Линейный слой `w2` для возврата к исходной размерности.
    - Дополнительную нормализацию на выходе.

    Args:
        model_dim (int): Размерность скрытого состояния (размерность эмбеддинга).
        linear_dim (int): Промежуточная размерность после первого линейного слоя.
        feature_dim (int): Количество признаков на входе.
        dropout (float): Вероятность dropout для регуляризации.

    Attributes:
        w1 (torch.nn.Linear): Первый линейный слой.
        act (torch.nn.SELU): Активационная функция SELU.
        norm (torch.nn.LayerNorm): Нормализация слоя по [feature_dim, linear_dim].
        dropout (torch.nn.Dropout): Dropout для регуляризации.
        w2 (torch.nn.Linear): Второй линейный слой, возвращающий к исходной размерности.
        last_norm (torch.nn.LayerNorm): Нормализация на выходе сети.

    Shape:
        - Input: (batch_size, feature_dim, model_dim)
        - Output: (batch_size, feature_dim, model_dim)

    Example:
        >>> ffn = FeatureFFN(model_dim=64, linear_dim=128, feature_dim=5, dropout=0.1)
        >>> x = torch.randn(32, 5, 64)  # batch_size=32, feature_dim=5, model_dim=64
        >>> output = ffn(x)  # shape (32, 5, 64)
    """

    def __init__(
        self: Self,
        model_dim: int,
        linear_dim: int,
        feature_dim: int,
        dropout: float,
    ) -> None:
        super().__init__(
            model_dim=model_dim,
            linear_dim=linear_dim,
            dropout=dropout,
        )
        self._feature_dim = feature_dim
        self.w1 = torch.nn.Linear(self.base_model_dim, self.base_linear_dim)
        self.act = torch.nn.SELU()
        self.norm = torch.nn.LayerNorm([feature_dim, self.base_linear_dim])
        self.dropout = torch.nn.Dropout(p=self.base_dropout_p)
        self.w2 = torch.nn.Linear(self.base_linear_dim, self.base_model_dim)
        self.last_norm = torch.nn.LayerNorm([feature_dim, self.base_model_dim])

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        """
        Выполняет прямой проход через фид-форвардную сеть.

        Args:
            x (torch.Tensor): Входной тензор размерности (batch_size, feature_dim, model_dim).

        Returns:
            torch.Tensor: Выходной тензор той же размерности, что и вход — (batch_size, feature_dim, model_dim).

        Raises:
            ValueError: Если форма входного тензора не соответствует ожидаемой.
        """
        if x.ndim != 3:
            msg: str = f"Expected input tensor to have 3 dimensions, got {x.ndim}."
            raise ValueError(
                msg,
            )
        if x.shape[-1] != self.base_model_dim:
            msg: str = f"Expected last dimension of input tensor to be {self.base_model_dim}, got {x.shape[-1]}."
            raise ValueError(
                msg,
            )
        if x.shape[-2] != self._feature_dim:
            msg: str = f"Expected second dimension of input tensor to be {self._feature_dim}, got {x.shape[-2]}."
            raise ValueError(
                msg,
            )
        return self.last_norm(self.w2(self.dropout(self.norm(self.act(self.w1(x))))))
