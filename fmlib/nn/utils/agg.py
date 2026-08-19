from typing import Self, Tuple

import torch


class BaseAggregation(torch.nn.Module):
    """
    Базовый класс для слоёв агрегации.

    Этот класс служит родительским для всех слоёв, выполняющих операцию агрегации над скрытыми состояниями.
    Реализует общие методы, такие как применение маски внимания к тензорам состояний.
    """

    def __init__(self):
        super().__init__()

    def apply_expanded_mask(
        self: Self,
        hidden_state: torch.Tensor,
        attention_mask: torch.BoolTensor,
        return_seq_len: bool,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Применяет расширенную маску внимания к скрытым состояниям.

        Args:
            hidden_state (torch.Tensor): Тензор скрытых состояний размерности (batch_size, seq_len, embedding_dim).
            attention_mask (torch.BoolTensor): Маска внимания размерности (batch_size, seq_len), где True — активные позиции.
            return_seq_len (bool): Флаг, определяющий, возвращать ли длины последовательностей.

        Returns:
            Tuple[torch.Tensor, Optional[torch.LongTensor]]:
                - output (torch.Tensor): Скрытые состояния с применённой маской.
                - seq_len (Optional[torch.LongTensor]): Длины последовательностей, если `return_seq_len` установлен в True.
        """
        output = hidden_state * attention_mask.unsqueeze(-1)
        if return_seq_len:
            return output, attention_mask.sum(dim=1)
        return output, None


class SumLayerNorm(BaseAggregation):
    """
    Слой агрегации, который суммирует скрытые состояния и применяет нормализацию слоя (LayerNorm).

    Этот класс реализует операцию агрегации, которая сначала маскирует скрытые состояния по attention_mask,
    затем суммирует их по оси последовательности (seq_len) и применяет нормализацию слоя.

    Args:
        embedding_dim (int): Размерность эмбеддинга (входных и выходных тензоров).
    """

    def __init__(self: Self, embedding_dim: int):
        """
        Инициализирует слой агрегации Sum + LayerNorm.

        Args:
            embedding_dim (int): Размерность эмбеддингов.
        """
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(embedding_dim)

    def forward(self: Self, hidden_state: torch.Tensor, attention_mask: torch.BoolTensor) -> torch.Tensor:
        """
        Выполняет прямой проход через слой агрегации.

        Args:
            hidden_state (torch.Tensor): Тензор скрытых состояний формы (batch_size, seq_len, embedding_dim).
            attention_mask (torch.BoolTensor): Маска внимания формы (batch_size, seq_len), где True — активные позиции.

        Returns:
            torch.Tensor: Результирующий тензор формы (batch_size, embedding_dim),
                представляющий агрегированные и нормализованные скрытые состояния.
        """
        masked_states, _ = self.apply_expanded_mask(hidden_state, attention_mask, return_seq_len=False)
        aggregate_hidden_state = masked_states.sum(dim=1)
        return self.layer_norm(aggregate_hidden_state)


class ConvAggregation(BaseAggregation):
    """
    Слой агрегации, использующий одномерную свёрточную сеть для агрегирования скрытых состояний.

    Реализует операцию агрегации, в которой скрытые состояния маскируются по attention_mask,
    затем преобразуются через свёрточный слой и усредняющий пуллинг. Результат сжимается до формы (batch_size, embedding_dim).

    Args:
        embedding_dim (int): Размерность эмбеддинга.
        kernel_size (int): Размер ядра свёрточного слоя. По умолчанию — 1.
    """

    def __init__(self: Self, embedding_dim: int, kernel_size: int = 1):
        """
        Инициализирует слой агрегации на основе свёртки.

        Args:
            embedding_dim (int): Размерность эмбеддингов.
            kernel_size (int): Размер ядра свёрточного слоя.
        """
        super().__init__()
        self.conv_layer = torch.nn.Conv1d(embedding_dim, embedding_dim, kernel_size)
        self.pool_layer = torch.nn.AdaptiveAvgPool1d(1)

    def forward(self: Self, hidden_state: torch.Tensor, attention_mask: torch.BoolTensor) -> torch.Tensor:
        """
        Выполняет прямой проход через слой агрегации.

        Args:
            hidden_state (torch.Tensor): Тензор скрытых состояний формы (batch_size, seq_len, embedding_dim).
            attention_mask (torch.BoolTensor): Маска внимания формы (batch_size, seq_len), где True — активные позиции.

        Returns:
            torch.Tensor: Результирующий тензор формы (batch_size, embedding_dim),
                представляющий агрегированные скрытые состояния после применения свёртки и пуллинга.
        """
        masked_states, _ = self.apply_expanded_mask(hidden_state, attention_mask, return_seq_len=False)
        aggregate_hidden_state = self.conv_layer(masked_states.permute(0, 2, 1))
        aggregate_hidden_state = self.pool_layer(aggregate_hidden_state)
        return aggregate_hidden_state.squeeze(-1)


class Sum(BaseAggregation):
    """
    Слой агрегации, который суммирует скрытые состояния.

    Этот класс реализует простую операцию агрегации, которая сначала маскирует скрытые состояния по attention_mask,
    затем суммирует их по оси последовательности (seq_len),
    возвращая результирующий тензор размерности (batch_size, embedding_dim).
    """

    def __init__(self):
        super().__init__()

    def forward(self: Self, hidden_state: torch.Tensor, attention_mask: torch.BoolTensor) -> torch.Tensor:
        """
        Выполняет прямой проход через слой агрегации.

        Args:
            hidden_state (torch.Tensor): Тензор скрытых состояний формы (batch_size, seq_len, embedding_dim).
            attention_mask (torch.BoolTensor): Маска внимания формы (batch_size, seq_len), где True — активные позиции.

        Returns:
            torch.Tensor: Результирующий тензор формы (batch_size, embedding_dim),
                представляющий агрегированные скрытые состояния.
        """
        masked_states, _ = self.apply_expanded_mask(hidden_state, attention_mask, return_seq_len=False)
        return masked_states.sum(dim=1)


class LastHiddenState(BaseAggregation):
    """
    Слой агрегации, который возвращает последнее скрытое состояние из последовательности.

    Этот класс используется для получения последнего элемента последовательности, соответствующего
    последнему активному токену (учитывая attention_mask). Обычно применяется в задачах,
    где важно финальное представление последовательности.
    """

    def __init__(self):
        super().__init__()

    def forward(self: Self, hidden_state: torch.Tensor, attention_mask: torch.BoolTensor) -> torch.Tensor:
        """
        Выполняет прямой проход через слой агрегации.

        Args:
            hidden_state (torch.Tensor): Тензор скрытых состояний формы (batch_size, seq_len, embedding_dim).
            attention_mask (torch.BoolTensor): Маска внимания формы (batch_size, seq_len), где True — активные позиции.

        Returns:
            torch.Tensor: Результирующий тензор формы (batch_size, embedding_dim),
                представляющий последнее скрытое состояние последовательности.
        """
        masked_states, seq_len = self.apply_expanded_mask(hidden_state, attention_mask, return_seq_len=True)
        aggregate_hidden_state = masked_states[:, seq_len - 1, :]
        return aggregate_hidden_state


class MeanHiddenState(BaseAggregation):
    """
    Слой агрегации, который возвращает среднее значение скрытых состояний последовательности.

    Этот класс реализует операцию агрегации, где скрытые состояния сначала маскируются по attention_mask,
    затем вычисляется их среднее значение по оси длины последовательности (seq_len), учитывая только активные токены.
    """

    def __init__(self):
        super().__init__()

    def forward(self: Self, hidden_state: torch.Tensor, attention_mask: torch.BoolTensor) -> torch.Tensor:
        """
        Выполняет прямой проход через слой агрегации.

        Args:
            hidden_state (torch.Tensor): Тензор скрытых состояний формы (batch_size, seq_len, embedding_dim).
            attention_mask (torch.BoolTensor): Маска внимания формы (batch_size, seq_len), где True — активные позиции.

        Returns:
            torch.Tensor: Результирующий тензор формы (batch_size, embedding_dim),
                представляющий среднее значение скрытых состояний.
        """
        masked_states, seq_len = self.apply_expanded_mask(hidden_state, attention_mask, return_seq_len=True)
        aggregate_hidden_state = masked_states.sum(dim=1) / seq_len.unsqueeze(1).expand(
            masked_states.shape[0],
            masked_states.shape[-1],
        )

        return aggregate_hidden_state


class LinearAggregation(BaseAggregation):
    """
    Слой агрегации, использующий линейные преобразования.

    Этот слой применяется к тензорам размерности (batch_size, n_features, embedding_dim), где количество
    признаков (n_features) фиксировано. Подходит для задач с табличными данными, где каждый признак
    представлен эмбеддингом. Реализует агрегацию через линейное преобразование по оси признаков,
    активацию SELU, проекцию и нормализацию.

    Args:
        num_features (int): Количество признаков во второй размерности входного тензора.
        embedding_dim (int): Размерность эмбеддингов.
    """

    def __init__(self: Self, num_features: int, embedding_dim: int):
        """
        Инициализирует слой линейной агрегации.

        Args:
            num_features (int): Количество признаков.
            embedding_dim (int): Размерность эмбеддингов.
        """
        super().__init__()
        self.agg_features = torch.nn.Linear(num_features, 1)
        self.act = torch.nn.SELU()
        self.proj = torch.nn.Linear(embedding_dim, embedding_dim)
        self.layer_norm = torch.nn.LayerNorm(embedding_dim)

    def forward(self: Self, hidden_state: torch.Tensor, attention_mask: torch.BoolTensor) -> torch.Tensor:
        """
        Выполняет прямой проход через слой агрегации.

        Args:
            hidden_state (torch.Tensor): Входной тензор скрытых состояний формы (batch_size, seq_len, embedding_dim).
            attention_mask (torch.BoolTensor): Маска внимания формы (batch_size, seq_len), где True — активные позиции.

        Returns:
            torch.Tensor: Результирующий тензор формы (batch_size, embedding_dim),
                представляющий агрегированные скрытые состояния
                после линейного преобразования, активации, проекции и нормализации.
        """
        masked_states, _ = self.apply_expanded_mask(hidden_state, attention_mask, return_seq_len=False)
        out = masked_states.permute(0, 2, 1)  # (batch_size, embedding_dim, n_features)
        out = self.agg_features(out)  # (batch_size, embedding_dim, 1)
        out = out.squeeze(-1)  # (batch_size, embedding_dim)
        out = self.layer_norm(self.proj(self.act(out)))
        return out
