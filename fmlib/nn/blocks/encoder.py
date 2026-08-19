from typing import Optional, Self

import torch

from fmlib.nn.blocks.attention import Attention
from fmlib.nn.blocks.ffn import BaseFFN
from fmlib.nn.blocks.utils.rotary_embeddings import RotaryPositionalEmbeddings


class BaseEncoderBlock(torch.nn.Module):
    """
    Базовый блок кодировщика, представляющий собой абстрактный класс для реализации
    конкретных архитектур кодирующих блоков (например, Transformer-слоев).

    Этот класс наследуется от `torch.nn.Module` и служит основой для построения
    пользовательских реализаций кодирующих блоков. Метод `forward` является
    чисто виртуальным и должен быть переопределен в подклассах.

    Примеры:
        >>> class MyEncoderBlock(BaseEncoderBlock):
        ...     def forward(self, x, key_padding_mask=None, attn_mask=None):
        ...         # Реализация конкретного кодирующего блока
        ...         return x
    """

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.BoolTensor] = None,
        attn_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Основной метод выполнения прямого прохода через кодирующий блок.

        Args:
            x (torch.Tensor): Входной тензор размерности [batch_size, seq_len, embed_dim].
            key_padding_mask (Optional[torch.BoolTensor]): Маска, указывающая на позиции
                заполнения (padding) в последовательности. Используется для игнорирования
                этих позиций при вычислениях внимания. Размерность: [batch_size, seq_len].
            attn_mask (Optional[torch.BoolTensor]): Маска для маскирования определенных
                позиций при вычислении внимания. Размерность: [seq_len, seq_len].

        Returns:
            torch.Tensor: Выходной тензор после обработки входных данных.
                Размерность: [batch_size, seq_len, embed_dim].

        Raises:
            NotImplementedError: Метод не реализован в базовом классе и должен быть
                переопределен в подклассе.
        """
        raise NotImplementedError()


class EncoderBlock(BaseEncoderBlock):
    """
    Блок кодировщика трансформерной архитектуры.

    Реализует одну стадию трансформерного кодировщика, включающую:
    1. Много-головое внимание (Multi-head Attention).
    2. Нормализацию и dropout после внимания.
    3. Фид-форвардную сеть (FFN).
    4. Дополнительную нормализацию и dropout после FFN.
    Используются остаточные соединения (skip connections) для улучшения градиентного потока.

    Args:
        hidden_dim (int): Размерность скрытого состояния (размерность эмбеддинга).
        num_heads (int): Количество голов внимания.
        attn_dropout (float): Вероятность dropout для слоя внимания.
        dropout (float): Вероятность dropout для FFN.
        ffn (BaseFFN): Экземпляр фид-форвардной сети.

    Attributes:
        multi_head_attn (torch.nn.MultiheadAttention): Слой много-голового внимания.
        attention_norm (torch.nn.LayerNorm): Нормализация выхода слоя внимания.
        attention_dropout (torch.nn.Dropout): Dropout для регуляризации внимания.
        ffn (BaseFFN): Экземпляр фид-форвардной сети.
        ffn_norm (torch.nn.LayerNorm): Нормализация перед FFN.
        dropout (torch.nn.Dropout): Dropout для регуляризации FFN.

    Shape:
        - Input: (batch_size, seq_len, hidden_dim)
        - Output: (batch_size, seq_len, hidden_dim)

    Пример:
        >>> encoder_block = EncoderBlock(
        ...     hidden_dim=512,
        ...     num_heads=8,
        ...     attn_dropout=0.1,
        ...     dropout=0.1,
        ...     ffn=PositionWiseFFN(...),
        ... )
        >>> x = torch.randn(32, 10, 512)  # batch_size=32, seq_len=10
        >>> output = encoder_block(x)  # shape (32, 10, 512)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        attn_dropout: float,
        dropout: float,
        ffn: BaseFFN,
    ):
        super().__init__()
        self.multi_head_attn = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.attention_norm = torch.nn.LayerNorm(hidden_dim)
        self.attention_dropout = torch.nn.Dropout(attn_dropout)

        self.ffn = ffn
        self.ffn_norm = torch.nn.LayerNorm(hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.BoolTensor] = None,
        attn_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Выполняет прямой проход через блок кодировщика.

        Args:
            x (torch.Tensor): Входной тензор размерности (batch_size, seq_len, hidden_dim).
            key_padding_mask (Optional[torch.BoolTensor]): Маска заполнения ключей.
                Используется для игнорирования позиций padding при вычислении внимания.
                Размерность: [batch_size, seq_len], где True означает, что позиция должна быть
                исключена из внимания.
            attn_mask (Optional[torch.BoolTensor]): Маска для игнорирования определённых
                позиций во время расчёта матрицы внимания. Размерность: [seq_len, seq_len], где True
                указывает на то, что соответствующая позиция должна быть проигнорирована.

        Returns:
            torch.Tensor: Выходной тензор той же размерности, что и вход —
                (batch_size, seq_len, hidden_dim).

        """
        x_norm = self.attention_norm(x)
        att_output, _ = self.multi_head_attn(
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        y = x + self.attention_dropout(att_output)
        z = y + self.dropout(self.ffn(self.ffn_norm(y)))
        return self.dropout(z)


class EventEncoderBlock(torch.nn.Module):
    """
    Блок агрегации эмбеддингов последовательностей событий, объединяющий Self Attention и Feed Forward слои.

    Этот блок обрабатывает эмбеддинги событий следующим образом:
    1. Self Attention для признаков с использованием skip connections для улучшения градиентного потока.
    2. Применение Feed Forward слоя.
    3. Усреднение значений по оси событий (mean pooling) и L2 нормализацию.

    Args:
        hidden_dim (int): Размерность входных/выходных признаков.
        attn_dropout (float): Вероятность dropout для слоя внимания.
        dropout (float): Вероятность dropout для фид-форвардной сети.
        ffn (BaseFFN): Экземпляр фид-форвардной сети.

    Example:
        >>> block = FeatureEncoderBlock(
        ...     hidden_dim=64,
        ...     attn_dropout=0.1,
        ...     dropout=0.1,
        ...     ffn=PositionWiseFFN(...),
        ... )
        >>> embeddings = torch.randn(32, 5, 64)  # (batch_size, num_features, hidden_dim)
        >>> output = block(embeddings)  # shape (32, 64)
    """

    def __init__(
        self, hidden_dim: int, attn_dropout: float, dropout: float, ffn: BaseFFN, attention: torch.nn.Module | None = None
    ):
        super().__init__()
        if attention is None:
            attention = torch.nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=1,
                dropout=attn_dropout,
                batch_first=True,
            )
        self.multi_head_attn = attention
        self.ffn = ffn

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.BoolTensor,
        key_padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Выполняет прямой проход через блок агрегации признаков.

        Args:
            x (torch.Tensor): Входные эмбеддинги признаков размерности (batch_size, num_features, hidden_dim).
            attn_mask (torch.BoolTensor): Маска для игнорирования определённых позиций во время расчёта матрицы внимания.
                Также применяется для корректного расчета среднего значения при pooling'е.
                Формат: [*, num_features, num_features], где True указывает на то, что
                соответствующая позиция игнорируется при подсчете внимания.
            key_padding_mask (Optional[torch.BoolTensor]): Маска заполнения ключей.
                Используется для игнорирования позиций padding при вычислении внимания.
                Формат: [batch_size, num_features], где True означает, что позиция должна быть исключена из внимания.

        Output Shape:
            - (batch_size, hidden_dim)  # Нормализованный выход после mean pooling

        Returns:
            torch.Tensor: L2-нормализованное среднее значение по последовательности.
        """
        att_output, _ = self.multi_head_attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False,
        )
        y = x + att_output
        z = y + self.ffn(y)

        feature_mask = attn_mask[:, -1, :].logical_not()
        num_useful_features = feature_mask.sum(dim=-1).unsqueeze(-1)

        masked_aggregated_output = z * feature_mask.unsqueeze(-1)
        aggregated_output = masked_aggregated_output.sum(dim=-2) / num_useful_features
        return torch.nn.functional.normalize(aggregated_output, dim=-1)


class RotaryEncoderBlock(torch.nn.Module):
    """
    Энкодерный блок. Поддерживает использование Rotary Positional Embeddings.

    *Примечание:* Код основан на имплементации DLT от команды NBA.

    Аргументы:
        hidden_dim (int): Размерность скрытого состояния.
        num_heads (int): Количество голов в мульти-головной attention.
        ffn (BaseFFN): Экземпляр Feed Forward сети.
        dropout (float): Вероятность dropout для входа.
            По умолчанию - 0.1.
        attn_dropout (float | None): Вероятность dropout для attention.
            По умолчанию - `None` т.е. `dropout`.
        use_rotary (bool): Использовать ли Rotary Positional Embeddings.

    Аттрибуты:
        input_norm (torch.nn.LayerNorm): Нормализация входа.
        attention_norm (torch.nn.LayerNorm): Нормализация выхода attention.
        self_attn (Attention): Self attention.
        ffn (BaseFFN): Feed Forward сеть.
        use_rotary (bool): Использовать ли Rotary Positional Embeddings.
        rotary_emb (torch.nn.Module | None): Rotary Positional Embeddings.
    """

    def __init__(
        self: Self,
        hidden_dim: int,
        num_heads: int,
        ffn: BaseFFN,
        dropout: float = 0.1,
        attn_dropout: float | None = None,
        use_rotary: bool = False,
    ) -> None:
        super().__init__()

        self.input_norm: torch.nn.LayerNorm = torch.nn.LayerNorm(hidden_dim)
        self.attention_norm: torch.nn.LayerNorm = torch.nn.LayerNorm(hidden_dim)
        self.self_attn: Attention = Attention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )
        self.ffn: BaseFFN = ffn
        self.use_rotary: bool = use_rotary
        self.rotary_emb: torch.nn.Module | None = None
        if self.use_rotary:
            head_dim: int = hidden_dim // num_heads
            self.rotary_emb = RotaryPositionalEmbeddings(head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Аргументы:
            hidden_states (torch.Tensor): Входные эмбеддинги размерности (batch_size, seq_len, hidden_dim).
            attn_mask (torch.Tensor | None): Маска для игнорирования определённых позиций во время расчёта матрицы внимания.
            positions (torch.Tensor | None): Позиции для позиционных эмбеддингов.

        Возвращает:
            tuple[torch.Tensor, torch.Tensor]: Вектор признаков и матрица внимания.
        """
        initial_shape: tuple[int, ...] = hidden_states.size()
        position_embeddings: torch.Tensor | None = None
        if self.use_rotary:
            if positions is None:
                _, seq_len, _ = initial_shape
                device: torch.device = hidden_states.device
                positions = torch.arange(seq_len, device=device).unsqueeze(0)
            position_embeddings = self.rotary_emb(positions, dtype=hidden_states.dtype)
        else:
            assert positions is None

        inp_hidden_states: torch.Tensor = self.input_norm(hidden_states)

        attn_output, attn_score = self.self_attn(
            query=inp_hidden_states,
            key=inp_hidden_states,
            value=inp_hidden_states,
            attn_mask=attn_mask,
            rotary_position_embeds=position_embeddings,
        )

        hidden_states = hidden_states + attn_output
        hidden_states = hidden_states + self.ffn(self.attention_norm(hidden_states))
        assert hidden_states.size() == initial_shape
        return (hidden_states, attn_score)
