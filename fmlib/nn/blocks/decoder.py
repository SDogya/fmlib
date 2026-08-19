from typing import Optional, Self

import torch

from fmlib.nn.blocks.ffn import BaseFFN
from fmlib.nn.utils.initialization import clone_module_to_modulelist


class DecoderBlock(torch.nn.Module):
    """
    Блок декодировщика трансформерной архитектуры.

    Реализует одну стадию трансформерного декодировщика, включающую:
    1. Cross Attention между выходом кодировщика и текущим входом.
    2. Self Attention для обработки текущей последовательности.
    3. Feed Forward слой.
    Все компоненты используют остаточные соединения (skip connections) и нормализацию.

    Args:
        hidden_dim (int): Размерность скрытого состояния (размерность эмбеддинга).
        num_heads (int): Количество голов внимания.
        attn_dropout (float): Вероятность dropout для слоёв внимания.
        dropout (float): Вероятность dropout для FFN.
        ffn (BaseFFN): Экземпляр фид-форвардной сети.

    Attributes:
        ln_0 (torch.nn.LayerNorm): Нормализация перед кросс-вниманием.
        ln_1 (torch.nn.LayerNorm): Нормализация ключей/значений для кросс-внимания.
        ln_2 (torch.nn.LayerNorm): Нормализация перед self-вниманием.
        cross_attention (torch.nn.MultiheadAttention): Слой кросс-внимания.
        self_attention (torch.nn.MultiheadAttention): Слой self-внимания.
        dropout (torch.nn.Dropout): Dropout для регуляризации.
        ffn (BaseFFN): Экземпляр фид-форвардной сети.

    Shape:
        - x: (batch_size, seq_len, hidden_dim)
        - x_kv: (batch_size, seq_len, hidden_dim)
        - Output: (batch_size, seq_len, hidden_dim)

    Пример:
        >>> decoder_block = DecoderBlock(
        ...     hidden_dim=512,
        ...     num_heads=8,
        ...     attn_dropout=0.1,
        ...     dropout=0.1,
        ...     ffn=PositionWiseFFN(...),
        ... )
        >>> x = torch.randn(32, 10, 512)  # batch_size=32, seq_len=10
        >>> x_kv = torch.randn(32, 10, 512)
        >>> output = decoder_block(x, x_kv)  # shape (32, 10, 512)
    """

    def __init__(
        self: Self,
        hidden_dim: int,
        num_heads: int,
        attn_dropout: float,
        dropout: float,
        ffn: BaseFFN,
    ):
        super().__init__()
        self.ln_0 = torch.nn.LayerNorm(hidden_dim)
        self.ln_1 = torch.nn.LayerNorm(hidden_dim)
        self.ln_2 = torch.nn.LayerNorm(hidden_dim)
        self.ffn_norm = torch.nn.LayerNorm(hidden_dim)

        self.cross_attention, self.self_attention = [
            torch.nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=attn_dropout,
                batch_first=True,
            )
            for _ in range(2)
        ]
        self.dropout = torch.nn.Dropout(dropout)
        self.ffn = ffn

    def forward(
        self: Self,
        x: torch.Tensor,
        x_kv: torch.Tensor,
        cross_attention_padding_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        self_attention_padding_mask: Optional[torch.Tensor] = None,
        self_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Функция прямого прохода через блок декодировщика.

        Args:
            x (torch.Tensor): Входная последовательность декодировщика.
            x_kv (torch.Tensor): Ключи и значения от кодировщика.
            cross_attention_padding_mask (Optional[torch.Tensor]): Маска заполнения для кросс-внимания.
            cross_attention_mask (Optional[torch.Tensor]): Маска внимания для кросс-внимания.
            self_attention_padding_mask (Optional[torch.Tensor]): Маска заполнения для self-внимания.
            self_attention_mask (Optional[torch.Tensor]): Маска внимания для self-внимания.

        Returns:
            torch.Tensor: Выходной тензор после применения всех слоёв.
        """
        x_norm = self.ln_0(x)
        x_kv_norm = self.ln_1(x_kv)
        cross_att_output, _ = self.cross_attention(
            x_norm,
            x_kv_norm,
            x_kv_norm,
            key_padding_mask=cross_attention_padding_mask,
            attn_mask=cross_attention_mask,
            need_weights=False,
        )
        y = x + self.dropout(cross_att_output)
        y_norm = self.ln_2(y)
        self_att_output, _ = self.self_attention(
            y_norm,
            y_norm,
            y_norm,
            key_padding_mask=self_attention_padding_mask,
            attn_mask=self_attention_mask,
            need_weights=False,
        )
        z = y + self.dropout(self_att_output)
        out = z + self.dropout(self.ffn(self.ffn_norm(z)))
        return out


class RotaryDecoderModel(torch.nn.Module):
    """
    Декодерная модель с использованием `RotaryEncoderBlock`.

    *Примечание:* Код основан на имплементации DLT от команды NBA.

    Аргументы:
        hidden_dim (int): Размерность скрытого состояния.
        layer_template (torch.nn.Module): Шаблон для создания `RotaryEncoder` блоков.
        num_layers (int): Количество `RotaryEncoder` блоков в модели.
        dropout (float): Вероятность dropout для регуляризации.

    Аттрибуты:
        input_dropout (torch.nn.Dropout): Dropout для входных данных.
        decoder (torch.nn.ModuleList): Список `RotaryEncoder` блоков, клонированных из `layer_template`.
        norm (torch.nn.LayerNorm): Нормализация выходных данных.
    """

    def __init__(
        self: Self, hidden_dim: int, layer_template: torch.nn.Module, num_layers: int = 1, dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.input_dropout: torch.nn.Dropout = torch.nn.Dropout(dropout)
        self.decoder: torch.nn.ModuleList = clone_module_to_modulelist(module=layer_template, copy_count=num_layers, init="nba")
        self.norm: torch.nn.LayerNorm = torch.nn.LayerNorm(hidden_dim)

    def forward(self: Self, hidden_states: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        Аргументы:
            hidden_states (torch.Tensor): Входные данные с размерностью (batch_size, seq_len, hidden_dim).
            positions (torch.Tensor | None): Позиции для позиционного кодирования.

        Возвращает:
            torch.Tensor: Выходные данные с размерностью (batch_size, seq_len, hidden_dim).
        """
        _, seq_len, _ = hidden_states.size()
        dtype: torch.dtype = hidden_states.dtype
        device: torch.device = hidden_states.device
        min_value: float = torch.finfo(dtype).min

        hidden_states = self.input_dropout(hidden_states)

        causal_mask: torch.Tensor = torch.ones(
            size=(seq_len, seq_len),
            device=device,
            dtype=dtype,
        )
        casual_mask = torch.tril(causal_mask).unsqueeze(0).unsqueeze(0)
        causal_mask = causal_mask.view(1, 1, seq_len, seq_len)
        casual_mask = (1.0 - casual_mask) * min_value

        for decoder_layer in self.decoder:
            hidden_states, _ = decoder_layer(
                hidden_states=hidden_states,
                attn_mask=casual_mask,
                positions=positions,
            )

        return self.norm(hidden_states)
