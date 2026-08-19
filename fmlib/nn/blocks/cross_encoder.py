from typing import Self

import torch

from fmlib.nn.blocks.attention import Attention
from fmlib.nn.blocks.ffn import BaseFFN
from fmlib.nn.utils.initialization import clone_module_to_modulelist

CrossEncoderOutput = tuple[torch.Tensor, torch.Tensor]


class CrossEncoderBlock(torch.nn.Module):
    """
    Блок для применения attention'а между основной последовательностью и reference'ом (patch).

    *Примечание:* Код основан на имплементации DLT от команды NBA.

    Аргументы:
        hidden_dim (int): Размерность скрытого состояния.
        num_heads (int): Количество голов в мульти-головной attention.
        ffn (BaseFFN): Экземпляр Feed Forward сети.
        dropout (float): Вероятность dropout для входа.
            По умолчанию - 0.1.
        attn_dropout (float | None): Вероятность dropout для attention.
            По умолчанию - `None` т.е. `dropout`.

    Аттрибуты:
        input_norm (torch.nn.LayerNorm): Нормализация входа.
        attention_norm (torch.nn.LayerNorm): Нормализация выхода attention.
        self_attn (Attention): Self attention.
        ffn (BaseFFN): Feed Forward сеть.
        use_rotary (bool): Использовать ли Rotary Positional Embeddings.
        rotary_emb (torch.nn.Module | None): Rotary Positional Embeddings.
    """

    def __init__(
        self: Self, hidden_dim: int, num_heads: int, ffn: BaseFFN, dropout: float = 0.1, attn_dropout: float | None = None
    ) -> None:
        super().__init__()
        self.query_norm: torch.nn.LayerNorm = torch.nn.LayerNorm(hidden_dim)
        self.key_value_norm: torch.nn.LayerNorm = torch.nn.LayerNorm(hidden_dim)
        self.attention_norm: torch.nn.LayerNorm = torch.nn.LayerNorm(hidden_dim)
        self.cross_attn: Attention = Attention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            attn_dropout=attn_dropout,
        )
        self.ffn: BaseFFN = ffn

    def forward(
        self: Self,
        query_states: torch.Tensor,
        key_value_states: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> CrossEncoderOutput:
        """
        Аргументы:
            query_states (torch.Tensor): Входной тензор для query.
                Размерность должна быть `(batch_size, query_seq_len, hidden_dim)`.
            key_value_states (torch.Tensor): Входной тензор для key и value.
                Размерность должна быть `(batch_size, key_value_seq_len, hidden_dim)`.
            attn_mask (torch.Tensor | None): Маска для attention. По умолчанию - `None`.
                Размерность должна быть `(batch_size, query_seq_len, key_value_seq_len)`.

        Возвращает:
            torch.Tensor: Выходной тензор по query.
            torch.Tensor: Выходной тензор attention'а.
        """
        initial_shape: tuple[int, ...] = query_states.size()
        q_norm: torch.Tensor = self.query_norm(query_states)
        kv_norm: torch.Tensor = self.key_value_norm(key_value_states)

        attn_output, attn_score = self.cross_attn(
            query=q_norm, key=kv_norm, value=kv_norm, attn_mask=attn_mask, rotary_position_embeds=None
        )

        query_states = query_states + attn_output
        query_states = query_states + self.ffn(self.attention_norm(query_states))
        assert query_states.size() == initial_shape
        return (query_states, attn_score)


class RotaryCrossEncoderModel(torch.nn.Module):
    """
    Модель для сбора информации из последовательности и добавления в reference (patch).

    *Примечание:* Код основан на имплементации DLT от команды NBA.

    Аргуменеты:
        encoder_layer_template (torch.nn.Module): Шаблон для создания слоя encoder'а.
        cross_layer_template (torch.nn.Module): Шаблон для создания слоя cross-attention'а.
        num_layers (int): Количество слоев в модели. По умолчанию - 1.
    """

    def __init__(
        self: Self, encoder_layer_template: torch.nn.Module, cross_layer_template: torch.nn.Module, num_layers: int = 1
    ) -> None:
        super().__init__()

        self.num_layers: int = num_layers
        self.encoder_layers: torch.nn.ModuleList = clone_module_to_modulelist(
            module=encoder_layer_template,
            copy_count=num_layers,
            init="nba",
        )

        self.cross_layers: torch.nn.ModuleList = clone_module_to_modulelist(
            module=cross_layer_template,
            copy_count=num_layers,
            init="nba",
        )

        assert len(self.cross_layers) == self.num_layers
        assert len(self.encoder_layers) == self.num_layers

    def forward(
        self: Self,
        hidden_states: torch.Tensor,
        patches: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> CrossEncoderOutput:
        """
        Аргументы:
            hidden_states (torch.Tensor): Входной тензор.
            patches (torch.Tensor): Тензор с патчами (векторами для сборки данных).
            attn_mask (torch.Tensor | None): Маска для attention. По умолчанию - `None`.
            cross_attn_mask (torch.Tensor | None): Маска для cross-attention. По умолчанию - `None`.
            positions (torch.Tensor | None): Позиции для позиционных эмбеддингов. По умолчанию - `None`.
                Будут использоваться только для `encoder_layers`.

        Возвращает:
            torch.Tensor: Выходной тензор состояний.
            torch.Tensor: Выходной тензор патчей.
        """
        for layer_idx in range(self.num_layers):
            hidden_states, _ = self.encoder_layers[layer_idx](
                hidden_states=hidden_states,
                attn_mask=attn_mask,
                positions=positions,
            )

            patches, _ = self.cross_layers[layer_idx](
                query_states=patches,
                key_value_states=hidden_states,
                attn_mask=cross_attn_mask,
            )

        return (hidden_states, patches)
