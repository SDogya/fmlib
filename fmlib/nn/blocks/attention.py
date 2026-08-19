import warnings
from typing import Self

import torch
import torch.nn.functional as func

from fmlib.nn.blocks.utils.rotary_embeddings import apply_rotary_pos_emb

AttentionOutput = tuple[torch.Tensor, torch.Tensor]
RotaryPositionEmbeds = tuple[torch.Tensor, torch.Tensor]


class RawAttention(torch.nn.Module):
    """
    Внутреняя наивная имплементация attention'а.
    Может быть заменена на FlexAttention в дальнейшем.
    """

    def __init__(
        self: Self,
        hidden_dim: int,
        num_heads: int,
        scale: float | None = None,
        attn_dropout: float = 0.0,
        mask_value: float | None = None,
    ) -> None:
        super().__init__()

        if hidden_dim < 1:
            msg: str = f"`hidden_dim` must be a positive integer. Got: {hidden_dim=}."
            raise ValueError(msg)
        elif hidden_dim < 128:
            msg: str = f"Suspiciously low value of `hidden_dim`. Got: {hidden_dim=}."
            warnings.warn(msg, stacklevel=2)

        self.hidden_dim: int = hidden_dim

        if num_heads < 1:
            msg: str = f"`hidden_dim` must be a positive integer. Got: {hidden_dim=}."
            raise ValueError(msg)

        head_dim: int = hidden_dim // num_heads
        if (head_dim * num_heads) != hidden_dim:
            msg: str = f"`hidden_dim` must be multiple of `num_heads`. Got: {hidden_dim=} vs {num_heads=}."
            raise ValueError(msg)
        elif head_dim < 16:
            msg: str = f"Suspiciously low value of `head_dim = hidden_dim // num_heads`. Got: {head_dim=}."
            warnings.warn(msg, stacklevel=2)

        self.head_dim: int = head_dim
        self.num_heads: int = num_heads

        if scale is not None and ((scale <= 0.0) or (scale > 1.0)):
            msg: str = f"`scale` must be in the range (0.0, 1.0]. Got: {scale=}."
            raise ValueError(msg)
        self.scale: float | None = scale

        self.attn_dropout: float = attn_dropout

        self.mask_value: float | None = mask_value

    def prepare_hidden_states(self: Self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        temp_shape: tuple[int, ...] = (batch_size, seq_len, self.num_heads, self.head_dim)
        target_shape: tuple[int, ...] = (batch_size, self.num_heads, seq_len, self.head_dim)
        result: torch.Tensor = hidden_states.view(*temp_shape).transpose(1, 2)
        assert result.shape == target_shape
        return result

    def get_mask_value(self: Self, dtype: torch.dtype) -> float:
        if self.mask_value is None:
            return torch.finfo(dtype).min
        else:
            return self.mask_value

    def forward(
        self: Self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        rotary_position_embeds: RotaryPositionEmbeds | None = None,
    ) -> AttentionOutput:
        query_preped: torch.Tensor = self.prepare_hidden_states(query)
        key_preped: torch.Tensor = self.prepare_hidden_states(key)
        value_preped: torch.Tensor = self.prepare_hidden_states(value)

        dropout_p: float = 0.0
        if self.training:
            dropout_p = self.attn_dropout

        if rotary_position_embeds is not None:
            cos, sin = rotary_position_embeds
            query_preped, key_preped = apply_rotary_pos_emb(query_preped, key_preped, cos, sin)

        if attn_mask is not None:
            assert torch.is_tensor(attn_mask)
            if attn_mask.dtype == torch.bool:
                mask_value: float = self.get_mask_value(query.dtype)
                attn_mask = torch.where(attn_mask, 0.0, mask_value)
            attn_mask = attn_mask.float()
            assert attn_mask.dtype.is_floating_point

            batch_size: int = query.size(0)
            if attn_mask.size(0) != batch_size:
                attn_mask = attn_mask.expand(batch_size, -1, -1, -1)

        output: torch.Tensor = func.scaled_dot_product_attention(
            attn_mask=attn_mask,
            query=query_preped,
            key=key_preped,
            value=value_preped,
            scale=self.scale,
            dropout_p=dropout_p,
        )

        output = output.transpose(1, 2).contiguous().view(*query.shape)

        return (output, None)


class Attention(torch.nn.Module):
    """
    Верхнеуровневый модуль attention'а. Возможны разные внутренние имплементации.

    Аргументы:
        hidden_dim (int): Размерность скрытого состояния.
            Должно быть кратно `num_heads`: `hidden_dim = num_heads * head_dim`.
        num_heads (int): Количество голов.
        scale (float | None): Коэффициент масштабирования attention score.
            По умолчанию - `None` т.е. `1 / math.sqrt(hidden_dim)`.
        dropout (float): Коэффициент dropout'а. Применяются к выводу.
            По умолчанию - `0.0` т.е. нет пропусков.
        attn_dropout (float | None): Коэффициент dropout'а. Применяются к attention score.
            По умолчанию - `None` т.е. будет использоваться значение параметра `dropout`.

    Аттрибуты:
        raw (torch.nn.Module): Модуль, реализующий внутреннюю имплементацию attention'а.
            *Примечание:* Тип и интерфейс аттрибута не гарантируется.
        query_layer (torch.nn.Linear): Слой преобразования query.
        key_layer (torch.nn.Linear): Слой преобразования key.
        value_layer (torch.nn.Linear): Слой преобразования value.
        proj_layer (torch.nn.Linear): Слой преобразования output.
        resid_dropout (torch.nn.Dropout): Dropout для выхода.
    """

    def __init__(
        self: Self,
        hidden_dim: int,
        num_heads: int,
        scale: float | None = None,
        dropout: float = 0.0,
        attn_dropout: float | None = None,
        mask_value: float | None = None,
    ) -> None:
        super().__init__()

        if attn_dropout is None:
            attn_dropout = dropout

        self.raw: RawAttention = RawAttention(
            attn_dropout=attn_dropout,
            hidden_dim=hidden_dim,
            mask_value=mask_value,
            num_heads=num_heads,
            scale=scale,
        )

        self.query_layer: torch.nn.Linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.key_layer: torch.nn.Linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.value_layer: torch.nn.Linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.proj_layer: torch.nn.Linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.resid_dropout: torch.nn.Dropout = torch.nn.Dropout(dropout)

    def forward(
        self: Self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        rotary_position_embeds: RotaryPositionEmbeds | None = None,
    ) -> AttentionOutput:
        """
        Вычисляет multi-head attention.

        Аргументы:
            query (torch.Tensor): Тензор векторов запроса.
                Должен иметь размер `batch_size x seq_len x hidden_dim`.
            key (torch.Tensor): Тензор векторов ключей.
                Должен иметь размер `batch_size x kv_len x hidden_dim`.
            value (torch.Tensor): Тензор векторов значений.
                Должен иметь размер `batch_size x kv_len x hidden_dim`.
            attn_mask (torch.Tensor | None): Маска attention'а.
                Должна иметь размер `... x seq_len x kv_len`.
                Поддерживаемые типы маски: bool или float.
            rotary_position_embeds (RotaryPositionEmbeds | None): Вектора cos, sin.

        Возвращает:
            torch.Tensor: Тензор векторов output.
                Будет иметь размер `batch_size x seq_len x hidden_dim`.
            torch.Tensor: Тензор attention score.
                Будет иметь размер `batch_size x ... x seq_len x kv_len`.
        """
        query_preped: torch.Tensor = self.query_layer(query)
        key_preped: torch.Tensor = self.key_layer(key)
        value_preped: torch.Tensor = self.value_layer(value)

        raw_output, attn_score = self.raw(
            query=query_preped,
            key=key_preped,
            value=value_preped,
            attn_mask=attn_mask,
            rotary_position_embeds=rotary_position_embeds,
        )

        output: torch.Tensor = self.proj_layer(raw_output)
        output = self.resid_dropout(output)

        return (output, attn_score)


class IntraFeatureAttention(torch.nn.Module):
    """Intra-feature attention mechanism for processing 2D event sequences.

    This module implements a scaled dot-product attention mechanism that operates
    across feature dimensions within each event in a sequence. Each feature attends
    to other features within the same event.

    Args:
        hidden_size (int): Dimensionality of input features and attention outputs.

    Input Shapes:
        - q, k, v: (batch_size, seq_len, num_features, hidden_size)
        - event_attention_mask: (batch_size, seq_len, num_features, num_features) or None

    Output Shape:
        - (batch_size, seq_len, num_features, hidden_size)

    Attributes:
        w_q (nn.Linear): Query transformation layer
        w_k (nn.Linear): Key transformation layer
        w_v (nn.Linear): Value transformation layer
        output (nn.Linear): Output projection layer

    Example:
        >>> attn = IntraFeatureAttention(hidden_size=64)
        >>> features = torch.randn(32, 10, 5, 64)  # (batch, seq_len, num_features, hidden)
        >>> output = attn(features, features, features)
    """

    def __init__(self: Self, hidden_size: int, bias: bool = True, mask_value: float | None = None) -> None:
        super().__init__()

        if mask_value is None:
            mask_value = float("-inf")
        self.mask_value: float = mask_value

        self.hidden_size = hidden_size
        self.norm: float = 1.0 / (self.hidden_size**0.5)
        self.w_q = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.w_k = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.w_v = torch.nn.Linear(hidden_size, hidden_size, bias=bias)
        self.output = torch.nn.Linear(hidden_size, hidden_size, bias=bias)

    def calcuate_attn_scores(
        self: Self, q: torch.Tensor, k: torch.Tensor, attn_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute attention scores between features.

        Args:
            q: Query tensor (batch_size, seq_len, num_features, hidden_size)
            k: Key tensor (batch_size, seq_len, num_features, hidden_size)
            attn_mask: Optional attention mask (batch_size, seq_len, num_features, num_features)

        Returns:
            Attention scores tensor (batch_size, seq_len, num_features, num_features)
        """
        k = k.transpose(-2, -1)
        scores: torch.Tensor = torch.einsum("bij,bjk->bik", [q, k]) * self.norm
        if attn_mask is not None:
            attn_mask = attn_mask.to(scores.device)
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask, self.mask_value)
            elif attn_mask.dtype.is_floating_point:
                scores = scores + attn_mask
            else:
                bool_mask: torch.Tensor = attn_mask == 0
                scores = scores.masked_fill(bool_mask, self.mask_value)
        scores = torch.nn.functional.softmax(scores, dim=-1)
        return scores

    def forward(
        self: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of the attention mechanism.

        Args:
            q: Query tensor
            k: Key tensor
            v: Value tensor
            attn_mask: Optional attention mask

        Returns:
            Attention-weighted output tensor
            Attention scores tensor
        """
        if key_padding_mask is not None:
            msg = "`key_padding_mask` is not supported"
            raise NotImplementedError(msg)

        query, key, value = self.w_q(query), self.w_k(key), self.w_v(value)
        scores: torch.Tensor = self.calcuate_attn_scores(query, key, attn_mask)
        output: torch.Tensor = torch.einsum("bij,bjk->bik", [scores, value])
        output = self.output(output)

        result = (output, scores) if need_weights else (output, None)

        return result
