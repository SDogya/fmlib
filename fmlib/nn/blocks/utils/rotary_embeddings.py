import math
from typing import Self

import torch

from fmlib.constants.models import DEFAULT_PARTIAL_ROPE_FACTOR, DEFAULT_ROPE_THETA


class RotaryPositionalEmbeddings(torch.nn.Module):
    """
    Модуль для получения `поворотов` эмбеддингов с использованием RoPE.

    Аргументы:
        hidden_size (int): Размерность эмбеддингов.
        partial_rope_factor (float): Коэффициент для вычисления размерности RoPE.
            По умолчанию `DEFAULT_PARTIAL_ROPE_FACTOR`.
        rope_theta (float): Параметр для вычисления RoPE. По умолчанию `DEFAULT_ROPE_THETA`.
    """

    def __init__(
        self: Self,
        hidden_size: int,
        partial_rope_factor: float = DEFAULT_PARTIAL_ROPE_FACTOR,
        rope_theta: float = DEFAULT_ROPE_THETA,
    ) -> None:
        super().__init__()

        if hidden_size < 1:
            msg: str = f"`hidden_size` must be greater than 0. Got {hidden_size}."
            raise ValueError(msg)

        if rope_theta < 0.0:
            msg: str = f"`rope_theta` must be greater than 0. Got {rope_theta}."
            raise ValueError(msg)

        if partial_rope_factor < 0.0:
            msg: str = f"`partial_rope_factor` must be greater than 0. Got {partial_rope_factor}."
            raise ValueError(msg)

        def _init_rope_parameters(hidden_size, rope_theta: float, partial_factor: float) -> torch.Tensor:
            dim: int = math.floor(hidden_size * partial_factor)

            if dim < 1:
                msg: str = f"`dim` must be greater than 0. Got {dim=}."
                raise ValueError(msg)

            powers: torch.FloatTensor = torch.div(torch.arange(0, dim, 2, dtype=torch.float32), dim)
            inv_freq: torch.FloatTensor = torch.div(torch.ones_like(powers), (rope_theta**powers))
            return inv_freq

        inv_freq: torch.Tensor = _init_rope_parameters(
            partial_factor=partial_rope_factor,
            hidden_size=hidden_size,
            rope_theta=rope_theta,
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self: Self, positions: torch.Tensor, dtype: torch.dtype) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Аргументы:
            positions (torch.Tensor): Позиции для которых нужно получить эмбеддинги.
            dtype (torch.dtype): Тип данных для эмбеддингов.

        Возвращает:
            torch.FloatTensor: Косинусы для эмбеддингов.
            torch.FloatTensor: Синусы для эмбеддингов.
        """
        batch_size: int = positions.size(0)
        device_type: str = positions.device.type
        inv_freq_expanded: torch.FloatTensor = self.inv_freq[None, :, None].expand(batch_size, -1, 1)
        positions_expanded: torch.FloatTensor = positions[:, None, :].to(dtype=torch.float32)
        with torch.autocast(device_type=device_type, enabled=False):
            freqs: torch.FloatTensor = (inv_freq_expanded @ positions_expanded).transpose(1, 2)
            emb: torch.FloatTensor = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        cos = cos.to(dtype=dtype)
        sin = sin.to(dtype=dtype)
        return (cos, sin)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Вспомогательная функция для деления эмбеддингов"""
    half_dim: int = x.size(-1) // 2
    x1: torch.Tensor = x[..., :half_dim]
    x2: torch.Tensor = x[..., half_dim:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Функция для применения RoPE к эмбеддингам.

    Аргументы:
        q (torch.Tensor): Эмбеддинги для `q`.
        k (torch.Tensor): Эмбеддинги для `k`.
        cos (torch.Tensor): Косинусы для RoPE.
        sin (torch.Tensor): Синусы для RoPE.
        unsqueeze_dim (int): Размерность для распаковки `sin, cos`. По умолчанию - 1.
    """
    cos: torch.Tensor = cos.unsqueeze(unsqueeze_dim)
    sin: torch.Tensor = sin.unsqueeze(unsqueeze_dim)
    q_embed: torch.Tensor = (q * cos) + (rotate_half(q) * sin)
    k_embed: torch.Tensor = (k * cos) + (rotate_half(k) * sin)
    return (q_embed, k_embed)
