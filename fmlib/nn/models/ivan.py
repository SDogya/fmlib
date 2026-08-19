from typing import Dict, List, Optional, Self

import torch

from fmlib.nn.blocks import BaseEncoderBlock, DecoderBlock
from fmlib.nn.utils.initialization import clone_module_to_modulelist, nba_init_weights


class IvanModel(torch.nn.Module):
    """
    Модель, реализующая трансформерную архитектуру с несколькими слоями кодировщика и декодировщика.

    Эта модель состоит из:
    - Входного эмбеддинга и позиционного кодирования.
    - Слоёв кодировщика для обработки входных данных.
    - Слоёв декодировщика для генерации выходной последовательности.
    - Используются нормализация, dropout и остаточные соединения.

    Args:
        vocab_size (int): Размер входного словаря.
        hidden_dim (int): Размерность скрытого состояния модели.
        positional_encoding_num_buckets (int): Количество бакетов для позиционного кодирования.
        num_encoder_layers (int): Количество слоёв в кодировщике.
        num_decoder_layers (int): Количество слоёв в декодировщике.
        dropout (float): Вероятность dropout для регуляризации.
        input_encoder_block (BaseEncoderBlock): Экземпляр входного блока кодировщика.
        encoder_block (BaseEncoderBlock): Экземпляр блока кодировщика.
            Экземпляр будет скопирован для каждого из слоев кодировщика.
            Ко всем весам копий будет применен `torch.nn.init.xavier_normal_`.
        decoder_block (DecoderBlock): Экземпляр блока декодировщика.
            Экземпляр будет скопирован для каждого из слоев декодировщика.
            Ко всем весам копий будет применен `torch.nn.init.xavier_normal_`.

    Attributes:
        input_dropout (torch.nn.Dropout): Dropout на входном уровне.
        input_embeddings (torch.nn.Embedding): Слой эмбеддингов для входных токенов.
        position_embeddings (torch.nn.Embedding): Слой позиционного кодирования.
        Encoder (torch.nn.ModuleList): Список слоёв кодировщика.
        Decoder (torch.nn.ModuleList): Список слоёв декодировщика.
        ln_0 (torch.nn.LayerNorm): Нормализация после кодировщика.
        ln_1 (torch.nn.LayerNorm): Нормализация после декодировщика.
        input_encoder (BaseEncoderBlock): Входной блок кодировщика.
        input_pad_vector (torch.nn.Parameter): Вектор заполнителя для дополнительных размерностей.

    Shape:
        - encoder_input: (batch_size, seq_len, ...)
        - decoder_input: (batch_size, seq_len, ...)
        - Output: (batch_size, seq_len, hidden_dim)

    Пример:
        >>> model = IvanModel(
        >>>     vocab_size=10000,
        >>>     hidden_dim=512,
        >>>     positional_encoding_num_buckets=1024,
        >>>     num_encoder_layers=6,
        >>>     num_decoder_layers=6,
        >>>     dropout=0.1,
        >>>     input_encoder_block=BaseEncoderBlock(...),
        >>>     encoder_block=BaseEncoderBlock(...),
        >>>     decoder_block=DecoderBlock(...)
        >>> )
        >>> encoder_input = torch.randint(0, 10000, (32, 10))
        >>> decoder_input = torch.randint(0, 10000, (32, 10))
        >>> output = model(encoder_input, decoder_input)
    """

    def __init__(
        self: Self,
        vocab_size: int,
        hidden_dim: int,
        positional_encoding_num_buckets: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout: float,
        input_encoder_block: BaseEncoderBlock,
        encoder_block: BaseEncoderBlock,
        decoder_block: DecoderBlock,
    ) -> None:
        super().__init__()
        self.input_dropout = torch.nn.Dropout(dropout)
        self.input_embeddings = torch.nn.Embedding(vocab_size, hidden_dim)
        self.position_embeddings = torch.nn.Embedding(positional_encoding_num_buckets, hidden_dim)

        self.Encoder = clone_module_to_modulelist(encoder_block, num_encoder_layers)
        self.ln_0 = torch.nn.LayerNorm(hidden_dim)

        self.Decoder = clone_module_to_modulelist(decoder_block, num_decoder_layers)
        self.ln_1 = torch.nn.LayerNorm(hidden_dim)

        self.input_encoder = input_encoder_block

        self.input_pad_vector = torch.nn.Parameter(torch.empty((1, 1, 1, hidden_dim), dtype=torch.float32))
        torch.nn.init.normal_(tensor=self.input_pad_vector, mean=0.0, std=0.1)

        # Сохраняем инициализацию весов от оригинальной модели
        self.apply(nba_init_weights)

    def get_input_names(self: Self) -> List[str]:
        """Возвращает список имен входных тензоров модели."""
        return list(self.get_dynamic_shapes().keys())

    def get_output_names(self: Self) -> List[str]:
        """Возвращает список имен выходных тензоров модели."""
        return ["decoder_output"]

    def get_dynamic_shapes(self: Self) -> Dict[str, Dict[int, str]]:
        """Описывает динамическую структуру тензоров на входе и выходе модели."""
        return {
            "encoder_input": {0: "batch_size", 1: "encoder_seq_len", 2: "event_count"},
            "decoder_input": {0: "batch_size", 1: "decoder_seq_len", 2: "event_count"},
            "positional_encoding": {0: "batch_size", 1: "encoder_seq_len"},
            "encoder_padding_mask": {0: "batch_size", 1: "encoder_seq_len"},
            "additional_embedding": {0: "batch_size", 1: "additional_embedding_seq_len", 2: "event_count"},
        }

    def forward(
        self: Self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
        positional_encoding: torch.Tensor,
        encoder_padding_mask: Optional[torch.BoolTensor] = None,
        additional_embedding: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        ########################### ENCODER PROCESSING STAGE START ###########################
        encoder_input = self.input_embeddings(encoder_input)
        encoder_input = torch.cat(
            [
                self.input_pad_vector.repeat(encoder_input.shape[0], encoder_input.shape[1], 1, 1),
                encoder_input,
            ],
            dim=2,
        )
        encoder_input = self.input_dropout(encoder_input)
        dim_0, dim_1, dim_2, dim_3 = encoder_input.shape
        encoder_input = encoder_input.view(-1, dim_2, dim_3)
        encoder_input = self.input_encoder(encoder_input)
        encoder_input = encoder_input.view(dim_0, dim_1, dim_2, dim_3)
        encoder_input = encoder_input[:, :, 0, :]

        position_emb = self.position_embeddings(positional_encoding)
        encoder_input = encoder_input + position_emb

        encoder_output = encoder_input
        for layer in self.Encoder:
            encoder_output = layer(
                encoder_output,
                key_padding_mask=(encoder_padding_mask.logical_not()) if encoder_padding_mask is not None else None,
            )

        encoder_output = self.ln_0(encoder_output)
        ########################### ENCODER PROCESSING STAGE END ###########################

        ########################### DECODER PROCESSING STAGE START ###########################
        decoder_input = self.input_embeddings(decoder_input)
        decoder_input = torch.cat(
            [
                self.input_pad_vector.repeat(decoder_input.shape[0], decoder_input.shape[1], 1, 1),
                decoder_input,
            ],
            dim=2,
        )
        decoder_input = self.input_dropout(decoder_input)
        dim_0, dim_1, dim_2, dim_3 = decoder_input.shape
        decoder_input = decoder_input.view(-1, dim_2, dim_3)
        decoder_input = self.input_encoder(decoder_input)
        decoder_input = decoder_input.view(dim_0, dim_1, dim_2, dim_3)
        decoder_input = decoder_input[:, :, 0, :]

        if additional_embedding is not None:
            if len(additional_embedding.shape) == 2:
                additional_embedding = additional_embedding.unsqueeze(1)
            decoder_input = torch.cat([decoder_input, additional_embedding], dim=1)
        ########################### DECODER PROCESSING STAGE END ###########################

        # TODO: создание этой маски можно вынести в буфер в конструктор
        decoder_seq_len = decoder_input.size(1)
        mask = ~torch.tril(
            torch.ones(decoder_seq_len, decoder_seq_len, device=decoder_input.device, dtype=torch.bool),
        )
        self_attention_attn_mask = torch.zeros_like(mask, dtype=decoder_input.dtype).masked_fill_(mask, float("-inf"))
        # увеличиваем внимание на первый элемент в декодер последовательности
        self_attention_attn_mask[..., 0] += 1

        decoder_output = decoder_input
        for layer in self.Decoder:
            decoder_output = layer(
                decoder_output,
                encoder_output,
                cross_attention_padding_mask=(~encoder_padding_mask) if encoder_padding_mask is not None else None,
                self_attention_mask=self_attention_attn_mask,
            )

        decoder_output = self.ln_1(decoder_output)

        output_values: List[torch.Tensor] = [decoder_output]
        output_names: List[str] = self.get_output_names()
        return dict(zip(output_names, output_values, strict=False))
