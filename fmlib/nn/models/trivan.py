from typing import Dict, List, Self

import torch

from fmlib.nn.blocks import BaseFFN, DecoderBlock, EncoderBlock
from fmlib.nn.models.heads import BaseHead
from fmlib.nn.utils.initialization import clone_module_to_moduledict, nba_init_weights

from .ivan import IvanModel


class Trivan(torch.nn.Module):
    """
    Расширенная трансформерная модель с адаптерами и несколькими выходными слоями.

    Модель состоит из:
    - Основного трансформерного модуля `IvanModel`.
    - Адаптеров, которые позволяют интегрировать эмбединги внешних моделей.
    - Выходных слоёв классификации для получения вероятностей по разным задачам.

    Args:
        vocab_size (int): Размер входного словаря.
        hidden_dim (int): Размерность скрытого состояния модели.
        positional_encoding_num_buckets (int): Количество бакетов для позиционного кодирования.
        num_encoder_layers (int): Количество слоёв в кодировщике.
        num_decoder_layers (int): Количество слоёв в декодировщике.
        dropout (float): Вероятность dropout для регуляризации.
        adapter_layer_names (List[str]): Список названий адаптеров.
        output_layer_names (List[str]): Список названий выходных слоёв классификации.
        adapter_ffn (BaseFFN): Экземпляр фид-форвардной сети для адаптеров.
            Экземпляр будет скопирован для каждого из адаптеров.
            Ко всем весам копий будет применен `torch.nn.init.xavier_normal_`.
        input_encoder_block (EncoderBlock): Экземпляр входного блок кодировщика.
            Ко всем весам экземпляра будет применен `torch.nn.init.xavier_normal_`.
        encoder_block (EncoderBlock): Экземпляр блока кодировщика.
            Экземпляр будет скопирован для каждого из слоев кодировщика.
            Ко всем весам копий будет применен `torch.nn.init.xavier_normal_`.
        decoder_block (DecoderBlock): Экземпляр блока декодировщика.
            Экземпляр будет скопирован для каждого из слоев декодировщика.
            Ко всем весам копий будет применен `torch.nn.init.xavier_normal_`.
        classification_head (BaseHead): Экземпляр головы классификации.
            Экземпляр будет скопирован для каждого из выходных слоёв.
            Ко всем весам копий будет применен `torch.nn.init.xavier_normal_`.

    Attributes:
        model (IvanModel): Основная трансформерная модель.
        adapter_layers (torch.nn.ModuleDict): Словарь адаптеров.
        output_layers (torch.nn.ModuleDict): Словарь выходных слоёв классификации.

    Methods:
        get_input_names: Возвращает список имен входных тензоров.
        get_output_names: Возвращает список имен выходных тензоров.
        get_dynamic_shapes: Возвращает динамические формы входных и выходных тензоров.
        forward: Функция прямого прохода через модель.
    """

    def __init__(
        self: Self,
        vocab_size: int,
        hidden_dim: int,
        positional_encoding_num_buckets: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        dropout: float,
        adapter_layer_names: List[str],
        output_layer_names: List[str],
        adapter_ffn: BaseFFN,
        input_encoder_block: EncoderBlock,
        encoder_block: EncoderBlock,
        decoder_block: DecoderBlock,
        classification_head: BaseHead,
    ):
        super().__init__()
        self.model = IvanModel(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            positional_encoding_num_buckets=positional_encoding_num_buckets,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            input_encoder_block=input_encoder_block,
            encoder_block=encoder_block,
            decoder_block=decoder_block,
        )
        self.adapter_layers = clone_module_to_moduledict(adapter_ffn, adapter_layer_names)
        self.output_layers = clone_module_to_moduledict(classification_head, output_layer_names)

        # Сохраняем инициализацию весов от оригинальной модели
        self.apply(nba_init_weights)

    def get_input_names(self: Self) -> List[str]:
        """Возвращает список имен входных тензоров модели."""
        return list(self.get_dynamic_shapes().keys())

    def get_output_names(self: Self) -> List[str]:
        """Возвращает список имен выходных тензоров модели."""
        return list(self.output_layers.keys())

    def get_dynamic_shapes(self: Self) -> Dict[str, Dict[int, str]]:
        """Описывает динамическую структуру тензоров на входе и выходе модели."""
        return {
            "encoder_input": {0: "batch_size", 1: "encoder_seq_len", 2: "event_count"},
            "decoder_input": {0: "batch_size", 1: "decoder_seq_len", 2: "event_count"},
            "positional_encoding": {0: "batch_size", 1: "encoder_seq_len"},
            "encoder_padding_mask": {0: "batch_size", 1: "encoder_seq_len"},
            "adapter_input": {name: {0: "batch_size", 1: "hidden_dim"} for name in self.adapter_layers},
        }

    def forward(
        self: Self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
        positional_encoding: torch.Tensor,
        encoder_padding_mask: torch.Tensor,
        adapter_input: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Функция прямого прохода через модель.

        Args:
            encoder_input (torch.Tensor): Входные данные для кодировщика.
            decoder_input (torch.Tensor): Входные данные для декодировщика.
            positional_encoding (torch.Tensor): Позиционное кодирование.
            encoder_padding_mask (torch.Tensor): Маска заполнения для кодировщика.
            adapter_input (Dict[str, torch.Tensor]): Входные данные для адаптеров.

        Returns:
            Dict[str, torch.Tensor]: Словарь выходных тензоров по ключам.
            К тензорам уже применен Softmax, поэтому их можно интерпретировать как вероятности принадлежности классам.
        """
        decoder_out = self.model(
            encoder_input=encoder_input,
            decoder_input=decoder_input,
            positional_encoding=positional_encoding,
            encoder_padding_mask=encoder_padding_mask,
        )

        decoder_out = decoder_out["decoder_output"][:, -1, :]

        additional_input = [decoder_out]
        for name in adapter_input:
            current_out = self.adapter_layers[name](adapter_input[name])
            additional_input.append(current_out)
        decoder_out = torch.stack(additional_input, dim=1)

        output = {}
        for key in self.output_layers:
            output[key] = self.output_layers[key](decoder_out)
        return output
