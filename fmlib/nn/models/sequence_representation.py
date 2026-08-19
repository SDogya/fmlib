import warnings
from typing import Any, Dict, List, Optional, Self, Tuple

import torch
from torch.export.dynamic_shapes import Dim
from transformers import PreTrainedModel

from fmlib.nn.blocks import BaseEncoderBlock
from fmlib.nn.embedding import BaseEventEmbedding, BasePositionalEmbedding, LinearEmbedding
from fmlib.nn.utils.agg import BaseAggregation
from fmlib.utils.named_adapters import OutputAdapter, validate_output_names


class SequenceRepresentationModel(torch.nn.Module):
    """
    Модель для предобучения на последовательностях событий.

    Эта модель предназначена для обработки временных последовательностей событий, представленных
    категориальными и числовыми признаками. Состоит из эмбеддингового слоя, блока кодирования признаков,
    трансформерной архитектуры, временного эмбеддинга и (опционально) позиционного эмбеддинга и слоя агрегации.

    Args:
        embedding (BaseEventEmbedding): Экземляр слоя эмбеддингов для событий.
        feature_encoder_block (BaseEncoderBlock): Экземпляр блок для кодирования признаков событий.
        transformer_model (PreTrainedModel): Предобученная трансформерная модель.
        time_embedding (LinearEmbedding): Экземпляр слоя линейного эмбеддинга для временных меток.
        positional_embedding (Optional[BasePositionalEmbedding]): Экземпляр необязательного слоя позиционных эмбеддингов.
        aggregation_layer (Optional[BaseAggregation]): Экземпляр необязательного слоя агрегации скрытых состояний.
        event_names (Tuple[str, ...]): Имена событий, используемых в модели.
            По умолчанию — ("evt_attr_9", "evt_attr_15", "channel_group", "sale_product_class_id").
        output_names (Tuple[str, ...]): Имена выходов модели. По умолчанию — ("logits",).

    Attributes:
        embedding (BaseEventEmbedding): Слой эмбеддингов событий.
        feature_encoder_block (BaseEncoderBlock): Блок кодирования признаков.
        transformer_model (PreTrainedModel): Трансформерная модель.
        time_embedding (LinearEmbedding): Временной эмбеддинг.
        positional_embedding (Optional[BasePositionalEmbedding]): Позиционный эмбеддинг.
        aggregation_layer (Optional[BaseAggregation]): Слой агрегации.
        event_names (Tuple[str, ...]): Имена событий.
        output_adapter (OutputAdapter): Адаптер выходов модели.
    """

    expected_output_count: int = 2

    def __init__(
        self: Self,
        embedding: BaseEventEmbedding,
        feature_encoder_block: BaseEncoderBlock,
        transformer_model: PreTrainedModel,
        time_embedding: LinearEmbedding,
        positional_embedding: Optional[BasePositionalEmbedding] = None,
        aggregation_layer: Optional[BaseAggregation] = None,
        event_names: Tuple[str, ...] = ("evt_attr_9", "evt_attr_15", "channel_group", "sale_product_class_id"),
        output_names: Tuple[str, ...] = ("logits", "last_hidden_state"),
    ) -> None:
        """
        Инициализирует модель для предобучения на последовательностях событий.

        Args:
            embedding (BaseEventEmbedding): Слой эмбеддингов.
            feature_encoder_block (BaseEncoderBlock): Блок кодирования признаков.
            transformer_model (PreTrainedModel): Трансформерная модель.
            time_embedding (LinearEmbedding): Слой временного эмбеддинга.
            positional_embedding (Optional[BasePositionalEmbedding]): Слой позиционных эмбеддингов.
            aggregation_layer (Optional[BaseAggregation]): Слой агрегации.
            event_names (Tuple[str, ...]): Имена событий.
            output_names (Tuple[str, ...]): Имена выходов модели.
        """
        super().__init__()
        self.embedding = embedding
        self.feature_encoder_block = feature_encoder_block
        self.transformer_model = transformer_model
        self.time_embedding = time_embedding
        self.positional_embedding = positional_embedding
        self.aggregation_layer = aggregation_layer

        self.event_names = event_names
        output_names = validate_output_names(output_names, self.expected_output_count)
        self.output_adapter: OutputAdapter = OutputAdapter(output_names)

    def get_input_names(self: Self) -> List[str]:
        """
        Возвращает список имён входных тензоров модели.
        Метод необходим для конвертации в ONNX.

        Returns:
            List[str]: Список имён входов.
        """
        return list(self.get_dynamic_shapes().keys())

    def get_output_names(self: Self) -> List[str]:
        """
        Возвращает список имён выходных тензоров модели.
        Метод необходим для конвертации в ONNX.

        Returns:
            List[str]: Список имён выходов.
        """
        return self.output_adapter.get_output_names()

    def get_dynamic_shapes(self: Self) -> Dict[str, Any]:
        """
        Возвращает информацию о динамических размерностях входных тензоров.
        Метод необходим для конвертации в ONNX.

        Returns:
            Dict[str, Any]: Словарь с описанием размерностей.
        """
        feature_dim = len(self.event_names) + 1
        seq_len_dim = Dim.STATIC
        events = {event: {0: Dim.AUTO, 1: seq_len_dim} for event in self.event_names}
        events["event_encoder_attention_mask"] = {0: Dim.AUTO, 1: seq_len_dim, 2: feature_dim, 3: feature_dim}
        events["padding_mask"] = {0: Dim.AUTO, 1: seq_len_dim}
        events["positional_timestamps"] = {0: Dim.AUTO, 1: seq_len_dim}
        events["encoding_timestamps"] = {0: Dim.AUTO, 1: seq_len_dim}
        return {"events": events}

    def forward(self: Self, events: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Прямой проход через модель.

        Args:
            events (Dict[str, torch.Tensor]): Словарь тензоров событий, включающий:
                - event_encoder_attention_mask
                - padding_mask
                - positional_timestamps
                - encoding_timestamps

        Returns:
            Dict[str, torch.Tensor]: Множество результатов работы (предсказаний) модели.
        """
        # Временное решение проблемы с компиляцией
        # TODO: исправить компиляцию и изменить формат входных данных в функцию
        event_encoder_attention_mask = events["event_encoder_attention_mask"]
        padding_mask = events["padding_mask"]
        positional_timestamps = events["positional_timestamps"]
        encoding_timestamps = events["encoding_timestamps"]

        embeds = self.embedding(events)  # (batch_size, seq_len, n_features, emb_dim)

        if self.positional_embedding is None:
            warnings.warn("Positional embedding layer is not given, but positional timestamps are provided.", stacklevel=2)

        if self.positional_embedding is not None:
            embeds += self.positional_embedding(positional_timestamps).unsqueeze(2)

        if encoding_timestamps.ndim != 2:
            msg: str = f"Encoding timestamps should be of shape (batch_size, seq_len), but got {encoding_timestamps.shape}"
            raise ValueError(msg)
        time_embeds = self.time_embedding(encoding_timestamps.unsqueeze(-1))

        embeds = torch.cat((embeds, time_embeds), dim=-2)  # (batch_size, seq_len, n_features + 1, emb_dim)
        dim_0, dim_1, dim_2, dim_3 = embeds.shape
        embeds = embeds.view(-1, dim_2, dim_3)
        event_encoder_attention_mask = event_encoder_attention_mask.view(-1, dim_2, dim_2)
        inputs_embeds = self.feature_encoder_block(
            x=embeds,
            attn_mask=event_encoder_attention_mask.logical_not(),
        )  # (batch_size * seq_len, emb_dim)

        inputs_embeds = inputs_embeds.view(dim_0, dim_1, dim_3)

        output: Dict[str, torch.Tensor] = self.transformer_model(
            inputs_embeds=inputs_embeds,
            attention_mask=padding_mask,
            output_hidden_states=False,
            return_dict=True,
        )
        last_hidden_state = output["last_hidden_state"]  # (batch_size, seq_len, emb_dim)

        if self.aggregation_layer is not None:
            output: torch.Tensor = self.aggregation_layer(last_hidden_state, padding_mask)  # (batch_size, emb_dim)
        else:
            output = last_hidden_state

        output_names: List[str] = self.get_output_names()
        output_values: List[torch.Tensor] = [output, last_hidden_state]
        return dict(zip(output_names, output_values, strict=False))
