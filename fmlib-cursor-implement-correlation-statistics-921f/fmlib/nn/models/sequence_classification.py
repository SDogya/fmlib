from typing import Any, Dict, List, Self

import torch

from fmlib.nn.models.heads.base import BaseHead
from fmlib.nn.models.sequence_representation import SequenceRepresentationModel


class SequenceClassificationModel(torch.nn.Module):
    """
    Модель классификации/регрессии на основе SequenceRepresentationModel.

    Оборачивает тело модели и добавляет голову для предсказания таргета из агрегированного
    скрытого состояния последовательности. Тело должно содержать aggregation_layer, чтобы
    свернуть последовательность (batch_size, seq_len, hidden_size) → (batch_size, hidden_size)
    перед подачей в голову.

    Args:
        body (SequenceRepresentationModel): Тело модели с обязательным aggregation_layer.
        head (BaseHead): Голова (MLP) для предсказания таргета из агрегированного представления.
        output_name (str): Имя выходного тензора в возвращаемом словаре. По умолчанию — "logits".

    Raises:
        ValueError: Если у body нет aggregation_layer.
    """

    def __init__(
        self: Self,
        body: SequenceRepresentationModel,
        head: BaseHead,
        output_name: str = "logits",
    ) -> None:
        super().__init__()
        if body.aggregation_layer is None:
            msg = (
                "SequenceRepresentationModel must have an aggregation_layer to produce a "
                "fixed-size representation for classification. Pass one of the aggregation "
                "classes from fmlib.nn.utils.agg (e.g. MeanHiddenState, SumLayerNorm)."
            )
            raise ValueError(msg)
        self.body = body
        self.head = head
        self.output_name = output_name
        self._body_repr_name: str = body.get_output_names()[0]

    def get_input_names(self: Self) -> List[str]:
        return self.body.get_input_names()

    def get_output_names(self: Self) -> List[str]:
        return [self.output_name]

    def get_dynamic_shapes(self: Self) -> Dict[str, Any]:
        return self.body.get_dynamic_shapes()

    def forward(self: Self, events: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Args:
            events (Dict[str, torch.Tensor]): Словарь тензоров событий (см. SequenceRepresentationModel.forward).

        Returns:
            Dict[str, torch.Tensor]: Словарь с ключом output_name и тензором логитов (batch_size, num_classes).
        """
        body_outputs: Dict[str, torch.Tensor] = self.body(events=events, **kwargs)
        aggregated: torch.Tensor = body_outputs[self._body_repr_name]  # (batch_size, hidden_size)
        logits: torch.Tensor = self.head(aggregated)  # (batch_size, num_classes)
        return {self.output_name: logits}
