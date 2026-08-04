from typing import Self

import torch

from fmlib.constants.batches import Batch
from fmlib.constants.losses import DEFAULT_TABULAR_OUTPUT, DEFAULT_TABULAR_TARGET


class WrappedLoss(torch.nn.Module):
    """
    Заворачивает стандартные лоссы в подходящую обёртку.

    Аргументы:
        loss_to_wrap (torch.nn.Module): Обёртываемый лосс.
        target_type (torch.dtype | None): Тип, к которому приводятся целевые значения.
            По умолчанию - `None`, то есть без конвертации типов.
            Нужен для многих классификационных лоссов, например CrossEntropy.
        output_name (str): Название выходного тензора модели.
            По умолчанию - `DEFAULT_TABULAR_OUTPUT`.
        target_name (str): Название тензора таргетов.
            По умолчанию - `DEFAULT_TABULAR_TARGET`.
    """

    def __init__(
        self: Self,
        loss_to_wrap: torch.nn.Module,
        target_type: torch.dtype | None = None,
        output_name: str = DEFAULT_TABULAR_OUTPUT,
        target_name: str = DEFAULT_TABULAR_TARGET,
    ) -> None:
        super().__init__()

        self.output_name: str = output_name
        self.target_name: str = target_name
        self.loss_to_wrap: torch.nn.Module = loss_to_wrap
        self.target_type: torch.dtype | None = target_type

    def forward(self: Self, outputs: Batch, targets: Batch) -> torch.Tensor:
        output: torch.Tensor = outputs[self.output_name]
        target: torch.Tensor = targets[self.target_name]

        if self.target_type is not None:
            target = target.to(dtype=self.target_type)

        result: torch.Tensor = self.loss_to_wrap(output, target.ravel())
        assert torch.numel(result) == 1
        return result
