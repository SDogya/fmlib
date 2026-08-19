from typing import Literal, Self

import torch
import torch.nn.functional as func

from fmlib.constants.batches import Batch
from fmlib.constants.io import DEFAULT_MAKE_MASK_NAME
from fmlib.constants.losses import DEFAULT_TABULAR_OUTPUT, DEFAULT_TABULAR_TARGET
from fmlib.constants.metadata import DEFAULT_PADDING


class CELoss(torch.nn.Module):
    """
    Готовый CELoss.

    Аргументы:
        target_type (torch.dtype | None): Тип, к которому приводятся целевые значения.
            По умолчанию - `None`, то есть без конвертации типов.
            Нужен для многих классификационных лоссов, например CrossEntropy.
        output_name (str): Название выходного тензора модели.
            По умолчанию - `DEFAULT_TABULAR_OUTPUT`.
        target_name (str): Название тензора таргетов.
            По умолчанию - `DEFAULT_TABULAR_TARGET`.
        target_mask_name (str | None): Название тензора масок таргетов.
            По умолчанию - `None`, т.е. будет получена применением
            к `target_name` функции `DEFAULT_MAKE_MASK_NAME`.
        ignore_index (int): Индекс, который игнорируется при подсчете лосса.
            По умолчанию - `DEFAULT_PADDING`.
        reduction (Literal["sum", "mean"]): Тип редукции.
            По умолчанию - "mean".
        other_kwargs (dict | None): Другие аргументы для `func.cross_entropy`.
            По умолчанию - `None`, т.е. никаких доп. параметров.
    """

    def __init__(
        self: Self,
        target_type: torch.dtype | None = None,
        output_name: str = DEFAULT_TABULAR_OUTPUT,
        target_name: str = DEFAULT_TABULAR_TARGET,
        target_mask_name: str | None = None,
        ignore_index: int = DEFAULT_PADDING,
        reduction: Literal["sum", "mean"] = "mean",
        other_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        if other_kwargs is None:
            other_kwargs = {}

        if target_mask_name is None:
            make_mask_name = DEFAULT_MAKE_MASK_NAME
            target_mask_name = make_mask_name(target_name)

        if reduction not in {"sum", "mean"}:
            msg: str = f"Reduction type is not supported. Got: {reduction}."
            raise ValueError(msg)

        self.other_kwargs: dict = other_kwargs
        self.reduction: str = reduction
        self.ignore_index: int = ignore_index
        self.output_name: str = output_name
        self.target_name: str = target_name
        self.target_name_mask: str = target_mask_name
        self.target_type: torch.dtype | None = target_type

    def forward(self: Self, outputs: Batch, targets: Batch) -> torch.Tensor:
        output: torch.Tensor = outputs[self.output_name]
        target: torch.Tensor = targets[self.target_name]

        if self.target_name_mask in targets:
            mask = targets[self.target_name_mask].float()
        else:
            mask = torch.ones_like(target, dtype=torch.float32)

        if self.target_type is not None:
            target = target.to(dtype=self.target_type)

        raw_loss: torch.Tensor = func.cross_entropy(
            input=output,
            target=target,
            reduction="none",
            ignore_index=self.ignore_index,
            **self.other_kwargs,
        )

        if self.reduction == "sum":
            result: torch.Tensor = torch.sum(raw_loss * mask.float())
        elif self.reduction == "mean":
            result: torch.Tensor = torch.mean(raw_loss * mask.float())
        else:
            raise ValueError()

        assert torch.numel(result) == 1
        return result
