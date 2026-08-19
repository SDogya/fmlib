from typing import Self

import torch

from fmlib.constants.batches import Batch
from fmlib.constants.losses import (
    DEFAULT_SLEARNER_CONTROL_OUTPUT,
    DEFAULT_SLEARNER_GROUP,
    DEFAULT_SLEARNER_TARGET,
    DEFAULT_SLEARNER_TREATMENT_OUTPUT,
)


class SLearnerLoss(torch.nn.Module):
    """
    Лосс специфичный для обучения Uplift модели в подходе SLearner.

    Аргументы:
        loss_to_wrap (torch.nn.Module): Loss, которую нужно обернуть.
            Как правило - это BCELoss.
        invert_group (bool): Инвертировать ли группу.
            По умолчанию - `True`, т.е. контрольная группа - 1, целевая  - 0.
        weight_groups (bool): Взвесить ли группы.
            По умолчанию - `True`.
            Группы могут быть несбалансированы и в таком случае нужно взвесить их
            согласно количеству семплов, относящемуся к каждому.
        target_type (torch.dtype | None): Тип, к которому приводить таргеты.
            По умолчанию - `None`, т.е. не проводить конвертацию.
        group_name (str): Название тензора флагов группы.
            По умолчанию - `DEFAULT_SLEARNER_GROUP`.
        target_name (str): Название тензора таргетов.
            По умолчанию - `DEFAULT_SLEARNER_TARGET`.
        control_output_name (str): Название выходного тензора control модели.
            По умолчанию - `DEFAULT_SLEARNER_CONTROL_OUTPUT`.
        treatment_output_name (str): Название выходного тензора treatment модели.
            По умолчанию - `DEFAULT_SLEARNER_TREATMENT_OUTPUT`.
    """

    def __init__(
        self,
        loss_to_wrap: torch.nn.Module,
        invert_group: bool = True,
        weight_groups: bool = True,
        target_type: torch.dtype | None = None,
        group_name: str = DEFAULT_SLEARNER_GROUP,
        target_name: str = DEFAULT_SLEARNER_TARGET,
        control_output_name: str = DEFAULT_SLEARNER_CONTROL_OUTPUT,
        treatment_output_name: str = DEFAULT_SLEARNER_TREATMENT_OUTPUT,
    ) -> None:
        super().__init__()
        self.target_type: torch.dtype | None = target_type
        self.loss_to_wrap: torch.nn.Module = loss_to_wrap
        self.weight_groups: bool = weight_groups
        self.invert_group: bool = invert_group

        self.group_name: str = group_name
        self.target_name: str = target_name
        self.control_output_name: str = control_output_name
        self.treatment_output_name: str = treatment_output_name

    def apply_loss(self: Self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        assert torch.numel(outputs) == torch.numel(targets)

        if torch.numel(outputs) == 0:
            return torch.tensor(0.0).to(device=outputs.device)

        if self.target_type is not None:
            targets = targets.to(dtype=self.target_type)

        return self.loss_to_wrap(outputs, targets)

    def forward(self: Self, outputs: Batch, targets: Batch) -> torch.Tensor:
        groups: torch.BoolTensor = targets[self.group_name].bool()
        if self.invert_group:
            groups = ~groups

        control_outputs: torch.Tensor = outputs[self.control_output_name][~groups]
        treatment_outputs: torch.Tensor = outputs[self.treatment_output_name][groups]

        control_targets: torch.Tensor = targets[self.target_name][~groups]
        treatment_targets: torch.Tensor = targets[self.target_name][groups]

        assert torch.numel(control_outputs) == torch.numel(control_targets)
        assert torch.numel(treatment_outputs) == torch.numel(treatment_targets)

        control_loss: torch.Tensor = self.apply_loss(control_outputs, control_targets)
        treatment_loss: torch.Tensor = self.apply_loss(treatment_outputs, treatment_targets)

        control_weight, treatment_weight = 0.5, 0.5
        if self.weight_groups:
            count: int = torch.numel(groups)
            control_count: int = torch.numel(control_outputs)
            treatment_count: int = torch.numel(treatment_outputs)
            assert (control_count + treatment_count) == count

            # Перевзвешивание.
            # Рассмотрим на примере control:
            # - больше treatment, меньше control -> control - более приоритетно
            # - больше control, меньше treatment -> treatment - более приоритетно
            # То же верно и для treatment.
            control_weight = treatment_count / count
            treatment_weight = control_count / count

        return (control_loss * control_weight) + (treatment_loss * treatment_weight)
