from typing import Self

import torch

from fmlib.constants.batches import GeneralBatch


class TrainingPipeline(torch.nn.Module):
    """
    Класс объединяюший трансформацию, модель и лосс.
    Применяемая трансформация зависит от флага `self.training`.

    В частности, при передаче параметра `training_transform` в конструкторе
    и модуле в режиме обучения будет использоваться `training_transform`,
    во всех иных случаях будет использоваться `transform`.

    Аргументы:
        model (torch.nn.Module): Модель для предсказания.
        losses (torch.nn.Module): Лосс для вычисления ошибки.
        eval_transform (torch.nn.Module): Трансформация для валидации и инференса.
            По умолчанию - `None`, не изменяет входные данные.
        training_transform (torch.nn.Module | None): Трансформация для тренировочных данных.
            По умолчанию - `None`, не изменяет входные данные.
            Может включать в себя маскирование, сдвиги и другие
            трансформации специфичные для *обучения* модели.

        Поля:
            model (torch.nn.Module): Модель для предсказания.
            losses (torch.nn.Module): Лосс для вычисления ошибки.
            eval_transform (torch.nn.Module | None): Трансформация для валидации.
            training_transform (torch.nn.Module | None): Трансформация для тренировочных данных.
    """

    def __init__(
        self: Self,
        model: torch.nn.Module,
        losses: torch.nn.Module,
        eval_transform: torch.nn.Module | None = None,
        training_transform: torch.nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.model: torch.nn.Module = model
        self.losses: torch.nn.Module = losses
        self.eval_transform: torch.nn.Module | None = eval_transform
        self.training_transform: torch.nn.Modul | None = training_transform

    @torch.jit.export
    def apply_transform(self: Self, batch: GeneralBatch) -> GeneralBatch:
        """
        Применяет трансформ и опционально переключается между трансформами
        для обучения и валидации.
        """
        if self.training:
            return self.apply_training_transform(batch)
        else:
            return self.apply_eval_transform(batch)

    @torch.jit.export
    def apply_eval_transform(self: Self, batch: GeneralBatch) -> GeneralBatch:
        """
        Применяет трансформ для валидации и инференса.
        """
        if self.eval_transform is None:
            return batch
        else:
            return self.eval_transform(batch)

    @torch.jit.export
    def apply_training_transform(self: Self, batch: GeneralBatch) -> GeneralBatch:
        """
        Применяет трансформ для тренировки.
        """
        if self.training_transform is None:
            return batch
        else:
            return self.training_transform(batch)

    def forward(self: Self, batch: GeneralBatch) -> tuple[GeneralBatch, GeneralBatch, torch.Tensor]:
        transformed: GeneralBatch = self.apply_transform(batch)
        outputs: GeneralBatch = self.model(**transformed)
        loss: torch.Tensor = self.losses(outputs, transformed)
        assert torch.numel(loss) == 1
        return (transformed, outputs, loss)
