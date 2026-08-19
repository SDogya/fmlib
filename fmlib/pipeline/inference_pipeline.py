from typing import Self, Tuple

import torch

from fmlib.constants.batches import Batch, GeneralBatch


class InferencePipeline(torch.nn.Module):
    """
    Класс, реализующий инференс-пайплайн на основе модели PyTorch.

    Позволяет использовать модуль преобразования (transform) и основную модель.
    Поддерживает как одиночный, так и множественный режимы применения transform.
    """

    def __init__(
        self: Self,
        model: torch.nn.Module,
        transform: torch.nn.Module | list[torch.nn.Module] | None = None,
    ) -> None:
        """
        Инициализация объекта InferencePipeline.

        Аргументы:
            model (torch.nn.Module): Основная модель, используемая для инференса.
            transform (torch.nn.Module | List[torch.nn.Module] | None):
                Модуль или список модулей для преобразования входных данных.
                По умолчанию: None.
        """
        super().__init__()
        self.model: torch.nn.Module = model
        self.transform: torch.nn.Module | torch.nn.ModuleList[torch.nn.Module] | None
        if isinstance(transform, list):
            self.transform = torch.nn.ModuleList(transform)
        elif transform is None:
            self.transform = None
        else:
            self.transform = torch.nn.ModuleList([transform])

    def forward(
        self: Self,
        batch: Batch,
    ) -> Tuple[list[GeneralBatch], list[Batch]]:
        """
        Выполняет полный инференс-пайплайн.

        Аргументы:
            batch (Dict[str, torch.Tensor]): Входной батч в виде словаря тензоров.

        Возвращает:
            Tuple[list[GeneralBatch], list[Batch]]:
                - Результаты после применения transform (или оригинальные данные).
                - Результаты после применения модели.
        """
        transform_result: list[GeneralBatch] = []
        model_result: list[Batch] = []
        for transform in self.transform:
            transformed_batch = transform(batch)
            transform_result.append(transformed_batch)
            model_result.append(self.model(**transformed_batch))
        return transform_result, model_result
