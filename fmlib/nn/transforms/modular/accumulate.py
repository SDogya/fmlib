from typing import Iterable, Self

import torch

from fmlib.constants.batches import GeneralBatch
from fmlib.nn.transforms.utils.join_batches import join_batches


class Accumulate(torch.nn.Module):
    """
    Применяет переданные трансформы по очереди и совмещает результат c входами.

    Аргументы:
        modules (Iterable[torch.nn.Module]): Трансформы (модули), которые необходимо применить.
        verbose (bool): Выводить ли информацию о пересечении в именах.
            По умолчанию - `True`.
        strict (bool): Является ли пересечение имен выводов ошибкой.
            По умолчанию - `False`.

    *Примечание:* Отличается от `Map` тем, что последовательно расширяет входной батч.
    Эквивалентно последовательному:
    ```python
        Seqauential(
            ...,
            Map([
                FilterIn(),
                modules[i],
            ]),
            ...
        )
    ```
    """

    def __init__(self: Self, modules: Iterable[torch.nn.Module], verbose: bool = True, strict: bool = False) -> None:
        super().__init__()

        self.strict: bool = strict
        self.verbose: bool = verbose
        self.modules: torch.nn.ModuleList = torch.nn.ModuleList(modules)

    def join_batches(self: Self, left: GeneralBatch, right: GeneralBatch) -> GeneralBatch:
        return join_batches(left, right, strict=self.strict, verbose=self.verbose)

    def forward(self: Self, batch: GeneralBatch) -> GeneralBatch:
        result: GeneralBatch = {**batch}

        for module in self.modules:
            addition: GeneralBatch = module(batch)
            result = self.join_batches(result, addition)

        return result
