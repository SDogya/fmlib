from typing import Self

import torch

from fmlib.constants.batches import Batch
from fmlib.nn.utils.initialization import InitFn, clone_module_to_moduledict


class HeadEnsemble(torch.nn.Module):
    """
    Множество классификационных голов.
    Принимает эмбеддинг и возвращает логиты в виде словаря.

    *Примечание:* Изначально предназначается для работы вместе с DLT моделью.

    Аргументы:
        output_layer_template (torch.nn.Module): Шаблон для классификационных голов.
        output_layers (list[str] | None): Список имен классификационных голов.
            По умолчанию - `None`, что соотвествует одной голове - `dummy`.
        init (str | InitFn): Инициализация для классификационных голов.
            По умолчанию - `nba`.
    """

    def __init__(
        self: Self, output_layer_template: torch.nn.Module, output_layers: list[str] | None = None, init: str | InitFn = "nba"
    ) -> None:
        if output_layers is None:
            output_layers = ["dummy"]

        super().__init__()

        if (actual_length := len(output_layers)) != (unique_len := len(set(output_layers))):
            msg: str = f"Output layers must be unique. Got {unique_len=} unique vs. {actual_length=} total."
            raise ValueError(msg)

        self.output_names: list[str] = sorted(output_layers)

        self.output_layers: torch.nn.ModuleDict = clone_module_to_moduledict(
            layer_names=self.output_names,
            module=output_layer_template,
            init=init,
        )

    def forward(self: Self, x: torch.Tensor, **kwargs) -> Batch:
        result: Batch = {}

        for name in self.output_names:
            result[name] = self.output_layers[name](x)

        return result
