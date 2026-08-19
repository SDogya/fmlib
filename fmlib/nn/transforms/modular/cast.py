import warnings
from typing import Callable, Dict, Self, Tuple

import torch

from fmlib.constants.batches import GeneralBatch
from fmlib.nn.transforms.modular.filtering import apply_prefix

CastingType = Dict[str, torch.dtype] | Callable[[str], torch.dtype | None]


class Cast(torch.nn.Module):
    """
    Приведение к типу выбранных колонок.

    Аргументы:
        casting (CastType): Маппинг приведения типов. Возможные значения:
            - Dict[str, torch.dtype] - прямой маппинг;
            - Callable[[str], torch.dtype | None] - маппинг по функции.
        bypass (bool): Пропускать ли неизменяемые колонки.
            По умолчанию - `True`.
        strict (bool): Вызывать ли исключение при некорректном типе.
        verbose (bool): Выводить ли предупреждения при некорректном типе.
    """

    def __init__(self: Self, casting: CastingType, bypass: bool = True, strict: bool = False, verbose: bool = True) -> None:
        super().__init__()

        self.bypass: bool = bypass
        self.strict: bool = strict
        self.verbose: bool = verbose
        self.casting: CastingType = casting

    def apply_casting(self: Self, key: str) -> torch.dtype | None:
        if isinstance(self.casting, dict):
            if key in self.casting:
                return self.casting[key]
            else:
                return None
        else:
            return self.casting(key)

    def cast(self: Self, batch: GeneralBatch, prefix: str | None = None) -> Tuple[GeneralBatch, bool]:
        result: GeneralBatch = {}

        casted_any: bool = False
        for key in sorted(batch.keys()):
            prefixed: str = apply_prefix(key, prefix)
            casting: torch.dtype | None = self.apply_casting(prefixed)
            value: GeneralBatch | torch.Tensor = batch[key]

            curr_casted: bool = False
            curr_result: GeneralBatch | torch.Tensor = value
            if isinstance(value, dict):
                curr_result, curr_casted = self.cast(value, prefixed)
            elif not torch.is_tensor(value):
                msg: str = f"Value is neither `Tensor` nor `dict`. Got: {type(value)=}."
                if self.strict:
                    raise TypeError(msg)
                elif self.verbose:
                    warnings.warn(msg, stacklevel=2)

            if casting is None:
                if self.bypass or curr_casted:
                    result[key] = curr_result
            else:
                curr_casted = True
                assert isinstance(casting, torch.dtype)
                result[key] = curr_result.to(dtype=casting)

            casted_any = casted_any or curr_casted

        return (result, casted_any)

    def forward(self: Self, batch: GeneralBatch) -> GeneralBatch:
        result, _ = self.cast(batch)
        return result
