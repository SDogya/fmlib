import warnings
from typing import Callable, Dict, Self, Tuple

import torch

from fmlib.constants.batches import GeneralBatch
from fmlib.nn.transforms.modular.filtering import apply_prefix

MappingType = Dict[str, str] | Callable[[str], str | None]


class Rename(torch.nn.Module):
    """
    Трансформ для переименования колонок.

    Аргументы:
        mapping (MappingType): Маппинг имен колонок. Это может быть:
            - `dict` с маппингом имен колонок;
            - `callable` с маппингом имен колонок.
        bypass (bool): Если `True`, то пропускать коллонки входного батча,
            если `False` - оригинальный батч возвращаться не будет. По умолчанию - `True`.
        clone (bool): Если `True`, то клонировать тензоры. Может замедлять трансформ.
            По умолчанию - `False`.
        strict (bool): Если `True`, то выбрасывать исключение, если типы не соответствуют.
            По умолчанию - `False`.
        verbose (bool): Если `True`, то выводить предупреждения.
            По умолчанию - `True`.
    """

    def __init__(
        self: Self, mapping: MappingType, bypass: bool = True, clone: bool = False, strict: bool = False, verbose: bool = True
    ) -> None:
        super().__init__()

        self.clone: bool = clone
        self.bypass: bool = bypass
        self.strict: bool = strict
        self.verbose: bool = verbose
        self.mapping: MappingType = mapping

    def clone_or_pass(self: Self, value: GeneralBatch | torch.Tensor) -> GeneralBatch | torch.Tensor:
        if self.clone:
            if torch.is_tensor(value):
                return value.clone()
            elif isinstance(value, dict):
                return {k: self.clone_or_pass(v) for k, v in value.items()}
            else:
                msg: str = f"Cannot clone {type(value)=}. It is neither `Tensor` nor `dict`."
                raise ValueError(msg)
        else:
            return value

    def apply_mapping(self: Self, key: str) -> str | None:
        if isinstance(self.mapping, dict):
            if key in self.mapping:
                return self.mapping[key]
            else:
                return None
        else:
            return self.mapping(key)

    def rename(self: Self, batch: GeneralBatch, prefix: str | None = None) -> Tuple[GeneralBatch, bool]:
        result: GeneralBatch = {}

        renamed_any: bool = False
        for key in sorted(batch.keys()):
            prefixed: str = apply_prefix(key, prefix)
            mapping: str | None = self.apply_mapping(prefixed)
            value: GeneralBatch | torch.Tensor = batch[key]

            curr_renamed: bool = False
            curr_result: GeneralBatch | torch.Tensor = value
            if isinstance(value, dict):
                curr_result, curr_renamed = self.rename(value, prefixed)
            elif torch.is_tensor(value):
                curr_result = self.clone_or_pass(value)
            else:
                msg: str = f"Value is neither `Tensor` nor `dict`. Got: {type(value)=}."
                if self.strict:
                    raise TypeError(msg)
                elif self.verbose:
                    warnings.warn(msg, stacklevel=2)

            if mapping is None:
                if self.bypass or curr_renamed:
                    result[key] = curr_result
            else:
                curr_renamed = True
                result[mapping] = curr_result

            renamed_any = renamed_any or curr_renamed

        return (result, renamed_any)

    def forward(self: Self, batch: GeneralBatch) -> GeneralBatch:
        result, _ = self.rename(batch)
        return result
