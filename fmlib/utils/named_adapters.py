import math
import warnings
from typing import Dict, List, Self, Sequence

import torch


def validate_output_names(
    output_names: list[str], expected_output_count: int, strict: bool = False, output_name_template: str | None = None
) -> list[str]:
    """
    Проверяет и нормализует список имен выходов модели.
    Если их не хватает - дополняет, если не хватает - удаляет.

    Аргументы:
        output_names (list[str]): Список имен выходов модели.
        expected_output_count (int): Ожидаемое количество выходов модели.
        strict (bool): Определяет режим обработки несоответствий: ошибка или warning. По умолчанию - `False`.
        output_name_template (str | None): Шаблон для имен выходов модели, если их не хватает. По умолчанию - `None`.
        **Примечание:** При создании i-того вывода будет сделано `output_name_template.format(i=i)`.
    """

    def report(msg: str) -> None:
        if strict:
            raise ValueError(msg)
        else:
            warnings.warn(msg, stacklevel=2)

    if output_name_template is None:
        length: int = math.floor(math.log10(expected_output_count) + 1)
        output_name_template = "generic_output_{i:0" + str(length) + "d}"

    def generic_name(i: int) -> str:
        return output_name_template.format(i=i)

    if len(output_names) < expected_output_count:
        added_output_names: list[str] = [generic_name(i) for i in range(len(output_names), expected_output_count)]
        msg: str = (
            "Number of outputs is less then expected. List of outputs will be extended. "
            f"Got: {len(output_names)=} vs. {expected_output_count=}; added: {added_output_names=}."
        )
        report(msg)

        output_names = output_names + added_output_names
    elif expected_output_count < len(output_names):
        removed_output_names: list[str] = output_names[expected_output_count:]
        msg: str = (
            "Number of outputs is greater then expected. List of outputs will be shortened. "
            f"Got: {len(output_names)=} vs. {expected_output_count=}; removed: {removed_output_names=}."
        )
        report(msg)

        output_names = output_names[:expected_output_count]
    assert len(output_names) == expected_output_count
    return output_names


class InputAdapter:
    """
    TODO: add docstring
    """

    def __init__(self: Self, input_names: Sequence[str]) -> None:
        self.input_names: List[str] = list(input_names)

    def get_input_names(self: Self) -> List[str]:
        return list(self.input_names)

    def __call__(self: Self, *args: torch.Tensor, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:
        result: Dict[str, torch.Tensor] = dict(zip(self.input_names, args, strict=False))
        result: Dict[str, torch.Tensor] = {**result, **kwargs}

        return result


class OutputAdapter:
    """
    TODO: add docstring
    """

    def __init__(self: Self, output_names: Sequence[str]) -> None:
        self.output_names: List[str] = list(output_names)

    def get_output_names(self: Self) -> List[str]:
        return list(self.output_names)

    def __call__(self: Self, **kwargs: torch.Tensor) -> List[torch.Tensor]:
        def _is_in(name: str) -> bool:
            return name in kwargs

        assert all(map(_is_in, self.output_names))

        result: List[torch.Tensor] = [kwargs[n] for n in self.output_names]

        assert len(result) == len(self.output_names)
        return result
