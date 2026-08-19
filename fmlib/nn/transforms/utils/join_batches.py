import warnings

import torch

from fmlib.constants.batches import GeneralBatch


def join_batches(left: GeneralBatch, right: GeneralBatch, strict: bool = False, verbose: bool = True) -> GeneralBatch:
    """
    Функция для "слития" двух батчей.

    Аргументы:
        left (GeneralBatch): "Левый" батч. Имеет низкий приоритет.
        right (GeneralBatch): "Правый" батч. Имеет более высокий приоритет.
        strict (bool, optional): Нужно ли выбрасывать ошибки при  конфликтах значений. По умолчанию `False`.
        verbose (bool, optional): Нужно ли сигнализировать о конфликтах значений. По умолчанию `True`.
    """
    result: GeneralBatch = {name: left[name] for name in sorted(left.keys())}

    for name in sorted(right.keys()):
        value: torch.Tensor | GeneralBatch = right[name]
        if name in result:
            if not isinstance(value, type(result[name])):
                msg: str = f"Different types for the same name. Got: {name=}:{type(result[name])=} vs {type(value)=}."
                if strict:
                    raise TypeError(msg)
                elif verbose:
                    warnings.warn(msg, stacklevel=2)
            if isinstance(value, dict):
                result[name] = join_batches(
                    left=result[name],
                    right=value,
                    strict=strict,
                    verbose=verbose,
                )
            elif torch.is_tensor(value):
                if id(result[name]) != id(value):
                    msg: str = f"There are different tensors sharing the same name: {name=}."
                    if strict:
                        raise ValueError(msg)
                    elif verbose:
                        warnings.warn(msg, stacklevel=2)
                result[name] = value
            else:
                msg: str = f"Value is neither `dict` nor `torch.Tensor`, got: {name=}:{type(value)=}."
                raise TypeError(msg)
        else:
            result[name] = value
    return result
