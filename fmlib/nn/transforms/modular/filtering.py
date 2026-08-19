import re
from typing import List, Self, Tuple

import torch

from fmlib.constants.batches import GeneralBatch

PatternType = str | re.Pattern
PatternInputeType = PatternType | List[PatternType] | Tuple[PatternType]


def normalize_pattern(pattern: PatternType) -> re.Pattern:
    result: re.Pattern
    if isinstance(pattern, str):
        result = re.compile(pattern)
    elif isinstance(pattern, re.Pattern):
        result = pattern
    else:
        msg: str = f"Pattern is neither `str` nor `re.Pattern`: {pattern=}"
        raise TypeError(msg)
    assert isinstance(result, re.Pattern)
    return result


def normalize_many_patterns(patterns: PatternInputeType) -> List[re.Pattern]:
    result: List[re.Pattern]
    if isinstance(patterns, (list, tuple)):
        result = [normalize_pattern(pattern) for pattern in patterns]
    else:
        result = [normalize_pattern(patterns)]
    assert all(isinstance(p, re.Pattern) for p in result)
    return result


def apply_prefix(name: str, prefix: str | None = None) -> str:
    if prefix is None:
        return name
    else:
        return f"{prefix}_{name}"


def fit_patterns(name: str, patterns: List[re.Pattern]) -> bool:
    return any(pattern.match(name) for pattern in patterns)


class FilteringIn(torch.nn.Module):
    """
    Трансформ, пропускеающий только колонки, соответствующие паттернам.

    Аргументы:
        patterns (str|list[str] | Pattern | list[Pattern]): Паттерны, которые нужно пропустить.
        Могут быть:
            - Строкой или списком строк, точно подходящих под имя колонки.
            - Регулярным выражением или списком выражений под имя колонки.
        По умолчанию - ".*" (пропускает всё).
    """

    def __init__(self: Self, patterns: PatternInputeType = ".*") -> None:
        super().__init__()

        self.patterns: List[re.Pattern] = normalize_many_patterns(patterns)

    def filter_plain(self: Self, batch: GeneralBatch, prefix: str | None = None) -> GeneralBatch:
        result: GeneralBatch = {}
        for name, value in batch.items():
            full_name: str = apply_prefix(name, prefix)
            if torch.is_tensor(value):
                if fit_patterns(full_name, self.patterns):
                    result[name] = value
            elif isinstance(value, dict):
                result[name] = self.filter_plain(value, full_name)
            else:
                msg: str = f"Value is neither `dict` nor `torch.Tensor`: {name=}:{type(value)=}."
                raise ValueError(msg)
        return result

    def forward(self: Self, batch: GeneralBatch) -> GeneralBatch:
        result: GeneralBatch = self.filter_plain(batch)
        return result


class FilteringOut(torch.nn.Module):
    """
    Трансформ, пропускеающий только колонки, соответствующие паттернам.

    Аргументы:
        patterns (str|list[str] | Pattern | list[Pattern] | None): Паттерны, которые нужно не пропускать.
        Могут быть:
            - Строкой или списком строк, точно подходящих под имя колонки.
            - Регулярным выражением или списком выражений под имя колонки.
        По умолчанию - None (пропускает всё).
    """

    def __init__(self: Self, patterns: PatternInputeType | None = None) -> None:
        super().__init__()

        if patterns is None:
            patterns = []

        self.patterns: List[re.Pattern] = normalize_many_patterns(patterns)

    def filter_plain(self: Self, batch: GeneralBatch, prefix: str | None = None) -> GeneralBatch:
        result: GeneralBatch = {}
        for name, value in batch.items():
            full_name: str = apply_prefix(name, prefix)
            if torch.is_tensor(value):
                if not fit_patterns(full_name, self.patterns):
                    result[name] = value
            elif isinstance(value, dict):
                result[name] = self.filter_plain(value, full_name)
            else:
                msg: str = f"Value is neither `dict` nor `torch.Tensor`: {name=}:{type(value)=}."
                raise ValueError(msg)
        return result

    def forward(self: Self, batch: GeneralBatch) -> GeneralBatch:
        return self.filter_plain(batch)
