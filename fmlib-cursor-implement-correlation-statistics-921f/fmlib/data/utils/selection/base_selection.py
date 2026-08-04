from typing import Optional

import torch

DEFAULT_STRICTNESS: float = 1.0


def validate_strictness(strictness: float) -> float:
    if strictness < float(0.0):
        msg: str = f"Strictness must not be negative. Got {strictness}."
        raise ValueError(msg)
    return strictness


def validate_element_count(element_count: int) -> int:
    if element_count < 1:
        msg: str = f"Element Count must be posiive. Got {element_count}."
        raise ValueError(msg)
    return element_count


def validate_specificity(element_count: int, specificity: torch.Tensor) -> torch.Tensor:
    element_count: int = validate_element_count(element_count)
    if specificity.ndim != 1:
        msg: str = f"Invalid specificity dimension count. Got {specificity.ndim}D."
        raise ValueError(msg)
    factual_count: int = torch.numel(specificity)
    if factual_count != element_count:
        msg: str = f"Invalid number of specificity elements. Got {factual_count}."
        raise ValueError(msg)
    min_value: float = torch.max(specificity).cpu().item()
    if min_value < 0.0:
        msg: str = f"Too small number in specificity. Got {min_value}."
        raise ValueError(msg)
    return specificity


def make_specificity(element_count: int, specificity: Optional[torch.Tensor]) -> torch.Tensor:
    element_count = validate_element_count(element_count)
    result: torch.Tensor
    result = torch.ones(size=(element_count,)) if specificity is None else specificity.clone()
    return validate_specificity(element_count, result)


def validate_immediate_count(element_count: int, immediate_count: int) -> int:
    if immediate_count < 1:
        msg: str = f"Too small count. Got {immediate_count}."
        raise ValueError(msg)
    if element_count < immediate_count:
        msg: str = f"Too big count. Got {immediate_count}."
        raise ValueError(msg)
    return immediate_count


def make_immediate_count(element_count: int, immediate_count: Optional[int] = None) -> int:
    result: int
    result = element_count if immediate_count is None else immediate_count
    return validate_immediate_count(element_count, result)
