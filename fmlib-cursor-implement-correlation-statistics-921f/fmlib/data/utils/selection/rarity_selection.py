from typing import List, Optional, Self

import torch

from .base_selection import (
    DEFAULT_STRICTNESS,
    make_immediate_count,
    make_specificity,
    validate_element_count,
    validate_strictness,
)


class RaritySelector(torch.nn.Module):
    """
    TODO: add docstring
    """

    def __init__(
        self: Self,
        element_count: int,
        specificity: Optional[torch.Tensor] = None,
        strictness: float = DEFAULT_STRICTNESS,
    ) -> None:
        super().__init__()

        self.element_count: int = validate_element_count(element_count)
        self.strictness: float = validate_strictness(strictness)

        specificity_tensor: torch.Tensor = make_specificity(self.element_count, specificity)
        self.specificity_buffer: torch.nn.Buffer = torch.nn.Buffer(data=specificity_tensor)

        counts_tensor: torch.LongTensor = torch.zeros(self.element_count, dtype=torch.int64)
        self.counts_buffer: torch.nn.Buffer = torch.nn.Buffer(data=counts_tensor)

    def forward(
        self: Self,
        immediate_count: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.BoolTensor:
        selected: List[torch.LongTensor] = []
        probas_denom: torch.tensor = self.strictness * self.counts_buffer
        raw_probas: torch.Tensor = self.specificity_buffer / (probas_denom + 1.0)

        immediate_count: int = make_immediate_count(
            element_count=self.element_count,
            immediate_count=immediate_count,
        )

        left_count: int = immediate_count
        while left_count > 0:
            probas: torch.Tensor = torch.cumsum(raw_probas, dim=-1)
            sampled: torch.Tensor = torch.max(probas) * torch.rand(
                size=(left_count,),
                generator=generator,
            )
            raw_indices: torch.LongTensor = torch.searchsorted(
                sorted_sequence=probas,
                input=sampled,
            )
            raw_indices = torch.clamp(raw_indices, max=(self.element_count - 1))
            indices: torch.LongTensor = torch.unique(raw_indices, sorted=True)
            left_count = left_count - torch.numel(indices)
            selected.append(indices)
            raw_probas[indices] = 0

        selected: torch.LongTensor = torch.flatten(torch.cat(selected, dim=-1))
        self.counts_buffer[selected] = self.counts_buffer[selected] + 1

        perm: torch.LongTensor = torch.randperm(immediate_count, generator=generator)
        selected: torch.LongTensor = torch.take(selected, perm)

        assert torch.numel(torch.unique(selected)) == immediate_count
        assert left_count == 0

        return selected
