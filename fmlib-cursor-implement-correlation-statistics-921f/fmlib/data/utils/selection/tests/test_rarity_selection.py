from math import ceil

import pytest
import torch

from ..rarity_selection import RaritySelector


@pytest.mark.parametrize("seed", [7, 42, 33, 777])
@pytest.mark.parametrize("size", [1, 32, 64, 222])
@pytest.mark.parametrize("share", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("iter_count", [1, 8, 24])
def test_rarity_selection(seed: int, size: int, share: float, iter_count: int) -> None:
    input_share: int = ceil(share * size)
    gen: torch.Generator = torch.Generator().manual_seed(seed)
    counts: torch.Tensor = torch.zeros(size, dtype=torch.int64)

    selector: RaritySelector = RaritySelector(size)

    for _ in range(iter_count):
        selected: torch.LongTensor = selector(input_share, gen)
        counts[selected] = counts[selected] + 1

    assert torch.all(selector.counts_buffer == counts).cpu().item()
    assert torch.sum(counts).cpu().item() == (input_share * iter_count)
