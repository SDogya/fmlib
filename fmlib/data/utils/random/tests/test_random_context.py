import pytest
import torch

from ..random_context import RandomContext


@pytest.mark.parametrize("seed", [3, 42, 777])
@pytest.mark.parametrize("count", [1, 2, 3, 16])
@pytest.mark.parametrize("size", [1, 32, 128, 1024])
def test_random_context(seed: int, count: int, size: int) -> None:
    gen_1: torch.Generator = torch.Generator().manual_seed(seed)
    gen_2: torch.Generator = torch.Generator().manual_seed(seed)

    initial_state: torch.ByteTensor = torch.get_rng_state().clone()
    with RandomContext(gen_1):
        for _ in range(count):
            vals: torch.Tensor = torch.rand((size,), dtype=torch.float32)
            gtrs: torch.Tensor = torch.rand((size,), dtype=torch.float32, generator=gen_2)

            assert torch.all(vals == gtrs).cpu().item()
            assert torch.all(torch.get_rng_state() == gen_2.get_state()).cpu().item()

    assert torch.all(torch.get_rng_state() == initial_state).cpu().item()

    assert torch.all(gen_1.get_state() == gen_2.get_state()).cpu().item()
