from typing import Any, Self

import torch

from fmlib.constants.random import DEFAULT_GENERATOR, DEFAULT_SEED


def swap_with_global(generator: torch.Generator) -> torch.Generator:
    global_state: torch.ByteTensor = torch.get_rng_state()
    torch.set_rng_state(generator.get_state())
    return generator.set_state(global_state)


class RandomContext:
    """
    TODO: add docstring
    """

    def __init__(self: Self, generator: torch.Generator = DEFAULT_GENERATOR) -> None:
        self.generator: torch.Generator = generator
        self.lock: Any = torch.multiprocessing.Lock()

    def swap(self: Self) -> None:
        try:
            self.lock.acquire()
            self.generator = swap_with_global(self.generator)
        finally:
            self.lock.release()

    def __enter__(self: Self) -> Self:
        self.swap()
        return self

    def __exit__(self: Self, type_of: object, value: object, traceback: object) -> None:
        self.swap()


class SeededRandomContext(RandomContext):
    """
    TODO: add docstring
    """

    def __init__(self: Self, seed: int = DEFAULT_SEED) -> None:
        generator: torch.Generator = torch.Generator().manual_seed(seed)
        super().__init__(generator=generator)
