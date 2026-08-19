from typing import Protocol, Self

import torch.distributed as dist


class DistributedInfo:
    def __iter__(self: Self):
        yield self.rank
        yield self.world_size

    @property
    def is_distributed(self: Self) -> bool:
        if dist.is_available():
            return dist.is_initialized()
        return False

    @property
    def rank(self: Self) -> int:
        if self.is_distributed:
            return dist.get_rank()
        return 0

    @property
    def world_size(self: Self) -> int:
        if self.is_distributed:
            return dist.get_world_size()
        return 1


class DistributedInfoProtocol(Protocol):
    @property
    def rank(self: Self) -> int: ...

    @property
    def world_size(self: Self) -> int: ...
