from typing import Any, Optional, Protocol, Self

import torch.utils.data as data


class WorkerInfo:
    def __iter__(self: Self):
        yield self.id
        yield self.world_size

    @property
    def worker_info(self: Self) -> Optional[Any]:
        return data.get_worker_info()

    @property
    def is_parallel(self: Self) -> bool:
        return self.worker_info is not None

    @property
    def id(self: Self) -> int:
        wi: Optional[data.WorkerInfo] = self.worker_info
        if wi is not None:
            return wi.id
        return 0

    @property
    def num_workers(self: Self) -> int:
        wi: Optional[data.WorkerInfo] = self.worker_info
        if wi is not None:
            return wi.num_workers
        return 1


class WorkerInfoProtocol(Protocol):
    @property
    def id(self: Self) -> int: ...

    @property
    def num_workers(self: Self) -> int: ...
