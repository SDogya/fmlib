from typing import Protocol, Self

from .distributed_info import DistributedInfo, DistributedInfoProtocol
from .worker_info import WorkerInfo, WorkerInfoProtocol

DEFAULT_WORKER_INFO: WorkerInfo = WorkerInfo()
DEFAULT_DISTRIBUTED_INFO: DistributedInfo = DistributedInfo()


def num_replicas(
    worker_info: WorkerInfoProtocol = DEFAULT_WORKER_INFO,
    distributed_info: DistributedInfoProtocol = DEFAULT_DISTRIBUTED_INFO,
) -> int:
    return worker_info.num_workers * distributed_info.world_size


def curr_replica(
    worker_info: WorkerInfoProtocol = DEFAULT_WORKER_INFO,
    distributed_info: DistributedInfoProtocol = DEFAULT_DISTRIBUTED_INFO,
) -> int:
    result: int = worker_info.id + worker_info.num_workers * distributed_info.rank
    assert result < num_replicas(worker_info, distributed_info)
    return result


class ReplicasInfo:
    def __init__(
        self: Self,
        worker_info: WorkerInfoProtocol = DEFAULT_WORKER_INFO,
        distributed_info: DistributedInfoProtocol = DEFAULT_DISTRIBUTED_INFO,
    ) -> None:
        self.worker_info: WorkerInfoProtocol = worker_info
        self.distributed_info: DistributedInfoProtocol = distributed_info

    @property
    def num_replicas(self: Self) -> int:
        return num_replicas(worker_info=self.worker_info, distributed_info=self.distributed_info)

    @property
    def curr_replica(self: Self) -> int:
        return curr_replica(worker_info=self.worker_info, distributed_info=self.distributed_info)


class ReplicasInfoProtocol(Protocol):
    @property
    def num_replicas(self: Self) -> int: ...

    @property
    def curr_replica(self: Self) -> int: ...
