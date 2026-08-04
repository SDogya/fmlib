import warnings
from typing import Iterable, Protocol, Self

import pyarrow.dataset as ds

from fmlib.data.io.partitioning.partitioning import partitioning_per_replica
from fmlib.data.io.pyarrow_partition_generator import BatchesIterator


class HasLengthProtocol(Protocol):
    def __len__(self: Self) -> int: ...


def compute_fixed_size_generic_length_from_sizes(partition_sizes: Iterable[int], batch_size: int, num_replicas: int) -> int:
    residue: int = 0
    batch_counter: int = 0
    for partition_size in partition_sizes:
        per_replica: int = partitioning_per_replica(partition_size, num_replicas)
        batch_count: int = per_replica // batch_size
        residue += per_replica % batch_size
        if batch_size < residue:
            batch_count += residue // batch_size
            residue = residue % batch_size
        batch_counter += batch_count
    batch_counter += residue > 0
    return batch_counter


def _compute_from_dataset(dataset: ds.Dataset, partition_size: int, batch_size: int, num_replicas: int) -> int:
    """Compute batch count by counting fragment rows and simulating partition splits."""

    def default_partitions(fragment_size: int) -> list[int]:
        full_partitions_count: int = fragment_size // partition_size
        result: list[int] = [partition_size] * full_partitions_count
        if (residue := (fragment_size % partition_size)) > 0:
            result.append(residue)
        return result

    partition_sizes: list[int] = []
    for fragment in dataset.get_fragments():
        partition_sizes.extend(default_partitions(fragment.count_rows()))

    return compute_fixed_size_generic_length_from_sizes(
        partition_sizes=partition_sizes,
        num_replicas=num_replicas,
        batch_size=batch_size,
    )


def compute_fixed_size_batches_length(iterable: BatchesIterator, batch_size: int, num_replicas: int) -> int:
    assert isinstance(iterable, BatchesIterator)
    return _compute_from_dataset(iterable.dataset, iterable.batch_size, batch_size, num_replicas)


def compute_fixed_size_generic_length(iterable: Iterable[HasLengthProtocol], batch_size: int, num_replicas: int) -> int:
    warnings.warn("Generic length computation. This may cause performance issues.", UserWarning, stacklevel=2)
    return compute_fixed_size_generic_length_from_sizes(map(len, iterable), batch_size, num_replicas)


def compute_fixed_size_length(iterable: Iterable[HasLengthProtocol], batch_size: int, num_replicas: int) -> int:
    # Use fast dataset-based computation for any iterator that exposes .dataset and .batch_size
    # (covers BatchesIterator and MultimodalBatchesIterator without a hard import dependency).
    if hasattr(iterable, "dataset") and hasattr(iterable, "batch_size"):
        return _compute_from_dataset(iterable.dataset, iterable.batch_size, batch_size, num_replicas)
    return compute_fixed_size_generic_length(iterable, batch_size, num_replicas)
