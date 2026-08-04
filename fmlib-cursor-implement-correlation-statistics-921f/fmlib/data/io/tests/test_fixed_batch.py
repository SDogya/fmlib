from typing import Any, Dict, Iterator, List, Optional, Self, Tuple, Union

import pytest
import torch
from torch.utils.data import IterableDataset

from fmlib.data.io.fixed_batch_dataset import FixedBatchSizeDataset, GeneralBatch, get_batch_size

SchemaType = Dict[str, Union[Tuple[int, ...], "SchemaType"]]

schemas: List[SchemaType] = [
    {"a": (1,), "b": (2, 3), "c": {"d": (4,), "e": (5, 6)}},
    {"a": (1,), "b": (2,), "c": (3,), "d": (4,)},
    {"a": {"b": {"c": (6, 7, 8)}}},
]


class FakeDataset(IterableDataset):
    def __init__(
        self: Self,
        schema: SchemaType,
        length: int = 256,
        min_size: int = 1,
        max_size: int = 16,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__()

        self.schema: SchemaType = schema

        self.length: int = length
        self.min_size: int = min_size
        self.max_size: int = max_size
        self.generator: Optional[torch.Generator] = generator

    def gen_batch(self: Self, schema: Dict[str, Any], size: int) -> GeneralBatch:
        batch: GeneralBatch = {}
        for key, value in schema.items():
            if isinstance(value, dict):
                batch[key] = self.gen_batch(value, size)
            else:
                batch[key] = torch.rand(
                    size=(size, *value),
                    generator=self.generator,
                )
        assert get_batch_size(batch) == size
        return batch

    def __iter__(self: Self) -> Iterator[GeneralBatch]:
        lengths: List[int] = (
            torch.randint(
                low=self.min_size,
                high=self.max_size,
                size=(self.length,),
                generator=self.generator,
            )
            .cpu()
            .tolist()
        )

        for length in lengths:
            batch: GeneralBatch = self.gen_batch(self.schema, length)

            yield batch


def slice_batches(batch: GeneralBatch, low: int, high: int) -> GeneralBatch:
    result: GeneralBatch = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            result[key] = value[low:high, ...]
        else:
            assert isinstance(value, dict)
            result[key] = slice_batches(value, low, high)
    return result


def compare(left: GeneralBatch, right: GeneralBatch) -> None:
    keys = set(left.keys())
    assert keys == set(right.keys())

    for key in keys:
        if torch.is_tensor(left[key]):
            assert torch.allclose(left[key], right[key])
        else:
            assert isinstance(left[key], dict)
            assert isinstance(right[key], dict)
            compare(left[key], right[key])


@pytest.mark.parametrize("schema", schemas)
@pytest.mark.parametrize("seed", [42, 777])
@pytest.mark.parametrize("length", [1, 2, 5, 7, 11])
@pytest.mark.parametrize("batch_size", [2, 3, 5, 7, 9, 11, 17])
def test_fixed_batch_size(schema: SchemaType, seed: int, length: int, batch_size: int) -> None:
    gtr_generator: torch.Generator = torch.Generator().manual_seed(seed)
    res_generator: torch.Generator = torch.Generator().manual_seed(seed)

    gtr_dataset: FakeDataset = FakeDataset(
        schema,
        length=length,
        generator=gtr_generator,
    )

    res_dataset: FakeDataset = FakeDataset(
        schema,
        length=length,
        generator=res_generator,
    )

    gtr_batches: List[GeneralBatch] = list(gtr_dataset)

    res_dataset: IterableDataset = FixedBatchSizeDataset(dataset=res_dataset, batch_size=batch_size)

    all_gtr_data: GeneralBatch = res_dataset.collate_fn(gtr_batches)

    start: int = 0
    sizes: List[int] = []
    for res_batch in iter(res_dataset):
        curr_batch_size: int = get_batch_size(res_batch)
        sizes.append(curr_batch_size)

        end = start + curr_batch_size

        gtr_slice: GeneralBatch = slice_batches(all_gtr_data, start, end)

        compare(res_batch, gtr_slice)

        start = end

    full_size: int = get_batch_size(all_gtr_data)
    assert sum(sizes) == full_size

    for size in sizes[:-1]:
        assert size == batch_size
