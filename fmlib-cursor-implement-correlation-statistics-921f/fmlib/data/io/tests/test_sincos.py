from typing import Dict, Iterator, Self

import pyarrow as pa
import pytest
import torch

from fmlib.data.io.implementation.flat_column import to_flat_columns
from fmlib.data.io.implementation.named_columns import NamedColumns
from fmlib.data.io.implementation.sequence_column import to_sequence_columns
from fmlib.data.io.metadata.metadata import Metadata

from .sincos_dataset import BatchGenerator, batch_to_pyarrow, make_metadata

Batch = Dict[str, torch.Tensor]

parts: list[str] = [
    "phase",
    "frequency",
    "sin_sin",
    "cos_cos",
    "sin_length",
    "cos_length",
]


@pytest.mark.parametrize("seed", [1, 3, 42, 777])
@pytest.mark.parametrize("batch_size", [1, 5, 32, 128])
@pytest.mark.parametrize("max_length", [1, 17, 64, 256])
def test_sincos(seed: int, batch_size: int, max_length: int):
    bg: BatchGenerator = BatchGenerator(
        batch_size=batch_size,
        max_length=max_length,
        generator=torch.Generator().manual_seed(seed),
    )
    gtr_bg: BatchGenerator = BatchGenerator(
        batch_size=batch_size,
        max_length=max_length,
        generator=torch.Generator().manual_seed(seed),
    )
    ids: torch.LongTensor = torch.arange(batch_size, dtype=torch.int64)

    batch: Batch = bg.generate_batch()
    pa_batch: Dict[str, pa.Array] = batch_to_pyarrow(batch)
    mt: Metadata = make_metadata(bg)
    named: NamedColumns = NamedColumns(
        columns={
            **to_flat_columns(pa_batch, mt),
            **to_sequence_columns(pa_batch, mt),
        },
    )
    samples: Batch = named[ids]
    gtr_samples: Batch = gtr_bg.generate_padded()

    part: str
    for part in parts:
        gen_part: torch.Tensor = samples[part].ravel()
        gtr_part: torch.Tensor = gtr_samples[part].ravel()
        assert gen_part.size() == gtr_part.size()
        assert torch.allclose(gen_part, gtr_part)


class BatchIterable:
    def __init__(self: Self, batch_generator: BatchGenerator, batch_count: int) -> None:
        self.metadata: Metadata = make_metadata(batch_generator)
        self.batch_generator: BatchGenerator = batch_generator
        self.batch_count: int = batch_count

    def __iter__(self: Self) -> Iterator[NamedColumns]:
        for _ in range(self.batch_count):
            raw_batch: Batch = self.batch_generator.generate_batch()
            batch: Batch = {part: raw_batch[part] for part in parts}
            pa_batch: pa.Table = batch_to_pyarrow(batch=batch)
            yield NamedColumns(
                columns={
                    **to_flat_columns(pa_batch, self.metadata),
                    **to_sequence_columns(pa_batch, self.metadata),
                },
            )
