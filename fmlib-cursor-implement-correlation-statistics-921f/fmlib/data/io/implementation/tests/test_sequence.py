import pytest
import torch

from ..indexing import get_mask, get_offsets
from ..sequence_column import SequenceColumn


@pytest.mark.parametrize("seed", [1, 19, 777])
@pytest.mark.parametrize("run_count", [1, 2, 3, 5])
@pytest.mark.parametrize("length", [1, 17, 33, 123])
@pytest.mark.parametrize("seq_len", [1, 2, 7, 77, 128])
def test_sequence(seed: int, run_count: int, length: int, seq_len: int):
    gen: torch.Generator = torch.Generator().manual_seed(seed)
    lengths: torch.LongTensor = torch.randint(
        low=1,
        high=int(1.5 * seq_len) + 1,
        size=(length,),
        generator=gen,
        dtype=torch.int64,
    )
    offsets: torch.LongTensor = get_offsets(lengths)
    data: torch.Tensor = torch.arange(offsets[-1].cpu().item(), dtype=torch.int64) * seed
    sequence = SequenceColumn(data, lengths, seq_len, padding=-seed)

    run_lengths: torch.LongTensor = torch.randint(
        low=1,
        high=2 * length,
        size=(run_count,),
        generator=gen,
        dtype=torch.int64,
    )

    run: int
    for run in range(run_count):
        indices_count: int = run_lengths[run].cpu().item()
        indices: torch.LongTensor = torch.randint(
            low=0,
            high=length,
            size=(indices_count,),
            generator=gen,
            dtype=torch.int64,
        )

        mask: torch.BoolTensor
        output: torch.LongTensor
        mask, output = sequence[indices]

        gtr_mask: torch.BoolTensor
        gtr_ids: torch.LongTensor
        gtr_mask, gtr_ids = get_mask(indices, offsets, seq_len)

        assert torch.all(gtr_mask == mask).cpu().all()

        gtr_vals: torch.LongTensor = torch.where(gtr_mask, gtr_ids * seed, -seed)

        assert torch.all(gtr_vals == output).cpu().item()
