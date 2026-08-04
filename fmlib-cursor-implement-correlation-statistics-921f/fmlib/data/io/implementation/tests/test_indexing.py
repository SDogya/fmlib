import pytest
import torch

from ..indexing import get_mask, get_offsets


@pytest.mark.parametrize("seed", [1, 42, 777])
@pytest.mark.parametrize("length", [1, 35, 42, 1023])
@pytest.mark.parametrize("max_seq_len", [1, 2, 17, 123])
def test_get_offsets(seed: int, length: int, max_seq_len: int):
    gen: torch.Generator = torch.Generator().manual_seed(seed)
    lengths: torch.LongTensor = torch.randint(
        low=0,
        high=max_seq_len,
        size=(length,),
        generator=gen,
        dtype=torch.int64,
    )

    offsets: torch.LongTensor = get_offsets(lengths)
    assert torch.numel(offsets) == length + 1

    test_lengths: torch.LongTensor = offsets[1:] - offsets[:-1]
    assert torch.all(test_lengths == lengths).cpu().item()


@pytest.mark.parametrize("seed", [1, 42, 777])
@pytest.mark.parametrize("run_count", [1, 2, 3, 7])
@pytest.mark.parametrize("length", [1, 35, 42, 1023])
@pytest.mark.parametrize("max_seq_len", [1, 2, 17, 123])
def test_get_mask(seed: int, run_count: int, length: int, max_seq_len: int):
    gen: torch.Generator = torch.Generator().manual_seed(seed)
    lengths: torch.LongTensor = torch.randint(
        low=0,
        high=int(1.5 * max_seq_len),
        size=(length,),
        generator=gen,
        dtype=torch.int64,
    )
    offsets: torch.LongTensor = get_offsets(lengths)
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

        output: torch.LongTensor
        mask: torch.BoolTensor
        mask, output = get_mask(indices, offsets, max_seq_len)

        assert mask.size() == (indices_count, max_seq_len)

        first_ids: torch.LongTensor = offsets[:-1][indices]
        last_ids: torch.LongTensor = offsets[1:][indices]
        len_ids: torch.LongTensor = last_ids - first_ids
        zero_len: torch.BoolTensor = len_ids == 0

        assert torch.all((output[..., 1:] - output[..., :-1]) <= 1).cpu().item()
        assert torch.all(first_ids <= torch.min(output, dim=-1).values).cpu().item()

        true_count: torch.LongTensor = torch.minimum(len_ids, torch.asarray(max_seq_len))
        assert torch.all(torch.sum(mask, dtype=torch.int64, dim=-1) == true_count).cpu().item()

        true_last: torch.LongTensor = torch.where(zero_len, first_ids, (last_ids - 1))
        assert torch.all(torch.max(output, dim=-1).values == true_last).cpu().item()


@pytest.mark.parametrize("seed", [1, 42, 777])
@pytest.mark.parametrize("length", [1, 35, 128, 256])
@pytest.mark.parametrize("max_seq_len", [1, 2, 17, 128, 512])
def test_raw_get_mask_random_slicing_basic(seed: int, length: int, max_seq_len: int):
    gen = torch.Generator().manual_seed(seed)
    lengths = torch.randint(
        low=0,
        high=int(1.5 * max_seq_len),
        size=(length,),
        generator=gen,
        dtype=torch.int64,
    )
    offsets = get_offsets(lengths)
    indices_count = max(1, length // 2)
    indices = torch.randint(
        low=0,
        high=length,
        size=(indices_count,),
        generator=gen,
        dtype=torch.int64,
    )
    mask, output = get_mask(indices, offsets, max_seq_len, random_slicing=True)
    assert mask.size() == (indices_count, max_seq_len)
    assert output.size() == (indices_count, max_seq_len)
    first_ids = offsets[indices]
    last_ids = offsets[indices + 1]
    len_ids = last_ids - first_ids
    zero_len = len_ids == 0

    diffs = output[..., 1:] - output[..., :-1]
    assert torch.all(diffs <= 1).cpu().item()
    assert torch.all(first_ids.unsqueeze(-1) <= output).cpu().item()

    valid_rows = ~zero_len
    if valid_rows.any():
        assert torch.all(output[valid_rows] < last_ids[valid_rows].unsqueeze(-1)).cpu().item()

    true_count = torch.minimum(len_ids, torch.asarray(max_seq_len))
    assert torch.all(torch.sum(mask, dtype=torch.int64, dim=-1) == true_count).cpu().item()


@pytest.mark.parametrize("seed", [1, 42, 777])
@pytest.mark.parametrize("length", [32, 64, 128])
@pytest.mark.parametrize("max_seq_len", [32, 64, 128])
def test_raw_get_mask_random_slicing_range(seed: int, length: int, max_seq_len: int):
    lengths = torch.full((length,), max_seq_len * 3, dtype=torch.int64)
    offsets = get_offsets(lengths)

    indices = torch.arange(length, dtype=torch.int64)

    _, output = get_mask(indices, offsets, max_seq_len, random_slicing=True)

    first_ids = offsets[indices]
    last_ids = offsets[indices + 1]

    assert torch.all(output >= first_ids.unsqueeze(-1)).cpu().item()
    assert torch.all(output < last_ids.unsqueeze(-1)).cpu().item()

    start_indices = output[:, 0]
    assert torch.all(start_indices <= last_ids - max_seq_len).cpu().item()
    assert torch.all(start_indices >= first_ids).cpu().item()


@pytest.mark.parametrize("seed", [1, 42, 777])
@pytest.mark.parametrize("max_seq_len", [16, 32, 64])
def test_raw_get_mask_random_slicing_short_segments(seed: int, max_seq_len: int):
    gen = torch.Generator().manual_seed(seed)
    lengths = torch.randint(
        low=0,
        high=max_seq_len + 1,
        size=(20,),
        generator=gen,
        dtype=torch.int64,
    )
    offsets = get_offsets(lengths)
    indices = torch.arange(len(lengths), dtype=torch.int64)
    for run in range(5):
        torch.manual_seed(seed + run)
        mask_random, output_random = get_mask(
            indices, offsets, max_seq_len, random_slicing=True
        )
        torch.manual_seed(seed + run)
        mask_fixed, output_fixed = get_mask(
            indices, offsets, max_seq_len, random_slicing=False
        )
        assert torch.all(mask_random == mask_fixed).cpu().item()

        non_empty = lengths > 0
        if non_empty.any():
            assert torch.all(
                output_random[non_empty] == output_fixed[non_empty]
            ).cpu().item()
