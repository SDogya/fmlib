import pytest
import torch

from fmlib.nn.transforms import SequentialTransform, make_for_sequence_representation

event_metadata = {
    "evt_attr_9": {"type": "categorical", "cardinality": 524, "event_id": 0},
    "evt_attr_15": {"type": "numerical", "event_id": 0},
    "channel_group": {"type": "categorical", "cardinality": 10, "event_id": 2},
}


@pytest.mark.parametrize("seed", [33, 77, 99])
@pytest.mark.parametrize("batch_size", [1, 3, 1024])
@pytest.mark.parametrize("seq_len", [1, 11, 101, 1001])
@pytest.mark.parametrize("time_encoding", [None, "delta", "absolute"])
def test_against_legacy(seed: int, batch_size: int, seq_len: int, time_encoding: str | None) -> None:
    params = {
        "events_meta": event_metadata,
        "time_encoding": time_encoding,
    }
    legacy = SequentialTransform(**params)
    modern = make_for_sequence_representation(**params)
    generator = torch.Generator().manual_seed(seed)
    epk_id = torch.randint(
        low=1,
        high=torch.iinfo(torch.int32).max,
        size=(batch_size, 1),
        generator=generator,
    )
    evt_attr_9 = torch.randint(
        low=-1,
        high=event_metadata["evt_attr_9"]["cardinality"],
        size=(batch_size, seq_len),
        generator=generator,
    )
    evt_attr_15 = 1.0 + torch.randn(
        size=(batch_size, seq_len),
        generator=generator,
    )
    channel_group = torch.randint(
        low=-1,
        high=event_metadata["channel_group"]["cardinality"],
        size=(batch_size, seq_len),
        generator=generator,
    )
    dttm_increments = torch.randint(low=0, high=100, size=(batch_size, seq_len), generator=generator)
    evt_dttm = torch.cumsum(dttm_increments, dim=-1)
    event_ids = torch.randint(
        low=0,
        high=3,
        size=(batch_size, seq_len),
        generator=generator,
    )
    batch = {
        "epk_id": epk_id,
        "evt_attr_9": evt_attr_9,
        "evt_attr_15": evt_attr_15,
        "channel_group": channel_group,
        "evt_dttm": evt_dttm,
        "event_ids": event_ids,
    }
    masks = {f"{k}_mask": (v >= 0) for k, v in batch.items()}
    batch = {**batch, **masks}
    legacy_results = legacy(batch)
    modern_results = modern(batch)

    assert torch.equal(legacy_results["epk_id"], modern_results["epk_id"])

    event_keys = set(legacy_results["events"].keys())
    for name in sorted(event_keys):
        legacy_result = legacy_results["events"][name]
        if legacy_result is None:
            assert name not in modern_results["events"]
        else:
            assert torch.equal(legacy_result, modern_results["events"][name])
