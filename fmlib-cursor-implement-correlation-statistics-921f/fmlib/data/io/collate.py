from typing import Dict, Sequence

import torch

from fmlib.constants.batches import GeneralBatch, GeneralValue


def dict_collate(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: torch.cat([d[k] for d in batch], dim=0) for k in batch[0]}


def general_collate(batch: Sequence[GeneralBatch]) -> GeneralBatch:
    result: GeneralBatch = {}
    test_sample: GeneralBatch = batch[0]

    if len(batch) == 1:
        return test_sample

    for key, test_value in test_sample.items():
        values: Sequence[GeneralValue] = [sample[key] for sample in batch]
        if torch.is_tensor(test_value):
            result[key] = torch.cat(values, dim=0)
        else:
            assert isinstance(test_value, dict)
            result[key] = general_collate(values)

    return result
