import os
import tempfile
import warnings
from copy import deepcopy
from typing import Iterable, TypedDict

import accelerate
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from fmlib.training.callbacks.epoch.save_checkpoint import SaveCheckpointCallback
from fmlib.utils.loading import load_model


class FakeTrainingState(TypedDict):
    config: DictConfig
    accelerator: accelerate.Accelerator
    pipeline: torch.nn.Module
    epoch_iterable: Iterable[int]


def compare_models(left: dict, right: dict) -> None:
    for left_name, left_param in left.items():
        right_param = right.get(left_name)
        assert left_param.dtype == right_param.dtype
        assert left_param.requires_grad == right_param.requires_grad
        assert torch.allclose(left_param, right_param), f"Error: {left_name=}."


@pytest.mark.parametrize("num_checkpoints", list(range(1, 5)))
@pytest.mark.parametrize("num_updates", list(range(1, 5)))
@pytest.mark.parametrize("direction", ["min", "max"])
@pytest.mark.parametrize("metric", ["test", "fake"])
@pytest.mark.parametrize("seed", [42, 777])
def test_save_checkpoint_max(num_checkpoints: int, num_updates: int, direction: str, metric: str, seed: int) -> None:
    temp_dir = tempfile.TemporaryDirectory()
    callback = SaveCheckpointCallback(
        num_checkpoints=num_checkpoints,
        base_path=temp_dir.name,
        metric_to_track=metric,
        direction=direction,
    )
    raw_model: torch.nn.Module = torch.nn.Linear(10, 10)

    accelerator: accelerate.Accelerator = accelerate.Accelerator()
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    fake_state: FakeTrainingState = FakeTrainingState(
        accelerator=accelerator,
        pipeline=raw_model,
        epoch_iterable=range(num_updates),
        config=OmegaConf.create(
            {
                "model": {
                    "_target_": "torch.nn.Linear",
                    "out_features": 10,
                    "in_features": 10,
                }
            }
        ),
    )

    states = {}

    for i in range(num_updates):
        torch.nn.init.normal_(fake_state["pipeline"].weight, generator=generator)
        torch.nn.init.normal_(fake_state["pipeline"].bias, generator=generator)
        states[i] = deepcopy(fake_state["pipeline"].state_dict())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            callback(fake_state, {metric: float(i)}, epoch=i)

    num_saved: int = min(num_checkpoints, num_updates)
    assert len(callback.state) == num_saved

    first: int = num_updates - num_saved

    for i, record in enumerate(callback.state):
        value = i if direction == "min" else first + i
        path, metric_value = record
        assert float(value) == metric_value
        assert os.path.exists(path)

        loaded: torch.nn.Module = load_model(model_path=path)
        compare_models(loaded.state_dict(), states[value])

        model_path: str = os.path.join(path, callback.checkpoint_name)
        assert os.path.exists(model_path)

        config_path: str = os.path.join(path, callback.config_name)
        assert os.path.exists(config_path)

    temp_dir.cleanup()
