import warnings

import pytest
import torch

from fmlib.utils.mode_context import mode_context


def test_train_to_eval():
    model: torch.nn.Module = torch.nn.Linear(10, 10).train()

    assert model.training
    with pytest.warns(UserWarning), mode_context(model, training=False):
        assert not model.training
    assert model.training


def test_eval_to_train():
    model: torch.nn.Module = torch.nn.Linear(10, 10).eval()

    assert not model.training
    with pytest.warns(UserWarning), mode_context(model, training=True):
        assert model.training
    assert not model.training


def test_eval_to_eval():
    model: torch.nn.Module = torch.nn.Linear(10, 10).eval()

    assert not model.training
    with warnings.catch_warnings(), mode_context(model, training=False):
        assert not model.training
    assert not model.training


def test_eval_to_train_silent():
    model: torch.nn.Module = torch.nn.Linear(10, 10).eval()

    assert not model.training
    with warnings.catch_warnings(), mode_context(model, training=True, verbose=False):
        assert model.training
    assert not model.training
