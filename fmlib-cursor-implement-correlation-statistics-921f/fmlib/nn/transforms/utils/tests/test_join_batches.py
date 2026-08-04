import pytest
import torch

from fmlib.nn.transforms.utils.join_batches import join_batches


def test_join_batches_plain() -> None:
    left = {
        "a": torch.asarray([1, 2, 3]),
        "b": torch.asarray([4, 5, 6]),
    }

    right = {
        "b": torch.asarray([7, 8, 9]),
        "c": torch.asarray([10, 11, 12]),
    }

    with pytest.raises(ValueError):
        _ = join_batches(left, right, strict=True, verbose=False)

    with pytest.warns(UserWarning):
        join = join_batches(left, right, strict=False, verbose=True)
        assert id(join["a"]) == id(left["a"])
        assert id(join["b"]) == id(right["b"])
        assert id(join["c"]) == id(right["c"])


def test_join_batches_nested() -> None:
    left = {
        "nested": {
            "a": torch.asarray([1, 2, 3]),
            "b": torch.asarray([4, 5, 6]),
        },
    }

    right = {
        "nested": {
            "b": torch.asarray([7, 8, 9]),
            "c": torch.asarray([10, 11, 12]),
        },
    }

    with pytest.raises(ValueError):
        _ = join_batches(left, right, strict=True, verbose=False)

    with pytest.warns(UserWarning):
        join = join_batches(left, right, strict=False, verbose=True)
        assert id(join["nested"]["a"]) == id(left["nested"]["a"])
        assert id(join["nested"]["b"]) == id(right["nested"]["b"])
        assert id(join["nested"]["c"]) == id(right["nested"]["c"])
