import torch

from fmlib.nn.transforms.modular.rename import Rename


def test_rename_dict():
    test_batch = {
        "a": torch.asarray([1]),
        "b": torch.asarray([2]),
        "nested": {
            "c": torch.asarray([3]),
            "d": torch.asarray([4]),
        },
    }

    test = Rename({"b": "beta"})(test_batch)
    assert set(test.keys()) == {"a", "beta", "nested"}

    test = Rename({"b": "beta"}, bypass=False)(test_batch)
    assert set(test.keys()) == {"beta"}

    test = Rename({"nested_c": "gamma"})(test_batch)
    assert set(test.keys()) == {"a", "b", "nested"}
    assert set(test["nested"].keys()) == {"gamma", "d"}

    test = Rename({"nested_d": "delta"}, bypass=False)(test_batch)
    assert set(test.keys()) == {"nested"}
    assert set(test["nested"].keys()) == {"delta"}
