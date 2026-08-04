import torch

from fmlib.nn.transforms.modular.filtering import FilteringIn, FilteringOut


def test_filter_in_simple() -> None:
    test_batch = {
        "a": torch.asarray([1]),
        "b": torch.asarray([2]),
        "c": torch.asarray([3]),
    }

    test_result = FilteringIn(["a", "b", "c"])(test_batch)
    assert set(test_result.keys()) == {"a", "b", "c"}

    test_result = FilteringIn(["a", "c"])(test_batch)
    assert set(test_result.keys()) == {"a", "c"}

    test_result = FilteringIn()(test_batch)
    assert set(test_result.keys()) == {"a", "b", "c"}

    test_result = FilteringIn([])(test_batch)
    assert set(test_result.keys()) == set()


def test_filter_out_simple() -> None:
    test_batch = {
        "a": torch.asarray([5]),
        "b": torch.asarray([8]),
        "c": torch.asarray([12]),
    }

    test_result = FilteringOut(["a", "c"])(test_batch)
    assert set(test_result.keys()) == {"b"}

    test_result = FilteringOut([])(test_batch)
    assert set(test_result.keys()) == {"a", "b", "c"}

    test_result = FilteringOut(["a", "b", "c"])(test_batch)
    assert set(test_result.keys()) == set()


def test_filter_in_pattern() -> None:
    test_batch = {
        "template_dead_template": torch.asarray([1]),
        "template_beaf_template": torch.asarray([2]),
    }

    test_result = FilteringIn([".*dead.*"])(test_batch)
    assert set(test_result.keys()) == {"template_dead_template"}

    test_result = FilteringIn(["template.*template"])(test_batch)
    assert set(test_result.keys()) == {"template_dead_template", "template_beaf_template"}


def test_filter_out_pattern() -> None:
    test_batch = {
        "template_dead_template": torch.asarray([1]),
        "template_beaf_template": torch.asarray([2]),
    }

    test_result = FilteringOut([".*dead.*"])(test_batch)
    assert set(test_result.keys()) == {"template_beaf_template"}

    test_result = FilteringOut(["template.*template"])(test_batch)
    assert set(test_result.keys()) == set()


def test_filter_in_nested() -> None:
    test_batch = {
        "a": torch.asarray([1]),
        "nested": {
            "b": torch.asarray([2]),
            "c": torch.asarray([3]),
        },
    }

    test_result = FilteringIn(["nested.*"])(test_batch)
    assert set(test_result.keys()) == {"nested"}
    assert set(test_result["nested"].keys()) == {"b", "c"}

    test_result = FilteringIn(["nested.*b"])(test_batch)
    assert set(test_result.keys()) == {"nested"}
    assert set(test_result["nested"].keys()) == {"b"}

    test_result = FilteringIn([".*b"])(test_batch)
    assert set(test_result.keys()) == {"nested"}
    assert set(test_result["nested"].keys()) == {"b"}


def test_filter_out_nested() -> None:
    test_batch = {
        "a": torch.asarray([1]),
        "nested": {
            "b": torch.asarray([2]),
            "c": torch.asarray([3]),
        },
    }

    test_result = FilteringOut(["nested.*"])(test_batch)
    assert set(test_result.keys()) == {"a", "nested"}
    assert set(test_result["nested"].keys()) == set()

    test_result = FilteringOut(["nested.*b"])(test_batch)
    assert set(test_result.keys()) == {"a", "nested"}
    assert set(test_result["nested"].keys()) == {"c"}

    test_result = FilteringOut([".*b"])(test_batch)
    assert set(test_result.keys()) == {"a", "nested"}
    assert set(test_result["nested"].keys()) == {"c"}
