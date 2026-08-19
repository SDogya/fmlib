import pytest

from ..batching import UniformBatching


def test_uniform_batching_match():
    batching: UniformBatching = UniformBatching(9, 3)
    assert list(batching) == [(0, 3), (3, 6), (6, 9)]


def test_uniform_batching_more():
    batching: UniformBatching = UniformBatching(9, 4)
    assert list(batching) == [(0, 4), (4, 8), (8, 9)]


def test_uniform_batching_one():
    batching: UniformBatching = UniformBatching(3, 1)
    assert list(batching) == [(0, 1), (1, 2), (2, 3)]


def test_uniform_batching_dummy():
    batching: UniformBatching = UniformBatching(1, 4)
    assert list(batching) == [(0, 1)]


def test_uniform_zero_length():
    with pytest.raises(ValueError):
        _ = UniformBatching(0, 4)


def test_uniform_zero_batch_size():
    with pytest.raises(ValueError):
        _ = UniformBatching(4, 0)


def test_uniform_negative_index():
    with pytest.raises(IndexError):
        batching: UniformBatching = UniformBatching(9, 4)
        _ = batching[-1]


def test_uniform_large_index():
    with pytest.raises(IndexError):
        batching: UniformBatching = UniformBatching(9, 4)
        _ = batching[120]
