"""Tests for stub_drop_features helper."""

from fmlib.feature_selection.base import apply_drop_decisions, stub_drop_features


def test_stub_drop_is_deterministic() -> None:
    candidates = [f"f_{i}" for i in range(40)]
    first = stub_drop_features(candidates, seed=42, stage="statistics", method="null_rate")
    second = stub_drop_features(candidates, seed=42, stage="statistics", method="null_rate")
    assert [item.feature for item in first] == [item.feature for item in second]
    assert 10 <= len(first) <= 20
    assert all(item.reason == "stub_random_drop" for item in first)
    assert all(not item.keep for item in first)


def test_stub_drop_preserves_at_least_one() -> None:
    candidates = [f"f_{i}" for i in range(15)]
    decisions = stub_drop_features(candidates, seed=1, stage="model", method="lasso")
    remaining = apply_drop_decisions(candidates, decisions)
    assert len(remaining) >= 1
    assert len(remaining) + len(decisions) == len(candidates)


def test_stub_drop_different_methods_differ() -> None:
    candidates = [f"f_{i}" for i in range(40)]
    a = {item.feature for item in stub_drop_features(candidates, seed=42, stage="s", method="a")}
    b = {item.feature for item in stub_drop_features(candidates, seed=42, stage="s", method="b")}
    assert a != b


def test_stub_drop_single_candidate() -> None:
    assert stub_drop_features(["only"], seed=0, stage="s", method="m") == []
