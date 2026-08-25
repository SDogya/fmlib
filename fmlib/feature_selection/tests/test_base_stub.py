"""Tests for stub_drop_features helper."""

import random

import numpy as np

from fmlib.feature_selection.base import (
    StageContext,
    apply_drop_decisions,
    bind_process_rng,
    resolve_step_seed,
    stub_drop_features,
)


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


def test_resolve_step_seed_prefers_params_over_execution() -> None:
    context = StageContext(
        spark=None,
        datasets={},
        schema=None,
        config=None,
        seed=42,
        candidates=[],
        run_seed=99,
    )
    assert resolve_step_seed(None, context) == 42
    assert resolve_step_seed({}, context) == 42
    assert resolve_step_seed({"n_jobs": -1}, context) == 42
    assert resolve_step_seed({"seed": 17}, context) == 17
    assert resolve_step_seed({"seed": 0}, context) == 0
    assert resolve_step_seed(
        {"params": {"seed": 17, "n_jobs": -1}, "method": "lightgbm"},
        context,
    ) == 17
    assert resolve_step_seed(
        {"seed": 3, "params": {"seed": 17}},
        context,
    ) == 3


def test_bind_process_rng_is_repeatable() -> None:
    bind_process_rng(123)
    first_py = random.random()
    first_np = float(np.random.random())
    bind_process_rng(123)
    assert random.random() == first_py
    assert float(np.random.random()) == first_np
