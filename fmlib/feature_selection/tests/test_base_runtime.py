"""Тесты общих вспомогательных функций выполнения шагов."""

import random

import numpy as np

from fmlib.feature_selection.base import (
    StageContext,
    bind_process_rng,
    resolve_step_seed,
)


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
