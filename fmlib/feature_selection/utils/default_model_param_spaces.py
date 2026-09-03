"""Fallback Optuna search spaces for selectors that tune.

These ranges are used only when ``params.optuna_params.enabled`` is true and
``params.parameters`` contains no mapping entries. Any mapping in the YAML
block fully replaces this file: unspecified default keys are not mixed in.

``learning_rate``, ``early_stopping_rounds`` and the tree cap are not part of
the grid. Selectors overlay them from ``lama_boost_defaults`` after ``n_rows``
is known.
"""

from __future__ import annotations

from typing import Any

LIGHTGBM_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "num_leaves": {"type": "int", "min": 16, "max": 255},
    "colsample_bytree": {"type": "float", "min": 0.5, "max": 1.0},
    "subsample": {"type": "float", "min": 0.5, "max": 1.0},
    "min_child_weight": {"type": "float", "min": 1e-3, "max": 10.0, "log": True},
}

CATBOOST_RFE_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "depth": {"type": "int", "min": 3, "max": 7},
    "l2_leaf_reg": {"type": "float", "min": 1e-8, "max": 10.0, "log": True},
    "min_data_in_leaf": {"type": "int", "min": 1, "max": 20},
}

BORUTA_LGBM_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    name: dict(spec) for name, spec in LIGHTGBM_SEARCH_SPACE.items()
}

BORUTA_RF_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "n_estimators": {"type": "int", "min": 50, "max": 200},
    "max_depth": {"type": "int", "min": 3, "max": 10},
    "min_samples_split": {"type": "int", "min": 2, "max": 20},
    "min_samples_leaf": {"type": "int", "min": 1, "max": 10},
}
