"""Fallback Optuna search spaces for selectors that tune.

These ranges are used only when ``params.optuna_params.enabled`` is true and
``params.parameters`` contains no mapping entries. Any mapping in the YAML
block fully replaces this file: unspecified default keys are not mixed in.
"""

from __future__ import annotations

from typing import Any

LIGHTGBM_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "n_estimators": {"type": "int", "min": 100, "max": 500},
    "learning_rate": {"type": "float", "min": 0.01, "max": 0.2, "log": True},
    "max_depth": {"type": "int", "min": 3, "max": 8},
    "num_leaves": {"type": "int", "min": 8, "max": 64},
    "subsample": {"type": "float", "min": 0.6, "max": 1.0},
    "colsample_bytree": {"type": "float", "min": 0.6, "max": 1.0},
}

CATBOOST_RFE_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "depth": {"type": "int", "min": 4, "max": 8},
    "min_data_in_leaf": {"type": "int", "min": 16, "max": 64},
    "learning_rate": {"type": "float", "min": 0.01, "max": 0.3, "log": True},
    "l2_leaf_reg": {"type": "float", "min": 0.5, "max": 30, "log": True},
    "random_strength": {"type": "float", "min": 0.05, "max": 0.2},
}

BORUTA_LGBM_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "n_estimators": {"type": "int", "min": 100, "max": 1_400},
    "num_leaves": {"type": "int", "min": 8, "max": 64},
    "max_depth": {"type": "int", "min": 4, "max": 7},
    "learning_rate": {
        "type": "float",
        "min": 0.01,
        "max": 0.1,
        "log": True,
    },
    "min_child_samples": {"type": "int", "min": 16, "max": 64},
    "subsample": {"type": "float", "min": 0.7, "max": 1.0},
    "colsample_bytree": {"type": "float", "min": 0.7, "max": 1.0},
    "reg_alpha": {
        "type": "float",
        "min": 0.01,
        "max": 1.0,
        "log": True,
    },
    "reg_lambda": {
        "type": "float",
        "min": 0.01,
        "max": 1.0,
        "log": True,
    },
    "boosting_type": {
        "type": "categorical",
        "values": ["gbdt", "dart", "goss"],
    },
}

BORUTA_RF_SEARCH_SPACE: dict[str, dict[str, Any]] = {
    "n_estimators": {"type": "int", "min": 50, "max": 200},
    "max_depth": {"type": "int", "min": 3, "max": 10},
    "min_samples_split": {"type": "int", "min": 2, "max": 20},
    "min_samples_leaf": {"type": "int", "min": 1, "max": 10},
}
