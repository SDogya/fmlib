"""Focused tests for shared categorical model preparation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fmlib.feature_selection.utils.categorical import (
    encode_categorical_frame,
    prepare_categorical_sample,
    resolve_categorical_handling,
)
from fmlib.feature_selection.utils.task_runtime import resolve_task


def _sample(mode: str, **settings: object) -> tuple[object, object, object]:
    frame = pd.DataFrame(
        {
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "category": ["a", "b", "a", None, "c", "b"],
            "target": [0, 1, 0, 1, 0, 1],
        },
    )
    handling = resolve_categorical_handling(
        {"categorical_handling": {"mode": mode, **settings}},
        method_name="test",
    )
    sample = prepare_categorical_sample(
        frame,
        target_col="target",
        feature_cols=["num", "category"],
        categorical_cols=["category"],
        max_rows=100,
        sample_fraction=None,
        seed=17,
        method_name="test",
        handling=handling,
    )
    task = resolve_task("binary_classification", frame["target"], method_name="test")
    return sample, task, handling


def test_max_cardinality_drops_high_cardinality_source_feature() -> None:
    sample, _task, _handling = _sample("max_cardinality", max_cardinality=3)

    assert sample.dropped_cardinality == {"category": 4}
    assert sample.feature_cols == ["num"]


def test_top_n_maps_unseen_fold_level_to_other_category() -> None:
    sample, task, handling = _sample("top_n", top_n=1)

    encoded = encode_categorical_frame(
        sample,
        target=task.encoded_target,
        task=task,
        handling=handling,
        seed=17,
        train_indices=np.asarray([0, 1, 2, 3]),
    )

    assert str(encoded.features["category"].dtype) == "category"
    assert encoded.features.loc[4, "category"] == "__FMLIB_OTHER__"


def test_target_encoding_is_oof_for_singleton_categories() -> None:
    frame = pd.DataFrame(
        {
            "category": [f"value_{index}" for index in range(10)],
            "target": [0, 1] * 5,
        },
    )
    handling = resolve_categorical_handling(
        {"categorical_handling": {"mode": "target_encoding", "target_encoding": {"folds": 5}}},
        method_name="test",
    )
    sample = prepare_categorical_sample(
        frame,
        target_col="target",
        feature_cols=["category"],
        categorical_cols=["category"],
        max_rows=100,
        sample_fraction=None,
        seed=17,
        method_name="test",
        handling=handling,
    )
    task = resolve_task("binary_classification", frame["target"], method_name="test")
    encoded = encode_categorical_frame(
        sample,
        target=task.encoded_target,
        task=task,
        handling=handling,
        seed=17,
    )

    # Every category is unseen in its OOF training partition, so every row
    # receives the fold prior instead of its own target.
    values = encoded.features["__fmlib_te__category"].to_numpy()
    assert not np.array_equal(values, frame["target"].to_numpy(dtype=float))


def test_multiclass_target_encoding_keeps_source_mapping() -> None:
    frame = pd.DataFrame(
        {
            "category": ["a", "b", "c"] * 3,
            "target": ["first", "second", "third"] * 3,
        },
    )
    handling = resolve_categorical_handling(
        {"categorical_handling": {"mode": "target_encoding"}},
        method_name="test",
    )
    sample = prepare_categorical_sample(
        frame,
        target_col="target",
        feature_cols=["category"],
        categorical_cols=["category"],
        max_rows=100,
        sample_fraction=None,
        seed=17,
        method_name="test",
        handling=handling,
    )
    task = resolve_task("classification", frame["target"], method_name="test")
    encoded = encode_categorical_frame(
        sample,
        target=task.encoded_target,
        task=task,
        handling=handling,
        seed=17,
    )

    assert len(encoded.model_features) == 3
    assert set(encoded.source_by_model_feature.values()) == {"category"}
