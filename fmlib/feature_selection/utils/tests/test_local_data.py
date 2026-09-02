"""Tests for bounded local numeric samples shared by LightGBM and Boruta."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from fmlib.feature_selection.utils.local_data import prepare_numeric_frame


def _numeric_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [5.0, 6.0, 7.0, 8.0],
            "y": [0, 1, 0, 1],
        },
    )


def test_prepare_reuses_compatible_column_subset(monkeypatch: pytest.MonkeyPatch) -> None:
    context = SimpleNamespace(local_numeric_sample=None)
    calls = {"n": 0}
    original = prepare_numeric_frame.__globals__["_prepare_pandas_frame"]

    def counting_prepare(*args: object, **kwargs: object) -> pd.DataFrame:
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "fmlib.feature_selection.utils.local_data._prepare_pandas_frame",
        counting_prepare,
    )

    first = prepare_numeric_frame(
        _numeric_frame(),
        target_col="y",
        feature_cols=["a", "b"],
        max_rows=10,
        sample_fraction=None,
        seed=7,
        method_name="lightgbm",
        context=context,
    )
    second = prepare_numeric_frame(
        _numeric_frame(),
        target_col="y",
        feature_cols=["a"],
        max_rows=10,
        sample_fraction=None,
        seed=7,
        method_name="boruta_shap",
        context=context,
    )

    assert calls["n"] == 1
    assert list(first.columns) == ["a", "b", "y"]
    assert list(second.columns) == ["a", "y"]
    pd.testing.assert_series_equal(second["a"], first["a"], check_names=True)


def test_prepare_does_not_reuse_when_max_rows_differ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = SimpleNamespace(local_numeric_sample=None)
    calls = {"n": 0}
    original = prepare_numeric_frame.__globals__["_prepare_pandas_frame"]

    def counting_prepare(*args: object, **kwargs: object) -> pd.DataFrame:
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        "fmlib.feature_selection.utils.local_data._prepare_pandas_frame",
        counting_prepare,
    )

    prepare_numeric_frame(
        _numeric_frame(),
        target_col="y",
        feature_cols=["a", "b"],
        max_rows=10,
        sample_fraction=None,
        seed=7,
        method_name="lightgbm",
        context=context,
    )
    prepare_numeric_frame(
        _numeric_frame(),
        target_col="y",
        feature_cols=["a"],
        max_rows=2,
        sample_fraction=None,
        seed=7,
        method_name="boruta_shap",
        context=context,
    )

    assert calls["n"] == 2


def test_mixed_frame_accepts_pandas_category_dtype() -> None:
    """A column already typed ``category`` must survive null filling.

    Parquet round-trips and ``astype("category")`` both produce this dtype, and
    filling one with a label outside its categories raises rather than adding
    the label. That took out every run whose categorical columns came from a
    cached parquet split.
    """
    import numpy as np

    from fmlib.feature_selection.utils.local_data import prepare_mixed_frame

    frame = pd.DataFrame(
        {
            "cat": pd.Series(["a", "b", None] * 10, dtype="category"),
            "num": np.arange(30, dtype=float),
            "response": [0, 1, 0] * 10,
        },
    )

    prepared = prepare_mixed_frame(
        frame,
        target_col="response",
        feature_cols=["cat", "num"],
        categorical_cols=["cat"],
        max_rows=100,
        sample_fraction=None,
        seed=0,
        method_name="probe",
    )

    assert sorted(prepared["cat"].unique()) == ["None", "a", "b"]
    assert len(prepared) == len(frame)
