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


def test_prepare_keeps_numeric_nulls() -> None:
    frame = _numeric_frame()
    frame.loc[0, "a"] = float("nan")

    prepared = prepare_numeric_frame(
        frame,
        target_col="y",
        feature_cols=["a", "b"],
        max_rows=10,
        sample_fraction=None,
        seed=7,
        method_name="lightgbm",
    )

    assert int(prepared["a"].isna().sum()) == 1
    assert not prepared["b"].isna().any()
