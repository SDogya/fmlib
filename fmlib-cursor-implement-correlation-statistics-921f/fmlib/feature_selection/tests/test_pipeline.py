"""Tests for FeatureSelectionPipeline skeleton."""

import pytest

from fmlib.feature_selection import (
    FeatureSchema,
    FeatureSelectionConfig,
    FeatureSelectionPipeline,
)
from fmlib.feature_selection.exceptions import ConfigError, SchemaError
from fmlib.feature_selection.tests.conftest import (
    FakeDataFrame,
    FakeSparkSession,
    make_pandas_frame,
    make_wide_schema_columns,
)


def _schema(categorical: list[str], continuous: list[str], *, with_split: bool = True) -> FeatureSchema:
    return FeatureSchema(
        categorical=tuple(categorical),
        continuous=tuple(continuous),
        target="response",
        task_type="binary_classification",
        time="event_date",
        split="dataset_split" if with_split else None,
        id_columns=("client_id",),
    )


def _config(**overrides: object) -> FeatureSelectionConfig:
    payload: dict = {
        "statistics": {
            "null_rate": {"enabled": False},
            "constants": {"enabled": False},
            "low_variance": {"enabled": False},
            "correlation": {"enabled": True},
            "psi": {"enabled": False},
            "stability_classifier": {"enabled": False},
        },
        "model": {"method": "lasso"},
        "precise": {"method": "none"},
        "execution": {"seed": 42},
    }
    payload.update(overrides)
    return FeatureSelectionConfig.from_dict(payload)


def test_fit_select_single_dataframe() -> None:
    categorical, continuous, columns = make_wide_schema_columns(40)
    spark = FakeSparkSession()
    data = make_pandas_frame(columns)
    pipeline = FeatureSelectionPipeline(_config())
    result = pipeline.fit_select(
        spark,
        data=data,
        schema=_schema(categorical, continuous, with_split=True),
    )
    assert result.datasets_mode == "single"
    assert len(result.selected_features) < len(categorical) + len(continuous)
    assert len(result.selected_features) >= 1
    assert all(name in categorical + continuous for name in result.selected_features)
    dropped_names = {item.feature for item in result.dropped_features}
    assert set(result.selected_features).isdisjoint(dropped_names)
    # inputs not mutated
    assert list(data.columns) == columns


def test_fit_select_datasets_mapping() -> None:
    categorical, continuous, columns = make_wide_schema_columns(40)
    spark = FakeSparkSession()
    train = make_pandas_frame(columns)
    pipeline = FeatureSelectionPipeline(_config())
    result = pipeline.fit_select(
        spark,
        datasets={"train": train},
        schema=_schema(categorical, continuous, with_split=False),
    )
    assert result.datasets_mode == "mapping"
    assert result.selected_features
    projected = pipeline.transform(train, result)
    for name in result.selected_features:
        assert name in projected.columns
    assert "response" in projected.columns
    assert "client_id" in projected.columns


def test_mutually_exclusive_inputs() -> None:
    categorical, continuous, columns = make_wide_schema_columns(20)
    spark = FakeSparkSession()
    frame = make_pandas_frame(columns)
    pipeline = FeatureSelectionPipeline(_config())
    with pytest.raises(ConfigError, match="either data"):
        pipeline.fit_select(
            spark,
            data=frame,
            datasets={"train": frame},
            schema=_schema(categorical, continuous),
        )


def test_single_dataframe_requires_split() -> None:
    categorical, continuous, columns = make_wide_schema_columns(20)
    spark = FakeSparkSession()
    pipeline = FeatureSelectionPipeline(_config())
    with pytest.raises(SchemaError, match="split"):
        pipeline.fit_select(
            spark,
            data=make_pandas_frame(columns),
            schema=_schema(categorical, continuous, with_split=False),
        )


def test_deterministic_across_runs() -> None:
    categorical, continuous, columns = make_wide_schema_columns(40)
    spark = FakeSparkSession()
    schema = _schema(categorical, continuous)
    config = _config()
    first = FeatureSelectionPipeline(config).fit_select(
        spark,
        data=make_pandas_frame(columns),
        schema=schema,
    )
    second = FeatureSelectionPipeline(config).fit_select(
        spark,
        data=make_pandas_frame(columns),
        schema=schema,
    )
    assert first.selected_features == second.selected_features
    assert [item.feature for item in first.dropped_features] == [item.feature for item in second.dropped_features]


def test_precise_boruta_stub_runs() -> None:
    categorical, continuous, columns = make_wide_schema_columns(50)
    spark = FakeSparkSession()
    config = _config(precise={"method": "boruta_shap"})
    result = FeatureSelectionPipeline(config).fit_select(
        spark,
        data=make_pandas_frame(columns),
        schema=_schema(categorical, continuous),
    )
    assert any(item.method == "boruta_shap" for item in result.dropped_features) or (len(result.selected_features) >= 1)


def test_correlation_drop_decisions_have_real_values() -> None:
    """Correlation selector should produce numeric value in FeatureDecision."""
    categorical, continuous, columns = make_wide_schema_columns(30)
    spark = FakeSparkSession()
    config = _config()
    result = FeatureSelectionPipeline(config).fit_select(
        spark,
        data=make_pandas_frame(columns),
        schema=_schema(categorical, continuous),
    )
    corr_drops = [item for item in result.dropped_features if item.method == "correlation"]
    for drop in corr_drops:
        assert drop.value is not None
        assert 0.0 <= drop.value <= 1.0
        assert drop.threshold is not None


def test_low_variance_selector_runs_in_pipeline() -> None:
    import pandas as pd

    frame = pd.DataFrame(
        {
            "constant": [1.0, 1.0, 1.0, 1.0],
            "varying": [1.0, 2.0, 3.0, 4.0],
            "response": [0, 1, 0, 1],
        },
    )
    schema = FeatureSchema(
        categorical=(),
        continuous=("constant", "varying"),
        target="response",
        task_type="binary_classification",
    )
    config = FeatureSelectionConfig.from_dict(
        {
            "statistics": {
                "null_rate": {"enabled": False},
                "constants": {"enabled": False},
                "low_variance": {"enabled": True},
                "correlation": {"enabled": False},
            },
            "model": {"method": "lasso"},
            "precise": {"method": "none"},
        },
    )

    result = FeatureSelectionPipeline(config).fit_select(
        FakeSparkSession(),
        datasets={"train": frame},
        schema=schema,
    )

    low_variance_drops = [item for item in result.dropped_features if item.method == "low_variance"]
    assert [item.feature for item in low_variance_drops] == ["constant"]
    assert result.selected_features == ["varying"]


def test_result_schema_apply_projection() -> None:
    """SelectionResult.apply should keep only selected + service columns."""
    categorical, continuous, columns = make_wide_schema_columns(30)
    spark = FakeSparkSession()
    config = _config()
    schema = _schema(categorical, continuous)
    result = FeatureSelectionPipeline(config).fit_select(
        spark,
        data=make_pandas_frame(columns),
        schema=schema,
    )
    fresh_data = make_pandas_frame(columns)
    projected = result.apply(fresh_data)
    keep = set(result.columns_to_keep())
    assert set(projected.columns) == keep & set(columns)


def test_no_service_columns_in_selected() -> None:
    """target, split, time, id columns must never appear in selected_features."""
    categorical, continuous, columns = make_wide_schema_columns(30)
    spark = FakeSparkSession()
    schema = _schema(categorical, continuous)
    result = FeatureSelectionPipeline(_config()).fit_select(
        spark,
        data=make_pandas_frame(columns),
        schema=schema,
    )
    forbidden = set(schema.service_columns())
    assert not (set(result.selected_features) & forbidden)


def test_apply_fake_dataframe_selects_columns() -> None:
    """SelectionResult.apply should work on FakeDataFrame for schema/result tests."""
    categorical, continuous, columns = make_wide_schema_columns(30)
    spark = FakeSparkSession()
    result = FeatureSelectionPipeline(_config()).fit_select(
        spark,
        data=make_pandas_frame(columns),
        schema=_schema(categorical, continuous),
    )
    fake = FakeDataFrame(columns=columns)
    projected = result.apply(fake)
    for name in result.selected_features:
        assert name in projected.columns
