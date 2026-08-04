"""Tests for FeatureSelectionConfig."""

from pathlib import Path

import pytest

from fmlib.feature_selection.config import FeatureSelectionConfig
from fmlib.feature_selection.exceptions import ConfigError


def test_defaults_validate() -> None:
    config = FeatureSelectionConfig()
    config.validate()
    assert config.model.method == "catboost_rfe"
    assert config.statistics.low_variance.enabled is True
    assert config.statistics.low_variance.scale_method == "standard"
    assert config.statistics.correlation.threshold == 0.95
    assert config.statistics.correlation.tie_break == "original_order"
    assert config.statistics.correlation.max_rows == 100_000
    assert config.statistics.psi.enabled is False
    assert config.precise.method == "none"


def test_from_dict_and_unknown_field() -> None:
    payload = {
        "statistics": {"null_rate": {"enabled": True, "threshold": 0.9}},
        "model": {"method": "lasso"},
        "precise": {"method": None},
        "execution": {"seed": 7},
    }
    config = FeatureSelectionConfig.from_dict(payload)
    assert config.model.method == "lasso"
    assert config.precise.method == "none"
    assert config.execution.seed == 7

    with pytest.raises(ConfigError, match="Unknown fields"):
        FeatureSelectionConfig.from_dict({"extra": 1})


def test_unsupported_model_method() -> None:
    with pytest.raises(ConfigError, match=r"model\.method"):
        FeatureSelectionConfig.from_dict({"model": {"method": "xgboost"}})


def test_low_variance_validation() -> None:
    with pytest.raises(ConfigError, match=r"low_variance\.scale_method"):
        FeatureSelectionConfig.from_dict({"statistics": {"low_variance": {"scale_method": "unit"}}})
    with pytest.raises(ConfigError, match=r"low_variance\.min_variance"):
        FeatureSelectionConfig.from_dict({"statistics": {"low_variance": {"min_variance": -0.1}}})


def test_constants_chunk_size_validation() -> None:
    with pytest.raises(ConfigError, match=r"constants\.chunk_size"):
        FeatureSelectionConfig.from_dict({"statistics": {"constants": {"chunk_size": 0}}})


def test_correlation_max_rows_validation() -> None:
    config = FeatureSelectionConfig.from_dict({"statistics": {"correlation": {"max_rows": 5000}}})
    assert config.statistics.correlation.max_rows == 5000

    with pytest.raises(ConfigError, match=r"correlation\.max_rows"):
        FeatureSelectionConfig.from_dict({"statistics": {"correlation": {"max_rows": 0}}})


def test_from_yaml_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "fs.yaml"
    path.write_text(
        "\n".join(
            [
                "statistics:",
                "  correlation:",
                "    enabled: true",
                "    method: spearman",
                "    threshold: 0.85",
                "model:",
                "  method: random_forest",
                "precise:",
                "  method: boruta_shap",
                "execution:",
                "  seed: 123",
                "  max_local_rows: 5000",
                "",
            ],
        ),
        encoding="utf-8",
    )
    config = FeatureSelectionConfig.from_yaml(path)
    assert config.statistics.correlation.method == "spearman"
    assert config.model.method == "random_forest"
    assert config.precise.method == "boruta_shap"
    assert config.execution.seed == 123
    assert config.to_dict()["execution"]["max_local_rows"] == 5000
