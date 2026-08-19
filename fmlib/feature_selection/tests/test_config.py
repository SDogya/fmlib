"""Tests for FeatureSelectionConfig."""

from pathlib import Path

import pytest

from fmlib.feature_selection.config import FeatureSelectionConfig
from fmlib.feature_selection.exceptions import ConfigError


def test_defaults_validate() -> None:
    config = FeatureSelectionConfig()
    config.validate()
    assert config.statistics.order == ()
    assert config.model.enabled is False
    assert config.model.method == "lightgbm"
    assert config.precise.enabled is False
    assert config.statistics.low_variance.scale_method == "standard"
    assert config.statistics.correlation.threshold == 0.95
    assert config.statistics.correlation.tie_break == "original_order"
    assert config.statistics.correlation.max_rows == 100_000
    assert config.precise.method == "none"


def test_from_dict_and_unknown_field() -> None:
    payload = {
        "statistics": {"null_rate": {"threshold": 0.9}},
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


def test_iv_config_validation_and_order() -> None:
    config = FeatureSelectionConfig.from_dict(
        {
            "statistics": {
                "order": ["iv"],
                "iv": {
                    "threshold": 0.05,
                    "num_bins": 8,
                    "max_threshold": 5.0,
                    "min_bin_share": 0.01,
                    "max_levels": 20,
                },
            },
        },
    )
    assert config.statistics.order == ("iv",)
    assert config.statistics.iv.threshold == 0.05
    assert config.statistics.iv.num_bins == 8
    assert config.statistics.iv.max_threshold == 5.0
    assert config.execution.verbose.iv is False

    with pytest.raises(ConfigError, match=r"iv\.threshold"):
        FeatureSelectionConfig.from_dict({"statistics": {"iv": {"threshold": -0.1}}})
    with pytest.raises(ConfigError, match=r"iv\.num_bins"):
        FeatureSelectionConfig.from_dict({"statistics": {"iv": {"num_bins": 1}}})
    with pytest.raises(ConfigError, match=r"iv\.max_threshold"):
        FeatureSelectionConfig.from_dict(
            {"statistics": {"iv": {"threshold": 0.5, "max_threshold": 0.1}}},
        )
    with pytest.raises(ConfigError, match=r"iv\.eps"):
        FeatureSelectionConfig.from_dict({"statistics": {"iv": {"eps": 0}}})
    with pytest.raises(ConfigError, match=r"iv\.min_bin_share"):
        FeatureSelectionConfig.from_dict({"statistics": {"iv": {"min_bin_share": 1}}})
    with pytest.raises(ConfigError, match=r"iv\.batch_size"):
        FeatureSelectionConfig.from_dict({"statistics": {"iv": {"batch_size": 0}}})


def test_from_yaml_roundtrip(tmp_path: Path) -> None:
    drop_path = tmp_path / "drop_features.txt"
    drop_path.write_text("legacy_feature\n", encoding="utf-8")
    path = tmp_path / "fs.yaml"
    path.write_text(
        "\n".join(
            [
                "preprocessing:",
                "  feature_drop:",
                "    enabled: true",
                "    path: drop_features.txt",
                "    strict: false",
                "statistics:",
                "  order:",
                "    - correlation",
                "  correlation:",
                "    method: spearman",
                "    threshold: 0.85",
                "model:",
                "  enabled: false",
                "  method: random_forest",
                "precise:",
                "  enabled: false",
                "  method: boruta_shap",
                "  params:",
                "    model_type: rf",
                "    boruta_trials: 12",
                "    optuna_params:",
                "      sampler: RANDOM",
                "      n_startup_trials: 2",
                "      n_trials: 4",
                "      timeout: 90",
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
    assert config.statistics.order == ("correlation",)
    assert config.preprocessing.feature_drop.enabled is True
    assert config.preprocessing.feature_drop.path == str(drop_path)
    assert config.model.enabled is False
    assert config.model.method == "random_forest"
    assert config.precise.enabled is False
    assert config.precise.method == "boruta_shap"
    assert config.precise.params["model_type"] == "rf"
    assert config.precise.params["boruta_trials"] == 12
    assert config.precise.params["optuna_params"]["n_trials"] == 4
    assert config.precise.params["optuna_params"]["timeout"] == 90
    assert config.precise.params["optuna_params"]["sampler"] == "RANDOM"
    assert config.execution.seed == 123
    assert config.to_dict()["execution"]["max_local_rows"] == 5000


def test_feature_drop_config_requires_path_when_enabled() -> None:
    with pytest.raises(ConfigError, match=r"feature_drop\.path"):
        FeatureSelectionConfig.from_dict(
            {
                "preprocessing": {
                    "feature_drop": {"enabled": True},
                },
            },
        )


def test_feature_drop_config_roundtrip_from_dict() -> None:
    config = FeatureSelectionConfig.from_dict(
        {
            "preprocessing": {
                "feature_drop": {
                    "enabled": True,
                    "path": "/tmp/drop.txt",
                    "strict": True,
                },
            },
        },
    )

    assert config.preprocessing.feature_drop.enabled is True
    assert config.preprocessing.feature_drop.path == "/tmp/drop.txt"
    assert config.preprocessing.feature_drop.strict is True
    assert config.to_dict()["preprocessing"]["feature_drop"]["enabled"] is True


def test_feature_drop_without_enabled_needs_no_path_and_rejects_unknown_fields() -> None:
    config = FeatureSelectionConfig.from_dict(
        {
            "preprocessing": {
                "feature_drop": {},
            },
        },
    )
    assert config.preprocessing.feature_drop.path is None
    assert config.preprocessing.feature_drop.enabled is False

    with pytest.raises(ConfigError, match="Unknown fields"):
        FeatureSelectionConfig.from_dict(
            {
                "preprocessing": {
                    "feature_drop": {
                        "path": "/tmp/drop.txt",
                        "extra": True,
                    },
                },
            },
        )

    with pytest.raises(ConfigError, match="Unknown fields"):
        FeatureSelectionConfig.from_dict(
            {
                "statistics": {
                    "null_rate": {"enabled": True},
                },
            },
        )


def test_test_run_preprocessing_config_roundtrip() -> None:
    config = FeatureSelectionConfig.from_dict(
        {
            "preprocessing": {
                "random_feature_drop": {
                    "enabled": True,
                    "n_features": 12,
                },
                "row_sample": {
                    "enabled": True,
                    "max_rows": 50_000,
                    "stratified": False,
                },
            },
        },
    )

    assert config.preprocessing.random_feature_drop.enabled is True
    assert config.preprocessing.random_feature_drop.n_features == 12
    assert config.preprocessing.row_sample.enabled is True
    assert config.preprocessing.row_sample.max_rows == 50_000
    assert config.preprocessing.row_sample.stratified is False
    payload = config.to_dict()["preprocessing"]
    assert payload["random_feature_drop"]["enabled"] is True
    assert payload["row_sample"]["max_rows"] == 50_000


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "preprocessing": {"random_feature_drop": {"enabled": True, "n_features": 0}},
            },
            "random_feature_drop.n_features",
        ),
        (
            {
                "preprocessing": {"row_sample": {"enabled": True, "max_rows": 0}},
            },
            "row_sample.max_rows",
        ),
        (
            {
                "preprocessing": {
                    "row_sample": {
                        "enabled": True,
                        "max_rows": 100,
                        "stratified": "yes",
                    },
                },
            },
            "row_sample.stratified",
        ),
    ],
)
def test_test_run_preprocessing_config_validation(
    payload: dict,
    message: str,
) -> None:
    with pytest.raises(ConfigError, match=message):
        FeatureSelectionConfig.from_dict(payload)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"model_type": "xgb"}, "model_type"),
        ({"boruta_trials": 0}, "boruta_trials"),
        ({"n_trials": 0}, "n_trials"),
        ({"max_rows": 0}, "max_rows"),
        ({"max_rows_limit": 0}, "max_rows_limit"),
        ({"sample_fraction": 1.5}, "sample_fraction"),
        ({"tentative_fix_method": "median"}, "tentative_fix_method"),
        ({"parameters": []}, "parameters"),
        ({"optuna_params": []}, "optuna_params"),
        (
            {"optuna_params": {"sampler": "CMAES"}},
            "sampler",
        ),
        (
            {"optuna_params": {"n_trials": 0}},
            "optuna_params.n_trials",
        ),
        (
            {"optuna_params": {"niter": 0}},
            "optuna_params.niter",
        ),
        ({"optuna_params": {"n_startup_trials": 0}},
            "optuna_params.n_startup_trials",
        ),
        ({"optuna_params": {"enabled": "yes"}}, "optuna_params.enabled"),
        ({"n_jobs": 0}, "n_jobs"),
        ({"n_jobs": -2}, "n_jobs"),
    ],
)
def test_boruta_precise_params_validation(
    params: dict,
    message: str,
) -> None:
    with pytest.raises(ConfigError, match=message):
        FeatureSelectionConfig.from_dict(
            {
                "precise": {
                    "method": "boruta_shap",
                    "params": params,
                },
            },
        )


def test_boruta_grid_requires_finite_values() -> None:
    with pytest.raises(ConfigError, match="non-empty 'values'"):
        FeatureSelectionConfig.from_dict(
            {
                "precise": {
                    "method": "boruta_shap",
                    "params": {
                        "parameters": {
                            "max_depth": {
                                "type": "int",
                                "min": 3,
                                "max": 5,
                            },
                        },
                        "optuna_params": {"sampler": "GRID"},
                    },
                },
            },
        )

    config = FeatureSelectionConfig.from_dict(
        {
            "precise": {
                "method": "boruta_shap",
                "params": {
                    "parameters": {
                        "max_depth": {"values": [3, 5]},
                    },
                    "optuna_params": {"sampler": "GRID"},
                },
            },
        },
    )
    assert config.precise.params["parameters"]["max_depth"]["values"] == [3, 5]


def test_optuna_params_roundtrip() -> None:
    config = FeatureSelectionConfig.from_dict(
        {
            "model": {
                "method": "lightgbm",
                "params": {
                    "optuna_mode": "global",
                    "optuna_params": {
                        "enabled": True,
                        "n_trials": 11,
                        "timeout": 45,
                        "sampler": "TPE",
                    },
                },
            },
            "precise": {
                "method": "boruta_shap",
                "params": {
                    "boruta_trials": 8,
                    "optuna_params": {
                        "enabled": False,
                        "n_trials": 6,
                        "timeout": 30,
                        "sampler": "TPE",
                    },
                },
            },
        },
    )

    assert config.model.params["optuna_params"]["n_trials"] == 11
    assert config.model.params["optuna_params"]["enabled"] is True
    assert config.precise.params["optuna_params"]["n_trials"] == 6
    assert config.precise.params["optuna_params"]["enabled"] is False
    payload = config.to_dict()
    assert payload["model"]["params"]["optuna_params"]["timeout"] == 45
    assert payload["precise"]["params"]["optuna_params"]["n_trials"] == 6
    assert payload["precise"]["params"]["optuna_params"]["enabled"] is False
    assert "tuning" not in payload["model"]
    assert "tuning" not in payload["precise"]


def test_tuning_block_is_rejected() -> None:
    with pytest.raises(ConfigError, match="params.optuna_params"):
        FeatureSelectionConfig.from_dict(
            {
                "model": {
                    "method": "lightgbm",
                    "tuning": {"enabled": True, "n_trials": 10},
                },
            },
        )
    with pytest.raises(ConfigError, match="params.optuna_params"):
        FeatureSelectionConfig.from_dict(
            {
                "precise": {
                    "method": "boruta_shap",
                    "tuning": {"enabled": True, "n_trials": 9},
                },
            },
        )


def test_legacy_n_trials_in_lightgbm_params_is_allowed() -> None:
    config = FeatureSelectionConfig.from_dict(
        {
            "model": {
                "method": "lightgbm",
                "params": {"n_trials": 5, "driver_n_jobs": 4},
            },
        },
    )
    assert config.model.params["n_trials"] == 5
    assert config.model.params["driver_n_jobs"] == 4


def test_verbose_false_by_default() -> None:
    config = FeatureSelectionConfig.from_dict({"execution": {"seed": 1}})
    assert config.execution.verbose.any_enabled() is False
    assert config.execution.verbose.lightgbm is False
    assert config.to_dict()["execution"]["verbose"]["pipeline"] is False


def test_verbose_true_enables_all_methods() -> None:
    config = FeatureSelectionConfig.from_dict({"execution": {"verbose": True}})
    flags = config.to_dict()["execution"]["verbose"]
    assert config.execution.verbose.any_enabled() is True
    assert flags["pipeline"] is True
    assert flags["lightgbm"] is True
    assert flags["boruta_shap"] is True
    assert flags["correlation"] is True


def test_verbose_per_method_mapping() -> None:
    config = FeatureSelectionConfig.from_dict(
        {
            "execution": {
                "verbose": {
                    "lightgbm": True,
                    "correlation": True,
                },
            },
        },
    )
    assert config.execution.verbose.lightgbm is True
    assert config.execution.verbose.correlation is True
    assert config.execution.verbose.pipeline is False
    assert config.execution.verbose.psi is False


def test_verbose_rejects_unknown_method_and_non_bool() -> None:
    with pytest.raises(ConfigError, match="execution.verbose"):
        FeatureSelectionConfig.from_dict(
            {"execution": {"verbose": {"not_a_method": True}}},
        )
    with pytest.raises(ConfigError, match="must be boolean"):
        FeatureSelectionConfig.from_dict(
            {"execution": {"verbose": {"lightgbm": 1}}},
        )
    with pytest.raises(ConfigError, match="must be a boolean or a mapping"):
        FeatureSelectionConfig.from_dict(
            {"execution": {"verbose": "lightgbm"}},
        )


def test_statistics_order_rejects_duplicates_and_unknown() -> None:
    with pytest.raises(ConfigError, match="Duplicate method in statistics.order"):
        FeatureSelectionConfig.from_dict(
            {"statistics": {"order": ["null_rate", "correlation", "null_rate"]}},
        )
    with pytest.raises(ConfigError, match="Unknown method in statistics.order"):
        FeatureSelectionConfig.from_dict(
            {"statistics": {"order": ["null_rate", "xgboost"]}},
        )
    with pytest.raises(ConfigError, match="statistics.order must be a list"):
        FeatureSelectionConfig.from_dict({"statistics": {"order": "lightgbm"}})
    with pytest.raises(ConfigError, match="must be a method name"):
        FeatureSelectionConfig.from_dict(
            {"statistics": {"order": [{"null_rate": {"threshold": 0.9}}]}},
        )


def test_top_level_order_is_rejected() -> None:
    with pytest.raises(ConfigError, match="Top-level order is not supported"):
        FeatureSelectionConfig.from_dict({"order": ["null_rate"]})


def test_statistics_order_roundtrip() -> None:
    config = FeatureSelectionConfig.from_dict(
        {
            "statistics": {
                "order": ["null_rate", "correlation"],
                "null_rate": {"threshold": 0.9},
            },
            "model": {"enabled": True, "method": "lightgbm"},
            "precise": {"enabled": False, "method": "none"},
        },
    )
    assert config.statistics.order == ("null_rate", "correlation")
    assert config.statistics.null_rate.threshold == 0.9
    assert config.model.enabled is True
    assert config.to_dict()["statistics"]["order"] == ["null_rate", "correlation"]


def test_precise_enabled_requires_boruta_shap() -> None:
    with pytest.raises(ConfigError, match="boruta_shap"):
        FeatureSelectionConfig.from_dict(
            {"precise": {"enabled": True, "method": "none"}},
        )
    config = FeatureSelectionConfig.from_dict(
        {"precise": {"enabled": False, "method": "boruta_shap"}},
    )
    assert config.precise.enabled is False
    assert config.precise.method == "boruta_shap"


def test_catboost_rfe_requires_max_features_when_enabled() -> None:
    with pytest.raises(ConfigError, match="max_features"):
        FeatureSelectionConfig.from_dict(
            {
                "model": {
                    "enabled": True,
                    "method": "catboost_rfe",
                    "selection": {"max_features": None},
                },
            },
        )


