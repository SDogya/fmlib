"""Tests for FeatureSelectionConfig."""

from pathlib import Path

import pytest

from fmlib.feature_selection.base import StageContext, resolve_step_seed
from fmlib.feature_selection.config import FeatureSelectionConfig
from fmlib.feature_selection.exceptions import ConfigError
from fmlib.feature_selection.schema import FeatureSchema


def test_defaults_validate() -> None:
    config = FeatureSelectionConfig()
    config.validate()
    assert config.order == ()
    assert config.statistics.order == ()
    assert config.model.enabled is False
    assert config.model.method == "lightgbm"
    assert config.precise.enabled is False
    # "standard" scales every variance to 1.0, which makes min_variance inert.
    assert config.statistics.low_variance.scale_method == "robust"
    assert config.statistics.correlation.threshold == 0.95
    assert config.statistics.correlation.tie_break == "original_order"
    assert config.statistics.correlation.max_rows == 100_000
    assert config.statistics.cache.enabled is False
    assert config.statistics.cache.path is None
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
    assert [step.method for step in config.order] == ["iv"]
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
    assert [step.method for step in config.order] == ["feature_drop", "correlation"]
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


def test_feature_drop_config_roundtrip_from_dict(tmp_path: Path) -> None:
    drop_path = tmp_path / "drop.txt"
    drop_path.write_text("legacy_feature\n", encoding="utf-8")
    config = FeatureSelectionConfig.from_dict(
        {
            "preprocessing": {
                "feature_drop": {
                    "enabled": True,
                    "path": str(drop_path),
                    "strict": True,
                },
            },
        },
    )

    assert config.preprocessing.feature_drop.enabled is True
    assert config.preprocessing.feature_drop.path == str(drop_path)
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
        ({"seed": True}, "seed"),
        ({"seed": 1.5}, "seed"),
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


def test_lightgbm_selection_mode_roundtrip() -> None:
    config = FeatureSelectionConfig.from_dict(
        {
            "model": {
                "enabled": True,
                "method": "lightgbm",
                "params": {
                    "selection_mode": "vote",
                    "min_set_share": 0.6,
                    "lgbm_threshold": 0.8,
                    "shap_threshold": 0.9,
                },
            },
        },
    )
    assert config.model.params["selection_mode"] == "vote"
    assert config.model.params["min_set_share"] == 0.6
    payload = config.to_dict()["model"]["params"]
    assert payload["selection_mode"] == "vote"
    assert payload["min_set_share"] == 0.6


def test_params_seed_roundtrip() -> None:
    config = FeatureSelectionConfig.from_dict(
        {
            "model": {
                "enabled": True,
                "method": "lightgbm",
                "params": {"seed": 17, "n_jobs": -1},
            },
            "precise": {
                "method": "boruta_shap",
                "params": {"seed": 0, "n_jobs": 1},
            },
            "statistics": {"psi": {"seed": 17, "n_jobs": -1}},
            "execution": {"seed": 42},
        },
    )
    assert config.model.params["seed"] == 17
    assert config.precise.params["seed"] == 0
    assert config.statistics.psi.seed == 17
    payload = config.to_dict()
    assert payload["model"]["params"]["seed"] == 17
    assert payload["precise"]["params"]["seed"] == 0
    assert payload["statistics"]["psi"]["seed"] == 17


def test_params_seed_omitted_keeps_execution_default() -> None:
    config = FeatureSelectionConfig.from_dict(
        {
            "model": {"enabled": True, "method": "lightgbm", "params": {"n_jobs": 1}},
            "execution": {"seed": 42},
        },
    )
    assert "seed" not in config.model.params
    assert config.statistics.psi.seed is None
    assert config.execution.seed == 42


def test_order_step_seed_roundtrip() -> None:
    config = FeatureSelectionConfig.from_dict(
        {
            "order": [
                {"lightgbm": {"n_jobs": 1, "seed": 17}},
                {"psi": {"seed": 0}},
            ],
            "execution": {"seed": 42},
        },
    )
    assert config.order[0].params["seed"] == 17
    assert config.order[1].params["seed"] == 0


def test_order_nested_model_params_seed_roundtrip() -> None:
    config = FeatureSelectionConfig.from_dict(
        {
            "order": [
                {
                    "lightgbm": {
                        "method": "lightgbm",
                        "params": {"n_jobs": 1, "seed": 17},
                    },
                },
            ],
            "execution": {"seed": 42},
        },
    )
    context = StageContext(
        spark=None,
        datasets={},
        schema=None,
        config=config,
        seed=42,
        candidates=[],
    )
    assert resolve_step_seed(config.order[0].params, context) == 17


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "model": {
                    "enabled": True,
                    "method": "lightgbm",
                    "params": {"seed": True},
                },
            },
            "seed",
        ),
        (
            {
                "model": {
                    "enabled": True,
                    "method": "lightgbm",
                    "params": {"seed": 1.5},
                },
            },
            "seed",
        ),
        (
            {
                "model": {
                    "enabled": True,
                    "method": "lightgbm",
                    "params": {"seed": "17"},
                },
            },
            "seed",
        ),
        (
            {
                "model": {
                    "enabled": True,
                    "method": "catboost_rfe",
                    "params": {"seed": True, "parameters": {"iterations": 10}},
                    "selection": {"max_features": 5},
                },
            },
            "seed",
        ),
        (
            {
                "precise": {
                    "method": "boruta_shap",
                    "params": {"seed": True},
                },
            },
            "seed",
        ),
        (
            {"statistics": {"psi": {"seed": True}}},
            "seed",
        ),
        (
            {"order": [{"psi": {"seed": 1.5}}]},
            "seed",
        ),
    ],
)
def test_params_seed_rejects_non_int(payload: dict, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        FeatureSelectionConfig.from_dict(payload)


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ({"selection_mode": "mean"}, "selection_mode"),
        ({"min_set_share": 0.0}, "min_set_share"),
        ({"min_set_share": 1.5}, "min_set_share"),
        ({"min_set_share": True}, "min_set_share"),
    ],
)
def test_lightgbm_selection_mode_validation(
    params: dict,
    message: str,
) -> None:
    with pytest.raises(ConfigError, match=message):
        FeatureSelectionConfig.from_dict(
            {
                "model": {
                    "enabled": True,
                    "method": "lightgbm",
                    "params": params,
                },
            },
        )


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


def test_top_level_order_accepts_repeats_and_inline_params() -> None:
    config = FeatureSelectionConfig.from_dict(
        {
            "order": [
                {"null_rate": {"threshold": 0.99}},
                {"null_rate": {"threshold": 0.9}},
            ],
            "model": {"enabled": True, "method": "lasso"},
            "statistics": {"order": ["correlation"]},
        },
    )
    assert [step.method for step in config.order] == ["null_rate", "null_rate"]
    assert config.order[0].params["threshold"] == 0.99
    assert config.order[1].params["threshold"] == 0.9
    assert config.to_dict()["order"] == [
        {"null_rate": {"threshold": 0.99}},
        {"null_rate": {"threshold": 0.9}},
    ]


def test_top_level_order_rejects_non_mapping_and_unknown_method() -> None:
    with pytest.raises(ConfigError, match="must be a mapping with exactly one"):
        FeatureSelectionConfig.from_dict({"order": ["null_rate"]})
    with pytest.raises(ConfigError, match="Unknown method in order"):
        FeatureSelectionConfig.from_dict({"order": [{"xgboost": {}}]})


def test_nested_layout_without_order_compiles_to_steps() -> None:
    config = FeatureSelectionConfig.from_dict(
        {
            "preprocessing": {
                "row_sample": {"enabled": True, "max_rows": 10, "stratified": False},
            },
            "statistics": {"order": ["null_rate"]},
            "model": {"enabled": True, "method": "lightgbm"},
            "precise": {"enabled": False, "method": "none"},
        },
    )
    assert [step.method for step in config.order] == [
        "row_sample",
        "null_rate",
        "lightgbm",
    ]
    assert config.order[0].params["max_rows"] == 10


def test_from_yaml_resolves_order_interpolations(tmp_path: Path) -> None:
    path = tmp_path / "fs.yaml"
    path.write_text(
        "\n".join(
            [
                "order:",
                "  - null_rate: ${null_rate.wide}",
                "  - null_rate: ${null_rate.tight}",
                "null_rate:",
                "  wide:",
                "    threshold: 0.99",
                "  tight:",
                "    threshold: 0.9",
                "execution:",
                "  seed: 7",
                "",
            ],
        ),
        encoding="utf-8",
    )
    config = FeatureSelectionConfig.from_yaml(path)
    assert [step.method for step in config.order] == ["null_rate", "null_rate"]
    assert config.order[0].params["threshold"] == 0.99
    assert config.order[1].params["threshold"] == 0.9
    assert config.execution.seed == 7


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
    assert [step.method for step in config.order] == [
        "null_rate",
        "correlation",
        "lightgbm",
    ]
    assert config.order[0].params["threshold"] == 0.9
    assert config.statistics.null_rate.threshold == 0.9
    assert config.model.enabled is True
    assert config.to_dict()["statistics"]["order"] == ["null_rate", "correlation"]


def test_empty_explicit_order_does_not_compile_nested() -> None:
    config = FeatureSelectionConfig.from_dict(
        {
            "order": [],
            "statistics": {"order": ["correlation"]},
            "model": {"enabled": True, "method": "lasso"},
        },
    )
    assert config.order == ()


def test_repeated_methods_get_indexed_score_keys() -> None:
    from types import SimpleNamespace

    from fmlib.feature_selection.runner import _relocate_step_scores

    context = SimpleNamespace(scores={"null_rate": {"threshold": 0.99}})
    _relocate_step_scores(context, "null_rate", 0)
    context.scores["null_rate"] = {"threshold": 0.9}
    _relocate_step_scores(context, "null_rate", 3)
    assert context.scores == {
        "null_rate#0": {"threshold": 0.99},
        "null_rate#3": {"threshold": 0.9},
    }


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


def test_feature_drop_missing_file_is_config_error(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.txt"
    with pytest.raises(ConfigError, match="file not found"):
        FeatureSelectionConfig.from_dict(
            {
                "preprocessing": {
                    "feature_drop": {
                        "enabled": True,
                        "path": str(missing),
                    },
                },
            },
        )


def test_catboost_rfe_requires_parameters_when_enabled() -> None:
    with pytest.raises(ConfigError, match="parameters is required"):
        FeatureSelectionConfig.from_dict(
            {
                "model": {
                    "enabled": True,
                    "method": "catboost_rfe",
                    "selection": {"max_features": 5},
                },
            },
        )


def test_catboost_rfe_requires_max_features_when_enabled() -> None:
    with pytest.raises(ConfigError, match="max_features"):
        FeatureSelectionConfig.from_dict(
            {
                "model": {
                    "enabled": True,
                    "method": "catboost_rfe",
                    "params": {"parameters": {"iterations": 10}},
                    "selection": {"max_features": None},
                },
            },
        )


def test_catboost_rfe_rejects_alias_clash() -> None:
    with pytest.raises(ConfigError, match="aliases together"):
        FeatureSelectionConfig.from_dict(
            {
                "model": {
                    "enabled": True,
                    "method": "catboost_rfe",
                    "params": {
                        "parameters": {"iterations": 10, "n_estimators": 10},
                    },
                    "selection": {"max_features": 5},
                },
            },
        )


def test_lightgbm_rejects_alias_clash_and_typo() -> None:
    with pytest.raises(ConfigError, match="aliases together"):
        FeatureSelectionConfig.from_dict(
            {
                "model": {
                    "enabled": True,
                    "method": "lightgbm",
                    "params": {
                        "parameters": {"n_estimators": 8, "num_iterations": 8},
                    },
                },
            },
        )
    with pytest.raises(ConfigError, match="unknown parameter"):
        FeatureSelectionConfig.from_dict(
            {
                "model": {
                    "enabled": True,
                    "method": "lightgbm",
                    "params": {"parameters": {"n_estmators": 8}},
                },
            },
        )
    with pytest.raises(ConfigError, match="unknown parameter"):
        FeatureSelectionConfig.from_dict(
            {
                "model": {
                    "enabled": True,
                    "method": "lightgbm",
                    "params": {"parameters": {"num_leavees": 16}},
                },
            },
        )


def test_lightgbm_rejects_forced_parameter_keys() -> None:
    with pytest.raises(ConfigError, match="must not set"):
        FeatureSelectionConfig.from_dict(
            {
                "model": {
                    "enabled": True,
                    "method": "lightgbm",
                    "params": {"parameters": {"random_state": 1}},
                },
            },
        )


def test_boruta_rejects_rf_key_on_lgbm() -> None:
    with pytest.raises(ConfigError, match="unknown parameter"):
        FeatureSelectionConfig.from_dict(
            {
                "precise": {
                    "method": "boruta_shap",
                    "params": {
                        "model_type": "lgbm",
                        "parameters": {"min_samples_split": 2},
                    },
                },
            },
        )


def test_malformed_optuna_spec_is_config_error() -> None:
    with pytest.raises(ConfigError, match="must define 'min' and 'max'"):
        FeatureSelectionConfig.from_dict(
            {
                "model": {
                    "enabled": True,
                    "method": "lightgbm",
                    "params": {
                        "parameters": {"num_leaves": {"type": "int"}},
                    },
                },
            },
        )


def test_psi_num_bins_must_be_at_least_two() -> None:
    with pytest.raises(ConfigError, match=r"psi\.num_bins"):
        FeatureSelectionConfig.from_dict({"statistics": {"psi": {"num_bins": 0}}})
    with pytest.raises(ConfigError, match=r"num_bins"):
        FeatureSelectionConfig.from_dict({"order": [{"psi": {"num_bins": 0}}]})


def test_statistics_cache_roundtrip() -> None:
    config = FeatureSelectionConfig.from_dict(
        {
            "statistics": {
                "cache": {
                    "enabled": True,
                    "path": "metrics.json",
                    "force_recompute": True,
                },
            },
        },
    )
    assert config.statistics.cache.enabled is True
    assert config.statistics.cache.path == "metrics.json"
    assert config.statistics.cache.force_recompute is True
    payload = config.to_dict()["statistics"]["cache"]
    assert payload == {
        "enabled": True,
        "path": "metrics.json",
        "force_recompute": True,
    }


def test_order_prerequisites_require_target_and_time() -> None:
    from fmlib.feature_selection.runner import validate_order_prerequisites

    lightgbm_config = FeatureSelectionConfig.from_dict({"order": [{"lightgbm": {}}]})
    no_target = FeatureSchema(
        categorical=(),
        continuous=("a",),
        target=None,
        task_type="binary_classification",
    )
    context = StageContext(
        spark=None,
        datasets={"train": None},
        schema=no_target,
        config=lightgbm_config,
        seed=0,
        candidates=["a"],
    )
    with pytest.raises(ConfigError, match="target"):
        validate_order_prerequisites(context)

    catboost_config = FeatureSelectionConfig.from_dict(
        {
            "order": [
                {
                    "catboost_rfe": {
                        "parameters": {"iterations": 10},
                        "selection": {"max_features": 5},
                    },
                },
            ],
        },
    )
    no_time = FeatureSchema(
        categorical=(),
        continuous=("a",),
        target="response",
        task_type="binary_classification",
    )
    context = StageContext(
        spark=None,
        datasets={"train": None},
        schema=no_time,
        config=catboost_config,
        seed=0,
        candidates=["a"],
    )
    with pytest.raises(ConfigError, match="FeatureSchema.time"):
        validate_order_prerequisites(context)


@pytest.mark.parametrize(
    ("relpath", "methods"),
    [
        ("examples/configs/feature_selection/iv.yaml", ("iv",)),
        ("examples/configs/feature_selection/null.yaml", ("null_rate", "lasso")),
        (
            "examples/big_c/main_conf.yaml",
            ("null_rate", "constants", "low_variance", "correlation", "psi", "lightgbm"),
        ),
        (
            "examples/configs/feature_selection/pipeline_lightgbm.yaml",
            (
                "null_rate",
                "constants",
                "low_variance",
                "correlation",
                "lightgbm",
                "boruta_shap",
            ),
        ),
    ],
)
def test_example_yamls_expose_top_level_order(
    relpath: str,
    methods: tuple[str, ...],
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    config = FeatureSelectionConfig.from_yaml(repo_root / relpath)
    assert tuple(step.method for step in config.order) == methods


