"""Typed configuration for the feature selection pipeline."""

from __future__ import annotations

from collections.abc import Sequence as AbcSequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from fmlib.feature_selection.exceptions import ConfigError

PSI_MODES = frozenset({"month_over_month", "train_valid"})
MODEL_METHODS = frozenset({"lasso", "random_forest", "catboost_rfe", "lightgbm"})
PRECISE_METHODS = frozenset({"boruta_shap", "none"})
BORUTA_MODEL_TYPES = frozenset({"lgbm", "rf"})
BORUTA_SAMPLERS = frozenset({"TPE", "RANDOM", "GRID"})
OPTUNA_SAMPLERS = frozenset({"TPE", "RANDOM", "GRID"})
CATBOOST_RFE_ALGORITHMS = frozenset(
    {
        "RecursiveByLossFunctionChange",
        "RecursiveByShapValues",
        "RecursiveByPredictionValuesChange",
    },
)
CV_STRATEGIES = frozenset({"random", "stratified", "group", "time_based"})
SAMPLE_STRATEGIES = frozenset({"random", "stratified"})
CORRELATION_METHODS = frozenset({"pearson", "spearman"})
CORRELATION_TIE_BREAKS = frozenset({"null_rate", "original_order"})
LOW_VARIANCE_SCALE_METHODS = frozenset({"standard", "minmax", "robust"})
VERBOSE_METHODS = (
    "pipeline",
    "feature_drop",
    "random_feature_drop",
    "row_sample",
    "null_rate",
    "constants",
    "low_variance",
    "correlation",
    "psi",
    "iv",
    "stability_classifier",
    "lightgbm",
    "lasso",
    "random_forest",
    "catboost_rfe",
    "boruta_shap",
)
STATISTICS_ORDER_METHODS = (
    "null_rate",
    "constants",
    "low_variance",
    "correlation",
    "psi",
    "iv",
    "stability_classifier",
)


def parse_statistics_order(raw: Any) -> tuple[str, ...]:
    """Parse ``statistics.order`` as unique method names.

    Parameter blocks live under ``statistics.<method>``, not in the list.
    Duplicates and unknown names are rejected.
    """
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, AbcSequence):
        msg = "statistics.order must be a list of method names."
        raise ConfigError(msg)
    names: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, str) or not item.strip():
            msg = (
                f"statistics.order[{index}] must be a method name. "
                "Put parameters under statistics.<method>, not in the list."
            )
            raise ConfigError(msg)
        if item in seen:
            msg = f"Duplicate method in statistics.order: {item!r}."
            raise ConfigError(msg)
        if item not in STATISTICS_ORDER_METHODS:
            msg = (
                f"Unknown method in statistics.order: {item!r}. "
                f"Expected one of: {list(STATISTICS_ORDER_METHODS)}."
            )
            raise ConfigError(msg)
        seen.add(item)
        names.append(item)
    return tuple(names)


def _reject_unknown(section: str, payload: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        message = f"Unknown fields in {section}: {unknown}. Remove them from the config."
        raise ConfigError(message)


def _require_bool(section: str, value: Any) -> None:
    if not isinstance(value, bool):
        msg = f"{section} must be boolean."
        raise ConfigError(msg)


def _reject_tuning_block(section: str, payload: Mapping[str, Any]) -> None:
    """Reject the removed ``tuning`` block in favour of ``params.optuna_params``."""
    if "tuning" not in payload:
        return
    msg = (
        f"{section}.tuning is not supported. Put Optuna settings in "
        f"{section}.params.optuna_params."
    )
    raise ConfigError(msg)


def _require_positive_int(value: Any, name: str) -> None:
    """Raise when a value is present but is not a positive integer."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = f"{name} must be a positive integer."
        raise ConfigError(msg)


def _validate_optuna_params_block(section: str, params: Mapping[str, Any]) -> None:
    """Validate the shared ``params.optuna_params`` mapping."""
    optuna_params = params.get("optuna_params", {})
    if not isinstance(optuna_params, Mapping):
        msg = f"{section}.params.optuna_params must be a mapping."
        raise ConfigError(msg)
    for name in ("n_trials", "n_startup_trials", "timeout"):
        _require_positive_int(
            optuna_params.get(name),
            f"{section}.params.optuna_params.{name}",
        )
    sampler = str(optuna_params.get("sampler", "TPE")).upper()
    if sampler not in OPTUNA_SAMPLERS:
        msg = (
            f"Unsupported {section}.params.optuna_params.sampler={sampler!r}. "
            f"Expected one of: {sorted(OPTUNA_SAMPLERS)}."
        )
        raise ConfigError(msg)


def _validate_catboost_rfe_params(params: Mapping[str, Any]) -> None:
    """Validate method-specific CatBoost RFE configuration.

    Only the shape of provided values is checked here. Presence of
    ``parameters`` is enforced by the selector at run time so a disabled
    default config stays constructible.
    """
    parameters = params.get("parameters", {})
    if not isinstance(parameters, Mapping):
        msg = (
            "model.params.parameters must be a mapping of CatBoost parameters, "
            "given as scalars and/or as search spaces such as "
            "{'type': 'int', 'min': 4, 'max': 8}."
        )
        raise ConfigError(msg)

    for name in ("eval_months", "max_rows"):
        _require_positive_int(params.get(name), f"model.params.{name}")

    sample_fraction = params.get("sample_fraction")
    if sample_fraction is not None and (
        isinstance(sample_fraction, bool)
        or not isinstance(sample_fraction, (int, float))
        or not 0.0 < float(sample_fraction) <= 1.0
    ):
        msg = "model.params.sample_fraction must be in (0, 1]."
        raise ConfigError(msg)

    _validate_optuna_params_block("model", params)

    selection_params = params.get("feature_selection_params", {})
    if not isinstance(selection_params, Mapping):
        msg = "model.params.feature_selection_params must be a mapping."
        raise ConfigError(msg)
    _require_positive_int(
        selection_params.get("steps"),
        "model.params.feature_selection_params.steps",
    )
    algorithm = selection_params.get("algorithm")
    if algorithm is not None and algorithm not in CATBOOST_RFE_ALGORITHMS:
        msg = (
            f"Unsupported model.params.feature_selection_params.algorithm="
            f"{algorithm!r}. Expected one of: {sorted(CATBOOST_RFE_ALGORITHMS)}."
        )
        raise ConfigError(msg)


def _validate_lightgbm_params(params: Mapping[str, Any]) -> None:
    """Validate method-specific LightGBM configuration."""
    _require_positive_int(params.get("n_trials"), "model.params.n_trials")
    _validate_optuna_params_block("model", params)
    for name in (
        "n_folds",
        "max_rows",
        "shap_max_rows",
        "n_jobs",
    ):
        value = params.get(name)
        if value is None:
            continue
        if name == "n_jobs":
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value == 0
                or value < -1
            ):
                msg = "model.params.n_jobs must be -1 or a positive integer."
                raise ConfigError(msg)
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        ):
            msg = f"model.params.{name} must be a positive integer."
            raise ConfigError(msg)
        if name == "n_folds" and value < 2:
            msg = "model.params.n_folds must be at least 2."
            raise ConfigError(msg)

    for name in ("lgbm_threshold", "shap_threshold", "sample_fraction"):
        value = params.get(name)
        if value is None:
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 < float(value) <= 1.0
        ):
            msg = f"model.params.{name} must be in (0, 1]."
            raise ConfigError(msg)

    optuna_mode = params.get("optuna_mode")
    if optuna_mode is not None and str(optuna_mode).lower() not in {
        "global",
        "per_fold",
    }:
        msg = (
            "model.params.optuna_mode must be one of "
            "['global', 'per_fold']."
        )
        raise ConfigError(msg)


def _validate_boruta_params(params: Mapping[str, Any]) -> None:
    """Validate method-specific BorutaSHAP configuration."""
    model_type = params.get("model_type", "lgbm")
    if model_type not in BORUTA_MODEL_TYPES:
        msg = (
            f"Unsupported precise.params.model_type={model_type!r}. "
            f"Expected one of: {sorted(BORUTA_MODEL_TYPES)}."
        )
        raise ConfigError(msg)

    n_jobs = params.get("n_jobs")
    if n_jobs is not None and (
        isinstance(n_jobs, bool)
        or not isinstance(n_jobs, int)
        or n_jobs == 0
        or n_jobs < -1
    ):
        msg = "precise.params.n_jobs must be -1 or a positive integer."
        raise ConfigError(msg)

    for name in (
        "max_rows",
        "max_rows_limit",
        "n_trials",
        "optuna_trials",
        "boruta_trials",
    ):
        value = params.get(name)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        ):
            msg = f"precise.params.{name} must be a positive integer."
            raise ConfigError(msg)

    sample_fraction = params.get("sample_fraction")
    if sample_fraction is not None and (
        isinstance(sample_fraction, bool)
        or not isinstance(sample_fraction, (int, float))
        or not 0.0 < float(sample_fraction) <= 1.0
    ):
        msg = "precise.params.sample_fraction must be in (0, 1]."
        raise ConfigError(msg)

    tentative = params.get("tentative_fix_method", "rough")
    if tentative not in {None, "rough"}:
        msg = "precise.params.tentative_fix_method must be 'rough' or null."
        raise ConfigError(msg)

    parameters = params.get("parameters", {})
    if not isinstance(parameters, Mapping):
        msg = "precise.params.parameters must be a mapping."
        raise ConfigError(msg)
    optuna_params = params.get("optuna_params", {})
    if not isinstance(optuna_params, Mapping):
        msg = "precise.params.optuna_params must be a mapping."
        raise ConfigError(msg)

    for name in ("n_trials", "niter", "n_startup_trials", "timeout"):
        value = optuna_params.get(name)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
        ):
            msg = (
                f"precise.params.optuna_params.{name} must be a positive "
                "integer."
            )
            raise ConfigError(msg)

    sampler = str(optuna_params.get("sampler", "TPE")).upper()
    if sampler not in BORUTA_SAMPLERS:
        msg = (
            f"Unsupported precise.params.optuna_params.sampler={sampler!r}. "
            f"Expected one of: {sorted(BORUTA_SAMPLERS)}."
        )
        raise ConfigError(msg)
    if sampler == "GRID":
        finite = bool(parameters) and all(
            isinstance(spec, Mapping)
            and isinstance(spec.get("values"), (list, tuple))
            and bool(spec["values"])
            for spec in parameters.values()
        )
        if not finite:
            msg = (
                "precise.params Optuna GRID sampler requires every custom "
                "parameter to define a non-empty 'values' list."
            )
            raise ConfigError(msg)


@dataclass(frozen=True)
class NullRateConfig:
    """Null-rate statistical filter settings."""

    threshold: float = 0.95


@dataclass(frozen=True)
class ConstantsConfig:
    """Constant / quasi-constant filter settings."""

    max_frequency: float = 0.999
    min_unique: Optional[int] = None
    chunk_size: int = 1000


@dataclass(frozen=True)
class LowVarianceConfig:
    """Low-variance filter settings for continuous features."""

    min_variance: float = 0.01
    scale_method: str = "standard"


@dataclass(frozen=True)
class CorrelationConfig:
    """Pairwise correlation filter settings.

    Only ``FeatureSchema.continuous`` candidates are evaluated. Categorical features
    pass through unchanged. Correlation is computed on at most
    ``min(max_rows, execution.max_local_rows)`` train rows.

    Tie-breaking for a correlated pair:

    - ``original_order`` (default): drop the later candidate;
    - ``null_rate``: drop the feature with more nulls; on tie fall back to
      ``original_order``.
    """

    method: str = "pearson"
    threshold: float = 0.95
    tie_break: str = "original_order"
    max_rows: int = 100_000


@dataclass(frozen=True)
class PsiConfig:
    """Population Stability Index filter settings.

    When mode='train_valid' and explicit test data is not provided,
    the selector can split the train dataset by month_part column:
    - latest N months go to test (for PSI comparison)
    - all earlier months go to train

    Args:
        mode: 'train_valid' or 'month_over_month' comparison mode.
        threshold: PSI value threshold for feature exclusion.
        num_bins: Number of quantile bins for PSI calculation.
        month_column: Column name containing month identifier (for train_valid mode).
        test_months: Number of latest months to use as test set.
        eps: Small constant for probability adjustment in Pandas PSI calculation.
        relative_error: Relative error for approximate quantiles in PySpark PSI.
        batch_size: Column batch size for PySpark PSI computation.
        n_jobs: Number of parallel jobs for Pandas PSI calculation.
        subsample_rows: Maximum number of rows for stratified subsampling (optional).
    """

    mode: str = "train_valid"
    threshold: float = 0.25
    num_bins: int = 10
    month_column: str = "month_part"
    test_months: int = 1
    eps: float = 1e-4
    relative_error: float = 0.001
    batch_size: int = 100
    n_jobs: int = -1
    subsample_rows: Optional[int] = None


@dataclass(frozen=True)
class IvConfig:
    """Information Value filter for binary classification.

    Continuous features are split into ``num_bins`` quantile bins; categorical
    features use distinct values (rare / excess levels can be merged). Nulls
    form a separate bin. A feature is dropped when IV is strictly below
    ``threshold`` (and optionally when it exceeds ``max_threshold``).

    Args:
        threshold: Drop when IV is strictly below this value.
        num_bins: Number of quantile bins for continuous features.
        max_threshold: Optional upper bound; drop when IV is strictly above
            (typical leakage / ID-like columns). ``null`` disables the rule.
        eps: Smoothing added to good/bad shares inside the WoE logarithm.
        min_bin_share: Merge categorical levels whose row share is below this
            value into an ``other`` bin. ``0`` disables the rule.
        max_levels: Keep at most this many non-null categorical levels
            (most frequent); the rest go to ``other``. ``null`` disables.
        relative_error: Relative error for Spark ``approxQuantile``.
        batch_size: How many columns to aggregate in one Spark job.
    """

    threshold: float = 0.02
    num_bins: int = 10
    max_threshold: Optional[float] = None
    eps: float = 1e-4
    min_bin_share: float = 0.0
    max_levels: Optional[int] = 50
    relative_error: float = 0.001
    batch_size: int = 50


@dataclass(frozen=True)
class StabilityClassifierConfig:
    """Per-feature stability classifier settings."""

    metric: str = "roc_auc"
    threshold: float = 0.8


@dataclass(frozen=True)
class StatisticsConfig:
    """Statistical stage configuration."""

    order: tuple[str, ...] = ()
    null_rate: NullRateConfig = field(default_factory=NullRateConfig)
    constants: ConstantsConfig = field(default_factory=ConstantsConfig)
    low_variance: LowVarianceConfig = field(default_factory=LowVarianceConfig)
    correlation: CorrelationConfig = field(default_factory=CorrelationConfig)
    psi: PsiConfig = field(default_factory=PsiConfig)
    iv: IvConfig = field(default_factory=IvConfig)
    stability_classifier: StabilityClassifierConfig = field(default_factory=StabilityClassifierConfig)


@dataclass(frozen=True)
class ModelSelectionRuleConfig:
    """Rule applied to aggregated model importances."""

    max_features: Optional[int] = 100


@dataclass(frozen=True)
class CrossValidationConfig:
    """Cross-validation settings for the model stage."""

    strategy: str = "stratified"
    folds: int = 5
    metric: str = "roc_auc"


@dataclass(frozen=True)
class ModelConfig:
    """Model-based selection stage configuration.

    Args:
        enabled: When false, the model stage is skipped.
        method: Exactly one selector when ``enabled``. Supported values:
            - ``"lightgbm"``: LightGBM + SHAP importance (params:
              lgbm_threshold, shap_threshold, n_folds, n_trials, max_rows,
              sample_fraction, optuna_mode, n_jobs, shap_max_rows,
              parameters). Optuna lives in ``params.optuna_params``.
            - ``"catboost_rfe"``: CatBoost recursive elimination on an
              out-of-time split, with optional Optuna tuning (params:
              eval_months, max_rows, sample_fraction, parameters,
              optuna_params, feature_selection_params; the target feature
              count comes from ``selection.max_features``)
            - ``"random_forest"``: Random Forest importance (stub)
            - ``"lasso"``: Lasso-based selection (stub)
        params: Method-specific parameters. Methods that tune with Optuna
            read the shared ``params.optuna_params`` block (``n_trials``,
            ``n_startup_trials``, ``sampler``, ``timeout``).
        selection: Rule for selecting top features from importances.
        cross_validation: Cross-validation settings.
    """

    enabled: bool = False
    method: str = "lightgbm"
    params: dict[str, Any] = field(default_factory=dict)
    selection: ModelSelectionRuleConfig = field(default_factory=ModelSelectionRuleConfig)
    cross_validation: CrossValidationConfig = field(default_factory=CrossValidationConfig)


@dataclass(frozen=True)
class PreciseConfig:
    """Optional precise/final selection stage configuration.

    Args:
        enabled: When false, the precise stage is skipped.
        method: ``"boruta_shap"`` when ``enabled``; ``"none"`` otherwise.
        params: Method-specific BorutaSHAP parameters. Optuna settings
            live in ``params.optuna_params``.
    """

    enabled: bool = False
    method: Optional[str] = "none"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureDropConfig:
    """Manual feature exclusions loaded from a text file or result JSON."""

    enabled: bool = False
    path: Optional[str] = None
    strict: bool = False


@dataclass(frozen=True)
class RowSampleConfig:
    """Optional test-run row cap applied to every dataset split."""

    enabled: bool = False
    max_rows: Optional[int] = None
    stratified: bool = True


@dataclass(frozen=True)
class RandomFeatureDropConfig:
    """Optional deterministic random candidate exclusion for test runs."""

    enabled: bool = False
    n_features: int = 0


@dataclass(frozen=True)
class PreprocessingConfig:
    """Input preparation performed before schema validation and selectors."""

    feature_drop: FeatureDropConfig = field(default_factory=FeatureDropConfig)
    random_feature_drop: RandomFeatureDropConfig = field(
        default_factory=RandomFeatureDropConfig,
    )
    row_sample: RowSampleConfig = field(default_factory=RowSampleConfig)


@dataclass(frozen=True)
class LocalSampleConfig:
    """Sampling used before controlled Spark → local materialization."""

    strategy: str = "stratified"
    cache_intermediate: bool = True


@dataclass(frozen=True)
class VerboseConfig:
    """Per-method debug flags. Omitted methods stay silent.

    ``execution.verbose`` accepts ``true`` (all methods), ``false`` (none),
    or a mapping such as ``{lightgbm: true, correlation: true}``.
    """

    pipeline: bool = False
    feature_drop: bool = False
    random_feature_drop: bool = False
    row_sample: bool = False
    null_rate: bool = False
    constants: bool = False
    low_variance: bool = False
    correlation: bool = False
    psi: bool = False
    iv: bool = False
    stability_classifier: bool = False
    lightgbm: bool = False
    lasso: bool = False
    random_forest: bool = False
    catboost_rfe: bool = False
    boruta_shap: bool = False

    def any_enabled(self: VerboseConfig) -> bool:
        """Return whether at least one method is verbose."""
        return any(bool(getattr(self, name)) for name in VERBOSE_METHODS)

    @classmethod
    def all_enabled(cls: type[VerboseConfig]) -> VerboseConfig:
        """Turn on every known method flag."""
        return cls(**dict.fromkeys(VERBOSE_METHODS, True))


def parse_verbose(raw: Any) -> VerboseConfig:
    """Parse ``execution.verbose`` from a boolean or per-method mapping.

    Args:
        raw: ``True`` / ``False`` / ``None`` / mapping of method flags.

    Returns:
        Normalized ``VerboseConfig``.

    Raises:
        ConfigError: On unknown methods or non-boolean flags.
    """
    if raw is None or raw is False:
        return VerboseConfig()
    if raw is True:
        return VerboseConfig.all_enabled()
    if isinstance(raw, Mapping):
        unknown = sorted(set(raw) - set(VERBOSE_METHODS))
        if unknown:
            msg = (
                f"Unknown fields in execution.verbose: {unknown}. "
                f"Expected one of: {list(VERBOSE_METHODS)}."
            )
            raise ConfigError(msg)
        flags: dict[str, bool] = {}
        for name, value in raw.items():
            if not isinstance(value, bool):
                msg = (
                    f"execution.verbose.{name} must be boolean, "
                    f"got {type(value).__name__}."
                )
                raise ConfigError(msg)
            flags[name] = value
        return VerboseConfig(**flags)
    msg = (
        "execution.verbose must be a boolean or a mapping of method flags."
    )
    raise ConfigError(msg)


@dataclass(frozen=True)
class ExecutionConfig:
    """Execution, reproducibility and capacity settings."""

    seed: int = 42
    allow_local_fallback: bool = True
    max_local_rows: int = 1_000_000
    local_memory_limit: float = 4.0
    local_sample: LocalSampleConfig = field(default_factory=LocalSampleConfig)
    verbose: VerboseConfig = field(default_factory=VerboseConfig)


@dataclass(frozen=True)
class FeatureSelectionConfig:
    """Top-level configuration for ``FeatureSelectionPipeline``.

    Args:
        statistics: Statistical filters. ``statistics.order`` lists unique
            method names; parameters live under ``statistics.<method>``.
        model: Optional model-based selector (``enabled`` + one ``method``).
        precise: Optional precise selector (``enabled`` + ``method``).
        execution: Seeds, backend fallback and capacity limits.
        preprocessing: Input-level feature exclusions and preparation.
            Each helper runs before statistics when its ``enabled`` flag
            is true.
    """

    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    precise: PreciseConfig = field(default_factory=PreciseConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    def validate(self: FeatureSelectionConfig) -> None:
        """Validate enums and numeric constraints.

        Raises:
            ConfigError: On unsupported values or inconsistent settings.
        """
        parse_statistics_order(self.statistics.order)
        _require_bool("preprocessing.feature_drop.enabled", self.preprocessing.feature_drop.enabled)
        _require_bool(
            "preprocessing.random_feature_drop.enabled",
            self.preprocessing.random_feature_drop.enabled,
        )
        _require_bool("preprocessing.row_sample.enabled", self.preprocessing.row_sample.enabled)
        _require_bool("model.enabled", self.model.enabled)
        _require_bool("precise.enabled", self.precise.enabled)

        feature_drop = self.preprocessing.feature_drop
        if feature_drop.enabled and (
            feature_drop.path is None or not str(feature_drop.path).strip()
        ):
            msg = (
                "preprocessing.feature_drop.path is required when "
                "preprocessing.feature_drop.enabled is true."
            )
            raise ConfigError(msg)
        random_drop = self.preprocessing.random_feature_drop
        if random_drop.enabled and (
            isinstance(random_drop.n_features, bool)
            or not isinstance(random_drop.n_features, int)
            or random_drop.n_features < 1
        ):
            msg = (
                "preprocessing.random_feature_drop.n_features must be "
                "positive when random_feature_drop.enabled is true."
            )
            raise ConfigError(msg)
        row_sample = self.preprocessing.row_sample
        if not isinstance(row_sample.stratified, bool):
            msg = "preprocessing.row_sample.stratified must be boolean."
            raise ConfigError(msg)
        if row_sample.enabled and (
            isinstance(row_sample.max_rows, bool)
            or not isinstance(row_sample.max_rows, int)
            or row_sample.max_rows < 1
        ):
            msg = (
                "preprocessing.row_sample.max_rows must be positive "
                "when row_sample.enabled is true."
            )
            raise ConfigError(msg)
        if self.statistics.correlation.method not in CORRELATION_METHODS:
            msg = (
                f"Unsupported correlation.method={self.statistics.correlation.method!r}. "
                f"Expected one of: {sorted(CORRELATION_METHODS)}."
            )
            raise ConfigError(msg)
        if self.statistics.correlation.tie_break not in CORRELATION_TIE_BREAKS:
            msg = (
                f"Unsupported correlation.tie_break={self.statistics.correlation.tie_break!r}. "
                f"Expected one of: {sorted(CORRELATION_TIE_BREAKS)}."
            )
            raise ConfigError(msg)
        if self.statistics.low_variance.scale_method not in LOW_VARIANCE_SCALE_METHODS:
            msg = (
                f"Unsupported low_variance.scale_method={self.statistics.low_variance.scale_method!r}. "
                f"Expected one of: {sorted(LOW_VARIANCE_SCALE_METHODS)}."
            )
            raise ConfigError(msg)
        if self.statistics.psi.mode not in PSI_MODES:
            msg = f"Unsupported psi.mode={self.statistics.psi.mode!r}. Expected one of: {sorted(PSI_MODES)}."
            raise ConfigError(
                msg,
            )
        if self.model.method not in MODEL_METHODS:
            msg = f"Unsupported model.method={self.model.method!r}. Expected one of: {sorted(MODEL_METHODS)}."
            raise ConfigError(
                msg,
            )
        if self.model.enabled and self.model.method == "lightgbm":
            _validate_lightgbm_params(self.model.params)
        if self.model.method == "catboost_rfe" and self.model.enabled:
            _validate_catboost_rfe_params(self.model.params)
            if self.model.selection.max_features is None:
                msg = (
                    "model.selection.max_features is required for catboost_rfe; "
                    "it sets num_features_to_select for recursive elimination."
                )
                raise ConfigError(msg)
        if self.model.cross_validation.strategy not in CV_STRATEGIES:
            msg = (
                f"Unsupported cross_validation.strategy="
                f"{self.model.cross_validation.strategy!r}. "
                f"Expected one of: {sorted(CV_STRATEGIES)}."
            )
            raise ConfigError(
                msg,
            )
        precise_method = self.precise.method
        if precise_method is None:
            precise_method = "none"
        if precise_method not in PRECISE_METHODS:
            msg = (
                f"Unsupported precise.method={self.precise.method!r}. "
                f"Expected one of: {sorted(PRECISE_METHODS)} or null."
            )
            raise ConfigError(
                msg,
            )
        if self.precise.enabled and precise_method != "boruta_shap":
            msg = (
                "precise.enabled: true requires precise.method='boruta_shap'. "
                "Set enabled: false or method: none to skip the precise stage."
            )
            raise ConfigError(msg)
        if precise_method == "boruta_shap":
            _validate_boruta_params(self.precise.params)
        if self.execution.local_sample.strategy not in SAMPLE_STRATEGIES:
            msg = (
                f"Unsupported local_sample.strategy="
                f"{self.execution.local_sample.strategy!r}. "
                f"Expected one of: {sorted(SAMPLE_STRATEGIES)}."
            )
            raise ConfigError(
                msg,
            )
        if self.execution.max_local_rows <= 0:
            msg = "execution.max_local_rows must be positive."
            raise ConfigError(msg)
        if self.execution.local_memory_limit <= 0:
            msg = "execution.local_memory_limit must be positive (GB)."
            raise ConfigError(msg)
        if not 0.0 <= self.statistics.null_rate.threshold <= 1.0:
            msg = "statistics.null_rate.threshold must be in [0, 1]."
            raise ConfigError(msg)
        if not 0.0 <= self.statistics.constants.max_frequency <= 1.0:
            msg = "statistics.constants.max_frequency must be in [0, 1]."
            raise ConfigError(msg)
        if self.statistics.constants.min_unique is not None and self.statistics.constants.min_unique < 1:
            msg = "statistics.constants.min_unique must be >= 1 when set."
            raise ConfigError(msg)
        if self.statistics.constants.chunk_size <= 0:
            msg = "statistics.constants.chunk_size must be positive."
            raise ConfigError(msg)
        if self.statistics.low_variance.min_variance < 0:
            msg = "statistics.low_variance.min_variance must be non-negative."
            raise ConfigError(msg)
        if not 0.0 <= self.statistics.correlation.threshold <= 1.0:
            msg = "statistics.correlation.threshold must be in [0, 1]."
            raise ConfigError(msg)
        if self.statistics.correlation.max_rows <= 0:
            msg = "statistics.correlation.max_rows must be positive."
            raise ConfigError(msg)
        _validate_iv_config(self.statistics.iv)

    def to_dict(self: FeatureSelectionConfig) -> dict[str, Any]:
        """Serialize config to a plain nested dictionary.

        Returns:
            JSON/YAML-compatible dictionary.
        """
        payload = asdict(self)
        payload["statistics"]["order"] = list(self.statistics.order)
        return payload

    @classmethod
    def from_dict(cls: type[FeatureSelectionConfig], payload: Mapping[str, Any]) -> FeatureSelectionConfig:
        """Build config from a nested mapping.

        Args:
            payload: Configuration dictionary (e.g. parsed YAML).

        Returns:
            Validated ``FeatureSelectionConfig``.

        Raises:
            ConfigError: On unknown fields or invalid values.
        """
        if not isinstance(payload, Mapping):
            msg = "Config payload must be a mapping."
            raise ConfigError(msg)
        if "order" in payload:
            msg = (
                "Top-level order is not supported. Put statistics methods in "
                "statistics.order; enable utils with preprocessing.*.enabled; "
                "enable model/precise with their enabled flags."
            )
            raise ConfigError(msg)
        _reject_unknown("root", payload, {f.name for f in fields(cls)})

        preprocessing_payload = payload.get("preprocessing", {})
        if preprocessing_payload is None:
            preprocessing_payload = {}
        if not isinstance(preprocessing_payload, Mapping):
            msg = "preprocessing must be a mapping."
            raise ConfigError(msg)
        preprocessing_raw = dict(preprocessing_payload)
        _reject_unknown(
            "preprocessing",
            preprocessing_raw,
            {f.name for f in fields(PreprocessingConfig)},
        )
        preprocessing = PreprocessingConfig(
            feature_drop=_build_section(
                FeatureDropConfig,
                preprocessing_raw.get("feature_drop", {}),
                "preprocessing.feature_drop",
            ),
            random_feature_drop=_build_section(
                RandomFeatureDropConfig,
                preprocessing_raw.get("random_feature_drop", {}),
                "preprocessing.random_feature_drop",
            ),
            row_sample=_build_section(
                RowSampleConfig,
                preprocessing_raw.get("row_sample", {}),
                "preprocessing.row_sample",
            ),
        )

        statistics_payload = payload.get("statistics") or {}
        if not isinstance(statistics_payload, Mapping):
            msg = "statistics must be a mapping."
            raise ConfigError(msg)
        statistics_raw = dict(statistics_payload)
        statistics_order = parse_statistics_order(statistics_raw.pop("order", ()))
        _reject_unknown(
            "statistics",
            statistics_raw,
            {f.name for f in fields(StatisticsConfig)} - {"order"},
        )
        statistics = StatisticsConfig(
            order=statistics_order,
            null_rate=_build_section(NullRateConfig, statistics_raw.get("null_rate", {}), "statistics.null_rate"),
            constants=_build_section(ConstantsConfig, statistics_raw.get("constants", {}), "statistics.constants"),
            low_variance=_build_section(
                LowVarianceConfig,
                statistics_raw.get("low_variance", {}),
                "statistics.low_variance",
            ),
            correlation=_build_section(
                CorrelationConfig,
                statistics_raw.get("correlation", {}),
                "statistics.correlation",
            ),
            psi=_build_section(PsiConfig, statistics_raw.get("psi", {}), "statistics.psi"),
            iv=_build_section(IvConfig, statistics_raw.get("iv", {}), "statistics.iv"),
            stability_classifier=_build_section(
                StabilityClassifierConfig,
                statistics_raw.get("stability_classifier", {}),
                "statistics.stability_classifier",
            ),
        )

        model_raw = dict(payload.get("model") or {})
        _reject_tuning_block("model", model_raw)
        _reject_unknown("model", model_raw, {f.name for f in fields(ModelConfig)})
        model = ModelConfig(
            enabled=model_raw.get("enabled", False),
            method=model_raw.get("method", ModelConfig.method),
            params=dict(model_raw.get("params", {})),
            selection=_build_section(
                ModelSelectionRuleConfig,
                model_raw.get("selection", {}),
                "model.selection",
            ),
            cross_validation=_build_section(
                CrossValidationConfig,
                model_raw.get("cross_validation", {}),
                "model.cross_validation",
            ),
        )

        precise_raw = dict(payload.get("precise") or {})
        _reject_tuning_block("precise", precise_raw)
        _reject_unknown("precise", precise_raw, {f.name for f in fields(PreciseConfig)})
        precise_method = precise_raw.get("method", PreciseConfig.method)
        if precise_method is None:
            precise_method = "none"
        precise = PreciseConfig(
            enabled=precise_raw.get("enabled", False),
            method=precise_method,
            params=dict(precise_raw.get("params", {})),
        )

        execution_raw = dict(payload.get("execution") or {})
        _reject_unknown("execution", execution_raw, {f.name for f in fields(ExecutionConfig)})
        local_sample_raw = execution_raw.get("local_sample", {})
        execution = ExecutionConfig(
            seed=execution_raw.get("seed", ExecutionConfig.seed),
            allow_local_fallback=execution_raw.get(
                "allow_local_fallback",
                ExecutionConfig.allow_local_fallback,
            ),
            max_local_rows=execution_raw.get("max_local_rows", ExecutionConfig.max_local_rows),
            local_memory_limit=execution_raw.get(
                "local_memory_limit",
                ExecutionConfig.local_memory_limit,
            ),
            local_sample=_build_section(LocalSampleConfig, local_sample_raw, "execution.local_sample"),
            verbose=parse_verbose(execution_raw.get("verbose", False)),
        )

        config = cls(
            preprocessing=preprocessing,
            statistics=statistics,
            model=model,
            precise=precise,
            execution=execution,
        )
        config.validate()
        return config

    @classmethod
    def from_yaml(cls: type[FeatureSelectionConfig], path: Union[str, Path]) -> FeatureSelectionConfig:
        """Load config from a YAML file without Hydra composition.

        Args:
            path: Path to a YAML file.

        Returns:
            Validated ``FeatureSelectionConfig``.
        """
        file_path = Path(path)
        try:
            from omegaconf import OmegaConf
        except ImportError as exc:  # pragma: no cover - hydra/omegaconf is a core dep
            msg = (
                "OmegaConf is required to load YAML configs. "
                "Install hydra-core or pass a dict to FeatureSelectionConfig.from_dict."
            )
            raise ConfigError(
                msg,
            ) from exc

        loaded = OmegaConf.to_container(OmegaConf.load(file_path), resolve=True)
        if loaded is None:
            loaded = {}
        if not isinstance(loaded, Mapping):
            msg = f"YAML root must be a mapping, got {type(loaded)!r}."
            raise ConfigError(msg)
        payload = dict(loaded)
        preprocessing_raw = payload.get("preprocessing", {})
        if isinstance(preprocessing_raw, Mapping):
            preprocessing = dict(preprocessing_raw)
            feature_drop_raw = preprocessing.get("feature_drop", {})
            if isinstance(feature_drop_raw, Mapping):
                feature_drop = dict(feature_drop_raw)
                drop_path = feature_drop.get("path")
                if drop_path is not None:
                    resolved_path = Path(str(drop_path)).expanduser()
                    if not resolved_path.is_absolute():
                        resolved_path = file_path.parent / resolved_path
                    feature_drop["path"] = str(resolved_path)
                    preprocessing["feature_drop"] = feature_drop
                    payload["preprocessing"] = preprocessing
        return cls.from_dict(payload)


def _validate_iv_config(config: IvConfig) -> None:
    """Validate Information Value filter settings."""
    if (
        isinstance(config.threshold, bool)
        or not isinstance(config.threshold, (int, float))
        or float(config.threshold) < 0.0
    ):
        msg = "statistics.iv.threshold must be a non-negative number."
        raise ConfigError(msg)
    if (
        isinstance(config.num_bins, bool)
        or not isinstance(config.num_bins, int)
        or config.num_bins < 2
    ):
        msg = "statistics.iv.num_bins must be an integer >= 2."
        raise ConfigError(msg)
    if config.max_threshold is not None and (
        isinstance(config.max_threshold, bool)
        or not isinstance(config.max_threshold, (int, float))
        or float(config.max_threshold) < 0.0
    ):
        msg = "statistics.iv.max_threshold must be a non-negative number or null."
        raise ConfigError(msg)
    if (
        config.max_threshold is not None
        and float(config.max_threshold) < float(config.threshold)
    ):
        msg = "statistics.iv.max_threshold must be >= statistics.iv.threshold."
        raise ConfigError(msg)
    if (
        isinstance(config.eps, bool)
        or not isinstance(config.eps, (int, float))
        or not 0.0 < float(config.eps) <= 1.0
    ):
        msg = "statistics.iv.eps must be in (0, 1]."
        raise ConfigError(msg)
    if (
        isinstance(config.min_bin_share, bool)
        or not isinstance(config.min_bin_share, (int, float))
        or not 0.0 <= float(config.min_bin_share) < 1.0
    ):
        msg = "statistics.iv.min_bin_share must be in [0, 1)."
        raise ConfigError(msg)
    if config.max_levels is not None and (
        isinstance(config.max_levels, bool)
        or not isinstance(config.max_levels, int)
        or config.max_levels < 2
    ):
        msg = "statistics.iv.max_levels must be an integer >= 2 or null."
        raise ConfigError(msg)
    if (
        isinstance(config.relative_error, bool)
        or not isinstance(config.relative_error, (int, float))
        or not 0.0 < float(config.relative_error) <= 1.0
    ):
        msg = "statistics.iv.relative_error must be in (0, 1]."
        raise ConfigError(msg)
    if (
        isinstance(config.batch_size, bool)
        or not isinstance(config.batch_size, int)
        or config.batch_size < 1
    ):
        msg = "statistics.iv.batch_size must be a positive integer."
        raise ConfigError(msg)


def _build_section(cls: type, payload: Any, section: str) -> Any:
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        msg = f"{section} must be a mapping."
        raise ConfigError(msg)
    _reject_unknown(section, payload, {f.name for f in fields(cls)})
    values = {f.name: payload[f.name] for f in fields(cls) if f.name in payload}
    return cls(**values)
