"""Typed configuration for the feature selection pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from fmlib.feature_selection.exceptions import ConfigError

PSI_MODES = frozenset({"month_over_month", "train_valid"})
MODEL_METHODS = frozenset({"lasso", "random_forest", "catboost_rfe", "lightgbm"})
PRECISE_METHODS = frozenset({"boruta_shap", "none"})
CV_STRATEGIES = frozenset({"random", "stratified", "group", "time_based"})
SAMPLE_STRATEGIES = frozenset({"random", "stratified"})
CORRELATION_METHODS = frozenset({"pearson", "spearman"})
CORRELATION_TIE_BREAKS = frozenset({"null_rate", "original_order"})
LOW_VARIANCE_SCALE_METHODS = frozenset({"standard", "minmax", "robust"})


def _reject_unknown(section: str, payload: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        message = f"Unknown fields in {section}: {unknown}. Remove them from the config."
        raise ConfigError(message)


@dataclass(frozen=True)
class NullRateConfig:
    """Null-rate statistical filter settings."""

    enabled: bool = True
    threshold: float = 0.95


@dataclass(frozen=True)
class ConstantsConfig:
    """Constant / quasi-constant filter settings."""

    enabled: bool = True
    max_frequency: float = 0.999
    min_unique: Optional[int] = None
    chunk_size: int = 1000


@dataclass(frozen=True)
class LowVarianceConfig:
    """Low-variance filter settings for continuous features."""

    enabled: bool = True
    min_variance: float = 0.01
    scale_method: str = "standard"


@dataclass(frozen=True)
class CorrelationConfig:
    """Pairwise correlation filter settings.

    Only ``FeatureSchema.continuous`` candidates are evaluated. Categorical features
    pass through unchanged. Correlation is computed on a seeded sample of about
    ``min(max_rows, execution.max_local_rows)`` train rows (Spark ``sample``, not
    ``limit``).

    Tie-breaking for a correlated pair:

    - ``original_order`` (default, draft behaviour): drop the later candidate;
    - ``null_rate``: drop the feature with more nulls; on tie fall back to
      ``original_order``.
    """

    enabled: bool = True
    method: str = "pearson"
    threshold: float = 0.95
    tie_break: str = "original_order"
    max_rows: int = 100_000


@dataclass(frozen=True)
class PsiConfig:
    """Population Stability Index filter settings."""

    enabled: bool = False
    mode: str = "train_valid"
    threshold: float = 0.25
    bins: int = 10


@dataclass(frozen=True)
class StabilityClassifierConfig:
    """Per-feature stability classifier settings."""

    enabled: bool = False
    metric: str = "roc_auc"
    threshold: float = 0.8


@dataclass(frozen=True)
class StatisticsConfig:
    """Statistical stage configuration."""

    null_rate: NullRateConfig = field(default_factory=NullRateConfig)
    constants: ConstantsConfig = field(default_factory=ConstantsConfig)
    low_variance: LowVarianceConfig = field(default_factory=LowVarianceConfig)
    correlation: CorrelationConfig = field(default_factory=CorrelationConfig)
    psi: PsiConfig = field(default_factory=PsiConfig)
    stability_classifier: StabilityClassifierConfig = field(default_factory=StabilityClassifierConfig)


@dataclass(frozen=True)
class ModelSelectionRuleConfig:
    """Rule applied to aggregated model importances."""

    max_features: Optional[int] = 100


@dataclass(frozen=True)
class ModelTuningConfig:
    """Optional Optuna tuning settings for the model stage."""

    enabled: bool = False
    n_trials: int = 30
    timeout_seconds: Optional[int] = 3600


@dataclass(frozen=True)
class CrossValidationConfig:
    """Cross-validation settings for the model stage."""

    strategy: str = "stratified"
    folds: int = 5
    metric: str = "roc_auc"


@dataclass(frozen=True)
class ModelConfig:
    """Model-based selection stage configuration."""

    method: str = "catboost_rfe"
    params: dict[str, Any] = field(default_factory=dict)
    selection: ModelSelectionRuleConfig = field(default_factory=ModelSelectionRuleConfig)
    tuning: ModelTuningConfig = field(default_factory=ModelTuningConfig)
    cross_validation: CrossValidationConfig = field(default_factory=CrossValidationConfig)


@dataclass(frozen=True)
class PreciseConfig:
    """Optional precise/final selection stage configuration."""

    method: Optional[str] = "none"


@dataclass(frozen=True)
class LocalSampleConfig:
    """Sampling used before controlled Spark → local materialization."""

    strategy: str = "stratified"
    cache_intermediate: bool = True


@dataclass(frozen=True)
class ExecutionConfig:
    """Execution, reproducibility and capacity settings."""

    seed: int = 42
    allow_local_fallback: bool = True
    max_local_rows: int = 1_000_000
    local_memory_limit: float = 4.0
    local_sample: LocalSampleConfig = field(default_factory=LocalSampleConfig)


@dataclass(frozen=True)
class FeatureSelectionConfig:
    """Top-level configuration for ``FeatureSelectionPipeline``.

    Args:
        statistics: Statistical filter settings.
        model: Model-based selector settings.
        precise: Optional precise selector settings.
        execution: Seeds, backend fallback and capacity limits.
    """

    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    precise: PreciseConfig = field(default_factory=PreciseConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    def validate(self: FeatureSelectionConfig) -> None:
        """Validate enums and numeric constraints.

        Raises:
            ConfigError: On unsupported values or inconsistent settings.
        """
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
            msg = f"Unsupported precise.method={self.precise.method!r}. Expected one of: {sorted(PRECISE_METHODS)} or null."
            raise ConfigError(
                msg,
            )
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

    def to_dict(self: FeatureSelectionConfig) -> dict[str, Any]:
        """Serialize config to a plain nested dictionary.

        Returns:
            JSON/YAML-compatible dictionary.
        """
        return asdict(self)

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
        _reject_unknown("root", payload, {f.name for f in fields(cls)})

        statistics_raw = dict(payload.get("statistics", {}))
        _reject_unknown("statistics", statistics_raw, {f.name for f in fields(StatisticsConfig)})
        statistics = StatisticsConfig(
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
            stability_classifier=_build_section(
                StabilityClassifierConfig,
                statistics_raw.get("stability_classifier", {}),
                "statistics.stability_classifier",
            ),
        )

        model_raw = dict(payload.get("model", {}))
        _reject_unknown("model", model_raw, {f.name for f in fields(ModelConfig)})
        model = ModelConfig(
            method=model_raw.get("method", ModelConfig.method),
            params=dict(model_raw.get("params", {})),
            selection=_build_section(
                ModelSelectionRuleConfig,
                model_raw.get("selection", {}),
                "model.selection",
            ),
            tuning=_build_section(ModelTuningConfig, model_raw.get("tuning", {}), "model.tuning"),
            cross_validation=_build_section(
                CrossValidationConfig,
                model_raw.get("cross_validation", {}),
                "model.cross_validation",
            ),
        )

        precise_raw = dict(payload.get("precise", {}))
        _reject_unknown("precise", precise_raw, {f.name for f in fields(PreciseConfig)})
        precise_method = precise_raw.get("method", PreciseConfig.method)
        if precise_method is None:
            precise_method = "none"
        precise = PreciseConfig(method=precise_method)

        execution_raw = dict(payload.get("execution", {}))
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
        )

        config = cls(statistics=statistics, model=model, precise=precise, execution=execution)
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
        return cls.from_dict(loaded)


def _build_section(cls: type, payload: Any, section: str) -> Any:
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        msg = f"{section} must be a mapping."
        raise ConfigError(msg)
    _reject_unknown(section, payload, {f.name for f in fields(cls)})
    values = {f.name: payload[f.name] for f in fields(cls) if f.name in payload}
    return cls(**values)
