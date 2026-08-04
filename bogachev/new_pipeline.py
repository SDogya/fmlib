#!/usr/bin/env python3
"""
New feature selection pipeline with clean architecture.

Usage:
    from new_pipeline import Pipeline, ConfigLoader
    
    config = ConfigLoader.load("config.yaml")
    pipeline = Pipeline(config)
    result_df = pipeline.run(spark, original_df, dataset_name)
"""

import os
import sys
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

sys.path.append('.')
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../../')
sys.path.insert(0, "/home/datalab/nfs/zaripov/autocampaignxfm")

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from tools.spark_session import create_spark_session
from methods import (
    drop_null_features,
    drop_constant_features,
    drop_low_variance_features,
    drop_correlated_features,
    select_robust_features,
    select_features_boruta_shap,
    apply_stratified_sampling,
)

# === CONFIGURATION ===
DEFAULT_RESULTS_DIR = "results"
PIPELINE_DIR = os.path.join(DEFAULT_RESULTS_DIR, "pipeline")


# === STEP BASE CLASS ===
class Step(ABC):
    """Base class for all pipeline steps."""
    
    def __init__(self, name: str, enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.dropped_cols: List[str] = []
        self.selected_cols: List[str] = []
        self.metadata: Dict[str, Any] = {}
    
    @abstractmethod
    def run(self, df: DataFrame, **kwargs) -> DataFrame:
        """Execute the step and return the transformed DataFrame."""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Return step information for logging."""
        return {
            "name": self.name,
            "enabled": self.enabled,
            "dropped_count": len(self.dropped_cols),
            "selected_count": len(self.selected_cols),
            "metadata": self.metadata,
        }


# === FEATURE DROPPER STEPS ===
class DropFeaturesStep(Step, ABC):
    """Base class for steps that drop features."""
    
    def __init__(self, name: str, enabled: bool = True):
        super().__init__(name, enabled)
        self.target_col: Optional[str] = None
        self.exclude_cols: List[str] = []
    
    def set_context(self, target_col: str, exclude_cols: List[str]) -> "DropFeaturesStep":
        """Set context for the step."""
        self.target_col = target_col
        self.exclude_cols = exclude_cols
        return self


class DropNullStep(DropFeaturesStep):
    """Drop features with high null fraction."""
    
    def __init__(self, max_null_fraction: float = 0.5, enabled: bool = True):
        super().__init__("drop_null", enabled)
        self.max_null_fraction = max_null_fraction
    
    def run(self, df: DataFrame, **kwargs) -> DataFrame:
        target_col = kwargs.get("target_col", self.target_col)
        exclude_cols = kwargs.get("exclude_cols", self.exclude_cols)
        
        result = drop_null_features(
            df,
            max_null_fraction=self.max_null_fraction,
            exclude_cols=exclude_cols,
            target_col=target_col,
        )
        
        self._update_dropped(df, result)
        return result
    
    def _update_dropped(self, original: DataFrame, result: DataFrame):
        orig_set = set(original.columns)
        result_set = set(result.columns)
        self.dropped_cols = list(orig_set - result_set)


class DropConstantStep(DropFeaturesStep):
    """Drop constant and quasi-constant features."""
    
    def __init__(self, tol: float = 0.98, chunk_size: int = 1000, enabled: bool = True):
        super().__init__("drop_constant", enabled)
        self.tol = tol
        self.chunk_size = chunk_size
    
    def run(self, df: DataFrame, **kwargs) -> DataFrame:
        target_col = kwargs.get("target_col", self.target_col)
        exclude_cols = kwargs.get("exclude_cols", self.exclude_cols)
        
        result = drop_constant_features(
            df,
            tol=self.tol,
            exclude_cols=exclude_cols,
            chunk_size=self.chunk_size,
            target_col=target_col,
        )
        
        self._update_dropped(df, result)
        return result
    
    def _update_dropped(self, original: DataFrame, result: DataFrame):
        orig_set = set(original.columns)
        result_set = set(result.columns)
        self.dropped_cols = list(orig_set - result_set)


class DropLowVarianceStep(DropFeaturesStep):
    """Drop low variance features."""
    
    def __init__(
        self,
        min_variance: float = 0.005,
        scale_method: str = "standard",
        chunk_size: int = 1000,
        enabled: bool = True,
    ):
        super().__init__("drop_low_variance", enabled)
        self.min_variance = min_variance
        self.scale_method = scale_method
        self.chunk_size = chunk_size
    
    def run(self, df: DataFrame, **kwargs) -> DataFrame:
        target_col = kwargs.get("target_col", self.target_col)
        exclude_cols = kwargs.get("exclude_cols", self.exclude_cols)
        
        result = drop_low_variance_features(
            df,
            min_variance=self.min_variance,
            scale_method=self.scale_method,
            exclude_cols=exclude_cols,
            chunk_size=self.chunk_size,
            target_col=target_col,
        )
        
        self._update_dropped(df, result)
        return result
    
    def _update_dropped(self, original: DataFrame, result: DataFrame):
        orig_set = set(original.columns)
        result_set = set(result.columns)
        self.dropped_cols = list(orig_set - result_set)


class DropCorrelatedStep(DropFeaturesStep):
    """Drop correlated features."""
    
    def __init__(self, corr_threshold: float = 0.95, enabled: bool = True):
        super().__init__("drop_correlated", enabled)
        self.corr_threshold = corr_threshold
    
    def run(self, df: DataFrame, **kwargs) -> DataFrame:
        target_col = kwargs.get("target_col", self.target_col)
        exclude_cols = kwargs.get("exclude_cols", self.exclude_cols)
        
        result = drop_correlated_features(
            df,
            corr_threshold=self.corr_threshold,
            exclude_cols=exclude_cols,
            target_col=target_col,
        )
        
        self._update_dropped(df, result)
        return result
    
    def _update_dropped(self, original: DataFrame, result: DataFrame):
        orig_set = set(original.columns)
        result_set = set(result.columns)
        self.dropped_cols = list(orig_set - result_set)


# === FEATURE SELECTOR STEPS ===
class FeatureSelectionStep(Step, ABC):
    """Base class for feature selection steps."""
    
    def __init__(self, name: str, enabled: bool = True):
        super().__init__(name, enabled)
        self.target_col: Optional[str] = None
        self.exclude_cols: List[str] = []
    
    def set_context(self, target_col: str, exclude_cols: List[str]) -> "FeatureSelectionStep":
        """Set context for the step."""
        self.target_col = target_col
        self.exclude_cols = exclude_cols
        return self


class StratifiedSamplingStep(FeatureSelectionStep):
    """Apply stratified sampling to reduce dataset size."""
    
    def __init__(self, max_rows: int = 100000, enabled: bool = True):
        super().__init__("stratified_sampling", enabled)
        self.max_rows = max_rows
    
    def run(self, df: DataFrame, **kwargs) -> DataFrame:
        target_col = kwargs.get("target_col", self.target_col)
        
        result = apply_stratified_sampling(
            df,
            target_col=target_col,
            max_rows=self.max_rows,
        )
        
        self.metadata["original_rows"] = df.count()
        self.metadata["result_rows"] = result.count()
        
        return result


class LGBMShapStep(FeatureSelectionStep):
    """LGBM + SHAP feature selection."""
    
    def __init__(
        self,
        threshold: float = 0.85,
        max_rows_limit: int = 250_000,
        enabled: bool = True,
    ):
        super().__init__("lgbm_shap", enabled)
        self.threshold = threshold
        self.max_rows_limit = max_rows_limit
    
    def run(self, df: DataFrame, **kwargs) -> DataFrame:
        target_col = kwargs.get("target_col", self.target_col)
        exclude_cols = kwargs.get("exclude_cols", self.exclude_cols)
        stratified_df = kwargs.get("stratified_df", df)
        
        result = select_robust_features(
            df,
            target_col=target_col,
            exclude_cols=exclude_cols,
            lgbm_threshold=self.threshold,
            shap_threshold=self.threshold,
            return_importances=True,
            max_rows_limit=self.max_rows_limit,
            stratified_df=stratified_df,
        )
        
        self.selected_cols = result["selected_features"]
        self.dropped_cols = list(set(df.columns) - set(self.selected_cols) - set(exclude_cols))
        self.metadata["lgbm_selected"] = result["lgbm_selected"]
        self.metadata["shap_selected"] = result["shap_selected"]
        
        return df.select(*self.selected_cols)


class BorutaShapStep(FeatureSelectionStep):
    """BorutaSHAP feature selection with Optuna tuning."""
    
    def __init__(
        self,
        model_type: str = "lgbm",
        boruta_trials: int = 50,
        parameters: Optional[Dict[str, Any]] = None,
        optuna_params: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
    ):
        super().__init__("boruta_shap", enabled)
        self.model_type = model_type
        self.boruta_trials = boruta_trials
        self.parameters = parameters or {}
        self.optuna_params = optuna_params or {}
    
    def run(self, df: DataFrame, **kwargs) -> DataFrame:
        target_col = kwargs.get("target_col", self.target_col)
        exclude_cols = kwargs.get("exclude_cols", self.exclude_cols)
        
        result = select_features_boruta_shap(
            df,
            target_col=target_col,
            exclude_cols=exclude_cols,
            model_type=self.model_type,
            boruta_trials=self.boruta_trials,
            parameters=self.parameters,
            optuna_params=self.optuna_params,
        )
        
        self.selected_cols = result
        self.dropped_cols = list(set(df.columns) - set(self.selected_cols) - set(exclude_cols))
        
        return df.select(*self.selected_cols)


# === PIPELINE ORCHESTRATOR ===
class Pipeline:
    """Pipeline orchestrator that executes steps in order."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        results_dir: str = DEFAULT_RESULTS_DIR,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.results_dir = results_dir
        self.logger = logger or self._create_logger()
        self.steps: List[Step] = []
        self._setup_steps()
    
    def _create_logger(self) -> logging.Logger:
        """Create default logger."""
        logger = logging.getLogger("new_pipeline")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _setup_steps(self):
        """Setup steps from config."""
        order = self.config.get("order", [])
        
        for step_config in order:
            step_name = step_config.get("name")
            params = step_config.get("params", {})
            enabled = step_config.get("enabled", True)
            
            step = self._create_step(step_name, params, enabled)
            if step:
                self.steps.append(step)
    
    def _create_step(self, name: str, params: Dict[str, Any], enabled: bool) -> Optional[Step]:
        """Factory method to create step instances."""
        if not enabled:
            return None
        
        step_map = {
            "drop_null": DropNullStep,
            "drop_constant": DropConstantStep,
            "drop_low_variance": DropLowVarianceStep,
            "drop_correlated": DropCorrelatedStep,
            "stratified_sampling": StratifiedSamplingStep,
            "lgbm_shap": LGBMShapStep,
            "boruta_shap": BorutaShapStep,
        }
        
        step_class = step_map.get(name)
        if not step_class:
            self.logger.warning(f"Unknown step type: {name}")
            return None
        
        return step_class(**params, enabled=enabled)
    
    def run(
        self,
        spark: Any,
        df: DataFrame,
        dataset_name: str,
        target_col: Optional[str] = None,
        exclude_cols: Optional[List[str]] = None,
    ) -> DataFrame:
        """
        Run the pipeline on the given DataFrame.
        
        Args:
            spark: Spark session
            df: Input DataFrame
            dataset_name: Name of the dataset (for logging and output paths)
            target_col: Optional target column name
            exclude_cols: Optional list of columns to exclude from processing
            
        Returns:
            Processed DataFrame after all steps
        """
        self.logger.info("=" * 80)
        self.logger.info(f"NEW PIPELINE: {dataset_name}")
        self.logger.info("=" * 80)
        self.logger.info(f"Input: {len(df.columns)} columns, {df.count():,} rows")
        
        # Auto-detect target column if not provided
        if target_col is None:
            target_col = self._find_target_column(df)
            self.logger.info(f"Auto-detected target column: {target_col}")
        
        # Auto-detect exclude columns if not provided
        if exclude_cols is None:
            exclude_cols = self._find_exclude_columns(df, target_col)
            self.logger.info(f"Auto-detected exclude columns: {exclude_cols}")
        
        # Prepare results directory
        dataset_dir = os.path.join(self.results_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        current_df = df
        step_num = 0
        
        for step in self.steps:
            if not step.enabled:
                self.logger.info(f"Step {step.name}: SKIPPED (disabled)")
                continue
            
            step_num += 1
            self.logger.info("-" * 80)
            self.logger.info(f"Step {step_num}: {step.name}")
            self.logger.info("-" * 80)
            
            # Set context for feature-related steps
            if isinstance(step, DropFeaturesStep) or isinstance(step, FeatureSelectionStep):
                step.set_context(target_col, exclude_cols)
            
            # Execute step
            start_time = datetime.now()
            try:
                current_df = step.run(
                    current_df,
                    target_col=target_col,
                    exclude_cols=exclude_cols,
                )
                elapsed = (datetime.now() - start_time).total_seconds()
                
                # Log results
                info = step.get_info()
                self.logger.info(f"  Output: {len(current_df.columns)} columns")
                self.logger.info(f"  Dropped: {info['dropped_count']} columns")
                self.logger.info(f"  Time: {elapsed:.2f}s")
                
                # Save dropped columns to file
                if step.dropped_cols:
                    filepath = os.path.join(dataset_dir, f"{step.name}_cols.txt")
                    self._save_columns(step.dropped_cols, filepath)
                    self.logger.info(f"  Saved: {filepath}")
                
                # Save selected columns for feature selection steps
                if step.selected_cols:
                    filepath = os.path.join(dataset_dir, f"{step.name}_selected.txt")
                    self._save_columns(step.selected_cols, filepath)
                    self.logger.info(f"  Saved: {filepath}")
                
            except Exception as e:
                self.logger.error(f"  ERROR: {str(e)}")
                self.logger.exception("Stack trace:")
                raise
        
        self.logger.info("=" * 80)
        self.logger.info("PIPELINE COMPLETED!")
        self.logger.info(f"Final: {len(current_df.columns)} columns")
        self.logger.info("=" * 80)
        
        return current_df
    
    def _find_target_column(self, df: DataFrame) -> str:
        """Auto-detect target column."""
        for col in df.columns:
            if "target" in col.lower() or col.lower() == "y":
                return col
        raise ValueError(f"Target column not found in DataFrame. Columns: {df.columns}")
    
    def _find_exclude_columns(self, df: DataFrame, target_col: str) -> List[str]:
        """Auto-detect columns to exclude (id, date, split, etc.)."""
        exclude = set()
        exclude_patterns = ["id", "date", "time", "split", "epoch", "seq", "report", "row"]
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Keep target_attr_1 (it's the target), drop other target_attr_*
            if "target_attr_" in col_lower and col_lower != "target_attr_1":
                exclude.add(col)
                continue
            
            # Always exclude the target column
            if col == target_col:
                exclude.add(col)
                continue
            
            # Check for exclude patterns
            for pattern in exclude_patterns:
                if pattern in col_lower:
                    exclude.add(col)
                    break
        
        return list(exclude)
    
    def _save_columns(self, columns: List[str], filepath: str):
        """Save column list to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            for col in columns:
                f.write(f"{col}\n")
    
    def get_step_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all steps."""
        return [step.get_info() for step in self.steps]


# === CONFIG LOADER ===
class ConfigLoader:
    """Loader for pipeline configuration."""
    
    @staticmethod
    def load(filepath: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        import yaml
        
        with open(filepath, "r") as f:
            config = yaml.safe_load(f)
        
        return config
    
    @staticmethod
    def get_dataset_config(
        config: Dict[str, Any],
        dataset_name: str,
    ) -> Dict[str, Any]:
        """Get configuration for a specific dataset."""
        datasets = config.get("datasets", {})
        return datasets.get(dataset_name, {})
    
    @staticmethod
    def get_pipeline_steps(config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract pipeline steps from config."""
        return config.get("steps", [])


# === MAIN ENTRY POINT ===
def main():
    """Main entry point for the pipeline."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup logging
    os.makedirs(PIPELINE_DIR, exist_ok=True)
    os.makedirs(os.path.join(PIPELINE_DIR, "logs"), exist_ok=True)
    
    log_file = os.path.join(PIPELINE_DIR, "logs", f"pipeline_{timestamp}.log")
    log_format = "%(asctime)s | %(levelname)-8s | %(message)s"
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    
    logger = logging.getLogger("new_pipeline")
    
    logger.info("=" * 80)
    logger.info("NEW FEATURE SELECTION PIPELINE")
    logger.info("=" * 80)
    
    # Load configuration
    config_path = "pipeline_conf.yaml"
    logger.info(f"Loading config from {config_path}")
    config = ConfigLoader.load(config_path)
    
    # Get datasets
    datasets = config.get("datasets", {})
    logger.info(f"Found {len(datasets)} datasets: {list(datasets.keys())}")
    
    # Start Spark
    logger.info("Starting Spark session...")
    spark = create_spark_session(
        app_name="new_feature_selection_pipeline",
        executor_instances=8,
    )
    logger.info("Spark session started!")
    
    # Process each dataset
    for dataset_name, dataset_config in datasets.items():
        if not dataset_config.get("enabled", True):
            logger.info(f"{dataset_name}: SKIPPED (disabled)")
            continue
        
        logger.info("=" * 80)
        logger.info(f"PROCESSING: {dataset_name}")
        logger.info("=" * 80)
        
        # Build path
        base_path = dataset_config.get("path", "")
        has_split_type = dataset_config.get("has_split_type", False)
        
        if has_split_type:
            train_path = os.path.join(base_path, "split_type=train")
        else:
            train_path = base_path
        
        logger.info(f"Loading data from {train_path}")
        
        try:
            # Load data
            original_df = spark.read.parquet(train_path)
            total_rows = original_df.count()
            total_cols = len(original_df.columns)
            
            logger.info(f"Loaded: {total_rows:,} rows, {total_cols} columns")
            
            # Get pipeline config for dataset
            pipeline_config = {
                "steps": ConfigLoader.get_pipeline_steps(config),
            }
            
            # Create pipeline
            pipeline = Pipeline(pipeline_config, results_dir=PIPELINE_DIR, logger=logger)
            
            # Run pipeline
            final_df = pipeline.run(
                spark=spark,
                df=original_df,
                dataset_name=dataset_name,
            )
            
            logger.info(f"Final columns: {len(final_df.columns)}")
            
        except Exception as e:
            logger.error(f"ERROR processing {dataset_name}: {str(e)}")
            logger.exception("Stack trace:")
            continue
    
    logger.info("=" * 80)
    logger.info("ALL DATASETS PROCESSED!")
    logger.info("=" * 80)
    
    spark.stop()


if __name__ == "__main__":
    main()
