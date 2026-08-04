"""
Example: Adding a new custom step to new_pipeline.py

To add a new feature selection method, follow these steps:

1. Import your method from methods module
2. Create a new Step class
3. Register it in Pipeline._create_step()
4. Configure in YAML

Example below shows how to add DropHighCardinality step.
"""

from typing import List
from pyspark.sql import DataFrame

# Example 1: Simple custom step
class DropHighCardinalityStep:
    """
    Drop features with cardinality higher than threshold.
    """
    
    def __init__(self, max_cardinality: int = 10000, enabled: bool = True):
        self.name = "drop_high_cardinality"
        self.enabled = enabled
        self.max_cardinality = max_cardinality
        self.dropped_cols: List[str] = []
        self.selected_cols: List[str] = []
    
    def set_context(self, target_col: str, exclude_cols: List[str]):
        """Set context (optional, if needed)."""
        self.target_col = target_col
        self.exclude_cols = exclude_cols
    
    def run(self, df: DataFrame, **kwargs) -> DataFrame:
        target_col = kwargs.get("target_col", getattr(self, "target_col", None))
        exclude_cols = kwargs.get("exclude_cols", getattr(self, "exclude_cols", []))
        
        # Get all columns to check (excluding target and exclude patterns)
        cols_to_check = [
            c for c in df.columns 
            if c not in exclude_cols and c != target_col
        ]
        
        # Calculate cardinality for each column
        cols_to_drop = []
        for col in cols_to_check:
            # Use approx_count_distinct for large datasets
            cardinality = df.selectExpr(f"approx_count_distinct({col}) as cnt").collect()[0]["cnt"]
            
            if cardinality > self.max_cardinality:
                cols_to_drop.append(col)
        
        self.dropped_cols = cols_to_drop
        self.selected_cols = [c for c in df.columns if c not in cols_to_drop]
        
        return df.select(*self.selected_cols)
    
    def get_info(self) -> dict:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "dropped_count": len(self.dropped_cols),
            "selected_count": len(self.selected_cols),
        }


# Example 2: More complex step with metadata
class CorrelationWithTargetStep:
    """
    Drop features with low correlation with target.
    Uses Pearson correlation for numerical features.
    """
    
    def __init__(self, min_corr: float = 0.01, enabled: bool = True):
        self.name = "correlation_with_target"
        self.enabled = enabled
        self.min_corr = min_corr
        self.dropped_cols: List[str] = []
        self.selected_cols: List[str] = []
        self.correlations: dict = {}
    
    def run(self, df: DataFrame, **kwargs) -> DataFrame:
        target_col = kwargs.get("target_col")
        if not target_col:
            raise ValueError("target_col is required for correlation_with_target step")
        
        # Get numerical columns only
        dtypes = dict(df.dtypes)
        num_cols = [
            c for c in df.columns 
            if dtypes.get(c) in ("double", "float", "int", "bigint")
            and c != target_col
        ]
        
        # Calculate correlation with target for each column
        self.correlations = {}
        for col in num_cols:
            corr_expr = f"corr({col}, {target_col})"
            corr = df.selectExpr(corr_expr).collect()[0][0]
            self.correlations[col] = abs(corr) if corr else 0.0
        
        # Drop columns with low correlation
        cols_to_drop = [
            col for col, corr in self.correlations.items() 
            if corr < self.min_corr
        ]
        
        self.dropped_cols = cols_to_drop
        self.selected_cols = [c for c in df.columns if c not in cols_to_drop]
        
        # Save correlations to metadata
        self.metadata = {"correlations": self.correlations}
        
        return df.select(*self.selected_cols)
    
    def get_info(self) -> dict:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "dropped_count": len(self.dropped_cols),
            "selected_count": len(self.selected_cols),
            "metadata": self.metadata or {},
        }


# Register custom steps in Pipeline class (modify new_pipeline.py):
#
# In Pipeline._create_step(), add to step_map:
#
# step_map = {
#     "drop_null": DropNullStep,
#     "drop_constant": DropConstantStep,
#     "drop_low_variance": DropLowVarianceStep,
#     "drop_correlated": DropCorrelatedStep,
#     "stratified_sampling": StratifiedSamplingStep,
#     "lgbm_shap": LGBMShapStep,
#     "boruta_shap": BorutaShapStep,
#     "drop_high_cardinality": DropHighCardinalityStep,  # <-- ADD HERE
#     "correlation_with_target": CorrelationWithTargetStep,  # <-- ADD HERE
# }
