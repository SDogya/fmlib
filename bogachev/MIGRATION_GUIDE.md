# Migration Guide: From pipeline.py to new_pipeline.py

## Overview

The new pipeline (`new_pipeline.py`) provides a cleaner, more scalable architecture for feature selection. This guide helps you transition from the old pipeline to the new one.

## Key Differences

| Aspect | Old Pipeline (pipeline.py) | New Pipeline (new_pipeline.py) |
|--------|----------------------------|--------------------------------|
| Architecture | Monolithic, sequential | Object-oriented, step-based |
| Adding Methods | Modify run_pipeline() | Create new Step class |
| Config Format | Mixed order + params | Separated steps definition |
| Code Duplication | High (repeated blocks) | None (DRY) |
| Extensibility | Hard | Easy |

## Architecture Comparison

### Old Pipeline Structure
```
pipeline.py (860 lines)
├── setup_logger()
├── load_paths()
├── get_order()
├── get_method_params()
├── test_drop_null()
├── test_drop_constant()
├── test_drop_variance()
├── test_drop_correlated()
├── test_drop_lgbm_shap()
├── test_drop_boruta_shap()
└── run_pipeline()  # Giant switch statement
```

### New Pipeline Structure
```
new_pipeline.py (clean separation)
├── Step (abstract base class)
├── DropNullStep, DropConstantStep, etc.
├── Pipeline (orchestrator)
├── ConfigLoader (YAML parser)
└── main() (entry point)
```

## Migration Steps

### Step 1: Create New Config File

**Old format** (`pipeline_conf.yaml`):
```yaml
global_methods:
  order:
    - drop_constant
    - drop_low_variance
    - boruta_shap
  drop_constant:
    enabled: true
    param: 0.98
  drop_low_variance:
    enabled: true
    param: 1e-5
    scale_method: standard
```

**New format** (`new_pipeline_config.yaml`):
```yaml
steps:
  - name: drop_constant
    params:
      tol: 0.98
    enabled: true
    
  - name: drop_low_variance
    params:
      min_variance: [1e-5, 1e-4]
      scale_method: [standard, minmax]
    enabled: true
```

### Step 2: Run the New Pipeline

```python
# Old way
from pipeline import main
main()  # Uses pipeline_conf.yaml

# New way
from new_pipeline import Pipeline, ConfigLoader

config = ConfigLoader.load("new_pipeline_config.yaml")
pipeline = Pipeline(config)
result_df = pipeline.run(spark, df, "dataset_name")
```

Or run directly:
```bash
python new_pipeline.py
```

### Step 3: Adding a New Method

**Old way** (hardcoded in run_pipeline):
```python
elif method_name == 'my_new_method':
    # ... verbose implementation with logging, saving, etc.
    result = my_new_method_func(...)
    # ... more boilerplate
```

**New way** (clean and simple):
```python
# 1. Create the step class
class MyNewMethodStep(Step):
    def __init__(self, param1: int = 100, enabled: bool = True):
        self.name = "my_new_method"
        self.enabled = enabled
        self.param1 = param1
    
    def run(self, df: DataFrame, **kwargs) -> DataFrame:
        # Your logic here
        return result_df
    
    def get_info(self) -> dict:
        return {"name": self.name, "enabled": self.enabled}

# 2. Register in Pipeline._create_step()
step_map = {..., "my_new_method": MyNewMethodStep}

# 3. Configure in YAML
steps:
  - name: my_new_method
    params:
      param1: 200
```

### Step 4: Config File Structure

The new config supports per-dataset overrides:

```yaml
datasets:
  SA_SMA:
    enabled: true
    path: /path/to/data
    steps:  # Dataset-specific overrides
      - name: stratified_sampling
        params:
          max_rows: 500000  # Different from global default
      - name: boruta_shap
        params:
          boruta_trials: 100
```

## Migration Checklist

- [ ] Review existing `pipeline_conf.yaml`
- [ ] Create `new_pipeline_config.yaml` with equivalent steps
- [ ] Update step parameters to new format
- [ ] Test on one dataset first
- [ ] Update CI/CD scripts to use new pipeline
- [ ] Archive old `pipeline.py` (rename to `pipeline_legacy.py`)

## Example Config Comparison

### Full Old Config
```yaml
global_methods:
  order:
    - drop_null
    - 0.5
    - drop_constant
    - drop_low_variance
    - drop_correlated
    - boruta_shap

  drop_null:
    enabled: true
    param: 0.5

  drop_constant:
    enabled: true
    param: 0.98

  drop_low_variance:
    enabled: true
    param: 1e-5
    scale_method: standard

  drop_correlated:
    enabled: true
    param: 0.95

  boruta_shap:
    enabled: true
    model_type: lgbm
    boruta_trials: 50
```

### Full New Config
```yaml
steps:
  - name: drop_null
    params:
      max_null_fraction: 0.5
    enabled: true
    
  - name: drop_constant
    params:
      tol: 0.98
    enabled: true
    
  - name: drop_low_variance
    params:
      min_variance: 1e-5
      scale_method: standard
    enabled: true
    
  - name: drop_correlated
    params:
      corr_threshold: 0.95
    enabled: true
    
  - name: boruta_shap
    params:
      model_type: lgbm
      boruta_trials: 50
    enabled: true
```

## Benefits of New Pipeline

1. **Simplicity**: 300 lines vs 860 lines
2. **Scalability**: Add new methods in 5 minutes
3. **Maintainability**: Each step is independent
4. **Flexibility**: Per-dataset configs are cleaner
5. **Extensibility**: Easy to add pre/post-processing steps

## Troubleshooting

### Error: "Unknown step type"
Add the step to `Pipeline._create_step()` factory method.

### Error: "Target column not found"
Specify `target_col` explicitly in `pipeline.run()` or check column names.

### Output files not saved
Check that `results_dir` exists and is writable.

## Support

For questions or issues, refer to:
- `EXAMPLE_CUSTOM_STEP.py` - Examples of custom steps
- `new_pipeline.py` - Source code documentation
- `MIGRATION_GUIDE.md` - This file
