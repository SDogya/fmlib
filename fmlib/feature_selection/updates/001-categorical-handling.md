# Categorical handling for model selectors

## Summary

Model-based feature selection now accepts categorical columns instead of
silently passing them through. The setting is local to a model step, so,
for example, LightGBM may use `top_n` while BorutaSHAP uses
`ordinal_campaign`.

```yaml
lightgbm:
  categorical_handling:
    mode: native
    # max_cardinality: 100     # required for max_cardinality
    # top_n: 50                # required for top_n
    # target_encoding:
    #   folds: 5
    #   smoothing: 20
```

The same `categorical_handling` block is supported in `catboost_rfe` and
`boruta_shap` parameters. If omitted, the default is `native`.

## Available modes

| Mode | Behaviour |
| --- | --- |
| `native` | Pass categorical columns to LightGBM/CatBoost as native categories. |
| `ordinal_campaign` | Convert category values to deterministic ordinal codes, matching Campaign's numeric-input approach. Unknown validation values become missing. |
| `max_cardinality` | Count distinct levels once on the bounded model sample. Columns with cardinality above `max_cardinality` receive a model-stage `drop`; eligible columns are native categories. |
| `top_n` | On each training partition keep the N most frequent levels; map all other and unseen values to an internal `__FMLIB_OTHER__` level, then use native categories. |
| `target_encoding` | Build leakage-aware out-of-fold encodings on training rows and apply the full train mapping to validation rows. Defaults: five folds and smoothing 20. |
| `skip` | Preserve the old behaviour: categorical columns receive no decision from that model selector. |

Missing values are normalised to one internal categorical level before
cardinality checks and transformations.

## Model integration

LightGBM now materializes a mixed pandas frame, keeps pandas categorical
dtype for native modes, and computes its fold matrix from the outer-fold
train partition. The same rule is used for its global Optuna hold-out. Split
and SHAP importances are aggregated back to original feature names.

CatBoost RFE uses its out-of-time fit partition to build categorical mappings.
Native and top-N fields are passed through CatBoost `cat_features`; ordinal
and target-encoded fields are numeric. Multiclass target encoding creates one
temporary numeric feature per class, then selection output is reconstructed to
source feature names.

BorutaSHAP with `model_type: lgbm` receives the mixed pandas frame, including
native categorical dtype. `model_type: rf` rejects native-producing modes
(`native`, `max_cardinality`, `top_n`) only when categorical candidates are
present; use `ordinal_campaign`, `target_encoding`, or `skip` there.

## Results, examples, and validation

Each selector records its categorical mode, measured cardinalities, and
cardinality-dropped columns in `scores`. Technical target-encoding column names
never appear in final feature decisions.

All LightGBM example YAML files now state `mode: native` explicitly. The
Campaign-matched example explicitly uses `ordinal_campaign` so its intent is
unchanged. CatBoost and Boruta examples also show the new block.

Added focused unit coverage for cardinality dropping, unseen top-N values,
OOF target encoding, and multiclass source mapping. The focused module passes
with `pytest --noconftest fmlib/feature_selection/utils/tests/test_categorical.py`.
The repository-wide feature-selection suite still requires PySpark, which is
not installed in the current environment.
