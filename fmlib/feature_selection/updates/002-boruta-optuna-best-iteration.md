# BorutaSHAP reuses the tuned LightGBM iteration count

## Summary

BorutaSHAP with LightGBM and enabled Optuna now records `best_iteration_` for
each completed tuning trial. The iteration count from the best trial replaces
the configured LightGBM tree cap in the model passed to BorutaSHAP.

Previously Optuna evaluated LightGBM with early stopping, but the subsequent
Boruta loop created a fresh model with the original `n_estimators` or an
automatically selected cap. Every Boruta iteration could therefore train many
more trees than the model Optuna had actually evaluated.

## API and behavior

No YAML changes are required. The behavior applies only when all of the
following are true:

- `boruta_shap.model_type` is `lgbm`;
- Optuna is enabled and has a non-empty search space;
- LightGBM reports a positive `best_iteration_` for the winning trial.

The selected count is stored as `best_iteration` in the BorutaSHAP score
payload, and `best_params.n_estimators` contains the same effective cap. Any
LightGBM tree-count alias in the original parameters is removed before the
canonical `n_estimators` value is set.

If early stopping is disabled or LightGBM does not report a positive best
iteration, the original tree cap remains unchanged. Random Forest and
Optuna-disabled paths are unaffected.

## Validation

Added regression coverage that simulates an Optuna trial stopping before the
configured cap and verifies that the model handed to BorutaSHAP receives the
tuned iteration count. Helper coverage checks invalid iteration values and
normalization of LightGBM tree-cap aliases.

The two focused regression tests pass with `pytest --noconftest`. A smoke test
with the installed LightGBM and Optuna also confirmed that the reported best
iteration becomes the final model cap. The complete BorutaSHAP test module
cannot run in the local environment because its autouse fixture requires the
PySpark optional dependency, which is not installed.
