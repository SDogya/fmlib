# Примеры конфигов отбора признаков

Каждый YAML в этой папке — готовый `order` плюс комментарии, **какой сценарий
он закрывает**. Schema в YAML не живёт: её передают в `fit_select()`.
`execution.task_type` должен совпасть с `FeatureSchema.task_type`.

```python
from fmlib.feature_selection import (
    FeatureSelectionConfig, FeatureSelectionPipeline, FeatureSchema,
)

config = FeatureSelectionConfig.from_yaml(
    "examples/configs/feature_selection/lightgbm.yaml"
)
schema = FeatureSchema(
    target="target",
    task_type="binary_classification",  # как в YAML execution.task_type
    categorical=[...],
    continuous=[...],
    time="month_part",  # нужен psi month_over_month и catboost_rfe
)
result = FeatureSelectionPipeline(config).fit_select(
    spark,  # или SimpleNamespace(version="local") на pandas
    schema=schema,
    datasets={"train": train_df},
    output_dir="runs/fs",
)
```

Передавайте только `{"train": ...}`. Форма `data=df` не режет по `schema.split`.

## Какой файл брать

### Задача (таргет)

| сценарий | файл | что ещё помнить |
|---|---|---|
| бинарная классификация | [`lightgbm.yaml`](lightgbm.yaml), [`catboost_rfe.yaml`](catboost_rfe.yaml) | дефолт `execution.task_type` |
| мультикласс (`classification`, не `multiclass`) | [`lightgbm_classification.yaml`](lightgbm_classification.yaml), [`catboost_rfe_classification.yaml`](catboost_rfe_classification.yaml) | IV **нельзя** |
| регрессия | [`lightgbm_regression.yaml`](lightgbm_regression.yaml), [`catboost_rfe_regression.yaml`](catboost_rfe_regression.yaml) | IV нельзя; сэмпл без стратификации по таргету |

LightGBM / CatBoost RFE / BorutaSHAP умеют все три задачи. `loss_function` /
`objective` пайплайн проставляет сам — в YAML их лучше не дублировать, кроме
исторического бинарного CatBoost (`Logloss` можно оставить, код для binary
ключ не навязывает).

### Как режет LightGBM

| сценарий | файл |
|---|---|
| среднее по фолдам, два кумулятива, пересечение (дефолт) | [`lightgbm.yaml`](lightgbm.yaml) |
| vote: кумулятив в каждом фолде, доля наборов `min_set_share` | [`lightgbm_vote.yaml`](lightgbm_vote.yaml) |
| как campaign: vote + `min_set_share: 1.0`, без Optuna, фиксированные деревья | [`lightgbm_pinned.yaml`](lightgbm_pinned.yaml) |
| Optuna отдельно на каждом фолде | [`lightgbm_per_fold.yaml`](lightgbm_per_fold.yaml) |

На одинаковых фолдах, порядке строк и сидах `lightgbm_pinned.yaml` совпадает с
campaign LGM∩SHAP. «Из коробки» campaign режет индексы по исходному порядку
строк и одним `random_state` на все фолды — наборы будут другими.

### CatBoost RFE

| сценарий | файл | нужно в schema |
|---|---|---|
| бинарный, Optuna, `steps` (доля за раунд) | [`catboost_rfe.yaml`](catboost_rfe.yaml) | `time` |
| мультикласс / регрессия | `catboost_rfe_classification.yaml` / `catboost_rfe_regression.yaml` | `time` |
| фиксированное число признаков за шаг | [`catboost_rfe_fixed_drop.yaml`](catboost_rfe_fixed_drop.yaml) | `time` |

`parameters.task_type: CPU` — это CPU/GPU CatBoost, не задача моделирования.

### Статистики по одному

[`null.yaml`](null.yaml), [`constants.yaml`](constants.yaml),
[`low_variance.yaml`](low_variance.yaml), [`correlations.yaml`](correlations.yaml),
[`psi.yaml`](psi.yaml) (month-over-month, нужен `time`),
[`psi_train_valid.yaml`](psi_train_valid.yaml) (train против valid),
[`iv.yaml`](iv.yaml) (только binary).

В нескольких старых файлах в `order` ещё стоит заглушка `lasso` — в рабочий
пайплайн её не копируйте.

### Полные пайплайны

| файл | состав |
|---|---|
| [`pipeline_lightgbm.yaml`](pipeline_lightgbm.yaml) | stats → lightgbm → boruta |
| [`pipeline_catboost_rfe.yaml`](pipeline_catboost_rfe.yaml) | stats → catboost_rfe → boruta |
| [`full_pipeline.yaml`](full_pipeline.yaml) | stats → catboost_rfe |
| [`pipeline_all_methods.yaml`](pipeline_all_methods.yaml) | feature_drop → stats (вкл. psi, iv) → boruta → catboost_rfe |
| [`boruta_shap.yaml`](boruta_shap.yaml) | только precise-стадия |

Для classification/regression возьмите соответствующий модельный YAML и
подставьте его шаг вместо `lightgbm` / `catboost_rfe` в pipeline-файл. Шаг
`iv` из classification/regression уберите.

Подробный контракт полей — [`docs/feature_selection.md`](../../../docs/feature_selection.md).
