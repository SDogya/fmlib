# 👨‍🎓 Обучение моделей c помощью Sber-AmazMe-FMLib

Для тренировки моделей библиотека использует абстракции из библиотеки `accelerate`. `Accelerate` - обертка над `PyTorch`, которая упрощает распределенное обучение. Подача данных в модель обеспечена с помощью `ParquetDataset`, подробнее с концепцией работы с данными можно ознакомиться [здесь](data_processing.md).

## Архитектура обучения

Концептуально инфраструктура для обучения состоит из следующих компонент:
- `instantiation` - создание объектов для процесса обучения из конфигурации
- `training_loop` - основная функциональность обучения в одном файле
- `losses` - набор лоссов
- `callbacks` - вызываемые во время обучения объекты
  - `step` - вызываются после каждого батча в обучении или валидации
    - `RocAucCallback` - считает `roc_auc_score` для задач классификации
  - `epoch` - вызываются в конце каждой эпохи
    - `EarlyStopping` - останавливает обучение, если метрика на валидации не обновляет лучшее значение
    - `SavingCheckpoint` - сохраняет лучшие контрольные точки для переиспользования во время инференса
    - `Logging` - печатает наблюдаемые метрики

## Обучение из Jupyter Lab/Notebook или скриптов

Для пользователя доступны следующие высокоуровневые примитивы:

- `TrainingState` - все объекты используемые для обучения
- `instantiate` - создание `TrainingState` из конфигурации
- `training_loop` - цикл обучения

Таким образом, весь пайплайн обучения можно уместить в следующие строки:
```python
with initialize(config_path = "examples/configs/train"):
    cfg: DictConfig = compose(config_name = "feature_transformer")
state: TrainingState = instantiate(cfg)
results = training_loop(state)
```

## Обучение из консоли

- Загрузка данных:
```bash
bash examples/load_data/feature_trasnformer.sh
```
- Конфигурация `accelerate`:
```bash
accelerate config
```
- Запуск с конфигурацией в `finetune.yaml`:
```bash
accelerate launch -m fmlib.training.train --config-dir=examples/configs/train --config-name=feature_transformer
```

## Для большего понимания рекомендуем ознакомиться с примерами в папке `examples/`