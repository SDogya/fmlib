try:
    from bootstrap import venv_bootstrap

    venv_bootstrap()
except ImportError:
    pass

import accelerate
import hydra
from omegaconf import DictConfig

from fmlib.training.instantiation import instantiate_state
from fmlib.training.training_loop import training_loop
from fmlib.training.training_types import MetricsType, TrainingState
from fmlib.training.utils.exit import register_cleanup
from fmlib.utils.resolvers import *  # noqa: F403
from fmlib.utils.seeding import force_seed_everything


@hydra.main(version_base=None)
def main(config: DictConfig) -> MetricsType:
    """
    Функция для запуска обучения модели при помощи
    фреймворка Hydra.

    Получение конфигурации эксперимента происходит
    при помощи библиотеки Hydra. Само обучение - с
    фреймворком Accelerate.

    Пайплайн:
        0. Загрузка конфигурации
        1. Инициализация глобальных ГПСЧ
        2. Создание состояния обучения
        3. Запуск обучения
        4. Вывод результатов обучения

    Возвращает:
        MetricsType: метрики обучения
    """
    force_seed_everything(config.get("seed", None))

    state: TrainingState = instantiate_state(config)
    accelerator: accelerate.Accelerator = state["accelerator"]

    register_cleanup(accelerator)

    results: MetricsType = training_loop(state)
    if accelerator.is_main_process:
        accelerator.print(results)

    return results


if __name__ == "__main__":
    main()
