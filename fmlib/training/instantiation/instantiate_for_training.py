import datetime
from typing import Any, Iterable, List, Tuple

import accelerate
import hydra
import torch
from accelerate.utils import InitProcessGroupKwargs, KwargsHandler
from omegaconf import DictConfig

from fmlib.pipeline import TrainingPipeline
from fmlib.training.training_types import DataType, EpochCallbackProtocol, StepCallbackProtocol, TrainingState
from fmlib.training.utils.autocast import no_autocast
from fmlib.utils.loading import load_model_from_config

# This is needed for the correct instantiation
from fmlib.utils.resolvers import *  # noqa: F403


def instantiate_from_key(cfg: DictConfig, key: str, optional: bool = True) -> Any:
    """
    Инстанциирует объект по ключу.
    Если ключ не указан и объект опционален, то возвращает `None`.
    """
    config: DictConfig | None = cfg.get(key, None)
    if (not optional) and (config is None):
        msg: str = f"Key {key} is not optional and is None."
        raise KeyError(msg)
    result: Any = None
    if config is not None:
        result = hydra.utils.instantiate(config)
    return result


def instantiate_accelerator(cfg: DictConfig) -> accelerate.Accelerator:
    """
    Сложная логика инстанциации из-за ошибки в имплемнтации accelerate.

    TODO: Нужно упростить в будущем, когда починят.
    """
    kwargs_handlers: List[KwargsHandler] = [
        InitProcessGroupKwargs(
            init_method="env://",
            timeout=datetime.timedelta(
                seconds=(24 * 60 * 60),
            ),
        )
    ]
    if "kwargs_handlers" in cfg.accelerator:
        other_handlers: List[KwargsHandler] = [
            hydra.utils.instantiate(kwargs_cfg) for kwargs_cfg in cfg.accelerator.kwargs_handlers
        ]
        kwargs_handlers.extend(other_handlers)
        del cfg.accelerator.kwargs_handlers
    assert "kwargs_handlers" not in cfg.accelerator
    accelerator: accelerate.Accelerator = hydra.utils.instantiate(
        cfg.accelerator,
        _partial_=True,
    )(kwargs_handlers=kwargs_handlers)
    return accelerator


def instantiate_pipeline(cfg: DictConfig) -> torch.nn.Module:
    """
    Инстанциация пайплайна и возможная подгрузка весов.

    Как правило состоит из:
    - тренировочных трансформаций
    - валидационных трансформаций
    - самой модели
    - лоссов
    """
    exact_weights_file: str | None = cfg.get("exact_weights_file", None)
    model: torch.nn.Module = load_model_from_config(config=cfg, exact_weights_file=exact_weights_file)
    losses: torch.nn.Module = instantiate_from_key(cfg, "losses")
    training_transform: torch.nn.Module | None = instantiate_from_key(cfg, "training_transform")
    validation_transform: torch.nn.Module | None = instantiate_from_key(cfg, "validation_transform")
    result: torch.nn.Module = TrainingPipeline(
        model=model,
        losses=losses,
        eval_transform=validation_transform,
        training_transform=training_transform,
    )
    return result


def instantiate_optimizer(cfg: DictConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Инстанциирует оптимизатор, используя параметры из предоставленной модели.
    """
    optimizer: torch.optim.Optimizer = hydra.utils.instantiate(
        cfg.optimizer,
        params=model.parameters(),
    )
    return optimizer


def instantiate_scheduler(cfg: DictConfig, optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Инстанциирует планировщик learning rate'а. Использует предоставленный оптимизатор.
    """
    scheduler: torch.optim.lr_scheduler.LRScheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
    return scheduler


def instantiate_for_training(
    cfg: DictConfig,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler, accelerate.Accelerator]:
    """
    Инстанциирует все необходимые объекты для обучения.
    """
    accelerator: accelerate.Accelerator = instantiate_accelerator(cfg)
    pipeline: torch.nn.Module = instantiate_pipeline(cfg).to(accelerator.device)

    optimizer: torch.optim.Optimizer = instantiate_optimizer(cfg, pipeline)
    scheduler: torch.optim.lr_scheduler.LRScheduler = instantiate_scheduler(cfg, optimizer)
    return (pipeline, optimizer, scheduler, accelerator)


def instantiate_state(cfg: DictConfig) -> TrainingState:
    """
    Инстанциирует все необходимые для обучения объекты:
        - модель
        - оптимизатор
        - планировщик learning rate'а
        - accelerator
        - данные для обучения
        - данные для валидации
        - трансформеры для обучения
    """
    training_data: DataType = instantiate_from_key(cfg, "training_data")
    validation_data: DataType = instantiate_from_key(cfg, "validation_data")
    epoch_callbacks: EpochCallbackProtocol = instantiate_from_key(cfg, "epoch_callbacks")
    training_step_callbacks: StepCallbackProtocol = instantiate_from_key(cfg, "training_step_callbacks")
    validation_step_callbacks: StepCallbackProtocol = instantiate_from_key(cfg, "validation_step_callbacks")

    pipeline, optimizer, scheduler, accelerator = instantiate_for_training(cfg)

    autocast: Any = instantiate_from_key(cfg, "autocast", optional=True)
    autocast = autocast if autocast is not None else no_autocast

    epoch_iterable: Iterable[int] = instantiate_from_key(cfg, "epoch_iterable")

    pipeline, optimizer, validation_data, training_data, scheduler = accelerator.prepare(
        pipeline, optimizer, validation_data, training_data, scheduler
    )

    return TrainingState(
        pipeline=pipeline,
        autocast=autocast,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        training_data=training_data,
        validation_data=validation_data,
        training_step_callbacks=training_step_callbacks,
        validation_step_callbacks=validation_step_callbacks,
        epoch_iterable=epoch_iterable,
        epoch_callbacks=epoch_callbacks,
        config=cfg,
    )
