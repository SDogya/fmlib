import warnings
from typing import Any

import torch
from omegaconf import DictConfig

from fmlib.pipeline import InferencePipeline
from fmlib.training.training_types import DataType, EvaluationState, StepCallbackProtocol
from fmlib.training.utils.autocast import no_autocast
from fmlib.utils.loading import load_model_from_config

# This is needed for the correct instantiation
from fmlib.utils.resolvers import *  # noqa: F403

from .instantiate_for_training import (
    instantiate_accelerator,
    instantiate_from_key,
)


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

    if exact_weights_file is None:
        msg: str = "No `exact_weights_file` provided. Model will default to random weights."
        warnings.warn(msg, stacklevel=2)

    model: torch.nn.Module = load_model_from_config(
        config=cfg,
        exact_weights_file=exact_weights_file,
    )
    evaluation_transform: torch.nn.Module | None = instantiate_from_key(cfg, "evaluation_transform")
    result: torch.nn.Module = InferencePipeline(
        model=model,
        transform=evaluation_transform,
    )
    return result


def instantiate_state_for_evaluation(cfg: DictConfig) -> EvaluationState:
    """
    Инстанциирует все необходимые для инференса/теста объекты:
        - модель
        - accelerator
        - данные для инференса/теста
        - трансформы для инференса/теста
    """
    evaluation_data: DataType = instantiate_from_key(cfg, "evaluation_data")
    evaluation_step_callbacks: StepCallbackProtocol = instantiate_from_key(cfg, "evaluation_step_callbacks")

    pipeline = instantiate_pipeline(cfg)
    accelerator = instantiate_accelerator(cfg)

    autocast: Any = instantiate_from_key(cfg, "autocast", optional=True)
    autocast = autocast if autocast is not None else no_autocast

    pipeline, evaluation_data = accelerator.prepare(pipeline, evaluation_data)

    return EvaluationState(
        pipeline=pipeline,
        autocast=autocast,
        accelerator=accelerator,
        evaluation_data=evaluation_data,
        evaluation_step_callbacks=evaluation_step_callbacks,
    )
