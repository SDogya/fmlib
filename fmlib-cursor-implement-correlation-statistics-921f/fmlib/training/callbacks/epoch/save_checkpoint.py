import warnings
from typing import Any, Dict, List, Literal, Self, Tuple

import accelerate
import pyarrow.fs as fs
import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save as to_bytes

from fmlib.constants.config_keys import WEIGHTS_PATH_KEY
from fmlib.constants.filenames import DEFAULT_CONFIG_FILE, DEFAULT_METRICS_FILE
from fmlib.constants.filesystem import DEFAULT_FILESYSTEM
from fmlib.constants.io import DEFAULT_CHECKPOINT_NAME, DEFAULT_TEMPLATE_CHECKPOINT_DIR
from fmlib.data.utils.prepare_directory import prepare_directory
from fmlib.training.training_types import MetricsType, TrainingState


class SaveCheckpointCallback:
    """
    Коллбек отвечающий за сохранение и очистку чекпоинтов во время обучения.
    Веса сохраняются в формате `.safetensors`, сопровождаются конфигом в формате `.yaml`
    и располагаются в отдельной папке по пути `base_path` т.е. формат доступный для чтения при помощи `fmlib.utils.load_model`.

    Аргументы:
        base_path (str): Путь к папке, в которой будут сохраняться чекпоинты.
        metric_to_track (str): Имя метрики, по которой будет определяться чекпоинт.
            Метрика должна считаться на предыдущем шаге.
        num_checkpoints (int): Количество сохраняемых чекпоинтов.
            По умолчанию - сохраняется 1 чекпоинт.
        direction (Literal["max", "min"]): Направление поиска чекпоинта.
            По умолчанию - сохраняются чекпоинты с наибольшей метрикой, `max`.
        submodule_to_save (str | None): Подмодуль, веса которого будут сохраняться.
            По умолчанию - `None`, модель сохраняется целиком.
            Если строка - то будет сохранен отдельный подмодуль.
        config_name (str): Имя конфига, который будет сохраняться вместе с весами.
            По умолчанию - DEFAULT_CONFIG_FILE.
        metrics_name (str): Имя дампа метрик, который будет сохраняться вместе с весами.
            По умолчанию - DEFAULT_METRICS_NAME.
        checkpoint_name (str): Имя чекпоинта, который будет сохраняться вместе с весами.
            По умолчанию - DEFAULT_CHECKPOINT_NAME.
        checkpoint_template (str): Шаблон имени папки чекпоинта.
            По умолчанию - DEFAULT_TEMPLATE_CHECKPOINT_DIR.
        filesystem (fs.FileSystem): Файловая система, в которой будут сохраняться чекпоинты.
            По умолчанию - DEFAULT_FILESYSTEM.
    """

    def __init__(
        self,
        base_path: str,
        metric_to_track: str,
        num_checkpoints: int = 1,
        direction: Literal["max", "min"] = "max",
        submodule_to_save: str | None = None,
        config_name: str = DEFAULT_CONFIG_FILE,
        metrics_name: str = DEFAULT_METRICS_FILE,
        checkpoint_name: str = DEFAULT_CHECKPOINT_NAME,
        checkpoint_template: str = DEFAULT_TEMPLATE_CHECKPOINT_DIR,
        filesystem: fs.FileSystem = DEFAULT_FILESYSTEM,
    ) -> None:
        self.metric_to_track: str = metric_to_track
        self.config_to_save: DictConfig | None = None
        self.config_name: str = config_name
        self.metrics_name: str = metrics_name
        self.checkpoint_name: str = checkpoint_name
        self.checkpoint_template: str = checkpoint_template

        if num_checkpoints < 1:
            msg: str = f"Invalid number of checkpoints. Must be at least 1, got {num_checkpoints=}"
            raise ValueError(msg)

        self.num_checkpoints: int = num_checkpoints

        if direction not in {"max", "min"}:
            msg: str = f"Invalid direction. Got {direction=}."
            raise ValueError(msg)

        self.default_value: float
        self.direction: str = direction
        if self.direction == "min":
            self.default_value = float("inf")
        else:
            self.default_value = float("-inf")

        self.state: List[Tuple[str, float]] = []
        self.filesystem: fs.FileSystem = filesystem
        self.submodule_to_save: str = submodule_to_save
        self.base_path: str = prepare_directory(path=base_path, strict=False, filesystem=filesystem)

    def get_checkpoint_path(self: Self, epoch: int) -> str:
        templated: str = self.checkpoint_template.format(epoch=epoch)
        return f"{self.base_path}/{templated}"

    def get_threshold(self: Self) -> float:
        if len(self.state) < self.num_checkpoints:
            return self.default_value
        values: List[float] = [v for _, v in self.state]
        values = sorted(values) if self.direction == "min" else sorted(values, reverse=True)
        return values[self.num_checkpoints - 1]

    def reset(self: Self) -> Self:
        self.cleanup()
        self.state = []
        return self

    def cleanup(self: Self) -> None:
        if len(self.state) < self.num_checkpoints:
            return

        state: List[Tuple[str, float]] = []
        threshold: float = self.get_threshold()
        for checkpoint, value in self.state:
            delete: bool
            delete = threshold < value if self.direction == "min" else value < threshold

            if delete:
                self.filesystem.delete_dir(checkpoint)
            else:
                state.append((checkpoint, value))
        self.state = state

    def get_directory(self: Self, epoch: int) -> str:
        dir_path: str = self.get_checkpoint_path(epoch)
        if self.filesystem.get_file_info(dir_path).type != fs.FileType.NotFound:
            if self.filesystem.get_file_info(dir_path).type != fs.FileType.Directory:
                msg: str = f"Sub-dirrectory {dir_path=} exists and is corrupted."
                raise ValueError(msg)
            msg: str = f"Sub-directory {dir_path=} already exists. Deleting."
            warnings.warn(msg, stacklevel=2)
            self.filesystem.delete_dir(dir_path)
        prepared: str = prepare_directory(dir_path, self.filesystem)
        return prepared

    def save_model(self: Self, model: torch.nn.Module, metrics: MetricsType, config: DictConfig, epoch: int) -> str:
        prepared: str = self.get_directory(epoch)
        checkpoint_path: str = f"{prepared}/{self.checkpoint_name}"
        with self.filesystem.open_output_stream(checkpoint_path) as stream:
            model_bytes: bytes = to_bytes(model.state_dict())
            stream.write(model_bytes)
        metrics_path: str = f"{prepared}/{self.metrics_name}"
        with self.filesystem.open_output_stream(metrics_path) as stream:
            metrics_obj: DictConfig = OmegaConf.create(metrics)
            yaml: str = OmegaConf.to_yaml(metrics_obj, sort_keys=True)
            stream.write(yaml.encode("utf-8"))
        config_path: str = f"{prepared}/{self.config_name}"
        with self.filesystem.open_output_stream(config_path) as stream:
            yaml: str = OmegaConf.to_yaml(config, sort_keys=True)
            stream.write(yaml.encode("utf-8"))
        assert self.filesystem.get_file_info(checkpoint_path).type == fs.FileType.File
        assert self.filesystem.get_file_info(metrics_path).type == fs.FileType.File
        assert self.filesystem.get_file_info(config_path).type == fs.FileType.File
        return prepared

    def get_config_to_save(self: Self, state: TrainingState) -> DictConfig:
        if self.config_to_save is None:
            config: DictConfig = state["config"]
            as_dict: Dict[str, Any] = OmegaConf.to_container(config, resolve=False)
            as_dict[WEIGHTS_PATH_KEY] = str(self.checkpoint_name)
            as_object: DictConfig = OmegaConf.create(as_dict)
            self.config_to_save = as_object
        assert self.config_to_save is not None
        return self.config_to_save

    def get_model_to_save(self: Self, state: TrainingState) -> torch.nn.Module:
        accelerator: accelerate.Accelerator = state["accelerator"]
        full_model: torch.nn.Module = accelerator.unwrap_model(state["pipeline"])

        if self.submodule_to_save is None:
            return full_model
        else:
            return full_model.get_submodule(self.submodule_to_save)

    def adapt_metric(self, metric_value: Any) -> float | int:
        # This is a deliberate act. The problem is that
        # `np.float64` is instance of `float`, but
        # can not be serialized by OmegaConf.
        if type(metric_value) not in [float, int]:
            warnings.warn(f"Metric value {type(metric_value)=} is not a primitive number.", stacklevel=2)
            float_metric_value: float = float(metric_value)
            warnings.warn("Trying to convert to `float`.", stacklevel=2)
            metric_value = float_metric_value
        return metric_value

    def get_metrics_to_save(self: Self, metrics: MetricsType) -> MetricsType:
        return {name: self.adapt_metric(value) for name, value in metrics.items()}

    def __call__(self: Self, state: TrainingState, metrics: MetricsType, epoch: int = 0) -> MetricsType:
        accelerator: accelerate.Accelerator = state["accelerator"]

        if not accelerator.is_main_process:
            return {}

        if self.metric_to_track not in metrics:
            warnings.warn("Metric to track not found. Defaulting.", stacklevel=2)

        epoch_iterable: Any = state["epoch_iterable"]
        length: int = len(epoch_iterable)
        if length <= self.num_checkpoints:
            warnings.warn(f"Suspicious number of checkpoints. Got {self.num_checkpoints=} vs {length=}.", stacklevel=2)

        raw_metric_value: Any = metrics.get(self.metric_to_track, self.default_value)
        metric_value: float | int = self.adapt_metric(raw_metric_value)

        save_model: bool
        threshold: float = self.get_threshold()
        save_model = metric_value < threshold if self.direction == "min" else threshold < metric_value

        if save_model:
            config_to_save: DictConfig = self.get_config_to_save(state)
            model_to_save: torch.nn.Module = self.get_model_to_save(state)
            metrics_to_save: MetricsType = self.get_metrics_to_save(metrics)
            file_path: str = self.save_model(
                metrics=metrics_to_save,
                config=config_to_save,
                model=model_to_save,
                epoch=epoch,
            )
            self.state.append((file_path, metric_value))

        self.cleanup()

        return {}

    def finalize(self: Self, state: TrainingState) -> MetricsType:
        self.reset()
        return {}
