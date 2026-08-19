from typing import Any

import hydra
import pyarrow.fs as fs
import torch
from omegaconf import DictConfig

from fmlib.constants.config_keys import INFERENCE_TRANSFORM_KEY
from fmlib.constants.filenames import DEFAULT_CONFIG_FILE
from fmlib.constants.filesystem import DEFAULT_FILESYSTEM
from fmlib.pipeline import InferencePipeline
from fmlib.utils.mlstorage.functions import generate_mlstorage_path, get_mlstorage_filesystem

from .load_config import load_config
from .load_model import load_model


def load_inference_pipeline_from_mlstorage(
    model_id: str,
    model_version: str,
    config_file: str = DEFAULT_CONFIG_FILE,
    filesystem: fs.FileSystem | None = None,
    transform_kwargs: list[dict[str, Any]] | None = None,
    check_path: bool = False,
):
    """
    Загружает готовый к использованию inference-пайплайн модели
    из хранилища MLStorage в бакете 'mlprod'.

    Функция автоматически определяет корректный путь к модели (детальное описание есть в
    `generate_mlstorage_path`), включая поддержку поиска последней версии по шаблону (через символ '*').
    При отсутствии переданной файловой системы (`filesystem`) создает клиент S3 для доступа к MLStorage
    (`pyarrow.s3.FileSystem`). В остальном работает аналогично функции
    `load_inference_pipeline`, но через MLStorage.

    Args:
        model_id (str): Уникальный идентификатор модели в формате
            f"{ID_бизнес_задачи}/{ID_версии_модели}" (генерируются в Библиотеке Моделей)
        model_version (str): Желаемая версия модели, может содержать подстановочные
            символы ('*') таким образом, что правее этих символов больше нет конкретных чисел
        config_file (str, optional): Имя файла конфигурации, находящегося в директории модели.
                                     По умолчанию используется значение `DEFAULT_CONFIG_FILE`.
        filesystem (fs.FileSystem | None, optional): Экземпляр файловой системы PyArrow.
                                                     Если не указан, будет создан
                                                     экземпляр S3FileSystem для MLStorage.
        transform_kwargs (list[dict[str, Any]] | None, optional): Список словарей с параметрами,
                                                                  используемыми при инициализации
                                                                  трансформаций в пайплайне.

    Returns:
        InferencePipeline: Готовый объект для выполнения инференса, содержащий загруженную
                           модель и необходимые преобразования. Конкретный тип возвращаемого
                           значения зависит от реализации функции `load_inference_pipeline`.

    Raises:
        RuntimeError: Если модель с указанным `model_id` и `model_version` не найдена
                      в MLStorage.
    """
    if filesystem is None:
        filesystem = get_mlstorage_filesystem()

    model_path = generate_mlstorage_path(
        operation="load",
        model_id=model_id,
        model_version=model_version,
    )

    return load_inference_pipeline(
        model_path=model_path,
        config_file=config_file,
        filesystem=filesystem,
        transform_kwargs=transform_kwargs,
        check_path=check_path,
    )


def load_inference_pipeline(
    model_path: str,
    config_file: str = DEFAULT_CONFIG_FILE,
    filesystem: fs.FileSystem = DEFAULT_FILESYSTEM,
    transform_kwargs: list[dict[str, Any]] | None = None,
    check_path: bool = True,
):
    """
    Загружает и возвращает готовый к использованию пайплайн для инференса.

    Parameters:
    -----------
    model_path : str
        Путь к папке с моделью.
    config_file : str, optional
        Имя файла конфигурации модели. По умолчанию используется `DEFAULT_CONFIG_FILE`.
    filesystem : fs.FileSystem, optional
        Система хранения данных (например, локальная или на основе S3).
        По умолчанию используется `DEFAULT_FILESYSTEM`.
    transform_kwargs : list[dict[str, Any]] | None, optional
        Список словарей с аргументами для создания нескольких объектов трансформаций.
        В конфигурации модели должен быть определен ключ `INFERENCE_TRANSFORM_KEY`,
            по которому будет создаваться объект трансформации.
        Если вы хотите создать несколько трансформаций с разными аргументами, то их нужно передать в виде списка словарей.
        Если вы хотите использовать трасформацию, определенную в конфиге, то используйте `None` или передайте пустой словарь.
        По умолчанию - `None`.

    Returns:
    --------
    InferencePipeline
        Объект пайплайна для инференса, содержащий загруженную модель и трансформации.

    Raises:
    -------
    FileNotFoundError
        Если не удаётся найти модель или конфигурационный файл.
    KeyError
        Если ключ `INFERENCE_TRANSFORM_KEY` отсутствует в конфиге.
    """
    model_path, config = load_config(
        model_path=model_path,
        config_file=config_file,
        filesystem=filesystem,
        check_path=check_path,
    )

    def make_transform(**kwargs) -> torch.nn.Module:
        transform_config: DictConfig = config[INFERENCE_TRANSFORM_KEY]
        return hydra.utils.instantiate(transform_config, _convert_="all", **kwargs)

    if transform_kwargs is not None:
        transforms: list[torch.nn.Module] = [make_transform(**kwargs) for kwargs in transform_kwargs]
    else:
        transforms = make_transform()
    return InferencePipeline(
        model=load_model(
            model_path=model_path,
            config_file=config_file,
            filesystem=filesystem,
            check_path=check_path,
        ),
        transform=transforms,
    )
