import re
import warnings
from pprint import pformat
from typing import Dict, List, Optional

import hydra
import pyarrow.fs as fs
import safetensors.torch as st
import torch
import torch.onnx
from omegaconf import DictConfig

from fmlib.constants.config_keys import (
    MODEL_KEY,
    WEIGHTS_IGNORE_PATTERN_KEY,
    WEIGHTS_MAPPING_KEY,
    WEIGHTS_PATH_KEY,
    WEIGHTS_PATTERN_KEY,
)
from fmlib.constants.filenames import DEFAULT_CONFIG_FILE
from fmlib.constants.filesystem import DEFAULT_FILESYSTEM
from fmlib.utils.import_context import import_context

from .load_config import load_config


def instantiate_model_from_config(
    config: DictConfig,
    model_path: str | None = None,
    filesystem: fs.FileSystem = DEFAULT_FILESYSTEM,
    check_path: bool = True,
) -> torch.nn.Module:
    model_config: DictConfig = config.get(MODEL_KEY)
    model: torch.nn.Module
    if model_path is None:
        model = hydra.utils.instantiate(model_config)
    else:
        with import_context(model_path, filesystem=filesystem, check_path=check_path):
            model = hydra.utils.instantiate(model_config)
    return model


def _load_state_dict(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = False,
) -> torch.nn.Module:
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if (len(missing) > 0) or (len(unexpected) > 0):
        assert not strict
        missing, unexpected = sorted(missing), sorted(unexpected)
        msg: str = (
            "Unexpected or missing keys found while loading model:\n"
            f"\tMissing keys ({len(missing)=}): \n{pformat(missing, compact=True)};\n"
            f"\tUnexpected keys ({len(unexpected)=}): \n{pformat(unexpected, compact=True)}."
        )
        warnings.warn(msg, stacklevel=2)
    return model


def _get_weights(
    path: str,
    pattern: Optional[str] = None,
    ignore_pattern: Optional[str] = None,
    mapping: Dict[str, str] | None = None,
    filesystem: fs.FileSystem = DEFAULT_FILESYSTEM,
) -> Dict[str, torch.Tensor]:
    if mapping is None:
        mapping = {}
    path: str = filesystem.normalize_path(path)
    info: fs.FileInfo = filesystem.get_file_info(path)
    if info.type != fs.FileType.File:
        msg: str = f"Path {path} is not a file."
        raise FileNotFoundError(msg)

    is_safetensors: bool = "safetensors" in info.extension

    state_dict: Dict[str, torch.Tensor]
    with filesystem.open_input_file(path) as f:
        if is_safetensors:
            warnings.warn("Safetensors will be loaded at once.", stacklevel=2)
            state_dict = st.load(f.readall())
        else:
            warnings.warn("Loading from torch file is not recommended.", stacklevel=2)
            state_dict = torch.load(f)

    new_state_dict: Dict[str, torch.Tensor] = filter_parameters(
        ignore_pattern=ignore_pattern,
        state_dict=state_dict,
        pattern=pattern,
        mapping=mapping,
    )
    return new_state_dict


def _load_weights(
    model: torch.nn.Module,
    path: str,
    strict: bool = False,
    pattern: Optional[str] = None,
    ignore_pattern: Optional[str] = None,
    mapping: Dict[str, str] | None = None,
    filesystem: fs.FileSystem = DEFAULT_FILESYSTEM,
) -> torch.nn.Module:
    if mapping is None:
        mapping = {}
    new_state_dict: Dict[str, torch.Tensor] = _get_weights(
        path=path,
        pattern=pattern,
        mapping=mapping,
        filesystem=filesystem,
        ignore_pattern=ignore_pattern,
    )
    return _load_state_dict(model, new_state_dict, strict)


def rename_by_pattern(keys: List[str], pattern: Optional[str] = None) -> Dict[str, str]:
    keys: List[str] = sorted(keys)

    if pattern is None:
        return {k: k for k in keys}

    compiled: re.Pattern[str] = re.compile(pattern)
    result: Dict[str, str] = {}
    for key in keys:
        curr_match: Optional[re.Match[str]] = compiled.match(key)

        if curr_match is None:
            result[key] = key
        else:
            result[key] = curr_match.group(1)

    return result


def filter_by_pattern(keys: List[str], ignore_pattern: Optional[str] = None) -> List[str]:
    if ignore_pattern is None:
        return keys

    compiled: re.Pattern[str] = re.compile(ignore_pattern)
    keys = [key for key in keys if compiled.match(key) is None]

    return keys


def get_mapping(
    keys: List[str], pattern: Optional[str] = None, ignore_pattern: Optional[str] = None, mapping: Dict[str, str] | None = None
) -> Dict[str, str]:
    if mapping is None:
        mapping = {}

    filtered: List[str] = filter_by_pattern(sorted(keys), ignore_pattern)
    fixed_mapping: Dict[str, str] = rename_by_pattern(filtered, pattern)
    fixed_mapping.update(mapping)
    return fixed_mapping


def filter_parameters(
    state_dict: Dict[str, torch.Tensor],
    pattern: Optional[str] = None,
    ignore_pattern: Optional[str] = None,
    mapping: Dict[str, str] | None = None,
) -> Dict[str, torch.Tensor]:
    if mapping is None:
        mapping = {}

    if (pattern is None) and (ignore_pattern is None) and (len(mapping) == 0):
        return state_dict

    keys: List[str] = sorted(state_dict.keys())
    fixed_mapping: Dict[str, str] = get_mapping(keys, pattern, ignore_pattern, mapping)

    result: Dict[str, torch.Tensor] = {new_key: state_dict[key] for key, new_key in fixed_mapping.items()}

    return result


def instantiate_model(
    model_path: str,
    config_file: str = DEFAULT_CONFIG_FILE,
    filesystem: fs.FileSystem = DEFAULT_FILESYSTEM,
    check_path: bool = True,
) -> torch.nn.Module:
    model_path, config = load_config(
        model_path=model_path,
        config_file=config_file,
        filesystem=filesystem,
        check_path=check_path,
    )
    model: torch.nn.Module = instantiate_model_from_config(
        filesystem=filesystem,
        model_path=model_path,
        config=config,
        check_path=check_path,
    )
    return model


def load_model_weights(
    config: DictConfig,
    model: torch.nn.Module,
    model_path: str | None = None,
    exact_weights_file: str | None = None,
    strict: bool = False,
    filesystem: fs.FileSystem = DEFAULT_FILESYSTEM,
) -> torch.nn.Module:
    """
    Инстанциирует модель и опционально заполняет весами.

    Аргументы:
        config (DictConfig): Конфигурация модели.
        model (torch.nn.Module): Модель, в которую нужно загрузить веса.
        model_path: (str | None): Путь к модели и, возможно, кастомным модулям.
        exact_weights_file: (str | None): Точный путь к файлу с весами.
        strict (bool): Если True, будет выброшено исключение, если весов не хватает.
        filesystem (fs.FileSystem): Система файлов для работы с файлами.

    *Примечание*: Приоритет загрузки моделей следующий:
        - `exact_weights_file` (если указан в аргументах)
        - файл из `model_path` (если указан в конфиге)
        - случайная инициализация
    """
    if exact_weights_file and model_path:
        msg: str = (
            "Exact weights file and model path are specified: "
            f"{exact_weights_file=} and {model_path=}. "
            "`exact_weights_path` takes precedence."
        )
        warnings.warn(msg, stacklevel=2)
    if exact_weights_file is None:
        weights_path: str = config.get(WEIGHTS_PATH_KEY)
        full_path: str = weights_path
        if model_path is not None:
            full_path = f"{model_path}/{weights_path}"
    else:
        msg: str = f"Model will be loaded from an override path: {exact_weights_file=}."
        warnings.warn(msg, stacklevel=2)
        full_path = exact_weights_file
    full_path = filesystem.normalize_path(full_path)
    result: torch.nn.Module = _load_weights(
        ignore_pattern=config.get(WEIGHTS_IGNORE_PATTERN_KEY, None),
        pattern=config.get(WEIGHTS_PATTERN_KEY, None),
        mapping=config.get(WEIGHTS_MAPPING_KEY, None),
        filesystem=filesystem,
        path=full_path,
        strict=strict,
        model=model,
    )
    return result


def load_model_from_config(
    config: DictConfig,
    model_path: str | None = None,
    exact_weights_file: str | None = None,
    strict: bool = False,
    filesystem: fs.FileSystem = DEFAULT_FILESYSTEM,
    check_path: bool = True,
) -> torch.nn.Module:
    result: torch.nn.Module = instantiate_model_from_config(
        model_path=model_path,
        filesystem=filesystem,
        config=config,
        check_path=check_path,
    )
    has_weights_path: bool = WEIGHTS_PATH_KEY in config
    has_exact_weights_file: bool = exact_weights_file is not None
    if has_weights_path or has_exact_weights_file:
        result = load_model_weights(
            model=result,
            config=config,
            strict=strict,
            model_path=model_path,
            filesystem=filesystem,
            exact_weights_file=exact_weights_file,
        )
    else:
        warnings.warn("No weights_path specified in the config.", stacklevel=2)
        if strict:
            msg: str = "Model weights are not specified in the config."
            raise ValueError(msg)
    return result


def load_model(
    model_path: str,
    strict: bool = False,
    config_file: str = DEFAULT_CONFIG_FILE,
    filesystem: fs.FileSystem = DEFAULT_FILESYSTEM,
    check_path: bool = True,
) -> torch.nn.Module:
    """
    TODO: add docstring
    """
    model_path, config = load_config(
        model_path=model_path, config_file=config_file, filesystem=filesystem, check_path=check_path
    )
    return load_model_from_config(
        config=config,
        model_path=model_path,
        strict=strict,
        filesystem=filesystem,
        check_path=check_path,
    )
