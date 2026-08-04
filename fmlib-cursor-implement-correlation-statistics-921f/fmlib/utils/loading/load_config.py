from typing import Tuple

import pyarrow.fs as fs
from omegaconf import DictConfig, OmegaConf

from fmlib.constants.filenames import DEFAULT_CONFIG_FILE
from fmlib.constants.filesystem import DEFAULT_FILESYSTEM


def _check_model_path(model_path: str, filesystem: fs.FileSystem = DEFAULT_FILESYSTEM, check_path: bool = True) -> str:
    model_path = filesystem.normalize_path(model_path)
    if check_path:
        info: fs.FileInfo = filesystem.get_file_info(model_path)
        if info.type == fs.FileType.NotFound:
            msg: str = f"Model folder {model_path} not found."
            raise FileNotFoundError(msg)
        if info.type != fs.FileType.Directory:
            msg: str = f"Model path {model_path} is not a directory."
            raise FileNotFoundError(msg)
    return model_path


def load_config(
    model_path: str,
    config_file: str = DEFAULT_CONFIG_FILE,
    filesystem: fs.FileSystem | None = None,
    check_path: bool = True,
) -> Tuple[str, DictConfig]:
    """
    TODO: add docstring
    """
    if filesystem is None:
        filesystem = DEFAULT_FILESYSTEM
    model_path: str = _check_model_path(model_path, filesystem, check_path)
    config_path = filesystem.normalize_path(f"{model_path}/{config_file}")
    with filesystem.open_input_file(config_path) as f:
        config: DictConfig = OmegaConf.load(f)
    return (model_path, config)
