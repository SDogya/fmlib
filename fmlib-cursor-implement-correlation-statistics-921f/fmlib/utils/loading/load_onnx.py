import warnings
from typing import Optional

import onnx
import onnxscript
import pyarrow.fs as fs
import torch
import torch.onnx

from fmlib.constants.config_keys import INPUT_NAMES_KEY, ONNX_PATH_KEY, OUTPUT_NAMES_KEY, WEIGHTS_PATH_KEY
from fmlib.constants.filenames import DEFAULT_CONFIG_FILE, ONNX_EXTENSION
from fmlib.constants.filesystem import DEFAULT_FILESYSTEM
from fmlib.onnx.compiling import CompiledModel, LazyCompiledModel

from .load_config import load_config
from .load_model import load_model


def _load_torch_onnx(
    path: str,
    filesystem: fs.FileSystem = DEFAULT_FILESYSTEM,
) -> torch.onnx.ONNXProgram:
    path: str = filesystem.normalize_path(path)
    info: fs.FileInfo = filesystem.get_file_info(path)
    if info.type != fs.FileType.File:
        msg: str = f"Path {path} is not a file."
        raise FileNotFoundError(msg)
    if ONNX_EXTENSION not in info.extension:
        msg: str = f"Incorrect file extension: {path}."
        raise IOError(msg)

    with filesystem.open_input_file(path) as f:
        loaded: onnx.ModelProto = onnx.load(f)

    ir_model: onnxscript.ir.Model = onnxscript.ir.from_proto(loaded)
    return torch.onnx.ONNXProgram(ir_model, exported_program=None)


def load_torch_onnx(
    model_path: str,
    config_file: str = DEFAULT_CONFIG_FILE,
    filesystem: fs.FileSystem = DEFAULT_FILESYSTEM,
) -> torch.onnx.ONNXProgram:
    """
    TODO: add docstring
    """
    model_path, config = load_config(
        model_path=model_path,
        config_file=config_file,
        filesystem=filesystem,
    )
    result: torch.onnx.ONNXProgram = _load_torch_onnx(
        path=f"{model_path}/{config.onnx_path}",
        filesystem=filesystem,
    )
    return result


def load_compiled_model(
    model_path: str,
    config_file: str = DEFAULT_CONFIG_FILE,
    filesystem: fs.FileSystem = DEFAULT_FILESYSTEM,
    check_path: bool = True,
) -> CompiledModel:
    """
    TODO: add docstring
    """
    model_path, config = load_config(
        model_path=model_path,
        config_file=config_file,
        filesystem=filesystem,
        check_path=check_path,
    )
    compiled: torch.onnx.ONNXProgram = _load_torch_onnx(
        path=f"{model_path}/{config.onnx_path}",
        filesystem=filesystem,
    )
    result: CompiledModel = CompiledModel(
        compiled=compiled,
        input_names=config.get(INPUT_NAMES_KEY, None),
        output_names=config.get(OUTPUT_NAMES_KEY, None),
    )
    return result


def load_lazy_compiled_model(
    model_path: str,
    config_file: str = DEFAULT_CONFIG_FILE,
    filesystem: fs.FileSystem | None = None,
    check_path: bool = True,
) -> LazyCompiledModel:
    """
    TODO: add docstring
    """
    model_path, config = load_config(
        model_path=model_path,
        config_file=config_file,
        filesystem=filesystem,
        check_path=check_path,
    )
    onnx_model: Optional[torch.onnx.ONNXProgram] = None
    if ONNX_PATH_KEY in config:
        full_path: str = f"{model_path}/{config.onnx_path}"
        info: fs.FileInfo = filesystem.get_file_info(full_path)
        if info.type == fs.FileType.File:
            onnx_model = load_torch_onnx(
                model_path=model_path,
                config_file=config_file,
                filesystem=filesystem,
            )
        else:
            warnings.warn(f"Missing file: {full_path}.", stacklevel=2)
    model: torch.nn.Module = load_model(
        model_path=model_path,
        config_file=config_file,
        filesystem=filesystem,
        check_path=check_path,
    )
    result: LazyCompiledModel = LazyCompiledModel(
        model=model,
        compiled=onnx_model,
        input_names=config.get(INPUT_NAMES_KEY, None),
        output_names=config.get(OUTPUT_NAMES_KEY, None),
    )
    result.compiled = onnx_model
    return result


def load_any_compiled_model(
    model_path: str,
    config_file: str = DEFAULT_CONFIG_FILE,
    filesystem: fs.FileSystem = DEFAULT_FILESYSTEM,
    check_path: bool = True,
) -> LazyCompiledModel | CompiledModel | None:
    model_path, config = load_config(
        model_path=model_path,
        config_file=config_file,
        filesystem=filesystem,
        check_path=check_path,
    )
    result: LazyCompiledModel | CompiledModel | None = None
    if WEIGHTS_PATH_KEY in config:
        weights_path: str = f"{model_path}/{config.weights_path}"
        weights_info: fs.FileInfo = filesystem.get_file_info(weights_path)
        if weights_info.type == fs.FileType.File:
            result = load_lazy_compiled_model(
                model_path=model_path,
                config_file=config_file,
                filesystem=filesystem,
                check_path=check_path,
            )
        else:
            warnings.warn(f"Missing file: {weights_path}.", stacklevel=2)
    if result is None:
        result = load_compiled_model(
            model_path=model_path,
            config_file=config_file,
            filesystem=filesystem,
            check_path=check_path,
        )
    return result
