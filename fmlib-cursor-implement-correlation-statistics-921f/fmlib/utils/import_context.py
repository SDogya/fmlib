import sys
import warnings
from contextlib import contextmanager
from typing import List

import pyarrow.fs as fs

from .loading.load_config import DEFAULT_FILESYSTEM


@contextmanager
def import_context(*paths: str, filesystem: fs.FileSystem = DEFAULT_FILESYSTEM, check_path: bool = True):
    """
    TODO: add docstring
    """
    original: List[str] = sys.path.copy()

    try:
        path: str
        for path in paths:
            if not filesystem.equals(DEFAULT_FILESYSTEM):
                warnings.warn("FS will not be used for imports.", stacklevel=2)
                continue
            if check_path:
                info: fs.FileInfo = filesystem.get_file_info(path)
                if info.type != fs.FileType.Directory:
                    warnings.warn(f"{path} is not a directory.", stacklevel=2)
            if path not in sys.path:
                sys.path.insert(0, path)
        yield
    finally:
        sys.path = original
