import pyarrow.fs as fs

from fmlib.constants.filesystem import DEFAULT_FILESYSTEM


def prepare_directory(
    path: str, strict: bool = True, exists_ok: bool = True, filesystem: fs.FileSystem = DEFAULT_FILESYSTEM
) -> str:
    """
    Прооверяет наличие и создаёт директорию.

    Аргументы:
        path (str): Путь к директории под вопросом.
        strict (bool): Проводить ли строгую проверку директории.
            По умолчанию - `True`, да, проводить проверку.
        exists_ok (bool): Является ли наличие директории ошибкой.
            По умолчанию - `True`, наличие директории - не ошибка.
        filesystem (fs.FileSystem): Файловая система, на которой производится проверка.
            По умолчанию - `DEFAULT_FILESYSTEM`, локальная файловая система.

    Возвращает:
        (str): Нормализованный путь к созданной директории.

    Ошибки:
        (IOError): Еслли директория существует и `exists_ok=False`.
        (IOError): Если директория повреждена и `strict=True`.
    """
    if filesystem.get_file_info(path).type == fs.FileType.NotFound:
        filesystem.create_dir(path, recursive=True)
    else:
        if filesystem.get_file_info(path).type == fs.FileType.Directory:
            if not exists_ok:
                msg: str = f"Directory already exists: {path=}."
                raise IOError(msg)
        else:
            if strict:
                msg: str = f"Directory {path=} is corrupted."
                raise IOError(msg)
    assert filesystem.get_file_info(path).type == fs.FileType.Directory
    return path
