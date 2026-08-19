import os
import re
import warnings
from functools import wraps
from pathlib import Path
from time import time
from typing import Any, Literal

import boto3
import requests
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from pyarrow import fs
from requests.auth import HTTPBasicAuth

from fmlib.constants.mlstorage import BOTO_CONFIG_DEFAULT_KWARGS, MLSTORAGE_REQUESTS_TIMEOUT


def _ttl_cache(ttl_seconds: int):
    def decorator(func):
        cache = {}

        @wraps(func)
        def wrapper(*args):
            now = time()

            if args in cache:
                result, timestamp = cache[args]
                if now - timestamp < ttl_seconds:
                    return result

            result = func(*args)
            cache[args] = (result, now)
            if len(cache) > 2:
                warnings.warn("There are at least 3 credential pairs in the cache", stacklevel=2)
            return result

        return wrapper

    return decorator


def _read_env_var(variable_name: str, is_required: bool = True) -> str | None:
    var = os.getenv(variable_name)
    if var is None and is_required:
        msg = f"Required environment variable not found: {variable_name}"
        raise RuntimeError(msg)
    return var


def _get_uz_credentials_from_env() -> tuple[str, str, str, str | None]:
    mls_url = _read_env_var("ML_STORAGE_URL")
    mls_username = _read_env_var("ML_STORAGE_LOGIN")
    mls_password = _read_env_var("ML_STORAGE_TOKEN")
    mls_verify = _read_env_var("ML_STORAGE_VERIFY", is_required=False)
    return mls_username, mls_password, mls_url, mls_verify


@_ttl_cache(ttl_seconds=86400)  # Do not ask for S3 ak sk twice - only the latest credentials work
def _get_s3_credentials_by_uz(username: str, password: str, url: str, verify: str) -> tuple[str, str, str]:
    auth_endpoint = "/api/v1/tokens/auth"
    full_url = url[: url.rfind("/")] + auth_endpoint
    headers = {"Content-Type": "application/json"}
    auth = HTTPBasicAuth(username, password)
    response = requests.get(full_url, headers=headers, auth=auth, verify=verify, timeout=MLSTORAGE_REQUESTS_TIMEOUT)
    if response.status_code == 200:
        credentials = response.json()["credentials"]
        return credentials["accessKeyId"], credentials["secretAccessKey"], credentials["sessionToken"]
    else:
        msg = f"Authorization error: {response.status_code} - {response.text}"
        raise RuntimeError(msg)


def _get_boto3_client(
    mls_ak: str,
    mls_sk: str,
    mls_token: str,
    url: str,
    verify: str,
) -> Any:  # Real return type requires additional types-boto3[s3] package
    session = boto3.session.Session()

    s3_client = session.client(
        service_name="s3",
        endpoint_url=url,
        aws_access_key_id=mls_ak,
        aws_secret_access_key=mls_sk,
        aws_session_token=mls_token,
        use_ssl=True,
        verify=verify,
        region_name="default",
        config=Config(connect_timeout=MLSTORAGE_REQUESTS_TIMEOUT, read_timeout=MLSTORAGE_REQUESTS_TIMEOUT),
    )

    return s3_client


def get_mlstorage_boto3_client() -> Any:
    """
    Возвращает клиент boto3 для взаимодействия с MLStorage,
    настроенный с использованием переменных окружения.

    Функция извлекает из окружения следующие переменные:
    - `ML_STORAGE_LOGIN`: Имя пользователя для аутентификации
    - `ML_STORAGE_TOKEN`: Пароль пользователя или токен ТУЗ для аутентификации
    - `ML_STORAGE_URL`: URL-адрес MLStorage в данном контуре
    - `ML_STORAGE_VERIFY`: Путь к сертификату для аутентификации
    Данные переменные используются для получения пары ключей доступа к S3 (Access Key и Secret Key)
    через сервис аутентификации, расположенный по тому же URL. Затем создаётся
    клиент boto3 для работы с MLStorage.

    Возвращает:
        Any: Настроенный экземпляр клиента boto3 для взаимодействия с MLStorage.
             Фактический тип возвращаемого значения — это 'boto3.client("s3")', однако указан как Any,
             чтобы избежать зависимости от пакета `types-boto3[s3]`.
    """
    mls_username, mls_password, mls_url, mls_verify = _get_uz_credentials_from_env()
    mls_ak, mls_sk, mls_token = _get_s3_credentials_by_uz(mls_username, mls_password, mls_url, mls_verify)
    mls_client = _get_boto3_client(mls_ak, mls_sk, mls_token, mls_url, mls_verify)

    return mls_client


def get_mlstorage_filesystem() -> fs.S3FileSystem:
    """
    Возвращает объект pyarrow S3FileSystem для взаимодействия с MLStorage,
    настроенный с использованием переменных окружения.

    Функция извлекает из окружения следующие переменные:
    - `ML_STORAGE_LOGIN`: Имя пользователя для аутентификации
    - `ML_STORAGE_TOKEN`: Пароль пользователя или токен ТУЗ для аутентификации
    - `ML_STORAGE_URL`: URL-адрес MLStorage в данном контуре
    - `ML_STORAGE_VERIFY`: Путь к сертификату для аутентификации
    Данные переменные используются для получения пары ключей доступа к S3 (Access Key и Secret Key)
    через сервис аутентификации, расположенный по тому же URL. Затем создаётся
    клиент pyarrow S3FileSystem для работы с MLStorage с использованием таймаутов из `MLSTORAGE_REQUESTS_TIMEOUT`

    Возвращает:
        fs.S3FileSystem: Сконфигурированный объект класса pyarrow S3FileSystem
                         для взаимодействия с MLStorage.
    """
    mls_username, mls_password, mls_url, mls_verify = _get_uz_credentials_from_env()
    mls_ak, mls_sk, mls_token = _get_s3_credentials_by_uz(mls_username, mls_password, mls_url, mls_verify)

    arrow_client = fs.S3FileSystem(
        endpoint_override=mls_url,
        access_key=mls_ak,
        secret_key=mls_sk,
        session_token=mls_token,
        region="default",
        tls_ca_file_path=mls_verify,
        request_timeout=MLSTORAGE_REQUESTS_TIMEOUT,
        connect_timeout=MLSTORAGE_REQUESTS_TIMEOUT,
    )

    return arrow_client


def _check_model_version_is_valid(model_version: str) -> bool:
    version_numbers = model_version.split(".")
    if len(version_numbers) != 3:
        return False

    if "*" in model_version:
        version_back = "*".join(model_version.split("*")[1:])
        version_back = version_back.replace("*", "").replace(".", "")
        if len(version_back) > 0:
            return False

    def version_number_is_valid(num: str):
        return len(num) >= 1 and len(num) <= 4 and (num.isdigit() or num == "*")

    return all(version_number_is_valid(num) for num in version_numbers)


def _check_model_id_is_valid(model_id: str) -> bool:
    return re.fullmatch(r"^\d{6}/\d{6}$", model_id) is not None


def generate_mlstorage_path(
    operation: Literal["save", "load"],
    model_id: str,
    model_version: str,
    boto3_client: Any = None,
) -> str:
    """
    Генерирует полный путь к хранилищу MLStorage (bucket/model_id/model_version) для модели
    в зависимости от операции и логики управления версиями.

    Функция определяет соответствующий бакет ("mlstage" для сохранения, "mlprod" для загрузки)
    и вычисляет корректный путь с учётом версионирования,
    используя логику чтения существующих объектов на S3.

    `model_version` Состоит из 3 числовых версий, разделённых точкой (major, minor, fix).
    Максимальная версия в каждой части: 9999.
    Если `model_version` содержит несколько символов ('*') в конце, поведение функции следующее:

    *   При `operation="load"` символы '*' заменяются на *наибольшую существующую*
        версию, соответствующую шаблону. Предположим, что есть следующие версии:
        0.1.0, 0.1.1, 0.2.0, 0.2.1, 1.0.0. Тогда:
        *   "0.1.0" станет "0.1.0"
        *   "0.1.*" станет "0.1.1"
        *   "0.*.*" станет "0.2.1"
        *   "0.3.*" вызовет RuntimeError("Model not found")
        *   "*.*.*" станет "1.0.0"
        *   "0.*.0", "*.0.0", "*.*.0" вызовет RuntimeError("Invalid version")
    *   При `operation="save"` символ '*' заменяется на *следующую доступную*
        версию, которая ещё не существует. Предположим, что есть следующие версии:
        0.1.0, 0.1.1, 0.2.0, 0.2.1, 1.0.0. Тогда:
        *   "0.1.1" вызовет RuntimeError("Model already exists")
        *   "0.1.*" станет "0.1.2"
        *   "0.*.*" станет "0.3.0"
        *   "0.3.*" станет "0.3.0"
        *   "*.*.*" станет "2.0.0"
        *   "*.*.*" станет "0.1.0", если в бакете нет ни одной модели
        *   "0.*.0", "*.0.0", "*.*.0" вызовет RuntimeError("Invalid version")

    Args:
        operation (Literal["save", "load"]): Операция, для которой генерируется путь.
            Определяет целевой бакет и поведение при разрешении версий.
        model_id (str): Уникальный идентификатор модели в формате
            f"{ID_бизнес_задачи}/{ID_версии_модели}" (генерируются в Библиотеке Моделей)
        model_version (str): Желаемая версия модели, может содержать подстановочные
            символы ('*') таким образом, что правее этих символов больше нет конкретных чисел
        boto3_client (Any, optional): Опциональный заранее настроенный клиент boto3 для S3.
            Если не передан, создаётся новый клиент с помощью `get_mlstorage_boto3_client()`.

    Raises:
        RuntimeError: Если указана неизвестная операция, если модель не найдена при загрузке,
                      или если при сохранении указана версия, которая уже существует.

    Returns:
        str: Полный путь к хранилищу в формате: `{bucket_name}/{model_id}/{model_version}`.
    """
    if operation == "save":
        bucket_name = "mlstage"
    elif operation == "load":
        bucket_name = "mlprod"
    else:
        msg = f"Unknown operation: {operation}"
        raise RuntimeError(msg)

    if not _check_model_id_is_valid(model_id):
        msg = f"Invalid model_id: {model_id}. It must be in the format: '123456/654321'"
        raise RuntimeError(msg)
    if not _check_model_version_is_valid(model_version):
        msg = f"Invalid model_version: {model_version}. It must be in the format: 'x.x.x' with possible '*' in the back"
        raise RuntimeError(msg)

    if boto3_client is None:
        boto3_client = get_mlstorage_boto3_client()

    prefix_path = f"{model_id}/{model_version.split('*')[0]}"
    response = boto3_client.list_objects_v2(
        Bucket="mlprod",  # Always mlprod
        Prefix=prefix_path,
    )
    if response["KeyCount"] == 0:
        if operation == "save":
            model_version = model_version.replace("*", "0")
            if model_version == "0.0.0":
                model_version = "0.1.0"
            path = f"{bucket_name}/{model_id}/{model_version}"
        else:
            msg = "Model not found"
            raise RuntimeError(msg)
    else:
        version_tree = {}
        for entry in response["Contents"]:
            versions = [int(version) for version in entry["Key"].split("/")[2].split(".")]
            if versions[0] not in version_tree:
                version_tree[versions[0]] = {}
            if versions[1] not in version_tree[versions[0]]:
                version_tree[versions[0]][versions[1]] = set()
            if versions[2] not in version_tree[versions[0]][versions[1]]:
                version_tree[versions[0]][versions[1]].add(versions[2])

        v1 = max(version_tree)
        v2 = max(version_tree[v1])
        v3 = max(version_tree[v1][v2])
        if operation == "load":
            path = f"{bucket_name}/{model_id}/{v1}.{v2}.{v3}"
        else:
            v_origs = model_version.split(".")
            if v_origs[0] == "*":
                path = f"{bucket_name}/{model_id}/{v1 + 1}.{0}.{0}"
            elif v_origs[1] == "*":
                path = f"{bucket_name}/{model_id}/{v1}.{v2 + 1}.{0}"
            elif v_origs[2] == "*":
                path = f"{bucket_name}/{model_id}/{v1}.{v2}.{v3 + 1}"
            else:
                msg = "Model already exists"
                raise RuntimeError(msg)

    return path


def save_model_to_mlstorage(
    model_files: list[str], model_id: str, model_version: str, boto_config_kwargs: dict[str, Any] = BOTO_CONFIG_DEFAULT_KWARGS
) -> None:
    """
    Загружает локальные файлы модели в указанное хранилище MLStorage.

    Данная функция определяет корректный путь назначения в бакете 'mlstage' с помощью
    `generate_mlstorage_path`, которая также разрешает версию, если
    в строке версии присутствуют подстановочные символы ('*'). Затем каждый файл из списка
    загружается в хранилище с помощью метода `boto3.client.upload_file`.

    Args:
        model_files (List[str]): Список полных локальных путей к файлам модели, которые необходимо
            загрузить.
        model_id (str): Уникальный идентификатор модели в формате
            f"{ID_бизнес_задачи}/{ID_версии_модели}" (генерируются в Библиотеке Моделей)
        model_version (str): Желаемая версия модели, может содержать подстановочные
            символы ('*') таким образом, что правее этих символов больше нет конкретных чисел
        boto_config_kwargs (dict[str, Any], опционально): Параметры конфигурации,
            передаваемые в `boto.s3.transfer.TransferConfig`. По умолчанию используется
            значение `BOTO_CONFIG_DEFAULT_KWARGS`.

    Raises:
        RuntimeError: Передаётся из функции `generate_mlstorage_path`, если не удалось
            разрешить версию или если модель с такой версией уже существует.

    Returns:
        None

    Пример:
        >>> save_model_to_mlstorage(
        ...     model_files=["/configs/config.yaml", "/models/model.safetensors"],
        ...     model_id="123456/654321",
        ...     model_version="0.1.*"
        ... )
        # Файлы будут загружены в соответствующую папку в MLStorage.
    """
    mls_client = get_mlstorage_boto3_client()

    mls_upload_path = generate_mlstorage_path("save", model_id, model_version, mls_client)
    bucket, s3_path = mls_upload_path.split("/", 1)

    for file_path in model_files:
        file_name = Path(file_path).name
        mls_client.upload_file(
            Filename=file_path, Bucket=bucket, Key=f"{s3_path}/{file_name}", Config=TransferConfig(**boto_config_kwargs)
        )
