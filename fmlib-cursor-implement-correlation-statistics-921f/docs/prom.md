# 🚴‍♂️ Использование библиотеки Sber-AmazMe-FMLib в ПРОМ

## Алгоритм работы:
1. Обучите модель и получите лучшую контрольную точку
2. Подготовьте конфигурацию и [запустите инференс](inference.md) на тестовых данных для проверки работоспособности.
3. [Сохраните](../examples/mlstorage/mlstorage_save_load.ipynb) конфигурацию и веса моделей на MLStorage

    **В данный момент работает в Sigma и Omega (личный ноутбук, ИФТ, ЛД). Появление MLStorage в ПРОМе ожидается в Q1**

    ```python
    from fmlib.utils.mlstorage.functions import save_model_to_mlstorage

    model_files = [
        "./model.safetensors",
        "./config.yaml",
    ]

    save_model_to_mlstorage(
        model_files=model_files,
        model_id="123456/654321",
        model_version="0.3.*",
    )
    ```

4. Подготовьте AirFlow DAG в соответствии с [требованиями команды AmazMe Финансовые рекомендации](https://confluence.sberbank.ru/pages/viewpage.action?pageId=17035923769).

    Контактные лица:
    - Семенов Александр Сергеевич - <ASergeeviSemenov@omega.sbrf.ru>; <ASergeeviSemenov@sberbank.ru>.
    - Васкан Владислав Денисович - <VDVaskan@omega.sbrf.ru>; <VDVaskan@sberbank.ru>.

5. [Загрузите](../examples/mlstorage/mlstorage_save_load.ipynb)  конфигурацию и веса модели из MLStorage внутри AirFlow тасок

    **В данный момент работает в Sigma и Omega (личный ноутбук, ИФТ, ЛД). Появление MLStorage в ПРОМе ожидается в Q1**

    ```python
    from fmlib.utils.loading.load_pipeline import load_inference_pipeline_from_mlstorage

    load_inference_pipeline_from_mlstorage(
        model_id="123456/654321",
        model_version="0.3.*",
    )
    ```

6. Во время подготовки докер образа для запуска AirFlow тасок конкретизируйте версии зависимостей и переменные окружения.

    Пример `requirements.txt`:
    ```
    sber-amazme-fmlib==0.0.5
    torch==2.6.0+cu124
    triton==3.2.0
    ```

    **Примечание:** если вы укажете только версию PyTorch, то установится версия cpu-only. Поэтому важно указывать еще и версию CUDA, как в примере.
    **Примечание:** если вы используете `torch.compile`, то важно указать зависимость на triton.

    Пример `Dockerfile`:
    ```
    FROM image-to-replace

    USER root

    ENV TORCHINDUCTOR_CACHE_DIR /tmp/torchinductor_cache
    ENV TRITON_CACHE_DIR /tmp/triton_cache
    COPY requirements.txt
    RUN pip install --no-cache-dir -r requirements.txt

    USER airflow:airflow
    ```

    **Примечание:** указание переменных окружения `TORCHINDUCTOR_CACHE_DIR` и `TRITON_CACHE_DIR` необходимо для корректной работы `torch.compile`
