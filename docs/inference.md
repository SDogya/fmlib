# 🤳 Инференс моделей c помощью Sber-AmazMe-FMLib

Инференс моделей в библиотеке обеспечивается следующими абстракциями:
- `ParquetDataset`

    Абстракция, отвечающая за загрузку данных из `parquet`-файлов и подачу их в модель. Более подробная информация содержится [здесь](data_processing.md).

- `InferencePipeline`

    Абстракция, реализующая инференс-пайплайн на основе `PyTorch` модели.
    Пайплайн представляет из себя последовательное применение `Transform` к прочитанным данным из `ParquetDataset` и дальнейшую подачу в модель.

    Пайплайн является наследником `torch.nn.Module`, поэтому с ним можно обращаться как с обычной моделью. Вы также можете компилировать этот модуль с помощью `torch.compile` для возможного повышения производительности вычислительных операций. В библиотеке не используется компиляция модулей по умолчанию, потому что есть примеры когда это, наоборот, замедляет производительность.

    Пайплайн может принимать несколько объектов `Transform` (массив объектов). В таком случае результат работы будет содержать столько значений, сколько объектов `Transform` было подано на вход пайплайну. Например, если в было дано 3 объекта `Transform`, то результат будет массив размера 3.
    ```
    batch -> transform[0] -> model -> result[0]
    batch -> transform[1] -> model -> result[1]
    batch -> transform[2] -> model -> result[2]
    ```

- `PartitionedParquetWriter`

    Абстракция, отвечащая за побатчевую запись результат модели в `parquet`-файлы.
    
    Ключевым параметром этой абстракции является результирующий размер `parquet`-файла - `write_every_n_batch`. **Примечание:** в целях повышения производительности дальнейшей обработки результата модели рекомендуется подбирать значение параметра так, чтобы результирующий размер одной партиции составлял 256-512 Мб.

    Позволяет записывать батчи не только в локальную файловую систему, но и на удаленные файловые системы (например, S3 или HDFS) - для этого необходимо передать параметр `filesystem`.
    
    Поддерживает работу в MultiGPU режиме, когда инференс модели происходит в нескольких процессах.


## Пример инференса модели

Перед запуском примера укажите корректные значения:
- `model_folder` - путь до папки с конфигурациями и весами модели. В папке должен быть файл `config.yaml`, где описана архитектура модели. 
    
    `parquet_metadata.yaml` - метаданные колонок из датасета для инференса.

- `dataset_path` - путь до датасета для инференса. Если датасет лежит в локальной файловой системе, то параметр `filesystem` не нужно указывать.

```python
import torch
import pyarrow.fs as fs
from tqdm.autonotebook import tqdm

from fmlib.data.io import ParquetDataset
from fmlib.data.io.writer import PartitionedParquetWriter
from fmlib.pipeline import InferencePipeline
from fmlib.utils.loading import load_config
from fmlib.utils.mode_context import mode_context

model_folder = "examples/configs/inference/feature_transformer"
dataset_path = "/user/team/team_ai_avatar/avatar_fm/examples/campaign_demo/td_processed/test"
device = "cuda:0"

_, metadata = load_config(model_folder, config_file="parquet_metadata.yaml")

dataloader = ParquetDataset(
    source=dataset_path,
    metadata=metadata,
    partition_size=4096,
    batch_size=512,
    filesystem=fs.HadoopFileSystem("hdfs://arnsdpsbx", port=0),
    device=device,
)

pipeline: InferencePipeline = (
    load_inference_pipeline(model_folder)
    .to(device)
)

writer = PartitionedParquetWriter(
    base_path="model_inference_result",
    write_every_n_batch=128,
)
with torch.no_grad(), mode_context(pipeline, training=False):
    for batch in tqdm(dataloader):
        transformed, model_logits = pipeline(batch)
        result = model_logits[0]
        result["epk_id"] = batch["epk_id"]
        writer.write(result)
writer.close()
```