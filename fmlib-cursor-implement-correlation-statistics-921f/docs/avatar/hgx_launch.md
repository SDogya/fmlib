# 🥏 Пример обучения на кластере HGX

Использование кластера HGX необходимо, когда вам не хватает ресурсов в Datalab AI (максимум можно взять 2 GPU). Кластер предоставляет возможность использования нескольких GPU нод, 1 нода содержит 4 GPU A100 80Gb. Количество доступных вам нод определяется командными доступами.

## В батчевом режиме (1 карта)
```python
job = osiris.create(
    name="compare-fmlib",
    image="registry.ca.sbrf.ru/ci02684173/ci02697916/notebooks/python3.12/cuda12.4/d-03.000.00:d-03.000.00-gigachat",
    restart=False,
    command=[
        "/home/datalab/nfs/sber-amazme-fmlib/env/bin/accelerate",
        "launch",
        "-m",
        "fmlib.training.train"
    ],
    args=[
        "--config-dir=/home/datalab/nfs/sber-amazme-fmlib/examples/configs/train",
        "--config-name=sequence_representation_2M",
    ],
    envs={
        "PYTHONPATH": "/home/datalab/nfs/sber-amazme-fmlib/env",
        "OMP_NUM_THREADS": "7",
        "NCCL_DEBUG": "INFO",
    },
    num_nodes=1,
    num_gpus=1,
    type="pytorchjob",
)
job
```
- В `command` нужно указать путь до *accelerate* в установленном окружении. Далее прописать команду `launch`, флаг `-m`, модуль для запуска `fmlib.training.train`
- `envs`: в параметр *PYTHONPATH* нужно указать путь до окружения

## В multi-gpu/multi-node режиме
Пример с запуском на 2 нодах по 4 карточки (всего 8 процессов).

```python
job = osiris.create(
    name="fmlib-mg",
    image="registry.ca.sbrf.ru/ci02684173/ci02697916/notebooks/python3.12/cuda12.4/d-03.000.00:d-03.000.00-gigachat",
    restart=False,
    pool="public",
    command=[
        "/home/datalab/nfs/sber-amazme-fmlib/env/bin/accelerate",
        "launch",
    ],
    args=[
        "--mixed_precision=no",
        "--dynamo_backend=no",
        "--num_machines=2",
        "--num_processes=8",
        "--main_process_ip=$(MASTER_ADDR).datalab.svc.cluster.local",
        "--main_process_port=$(MASTER_PORT)",
        "--machine_rank=$(RANK)",
        "-m",
        "fmlib.training.train",
        "--config-dir=/home/datalab/nfs/sber-amazme-fmlib/examples/configs/train",
        "--config-name=sequence_representation_2M",
    ],
    envs={
        "PYTHONPATH": "/home/datalab/nfs/sber-amazme-fmlib/env",
        "OMP_NUM_THREADS": "28",
        "NCCL_DEBUG": "INFO",
    },
    num_nodes=2,
    num_gpus=4,
    type="pytorchjob",
)
job
```