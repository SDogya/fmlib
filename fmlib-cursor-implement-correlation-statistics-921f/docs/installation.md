# 💾 Инструкция по установке Sber-AmazMe-FMLib в различных средах.

**Рекомендуемый образ для запуска в ЛД**: Python 3.12 CUDA 12.4 (`GigaChat:D-03.004.01`); `registry.ca.sbrf.ru/ci02684173/ci02697916/notebooks/python3.12/cuda12.4/d-03.000.00:d-03.000.00-gigachat`

<a name="toc"></a>
# Содержание
* [Установка из исходного кода](#source-installation)
* [Установка из Nexus](#nexus-installation)

<a name="source-installation"></a>
## 📀 Установка из исходного кода
**Общие шаги:**

0. Склонируйте репозиторий (пример для ЛД)
```bash
git clone https://df-bitbucket.ca.sbrf.ru/scm/amazmefm/sber-amazme-fmlib.git
cd sber-amazme-fmlib
```
1. Установите виртуальное окружение и активируйте его.
```bash
python -m venv env
source env/bin/activate
```
2. [Получите токен для скачивания из SberOSC](access.md#sber-osc) и установите переменную окружения.
```bash
export OSC_TOKEN="YOUR_PRIVATE_OSC_TOKEN"
```

Существует два варианта установки из источника:
- **Установка изменяемой версии**

    Этот вариант установки необходим в случае разработки нового функционала библиотеки. При установке создается ссылка на ваш репозиторий, откуда берется весь актуальный код. Этот вариант избавит вас от постоянной переустановки библиотеки, перезапуска Jupyter-ноутбуков и позволит сконцентрироваться на разработке.

    3. Установите Poetry.
    ```bash
    pip install poetry==2.1.3 --index-url https://token:${OSC_TOKEN}@sberosc.ca.sbrf.ru/repo/pypi/simple
    ```

    4. Настройка окружения Poetry для скачивания из SberOSC и отключение проверки сертификатов.
    ```bash
    export POETRY_HTTP_BASIC_SBER_OSC_PASSWORD=$OSC_TOKEN
    poetry config certificates.sber-osc.cert false
    ```

    5. Установка библиотеки.
    ```bash
    poetry install
    ```

- **Установка неизменяемой версии**

    Этот вариант установки необходим при использовании в окружениях, где невозможно ссылаться на акутальную реализацию. Например, при запуске на кластере HGX. Подробную инструкцию по работе с кластером HGX можно найти [здесь](HGX.md).

    При внесении изменений в код библиотеки необходимо повторить процедуру установки. Создавать новое окружение не нужно.

    3. Установка библиотеки
    ```bash
    pip install -e . --extra-index-url https://token:${OSC_TOKEN}@sberosc.ca.sbrf.ru/repo/pypi/simple
    ```


<a name="nexus-installation"></a>
## 💽 Установка из Nexus
Перед установкой убедитесь, что необходимая вам версия доступна в Nexus. Для этого перейдите в Nexus ([ЛД](https://df-nexus.ca.sbrf.ru/#browse/browse), [Сигма](https://nexus-ci.delta.sbrf.ru/#browse/browse)) в каталог `pypi-dev` и найдите там `sber-amazme-fmlib`.
- **Установка в ЛД**
    1. [Получите токен для скачивания из SberOSC](access.md#sber-osc) и установите переменную окружения.
        ```bash
        export OSC_TOKEN="YOUR_PRIVATE_OSC_TOKEN"
        ```
    2. [Получите credentials для скачивания из Nexus](access.md#nexus) и установите переменные окружения.
        ```bash
        export NEXUS_NAME="YOUR_NEXUS_NAME"
        export NEXUS_PASSWORD="YOUR_PRIVATE_NEXUS_TOKEN"
        ```
    3. Установите библиотеку с использованием стандартного пакетного менеджера `pip`.
        ```bash
        pip install sber-amazme-fmlib \
        --index-url https://${NEXUS_NAME}:${NEXUS_PASSWORD}@df-nexus.ca.sbrf.ru/repository/pypi-dev/simple \
        --extra-index-url https://token:${OSC_TOKEN}@sberosc.ca.sbrf.ru/repo/pypi/simple
        ```
- **Установка в Sigma**
    1. [Получите токен для скачивания из SberOSC](access.md#sber-osc) и установите переменную окружения.
        ```bash
        export OSC_TOKEN="YOUR_PRIVATE_OSC_TOKEN"
        ```
    2. [Получите credentials для скачивания из Nexus](access.md#nexus) и установите переменные окружения.
        ```bash
        export NEXUS_NAME="YOUR_NEXUS_NAME"
        export NEXUS_PASSWORD="YOUR_PRIVATE_NEXUS_TOKEN"
        ```
    3. Установите библиотеку с использованием стандартного пакетного менеджера `pip`.
        ```bash
        pip install sber-amazme-fmlib \
        --index-url https://${NEXUS_NAME}:${NEXUS_PASSWORD}@nexus-ci.sigma.sbrf.ru/repository/pypi-dev/simple \
        --extra-index-url https://token:${OSC_TOKEN}@sberosc.sigma.sbrf.ru/repo/pypi/simple
        ```