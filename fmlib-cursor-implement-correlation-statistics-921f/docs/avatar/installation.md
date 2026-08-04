# 💾 Инструкция по установке Sber-AmazMe-FMLib для команды Avatar.

**Нужный образ для запуска в ЛД**: Python 3.12 CUDA 12.4 (`GigaChat:D-03.004.01`); `registry.ca.sbrf.ru/ci02684173/ci02697916/notebooks/python3.12/cuda12.4/d-03.000.00:d-03.000.00-gigachat`

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
6. Установка в окружение сторонних библиотек.
```bash
pip install poetry pre-commit polars scikit-uplift scikit-learn --extra-index-url https://token:${OSC_TOKEN}@sberosc.ca.sbrf.ru/repo/pypi/simple
```