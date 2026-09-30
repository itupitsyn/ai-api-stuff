ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=12.5.1
ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

FROM ${BASE_CUDA_RUN_CONTAINER} AS base

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libcudnn8 \
    python3 \
    python3-pip \
    libpython3.10 \
    git

# Один слой на все зависимости. Раньше psycopg ставился отдельно, чтобы правка
# requirements.txt не инвалидировала слой с torch и diffusers: тот тянул
# diffusers из git без версии, и пересборка была лотереей. С 27.09.2026 в
# requirements.txt прибито всё, включая коммит diffusers, — пересборка
# воспроизводима, и дробить слои больше незачем.
COPY requirements.txt .
# Второй командой в ТОМ ЖЕ слое, а не отдельным: onnxruntime и onnxruntime-gpu
# кладутся в один каталог пакета и затирают друг друга, а порядок установки
# внутри одного pip-вызова не задаётся. Процессорный приезжает зависимостью
# whisperx и в сборке лёг поверх GPU-варианта — детекция позы уехала на
# процессор и стала медленнее в одиннадцать раз (88 с против 8). Поэтому
# GPU-сборку ставим принудительно последней; --no-deps, чтобы она не тянула
# за собой разрешение зависимостей заново.
RUN pip3 install -r requirements.txt \
 && pip3 install --no-deps --force-reinstall onnxruntime-gpu==1.23.2

COPY .env main.py scheduler.py gpu_runner.py comfy_client.py stats.py result_store.py cutout.py swap_prep.py swap_workflow.py ./
COPY comfy_workflows ./comfy_workflows
# Препроцессор Wan как есть, из их репозитория. Веса к нему монтируются с
# хоста и в образ не идут — там 2.5 ГБ.
COPY wan_preprocess ./wan_preprocess

ENV PYTHONUNBUFFERED=1

CMD ["fastapi", "run", "main.py", "--port", "8000", "--no-reload"]
