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
RUN pip3 install -r requirements.txt

COPY .env main.py scheduler.py gpu_runner.py comfy_client.py stats.py result_store.py ./
COPY comfy_workflows ./comfy_workflows

ENV PYTHONUNBUFFERED=1

CMD ["fastapi", "run", "main.py", "--port", "8000", "--no-reload"]
