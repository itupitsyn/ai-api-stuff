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

COPY requirements.txt .
RUN pip3 install -r requirements.txt

# psycopg отдельной строкой, а НЕ в requirements.txt: тот участвует в COPY
# выше, и любая его правка инвалидирует слой с torch, whisperx и diffusers.
# Последний ставится из git без закрепления версии, так что пересборка того
# слоя — это ещё и лотерея с новым diffusers. Дешевле и безопаснее довезти
# одну мелкую зависимость своим слоем.
RUN pip3 install --no-cache-dir "psycopg[binary]"

COPY .env main.py scheduler.py gpu_runner.py comfy_client.py stats.py ./
COPY comfy_workflows ./comfy_workflows

ENV PYTHONUNBUFFERED=1

CMD ["fastapi", "run", "main.py", "--port", "8000", "--no-reload"]
