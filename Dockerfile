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
    git

COPY .env main.py requirements.txt ./

RUN pip3 install -r requirements.txt

ENV PYTHONUNBUFFERED=1

CMD ["fastapi", "run", "main.py", "--port", "8000"]
