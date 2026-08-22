# Минимальный ComfyUI из исходников. Ноды workflow (SaveVideo, CreateVideo,
# WanImageToVideo, UNETLoader, LoraLoaderModelOnly, KSamplerAdvanced) — comfy-core,
# в свежем ComfyUI есть из коробки, кастомные ноды не нужны.
ARG CUDA_VERSION=12.5.1
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git ffmpeg libgl1 libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
# build-essential + python3-dev нужны Triton'у: он JIT-компилирует CUDA-кернелы
# gcc'ом на лету (текст-энкодер LTX Gemma-12B этого требует). Без компилятора —
# RuntimeError: Failed to find C compiler.

WORKDIR /opt
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git

WORKDIR /opt/ComfyUI
# cu130, а не cu126: на cu126 ComfyUI пишет на старте
#   "You need pytorch with cu130 or higher to use optimized CUDA operations"
# и часть форматов H3 идёт эмуляцией — в логе видно
#   "emulated ops: float8_e5m2, mxfp8, float8_e4m3fn, nvfp4"
# (модель minimax_h3_..._int8_convrot + текст-энкодер qwen3vl_32b_nvfp4_awq).
# Требует драйвер >= 580; на user-ml стоит 610.43, запас есть.
# Базовый образ намеренно оставлен на CUDA 12.5: колёса torch везут свои
# CUDA-библиотеки, база нужна только под системные зависимости.
# Откат — вернуть cu126 здесь; рабочий образ помечен тегом
# ai-api-stuff-comfyui:known-good-0.31.0.
RUN pip3 install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu130 \
    torch torchvision torchaudio
RUN pip3 install --no-cache-dir -r requirements.txt

# Кастомные ноды: ComfyUI-Manager (ставить ноды/модели из UI) + ComfyUI-LTXVideo (ноды LTX).
WORKDIR /opt/ComfyUI/custom_nodes
RUN git clone --depth 1 https://github.com/ltdrdata/ComfyUI-Manager.git \
 && git clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo.git
RUN for d in ComfyUI-Manager ComfyUI-LTXVideo; do \
      if [ -f "$d/requirements.txt" ]; then pip3 install --no-cache-dir -r "$d/requirements.txt"; fi; \
    done
WORKDIR /opt/ComfyUI

EXPOSE 8188
# --listen 0.0.0.0 — чтобы ai-api-stuff достучался по сети compose.
# --fp16-intermediates — промежуточные тензоры в fp16, экономит VRAM.
#
# ВНИМАНИЕ: рабочие флаги заданы в docker-compose.yml (command:), этот CMD —
# только дефолт для запуска образа напрямую. Раньше здесь стоял
# --disable-pinned-memory; он убран, причины и замеры — в комментарии у command:.
CMD ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--fp16-intermediates"]
