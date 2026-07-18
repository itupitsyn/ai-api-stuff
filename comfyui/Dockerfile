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
RUN pip3 install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu126 \
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
# --listen 0.0.0.0 — чтобы ai-api-stuff достучался по сети compose
CMD ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
