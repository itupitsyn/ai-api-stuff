# Минимальный ComfyUI из исходников. Ноды workflow (SaveVideo, CreateVideo,
# WanImageToVideo, UNETLoader, LoraLoaderModelOnly, KSamplerAdvanced) — comfy-core,
# в свежем ComfyUI есть из коробки, кастомные ноды не нужны.
ARG CUDA_VERSION=12.5.1
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git

WORKDIR /opt/ComfyUI
RUN pip3 install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu126 \
    torch torchvision torchaudio
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 8188
# --listen 0.0.0.0 — чтобы ai-api-stuff достучался по сети compose
CMD ["python3", "main.py", "--listen", "0.0.0.0", "--port", "8188"]
