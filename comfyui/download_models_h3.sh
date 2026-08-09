#!/usr/bin/env bash
# Докачивает модели MiniMax H3 (33B, видео + нативное стерео-аудио) для ComfyUI.
# Подобрано под одну 3090 (24 ГБ VRAM) + 31 ГБ RAM: pruned int8 UNET, nvfp4 текст-энкодер.
#
# Запуск с хоста ОДИН раз перед стартом comfyui-контейнера:
#     bash comfyui/download_models_h3.sh
# Целевую папку можно переопределить:  bash comfyui/download_models_h3.sh /path/to/models
# Ref2VA (генерация по референсам) — опционально:  H3_REF2VA=1 bash comfyui/download_models_h3.sh
#
# Объём: UNET 21 ГБ + энкодер 15.7 ГБ + 2×VAE 5.8 ГБ ≈ 42.5 ГБ (+21 ГБ если ref2va).
# Требуется ComfyUI >= 0.30.0 (ноды MiniMaxH3ImageToVideo / MiniMaxH3ReferenceToVideo).
set -euo pipefail

MODELS_DIR="${1:-$(dirname "$0")/models}"
REPO="https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main"

FILES=(
  # T2V / I2V (first-frame, last-frame, first+last). 20.97 ГБ.
  # Альтернативы в том же репо: pruned_fp8_scaled (20.96 ГБ, ~то же),
  # int8_convrot (34 ГБ, непруненый), bf16 (66 ГБ) — на 3090 не влезут.
  "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors|${REPO}/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"

  # Текст-энкодер Qwen3-VL-32B. nvfp4 (15.7 ГБ) на Ampere эмулируется софтом —
  # прироста скорости нет, но и потерь тоже; берём ради размера: int8-версия
  # весит 27 ГБ и с 31 ГБ RAM оффлоад начинает свопиться.
  "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors|${REPO}/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"

  # Видео- и аудио-VAE — нужны оба, аудио генерится тем же forward pass.
  "vae/minimax_h3_video_vae_fp16.safetensors|${REPO}/vae/minimax_h3_video_vae_fp16.safetensors"
  "vae/minimax_h3_audio_vae_fp32.safetensors|${REPO}/vae/minimax_h3_audio_vae_fp32.safetensors"

  # Turbo LoRA — аналог lightx2v для Wan: 4-8 шагов вместо 20. Это уже
  # переконверченный под ComfyUI и под pruned-базу вариант (оригинал larryvrh
  # ругается на shape adaln_proj при загрузке в pruned). 591 МБ.
  # ВАЖНО: с ней sampler euler + scheduler beta; res_multistep из дефолтного
  # воркфлоу даёт "диско" и портит аудио.
  "loras/minimax_h3_turbo_4step_ckpt500_comfyui_pruned.safetensors|https://huggingface.co/QrusherZA/H3_Turbo_ComfyUI/resolve/main/minimax_h3_turbo_4step_ckpt500_comfyui_pruned.safetensors"
)

if [ "${H3_REF2VA:-0}" = "1" ]; then
  FILES+=("diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors|${REPO}/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors")
fi

AUTH=()
if [ -n "${HF_TOKEN:-}" ]; then
  AUTH=(--header "Authorization: Bearer ${HF_TOKEN}")
fi

echo "Качаю модели MiniMax H3 в: ${MODELS_DIR}"
for entry in "${FILES[@]}"; do
  rel="${entry%%|*}"
  url="${entry##*|}"
  dest="${MODELS_DIR}/${rel}"
  mkdir -p "$(dirname "${dest}")"
  echo "==> ${rel}"
  wget -c "${AUTH[@]}" -O "${dest}" "${url}"
done

echo "Готово. Модели в ${MODELS_DIR}"
echo "Воркфлоу для UI: comfy_workflows/minimax_h3_{t2v,i2v}_turbo_ui.json"
