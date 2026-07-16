#!/usr/bin/env bash
# Докачивает модели Wan2.2 (fp8_scaled) для ComfyUI в структуру ComfyUI/models/.
# Запуск с хоста ОДИН раз перед стартом comfyui-контейнера:
#     bash comfyui/download_models.sh
# Целевую папку можно переопределить:  bash comfyui/download_models.sh /path/to/models
# HF_TOKEN (если задан в окружении) используется для снятия rate-limit.
#
# Объём: 4×UNET (~14 ГБ) + энкодер (~6 ГБ) + vae + loras ≈ 63 ГБ. Нужно место на диске.
set -euo pipefail

MODELS_DIR="${1:-$(dirname "$0")/models}"

FILES=(
  "diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors|https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
  "diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors|https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
  "diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors|https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
  "diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors|https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"
  "loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors|https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
  "loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors|https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
  "loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors|https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors"
  "loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors|https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors"
  "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors|https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
  "vae/wan_2.1_vae.safetensors|https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors"
)

AUTH=()
if [ -n "${HF_TOKEN:-}" ]; then
  AUTH=(--header "Authorization: Bearer ${HF_TOKEN}")
fi

echo "Качаю модели в: ${MODELS_DIR}"
for entry in "${FILES[@]}"; do
  rel="${entry%%|*}"
  url="${entry##*|}"
  dest="${MODELS_DIR}/${rel}"
  mkdir -p "$(dirname "${dest}")"
  echo "==> ${rel}"
  wget -c "${AUTH[@]}" -O "${dest}" "${url}"
done

echo "Готово. Модели в ${MODELS_DIR}"
