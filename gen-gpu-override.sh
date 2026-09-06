#!/usr/bin/env bash
# Генерирует docker-compose.override.yml под фактическое число видеокарт.
#
# Зачем: compose статичен и не умеет поднимать N сервисов по числу GPU. Ни
# replicas, ни --scale не годятся — они дают одинаковые контейнеры, а каждому
# ComfyUI нужна СВОЯ карта, своё имя и свой порт. Поэтому список генерируем.
#
# Базовый docker-compose.yml рассчитан на одну карту и работает сам по себе.
# Этот скрипт добавляет остальные и переписывает у API две переменные —
# GPU_DEVICES и COMFYUI_URLS. Compose подхватывает override автоматически.
#
#     bash gen-gpu-override.sh          # по тому, что видит nvidia-smi
#     bash gen-gpu-override.sh 0,1,2,3  # явный список (для примерки под облако)
#     docker compose up -d --remove-orphans
#
# Перезапускать после смены железа — и всё, правок в compose и коде не нужно.
#
# ---------------------------------------------------------------------------
# ФЛАГИ ЗАПУСКА ComfyUI — единственное место, где они заданы.
#
# Переопределяем CMD образа, чтобы менять флаги без пересборки: Dockerfile
# клонирует ComfyUI по --depth 1 с HEAD, и пересборка притащила бы другую
# версию вместо работающей 0.31.0.
#
# --disable-pinned-memory УБРАН. Он ставился против OOM-killer'а на длинных
# клипах H3, но наша нагрузка — 124 кадра (5 с), там до потолка RAM далеко.
# Ценой была скорость: без page-locked памяти async weight offloading не может
# делать асинхронный DMA, копии CPU<->GPU идут через промежуточный буфер драйвера
# и не перекрываются со счётом. Замер на медленной фазе: GPU 0%, шина ~0, одно
# ядро на 75-100%. Если OOM всё же вернётся — не возвращай флаг, а ограничь
# потолок пиннинга: --cache-ram 6 32 (по умолчанию порог inactive/pin = 100%
# системной RAM).
#
# --reserve-vram НЕ ставить. Проверено 2026-08-22: с --reserve-vram 14 (столько
# нужно картиночной модели с enable_model_cpu_offload, пик 12.9 ГБ) у ComfyUI
# остаётся ~10 ГБ на 20-гигабайтную H3, и фаза диффузии растягивается со 175 с
# до 450+ с при GPU в нуле и 127 Вт — модель стримится через шину непрерывно.
# Размен невыгоден: чинит редкий случай (картинка между видео) ценой частого
# (серия видео за 160 с).
#
# --disable-mmap НЕ ставить, хотя соблазн есть. Проверено 2026-08-22: тёплое
# видео ускоряется вдвое (159.7 -> 88.0 с), потому что без mmap ядро не вытесняет
# страницы весов прямо во время генерации. НО веса становятся анонимной памятью:
# ComfyUI держит ~45 ГБ, page cache вытесняется, и следующая КАРТИНКА грузится с
# диска — 338+ с вместо 14. То есть флаг возвращает исходную жалобу «картинка
# висит минутами». Плюс анонимную память ядро не может отдать под давлением —
# только в своп.
# ---------------------------------------------------------------------------
set -euo pipefail

# Флаги одни на все карты: разъехавшись, они проявились бы не ошибкой, а тем,
# что «одна карта почему-то медленнее». Поэтому единым списком и здесь.
COMFY_FLAGS='"--listen", "0.0.0.0", "--port", "8188", "--fp16-intermediates"'

cd "$(dirname "$0")"
OVERRIDE="docker-compose.override.yml"
BASE_PORT=8188

if [ $# -ge 1 ]; then
  IFS=',' read -ra GPUS <<< "$1"
else
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)
fi

if [ "${#GPUS[@]}" -eq 0 ]; then
  echo "Видеокарт не найдено. Проверь nvidia-smi или задай список явно:" >&2
  echo "    bash $0 0,1" >&2
  exit 1
fi

# Каталоги под вывод — заводим сами: если их создаст docker, они окажутся
# root:root, и ComfyUI внутри контейнера не сможет туда писать.
for gpu in "${GPUS[@]}"; do
  mkdir -p "comfyui/output-${gpu}"
done

urls=""
for gpu in "${GPUS[@]}"; do
  urls="${urls}${urls:+,}http://comfyui-${gpu}:8188"
done
devices="$(IFS=,; echo "${GPUS[*]}")"

{
  echo "# СГЕНЕРИРОВАНО gen-gpu-override.sh — руками не править."
  echo "# Здесь описаны ВСЕ ComfyUI: в docker-compose.yml их нет намеренно,"
  echo "# чтобы строка запуска не жила в двух местах и не разъехалась."
  echo "# Карты: ${devices}. Перегенерировать после смены железа."
  echo "services:"
  echo "  ai-api-stuff:"
  echo "    environment:"
  echo "      - GPU_DEVICES=${devices}"
  echo "      - COMFYUI_URLS=${urls}"

  for gpu in "${GPUS[@]}"; do
    cat <<YAML

  comfyui-${gpu}:
    build:
      context: ./comfyui
      dockerfile: Dockerfile
    container_name: comfyui-${gpu}
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['${gpu}']
              capabilities: [gpu, utility, compute]
    devices:
      - /dev/nvidia${gpu}:/dev/nvidia${gpu}
      - /dev/nvidiactl:/dev/nvidiactl
      - /dev/nvidia-uvm:/dev/nvidia-uvm
      - /dev/nvidia-uvm-tools:/dev/nvidia-uvm-tools
    environment:
      - NVIDIA_VISIBLE_DEVICES=${gpu}
      - NVIDIA_DRIVER_CAPABILITIES=all
    volumes:
      # каталог моделей общий: страницы mmap делятся между процессами, так что
      # второй экземпляр не стоит ни гигабайта лишнего кэша
      - ./comfyui/models:/opt/ComfyUI/models
      - ./comfyui/output-${gpu}:/opt/ComfyUI/output
    # Почему флаги именно такие и что уже пробовали — в шапке этого скрипта.
    command: ["python3", "main.py", ${COMFY_FLAGS}]
    ports:
      - $((BASE_PORT + gpu)):8188
    restart: unless-stopped
YAML
  done
} > "$OVERRIDE"

echo "Записан ${OVERRIDE}"
echo "  карты:        ${devices}"
echo "  ComfyUI:      ${urls}"
echo "  порты наружу: $(for g in "${GPUS[@]}"; do printf '%s ' "$((BASE_PORT + g))"; done)"
echo
echo "Дальше:  docker compose up -d --remove-orphans"
