#!/usr/bin/env bash
# Качает LoRA и чекпойнты с civitai в ./loras и ./checkpoints.
#
# Зачем отдельный скрипт: файлы весят сотни мегабайт и в гит не идут, поэтому
# на свежей машине их нужно дотянуть. Ключ берётся из .env (CIVIT_AI_API_KEY) —
# без него civitai отдаёт 401 даже на публичные файлы.
#
#     bash fetch-loras.sh
#
# СКРИПТ НЕ ПАДАЕТ, если что-то не скачалось: печатает предупреждение и идёт
# дальше. Сервис тоже переживает отсутствие файла — рисует без LoRA, о чём
# пишет в лог. Так забытый ключ ломает качество картинок, а не весь бот.
set -uo pipefail          # без -e: одна неудача не должна прерывать остальные

cd "$(dirname "$0")"
mkdir -p loras checkpoints

KEY=""
[ -f .env ] && KEY=$(grep -E '^CIVIT_AI_API_KEY=' .env 2>/dev/null | cut -d= -f2- | tr -d '\r"'"'"'')
if [ -z "$KEY" ]; then
  echo "ВНИМАНИЕ: CIVIT_AI_API_KEY не найден в .env." >&2
  echo "          Скачивание почти наверняка вернёт 401. Ключ берётся в профиле" >&2
  echo "          civitai -> API Keys. Продолжаю, вдруг файл публичный." >&2
fi

fails=0

# каталог | имя файла | id версии | id файла | зачем
FILES=(
  "loras|better-nipples.safetensors|2740476|2626892|чинит ареолы на коротких промптах; включена по умолчанию с силой 0.5"
)

for row in "${FILES[@]}"; do
  IFS='|' read -r dir name vid fid note <<< "$row"
  dest="${dir}/${name}"
  if [ -s "$dest" ]; then
    echo "уже есть: ${dest}"
    continue
  fi
  echo "качаю ${dest} — ${note}"
  code=$(curl -sL --max-time 900 \
    ${KEY:+-H "Authorization: Bearer ${KEY}"} \
    "https://civitai.com/api/download/models/${vid}?fileId=${fid}" \
    -o "${dest}.part" -w '%{http_code}') || code="сеть"

  # civitai на отказ отдаёт короткий JSON вместо весов — ловим по размеру
  size=$(stat -c %s "${dest}.part" 2>/dev/null || echo 0)
  if [ "$code" = "200" ] && [ "$size" -gt 1000000 ]; then
    mv "${dest}.part" "$dest"
    echo "  готово: $((size / 1048576)) МБ"
  else
    rm -f "${dest}.part"
    echo "  ВНИМАНИЕ: не скачалось (http=${code}, ${size} байт)." >&2
    echo "            Сервис поднимется и без неё — будет рисовать базовой" >&2
    echo "            моделью, предупредив в логе." >&2
    fails=$((fails + 1))
  fi
done

if [ "$fails" -gt 0 ]; then
  echo
  echo "Не скачалось файлов: ${fails}. Это не ошибка запуска — сервис работает." >&2
fi
exit 0
