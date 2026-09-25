import torch
import sys
import queue
import random
import os
import threading
import multiprocessing
import uuid
import base64
import tempfile
import time
import gc
import traceback

from diffusers import (ZImagePipeline, ZImageTransformer2DModel, ChromaPipeline, PipelineQuantizationConfig,
                       QwenImageEditPlusPipeline,
                       WanPipeline, AutoencoderKLWan, WanImageToVideoPipeline,
                       UniPCMultistepScheduler, WanTransformer3DModel, BitsAndBytesConfig)
from diffusers.utils import export_to_video, load_image
from fastapi import FastAPI, HTTPException, UploadFile, Form
from fastapi.responses import Response
from io import BytesIO
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from enum import Enum
from typing import Literal

# whisperx / pyannote / pandas импортируются ЛЕНИВО внутри _run_transcription.
# Это тяжёлый и капризный аудио-стек (torchcodec/ffmpeg): грузим только когда
# реально нужен. Плюсы — spawn-потомок сборки fp8-кэша не тащит его в память,
# и поломка аудио-стека не роняет весь сервер на старте (страдает только
# транскрипция, видео работает). Цена — ~пара секунд на первой транскрипции.

import huggingface_hub
from huggingface_hub.utils import http_backoff

from scheduler import Scheduler
from gpu_runner import GpuRunner
import stats
from comfy_client import (ComfyClient, MODELS as VIDEO_MODELS, DEFAULT_MODEL,
                          build_video_workflow, load_template, prepare_image,
                          template_name)

# Магия: перехватываем вызовы к HF и перенаправляем старый аргумент в новый
original_hf_hub_download = huggingface_hub.hf_hub_download

def patched_hf_hub_download(*args, **kwargs):
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token')
    return original_hf_hub_download(*args, **kwargs)

huggingface_hub.hf_hub_download = patched_hf_hub_download


class ProcessType(Enum):
    IMAGE_GENERATION = "img_gen"
    IMAGE_EDIT = "img_edit"
    TRANSCRIPTION = "trans"
    T2V = 't2v'
    I2V = 'i2v'


class Status(Enum):
    IN_PROGRESS = "in_progress"
    ERROR = "error"
    DONE = "done"
    PENDING = "pending"


class Item(BaseModel):
    prompt: str
    # Владелец задачи: по нему считается потолок и строится круг обслуживания.
    # Не задан — задача попадает к общему анонимному пользователю.
    user: int | None = None
    width: int = 832
    height: int = 480
    # fps не задан → берётся дефолт модели (Wan 30, H3 24). У H3 24 fps нативные:
    # другое значение не ускоряет генерацию, а меняет скорость воспроизведения.
    fps: int | None = None
    model: Literal[tuple(VIDEO_MODELS)] = DEFAULT_MODEL
    # Сырой текст пользователя и выбранный ботом шаблон. На генерацию не
    # влияют совсем: нужны для статистики и корпуса тест-кейсов, потому что
    # сюда приезжает уже переведённый и обёрнутый в шаблон промпт, а по нему
    # не видно, что человек написал на самом деле.
    source_prompt: str | None = None
    style: str | None = None
    # Переопределения для картинок; не заданы — берётся дефолт модели из
    # IMAGE_MODELS. Нужны, чтобы подбирать параметры без пересборки образа.
    steps: int | None = None
    guidance: float | None = None
    lora_scale: float | None = None
    # Оставить включённой только эту LoRA по имени
    # (nipples | cosplay | mystic | zpenis).
    lora_only: str | None = None
    # Вес каждой LoRA по отдельности: {"cosplay": 0.3}. Переопределяет реестр
    # для названных, остальные берут своё. Нужно, чтобы подбирать баланс между
    # адаптерами без пересборки — общий множитель lora_scale двигает все разом.
    lora_weights: dict[str, float] | None = None
    # Негативный промпт. РАБОТАЕТ ТОЛЬКО при guidance > 0: у Z-Image
    # do_classifier_free_guidance == (guidance > 0), а дефолт для turbo — 0,
    # и тогда негатив молча игнорируется. Поднятие guidance удваивает счёт
    # (два прохода на шаг) и для дистиллированной turbo-модели рискованно.
    negative: str | None = None
    # Сид: не задан — случайный, но в лог он пишется всегда, поэтому удачную
    # картинку можно повторить задним числом.
    seed: int | None = None


NEG_PROMPT = (
    "яркие цвета, засветка, статичность, размытые детали, субтитры, "
    "низкое качество, деформированные конечности, сросшиеся пальцы"
)

# Видео-задачи разделяют один резидентный «видео-слот» (t2v/i2v по ~50 ГБ,
# вместе не влезают). Картинки/аудио грузятся транзиентно и слот не трогают.
VIDEO_TYPES = (ProcessType.T2V, ProcessType.I2V)

# Видео гоним через ComfyUI (fp8_scaled, качество лучше nf4, offload/VRAM разруливает
# сам ComfyUI через /free). False → видео на diffusers (путь USE_FP8 ниже, откат на nf4).
USE_COMFYUI = True

# Держать картиночную модель не целиком в VRAM, а в RAM с помодульной подкачкой.
# Зачем: резидентная Z-Image берёт пиком 23.2 ГБ из 23.5, поэтому перед каждой
# картинкой приходится звать comfy.free() — а он стоит следующему видео ~6 минут
# на перечитывание 40 ГБ весов H3 с диска. Если ужать картинку так, чтобы она
# помещалась рядом с ComfyUI (--reserve-vram), free() станет не нужен вовсе.
# Цена — инференс картинки замедлится (веса ездят по PCIe помодульно).
# Откат: False, и всё возвращается к pipe.to("cuda").
# Замер 2026-08-22: с offload peak VRAM 12.9 ГБ (вместо 23.2), но инференс
# 26.2 с вместо 8.9, а картинка целиком 48.5 с вместо 14.1. Смысл появляется
# только вместе с --reserve-vram 14 у ComfyUI и отказом от comfy.free() —
# и тогда надо проверять, во что это обойдётся серии видео (сейчас 160 с).
IMAGE_CPU_OFFLOAD = False

# Реестр картиночных моделей: всё, что отличает одну от другой, — здесь, чтобы
# смена была конфигом, а не правкой кода. Выбор — IMAGE_MODEL в окружении.
#
# Z-Image (Tongyi/Alibaba) отказывается рисовать NSFW: safety-тренировка зашита
# в веса, промптом не обходится. Chroma — 8.9B на базе FLUX.1-schnell,
# переученная на 5 млн изображений без фильтра ("has not been aligned with a
# specific safety filter" в карточке), Apache 2.0.
#
# Берём вариант Flash: у lodestones это отдельный репозиторий с УЖЕ вмерженными
# весами в diffusers-формате, поэтому LoRA накатывать не нужно. Он считает за
# 8 шагов вместо 40 у базовой Chroma — на 3090 это ~13 с, то есть примерно
# нынешняя скорость Z-Image. Автор требует heun и CFG=1: с дефолтным euler и
# высоким CFG низкошаговые веса разносит.
#
# CFG=1 означает, что classifier-free guidance выключен (diffusers включает его
# при guidance > 1), поэтому негативный промпт Chroma не увидит — и один проход
# вместо двух, отсюда и скорость.
# ``quantize`` — какие компоненты ужать, чтобы модель влезла в VRAM целиком.
#
# Chroma в bf16 не помещается: трансформер 16.6 ГБ + T5-XXL 8.9 ГБ + VAE 0.2 =
# 25.7 ГБ против 23.5 доступных. С cpu_offload она работает, но плохо: модель
# живёт в RAM, и первый шаг каждой генерации тратит ~15 с на перекачку 16.6 ГБ
# трансформера по шине (замер 12.09.2026: инференс 37-47 с против 9 у Z-Image).
# Хуже того, каждый картиночный процесс держит тогда ~21 ГБ RSS, и две карты
# разом выбирают 60 ГБ машины — earlyoom убивает воркер.
#
# Поэтому квантуем ИЗБИРАТЕЛЬНО. T5 отрабатывает ОДИН раз за картинку — его
# размер важен, скорость нет; int8 ужимает его вдвое, до ~4.5 ГБ. Трансформер
# крутится 8 раз — его оставляем в bf16, потому что bitsandbytes-int8 в счёте
# МЕДЛЕННЕЕ bf16 (деквантизация на каждом проходе). Итого 16.6 + 4.5 + 0.2 =
# 21.3 ГБ, ~2 ГБ остаётся на активации, offload не нужен.
#
# int8 выбран ещё и потому, что на Ampere он нативный, в отличие от fp8/nvfp4 —
# те у наших карт идут эмуляцией (видно в логе ComfyUI на старте).
#
# ``cpu_offload`` остаётся как запасной путь: если квантованная модель всё же
# не влезет, ставим True и получаем медленно, но работающе.
IMAGE_MODELS = {
    "z_image": {
        "pipe_cls": ZImagePipeline,
        "model_id": "Tongyi-MAI/Z-Image-Turbo",
        "steps": 9,          # даёт 8 проходов DiT
        "guidance": 0.0,     # для turbo-моделей guidance должен быть 0
        "scheduler": None,
        "cpu_offload": False,
        "quantize": None,
        "quant_backend": None,
        "quant_kwargs": None,
        # Стек LoRA: каждая со своим весом, применяются одновременно.
        # Репозиторий смонтирован в /root, файлы лежат в ./loras и в гит не идут.
        # Пустой список — работаем на голой модели.
        #
        # Z-Image НЕ отказывает на анатомии (проверено 13.09.2026 с выключенными
        # LoRA) — эти нужны не чтобы «разблокировать», а чтобы она рисовала тела
        # достовернее: база даёт неправдоподобные ареолы и совсем плохо
        # справляется с мужской анатомией.
        #
        # Порядок и веса подобраны от общего к частному: сперва общий
        # NSFW-реализм, поверх — точечные правки. Косплейная
        # zimage_cos-NSFW-lora осталась в ./loras, но не подключена: она про
        # костюмы, а не про анатомию.
        # better-nipples на 0.5 — постоянно включена.
        #
        # Зачем: на КОРОТКИХ промптах вроде «boobs» описания нет, модель
        # достраивает по своему приору, и он даёт огромные тёмные ареолы и
        # пластиковую кожу. Ни позитивные уточнения, ни негатив с cfg этого не
        # чинят (проверено на фиксированном сиде 13.09.2026) — а адаптер чинит.
        #
        # Почему именно 0.5 и почему всегда: на развёрнутых промптах эта сила
        # практически не отличима от базы, на 1.0 уже видно (кожа глаже). На
        # посторонних сюжетах — одетый портрет, пейзаж, предмет — при 0.5 ущерба
        # нет, проверено попарно на одном сиде. То есть включать по условию
        # (короткий промпт / длинный) смысла нет: 0.5 полезна там, где нужна, и
        # безвредна там, где нет.
        #
        # Раньше здесь стоял стек из трёх адаптеров суммарным весом 1.7 — он
        # ломал анатомию и уводил кожу в пластик. mystic-xxx-v7 и zpenis-v2
        # лежат в ./loras, но по отдельности не проверены; чтобы испытать,
        # дописать сюда запись и гонять через lora_only.
        "lora_default_mult": 1.0,
        # Выбор адаптера по словам промпта УБРАН: подстроки ненадёжны — промпт
        # приходит и на русском, и без маркеров, и со словом «his» про женщину.
        # Включены две: better-nipples и косплейная. Друг другу не мешают (в
        # отличие от zpenis, который тянул nipples на себя), и вторая не стоит
        # почти ничего: пик VRAM тот же 24.2 ГБ, кадр 11.8 с против 11.7 с.
        # Вес косплейной подобран перебором 0.3/0.5/1.0 на двух сидах: 0.5 —
        # максимум, где костюм уже читается, а тело ещё целое; на 1.0 стиль
        # продавливает анатомию. Остальные лежат с нулевым весом и доступны для
        # проверки через lora_only или lora_weights.
        "lora": [
            {"path": "/root/loras/better-nipples.safetensors", "scale": 0.5,
             "name": "nipples"},
            {"path": "/root/loras/zimage_cos-NSFW-lora.safetensors", "scale": 0.5,
             "name": "cosplay"},
            {"path": "/root/loras/mystic-xxx-v7.safetensors", "scale": 0.0,
             "name": "mystic"},
            {"path": "/root/loras/zpenis-v2.safetensors", "scale": 0.0,
             "name": "zpenis"},
        ],
    },
    # Файнтюн Z-Image с civitai. Чекпойнт содержит ТОЛЬКО трансформер (453
    # тензора с префиксом model.diffusion_model), поэтому VAE, текст-энкодер и
    # токенизатор берём из базового Tongyi-MAI/Z-Image-Turbo — он уже в кэше.
    # Смысл захода: правки анатомии вплавлены в веса, значит не нужны ни стек
    # LoRA, ни выбор адаптера по промпту.
    "pornmaster": {
        "pipe_cls": ZImagePipeline,
        "model_id": "Tongyi-MAI/Z-Image-Turbo",
        "transformer_file": "/root/checkpoints/pornmaster-v35-bf16.safetensors",
        "steps": 9,
        "guidance": 0.0,
        "scheduler": None,
        "cpu_offload": False,
        "quantize": None,
        "quant_backend": None,
        "quant_kwargs": None,
        "lora": [],
        "lora_default_mult": 0.0,
    },
    "chroma_flash": {
        "pipe_cls": ChromaPipeline,
        "model_id": "lodestones/Chroma1-Flash",
        # 20, а не 8. Автор обещает 8 шагов, но с решателем heun — тот второго
        # порядка и делает два прохода на шаг. В diffusers heun недоступен:
        # ChromaPipeline всегда сам строит сигмы, а FlowMatchHeunDiscreteScheduler
        # кастомные сигмы не принимает. Со штатным euler (первый порядок) восьми
        # шагов не хватает: ODE недоинтегрирована, картинка выходит вымытой и
        # «живописной», вплоть до поддельной подписи художника в углу.
        # Замер 12.09.2026 на одном промпте: 8 шагов — мыло, 20 — честное фото
        # (31 с), 35 — чуть лучше (53 с). 20 взято как компромисс; клиент может
        # переопределить полем "steps" в запросе, если нужно быстрее или лучше.
        "steps": 20,
        "guidance": 1.0,
        # Автор Chroma советует heun, но это термин ComfyUI: в diffusers
        # ChromaPipeline всегда сам строит сигмы и передаёт их планировщику, а
        # FlowMatchHeunDiscreteScheduler кастомные сигмы не принимает и падает.
        # Оставляем штатный FlowMatchEuler, с которым интеграция и писалась;
        # если 8 шагов дадут грязь — поднимать steps, а не менять планировщик.
        "scheduler": None,
        "cpu_offload": False,
        "quantize": ["text_encoder"],
        # 4 бита, а не 8: с int8 энкодер занимал 4.5 ГБ, модель целиком 23.1 из
        # 23.5 ГБ, и активациям не хватало ~2 ГБ (падало в apply_rotary_emb на
        # 54 МБ). nf4 ужимает T5 до ~2.3 ГБ и освобождает нужный запас.
        # Двойная квантизация (bnb_4bit_use_double_quant) снимает ещё немного,
        # compute в bf16 — считать всё равно в полной точности.
        "quant_backend": "bitsandbytes_4bit",
        "quant_kwargs": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": torch.bfloat16,
        },
        "lora": [],
        "lora_default_mult": 0.0,
    },
}
# Обратно на z_image: Chroma даёт анатомию, но её «живописный» приор перебить
# не удалось — на простых промптах она уходит в иллюстрацию, а фотографичность
# Z-Image недостижима. План: вернуть Z-Image и снять отказы через LoRA (см.
# "lora" в реестре). Chroma остаётся доступной через IMAGE_MODEL=chroma_flash.
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "z_image")

# ==========================================================================
# Правка изображений по инструкции (/api/edit)
# ==========================================================================
# Отдельный реестр, а не запись в IMAGE_MODELS: тут другой смысл промпта и
# другие параметры. Z-Image так НЕ умеет и не научится, пока не выложат веса
# Omni — у неё второй картинке физически некуда попасть, в трансформер идут
# голые латенты (проверено по исходнику 14.09.2026). Qwen-Image-Edit обучена
# на парах «было → стало» и принимает список картинок как контекст.
#
# Промпт здесь — ИНСТРУКЦИЯ («сделай волосы рыжими»), а не описание кадра
# целиком, как у txt2img. Это разные вещи, и путать их нельзя: описание кадра
# модель тоже выполнит, но хуже.
EDIT_MODELS = {
    "qwen_edit": {
        "pipe_cls": QwenImageEditPlusPipeline,
        # Репозиторий уже квантован в nf4 — качать 15.8 ГиБ вместо 53.7 и не
        # жать на загрузке (а на этой машине жать 20B в RAM ещё и рискованно).
        "model_id": "ovedrive/Qwen-Image-Edit-2511-4bit",
        # 4 шага, а не штатные 20: Lightning — дистилляция, воспроизводит
        # длинную траекторию за несколько шагов. Замерено 14.09.2026 на четырёх
        # задачах: 13 с против 57 с, качество то же, а вязка свитера на четырёх
        # шагах даже рельефнее. Без LoRA поднимется на 20 (см. steps_no_lora).
        "steps": 4,
        "steps_no_lora": 20,
        # CFG выключен намеренно. Включается условием negative_prompt is not
        # None, то есть даже пустая строка его зажигает — и удваивает время.
        # Проверено: с негативом и без него разница 3–6 единиц из 255, то есть
        # уровень вариации, а не улучшения. 40 шагов вместо 20 тоже ничего не
        # дали. Единственный рычаг, который сработал, — Lightning.
        "negative_default": None,
        "true_cfg": 4.0,
        # Lightning тянется с HF при первом старте и кладётся в общий кэш.
        # Не нашлась — работаем на 20 шагах, см. _build_edit_pipe.
        "lora_repo": "lightx2v/Qwen-Image-Edit-2511-Lightning",
        "lora_file": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        # NSFW-адаптеров здесь нет и быть не должно: на вход идут фотографии
        # живых людей, которые загрузил пользователь. Реестр z_image со своими
        # LoRA сюда не подмешивается — это отдельная модель со своим списком.
    },
}
EDIT_MODEL = os.getenv("EDIT_MODEL", "qwen_edit")

# Сколько картинок принимаем за раз и до какой площади ужимаем каждую.
# Ограничения не косметические: на 1152×896 пик был 22.2 ГБ из 23.5 доступных,
# и снимок с телефона в исходном размере эту карту положит.
EDIT_MAX_IMAGES = 3
EDIT_MAX_PIXELS = 1152 * 896

# Квантизация трансформеров Wan через bitsandbytes (4-бит NF4): ~7 ГБ/эксперт, оба
# влезают в 24 ГБ. В отличие от torchao у bnb штатно работают save/load (быстрый
# холодный старт из кэша) и LoRA (QLoRA). Актуально только при USE_COMFYUI=False.
USE_FP8 = True

# Wan VAE в fp32 стабильнее (меньше артефактов), но декод медленнее и жрёт память.
# ComfyUI гоняет VAE в bf16. Ставим False ради скорости — верни True, если видео поплывёт.
VAE_FP32 = False

# Куда кэшировать квантованные веса. Первый раз: bf16 → квант → save_pretrained
# (медленно). Дальше КАЖДАЯ загрузка (в т.ч. после recycle) читает готовый nf4 отсюда
# — быстро, без повторной квантизации. bnb save/load работает штатно (не как torchao).
QUANT_CACHE_DIR = os.getenv("QUANT_CACHE_DIR", os.path.expanduser("~/.cache/wan_nf4"))


# ==========================================================================
#  GPU-СТОРОНА: исполняется в одном долгоживущем процессе.
#  - видео-модель (t2v ИЛИ i2v) кэшируется резидентно в _video_slot;
#  - картинки и транскрипция грузятся на время запроса и освобождаются,
#    НЕ вытесняя видео-модель.
# ==========================================================================

_slot = {"type": None, "pipe": None, "meta": None}


def _get_pipe(ptype, builder):
    """Кэш модели в рамках одного процесса. ВНУТРИ процесса модель не выгружаем:
    VRAM надёжно освобождается только смертью процесса, поэтому при смене модели
    host пересоздаёт процесс (gpu.recycle()). Сюда попадаем на первой задаче
    свежего процесса, дальше однотипные запросы переиспользуют тёплую модель."""
    if _slot["type"] != ptype:
        t0 = time.time()
        pipe, meta = builder()
        _slot["type"] = ptype
        _slot["pipe"] = pipe
        _slot["meta"] = meta
        print(f"[{ptype.value}] load: {time.time() - t0:.1f}s", flush=True)
    return _slot["pipe"], _slot["meta"]


def _vae_dtype():
    return torch.float32 if VAE_FP32 else torch.bfloat16


# Реестр видео-моделей (t2v/i2v в одном месте, без дублей в билдерах).
WAN_MODELS = {
    "t2v": {
        "pipe_cls": WanPipeline,
        "model_id": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "lora_high": "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors",
        "lora_low": "Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors",
    },
    # ВНИМАНИЕ: проверь точные имена I2V-весов в репо lightx2v/Wan2.2-Lightning
    "i2v": {
        "pipe_cls": WanImageToVideoPipeline,
        "model_id": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "lora_high": "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors",
        "lora_low": "Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors",
    },
}

WAN_META = {"num_steps": 4, "guidance": 1.0, "guidance_2": 1.0}


def _vae_of(model_id):
    return AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=_vae_dtype())


def _quant_config():
    # bitsandbytes 4-бит NF4: ~7 ГБ/эксперт (оба = 14 ГБ влезают в 24 ГБ — нет OOM
    # при квантизации/загрузке, оффлоадится лучше 8-бита). Компьют в bf16.
    # 8-бит не подошёл: ~14 ГБ/эксперт, два не влезают, а bnb квантует на GPU.
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _quantize_transformer(model_id, subfolder, cache_dir):
    """int8-квантованный трансформер с дисковым кэшом (bnb save/load работает штатно).

    Кэш есть → from_pretrained(cache) грузит готовый int8 (конфиг квантизации лежит
    в его config.json, метаданные на месте → LoRA и offload работают). Нет → квантуем
    bnb на лету и save_pretrained. LoRA вешаем поверх в _build_wan_fp8.
    """
    if os.path.isdir(cache_dir):
        print(f"[quant] load cached {os.path.basename(cache_dir)}", flush=True)
        return WanTransformer3DModel.from_pretrained(cache_dir, torch_dtype=torch.bfloat16)

    print(f"[quant] quantize+cache {os.path.basename(cache_dir)} (one-time)", flush=True)
    t = WanTransformer3DModel.from_pretrained(
        model_id, subfolder=subfolder,
        quantization_config=_quant_config(), torch_dtype=torch.bfloat16)
    t.save_pretrained(cache_dir)
    return t


def _build_wan_fp8(kind):
    """int8-путь (bnb): квантованные трансформеры (из кэша или на лету) + LoRA поверх
    + model_cpu_offload. bnb хранит метаданные квантизации в чекпойнте, поэтому и
    LoRA (QLoRA), и offload на GPU работают."""
    m = WAN_MODELS[kind]
    high_dir = os.path.join(QUANT_CACHE_DIR, f"wan_{kind}_transformer")
    low_dir = os.path.join(QUANT_CACHE_DIR, f"wan_{kind}_transformer_2")
    transformer = _quantize_transformer(m["model_id"], "transformer", high_dir)
    transformer_2 = _quantize_transformer(m["model_id"], "transformer_2", low_dir)

    pipe = m["pipe_cls"].from_pretrained(
        m["model_id"], transformer=transformer, transformer_2=transformer_2,
        vae=_vae_of(m["model_id"]), torch_dtype=torch.bfloat16)

    # LoRA-ускоритель поверх квантованной базы (как LoraLoaderModelOnly у ComfyUI)
    pipe.load_lora_weights("lightx2v/Wan2.2-Lightning", weight_name=m["lora_high"], adapter_name="high")
    pipe.load_lora_weights("lightx2v/Wan2.2-Lightning", weight_name=m["lora_low"], adapter_name="low")
    pipe.set_adapters(["high"])
    pipe.transformer_2.set_adapters(["low"])

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)
    pipe.enable_model_cpu_offload()
    return pipe


def _build_wan_bf16(kind):
    """Рабочий bf16-путь (USE_FP8=False): LoRA как адаптеры + sequential offload."""
    m = WAN_MODELS[kind]
    pipe = m["pipe_cls"].from_pretrained(m["model_id"], vae=_vae_of(m["model_id"]), torch_dtype=torch.bfloat16)
    pipe.load_lora_weights("lightx2v/Wan2.2-Lightning", weight_name=m["lora_high"], adapter_name="high")
    pipe.load_lora_weights("lightx2v/Wan2.2-Lightning", weight_name=m["lora_low"], adapter_name="low")
    pipe.set_adapters(["high"])
    pipe.transformer_2.set_adapters(["low"])
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)
    pipe.enable_sequential_cpu_offload()
    return pipe


def _build_wan(kind):
    return _build_wan_fp8(kind) if USE_FP8 else _build_wan_bf16(kind)


def _build_t2v_pipe():
    return _build_wan("t2v"), dict(WAN_META)


def _build_i2v_pipe():
    return _build_wan("i2v"), dict(WAN_META)


def _from_cache_first(cls, model_id, **kwargs):
    """``from_pretrained``, который сперва пробует строго локальный кэш.

    Без ``local_files_only`` huggingface_hub на КАЖДОЙ загрузке сверяет ревизии —
    по HEAD-запросу на файл. Когда сеть недоступна, каждый упирается в
    HF_HUB_ETAG_TIMEOUT, и загрузка давно скачанной модели растягивается на
    минуты: замеренный случай — Z-Image за 978 с вместо 5.5 с при пустой очереди.
    Промах кэша (первый запуск, новая модель) штатно уходит в сеть — поэтому
    именно так, а не через HF_HUB_OFFLINE, который скачивание запрещает вовсе.
    """
    try:
        return cls.from_pretrained(model_id, local_files_only=True, **kwargs)
    except Exception as e:
        # Широкий except намеренно: промах кэша прилетает то OSError, то
        # ValueError в зависимости от версии hub. Цена ошибки — обычная загрузка.
        print(f"[hf] {model_id} не поднялся из кэша ({type(e).__name__}), качаем",
              flush=True)
        return cls.from_pretrained(model_id, **kwargs)


def _build_image_pipe():
    """Поднимает картиночную модель, выбранную через IMAGE_MODEL."""
    spec = IMAGE_MODELS[IMAGE_MODEL]

    # Use bfloat16 for optimal performance on supported GPUs
    # low_cpu_mem_usage=True — дефолт diffusers при установленном accelerate;
    # стоявший здесь False заставлял сперва собрать пустую модель в RAM, а потом
    # залить в неё state dict, то есть держать ~20 ГБ лишних и выбивать page cache
    # (после чего веса H3 перечитывались с диска на 15 МБ/с). На инференс не
    # влияет — только на загрузку.
    extra = {}
    if spec["quantize"]:
        # Квантуем на загрузке, а не после: bitsandbytes подменяет слои в момент
        # материализации весов, постфактум модель уже не ужать.
        extra["quantization_config"] = PipelineQuantizationConfig(
            quant_backend=spec["quant_backend"],
            quant_kwargs=spec["quant_kwargs"],
            components_to_quantize=spec["quantize"],
        )

    tf_file = spec.get("transformer_file")
    if tf_file:
        # Трансформер из одиночного файла, остальное — из базового репозитория.
        # Не упали, а предупредили и взяли базовый: чекпойнты лежат вне гита, и
        # на свежей машине их может не быть. Лучше рисовать базой, чем не
        # рисовать вообще.
        if not os.path.exists(tf_file):
            print(f"[img_gen] чекпойнт не найден: {tf_file} — "
                  f"работаем на базовой модели", flush=True)
        else:
            try:
                extra["transformer"] = ZImageTransformer2DModel.from_single_file(
                    tf_file, torch_dtype=torch.bfloat16)
            except Exception as e:
                print(f"[img_gen] чекпойнт не загрузился ({type(e).__name__}: "
                      f"{str(e)[:160]}) — работаем на базовой модели", flush=True)

    pipe = _from_cache_first(
        spec["pipe_cls"],
        spec["model_id"],
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        **extra,
    )

    # Низкошаговым весам нужен свой планировщик: Chroma-Flash обучена под heun,
    # на дефолтном euler при 8 шагах картинку разносит.
    if spec["scheduler"] is not None:
        pipe.scheduler = spec["scheduler"].from_config(pipe.scheduler.config)

    # Грузим адаптерами, а не fuse_lora: так веса можно менять на лету
    # (lora_scale в запросе), не перезагружая модель. Каждая LoRA — отдельный
    # адаптер со своим именем, diffusers складывает их поправки.
    #
    # Каждая под своим try: файлы лежат вне репозитория и приходят с civitai в
    # формате kohya, который конвертер понимает не всегда. Отвалившаяся LoRA не
    # должна ронять генерацию — просто работаем без неё.
    loaded_loras = []
    for item in (spec.get("lora") or []):
        if not os.path.exists(item["path"]):
            print(f"[img_gen] LoRA не найдена: {item['path']} — пропускаю",
                  flush=True)
            continue
        try:
            pipe.load_lora_weights(item["path"], adapter_name=item["name"])
            loaded_loras.append(item)
        except Exception as e:
            print(f"[img_gen] LoRA {item['name']} не загрузилась "
                  f"({type(e).__name__}: {str(e)[:160]}) — пропускаю", flush=True)

    if loaded_loras:
        pipe.set_adapters([i["name"] for i in loaded_loras],
                          [i["scale"] for i in loaded_loras])
        print("[img_gen] LoRA: " + ", ".join(
            f"{i['name']}={i['scale']}" for i in loaded_loras), flush=True)

    if spec["cpu_offload"] or IMAGE_CPU_OFFLOAD:
        # accelerate двигает модули по одному; в VRAM живёт самый большой из них
        # плюс активации — это и покажет "peak VRAM" в логе инференса. Для Chroma
        # раскладка удачная: T5 отрабатывает один раз за генерацию и уезжает,
        # дальше 8 шагов крутится только трансформер.
        pipe.enable_model_cpu_offload()
    else:
        # Квантованные компоненты bitsandbytes размещает сам на загрузке и
        # переносить их запрещает; to() это знает и трогает только остальные.
        pipe.to("cuda")

    print(f"[img_gen] модель {IMAGE_MODEL} ({spec['model_id']}), "
          f"{spec['steps']} шагов, cfg {spec['guidance']}, "
          f"offload={spec['cpu_offload'] or IMAGE_CPU_OFFLOAD}, "
          f"quant={spec['quant_backend'] or 'нет'} {spec['quantize'] or ''}", flush=True)
    return pipe, spec


def _build_edit_pipe():
    """Поднимает модель правки по инструкции (EDIT_MODEL)."""
    spec = EDIT_MODELS[EDIT_MODEL]

    # Веса в репозитории уже в nf4, PipelineQuantizationConfig не нужен:
    # bitsandbytes читает свой quantization_config из конфигов компонентов.
    pipe = _from_cache_first(
        spec["pipe_cls"],
        spec["model_id"],
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # Lightning. Отдельный try по той же причине, что и у картиночных LoRA:
    # адаптер живёт вне репозитория, и его отсутствие не должно ронять правку —
    # без него просто считаем 20 шагов вместо четырёх.
    steps = spec["steps_no_lora"]
    if spec.get("lora_repo"):
        try:
            path = huggingface_hub.hf_hub_download(spec["lora_repo"], spec["lora_file"])
            pipe.load_lora_weights(path, adapter_name="lightning")
            pipe.set_adapters(["lightning"], [1.0])
            steps = spec["steps"]
            print(f"[img_edit] Lightning подключён, {steps} шагов", flush=True)
        except Exception as e:
            print(f"[img_edit] Lightning не загрузился ({type(e).__name__}: "
                  f"{str(e)[:160]}) — работаем на {steps} шагах", flush=True)

    # Квантованные компоненты bitsandbytes размещает сам; to() трогает остальные.
    pipe.to("cuda")

    spec = dict(spec, steps=steps)
    print(f"[img_edit] модель {EDIT_MODEL} ({spec['model_id']}), {steps} шагов, "
          f"cfg {'выкл' if spec['negative_default'] is None else spec['true_cfg']}",
          flush=True)
    return pipe, spec


def _video_to_bytes(video, fps):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        export_to_video(video, tmp_path, fps=fps)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


def _free_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_t2v(data):
    pipe, meta = _get_pipe(ProcessType.T2V, _build_t2v_pipe)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_inf = time.time()
    video = pipe(
        prompt=data.prompt,
        negative_prompt=NEG_PROMPT,
        height=data.height,
        width=data.width,
        num_frames=81,           # 4*k+1 кадров, тут k=20
        guidance_scale=meta["guidance"],
        guidance_scale_2=meta["guidance_2"],   # отдельный guidance для low-noise эксперта
        num_inference_steps=meta["num_steps"],
    ).frames[0]
    # peak VRAM ~0 → инференс идёт на CPU; ~14 ГБ → на GPU (диагностика fp8+offload)
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(f"[t2v] inference: {time.time() - t_inf:.1f}s | peak VRAM {peak:.1f} GB | cuda={torch.cuda.is_available()}", flush=True)

    return _video_to_bytes(video, data.fps or VIDEO_MODELS["wan"]["fps"])


def _run_i2v(data):
    from PIL import Image

    pipe, meta = _get_pipe(ProcessType.I2V, _build_i2v_pipe)

    image = Image.open(BytesIO(data["image"])).convert("RGB")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_inf = time.time()
    video = pipe(
        image=image,
        prompt=data["prompt"],
        negative_prompt=NEG_PROMPT,
        height=data["height"],
        width=data["width"],
        num_frames=81,
        guidance_scale=meta["guidance"],
        guidance_scale_2=meta["guidance_2"],
        num_inference_steps=meta["num_steps"],
    ).frames[0]
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(f"[i2v] inference: {time.time() - t_inf:.1f}s | peak VRAM {peak:.1f} GB | cuda={torch.cuda.is_available()}", flush=True)

    return _video_to_bytes(video, data.get("fps") or VIDEO_MODELS["wan"]["fps"])


def _run_image(data):
    pipe, spec = _get_pipe(ProcessType.IMAGE_GENERATION, _build_image_pipe)

    # Сила LoRA на лету: подбирать её приходится на глаз, и пересобирать образ
    # ради каждой пробы бессмысленно.
    #
    # Выставляем ВСЕГДА, а не только когда запрос её задал: пайплайн кэшируется
    # между задачами, и заданная однажды сила иначе залипает на всех следующих
    # запросах, включая чужие. Не задана — возвращаем дефолт из реестра.
    # lora_scale в запросе — общий МНОЖИТЕЛЬ к весам из реестра, а не замена их
    # одним числом: иначе потерялся бы подобранный баланс между адаптерами.
    # 0 выключает стек целиком, 1 (по умолчанию) — веса как заданы.
    #
    # Выставляем ВСЕГДА, а не только когда запрос попросил: пайплайн кэшируется
    # между задачами, и заданный однажды множитель иначе залипал бы на всех
    # следующих запросах, включая чужие.
    lora_mult = None
    applied = ""
    active = spec.get("lora") or []
    if active and getattr(pipe, "get_active_adapters", None) and pipe.get_active_adapters():
        lora_mult = getattr(data, "lora_scale", None)
        if lora_mult is None:
            lora_mult = spec.get("lora_default_mult", 0.0)

        # lora_only в запросе — оставить включённой ровно одну LoRA по имени.
        # Нужно, чтобы сравнивать их поодиночке: втроём они конфликтуют, и по
        # общей картинке не понять, какая именно портит.
        only = getattr(data, "lora_only", None)
        by_name = {i["name"]: i["scale"] for i in active}
        names = pipe.get_active_adapters()
        weights = []
        for n in names:
            if only is not None:
                # В режиме проверки вес берём ЦЕЛИКОМ из запроса, минуя реестр:
                # иначе адаптер с нулём в реестре (испытуемый) так и остался бы
                # выключенным, что уже однажды дало пустой прогон.
                w = lora_mult if n == only else 0.0
            else:
                per = (getattr(data, "lora_weights", None) or {})
                base = per[n] if n in per else by_name.get(n, 1.0)
                w = base * lora_mult
            weights.append(w)
        applied = ",".join(n for n, w in zip(names, weights) if w)
        try:
            pipe.set_adapters(names, weights)
        except Exception as e:
            print(f"[img_gen] set_adapters(x{lora_mult}) не удался: {e}", flush=True)
            lora_mult = None

    t_inf = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    seed = getattr(data, "seed", None)
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    image = pipe(
        prompt=data.prompt,
        height=896,
        width=1152,
        negative_prompt=getattr(data, "negative", None),
        num_inference_steps=getattr(data, "steps", None) or spec["steps"],
        guidance_scale=(spec["guidance"] if getattr(data, "guidance", None) is None
                        else data.guidance),
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]

    # 8 шагов Z-Image Turbo на 3090 — единицы секунд. Десятки/сотни секунд при
    # нормальном peak VRAM = карту делят или троттлит; peak ~0 = инференс уехал
    # на CPU (VRAM занял ComfyUI, /free не сработал).
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    steps = getattr(data, "steps", None) or spec["steps"]
    # lora в строке — чтобы по логу было видно, с какой силой считалась КАЖДАЯ
    # картинка: значение приходит из запроса и глазами в картинке неразличимо.
    only_note = f" only={data.lora_only}" if getattr(data, "lora_only", None) else ""
    lora_note = ""
    if lora_mult is not None:
        lora_note = f" | lora x{lora_mult}{only_note} [{applied or 'нет'}]"
    cfg = spec["guidance"] if getattr(data, "guidance", None) is None else data.guidance
    neg_note = f" | cfg {cfg} neg«{str(data.negative)[:24]}»" if getattr(data, "negative", None) else ""
    print(f"[img_gen] inference: {time.time() - t_inf:.1f}s | peak VRAM {peak:.1f} GB "
          f"| cuda={torch.cuda.is_available()} | {IMAGE_MODEL} | {steps} шагов"
          f"{lora_note}{neg_note} | seed {seed}", flush=True)
    return image  # PIL.Image — в base64/PNG превращает host-сторона


def _prep_edit_image(raw):
    """Байты с телефона -> PIL под размер, который карта переживёт.

    Ужимаем по ПЛОЩАДИ, сохраняя пропорции, и округляем стороны до кратности
    32: VAE ужимает в 8 раз и потом патчит по 2, некратное приходится
    подрезать, а подрезка съезжает на границе маски.
    """
    from PIL import Image

    im = Image.open(BytesIO(raw)).convert("RGB")
    w, h = im.size
    scale = (EDIT_MAX_PIXELS / (w * h)) ** 0.5
    if scale < 1:
        w, h = int(w * scale), int(h * scale)
    w, h = max(32, w // 32 * 32), max(32, h // 32 * 32)
    return im.resize((w, h), Image.LANCZOS) if (w, h) != im.size else im


def _run_image_edit(data):
    pipe, spec = _get_pipe(ProcessType.IMAGE_EDIT, _build_edit_pipe)

    images = [_prep_edit_image(b) for b in data["images"][:EDIT_MAX_IMAGES]]

    t_inf = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    seed = data.get("seed")
    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    out = pipe(
        image=images,
        prompt=data["prompt"],
        negative_prompt=spec["negative_default"],
        num_inference_steps=data.get("steps") or spec["steps"],
        true_cfg_scale=spec["true_cfg"],
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]

    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    sizes = ",".join(f"{i.width}x{i.height}" for i in images)
    print(f"[img_edit] inference: {time.time() - t_inf:.1f}s | peak VRAM {peak:.1f} GB "
          f"| {EDIT_MODEL} | {data.get('steps') or spec['steps']} шагов "
          f"| {len(images)} вход [{sizes}] | seed {seed}", flush=True)
    return out


def _run_transcription(data):
    # Транзиентно: whisperx грузит свои модели и освобождает после, видео-слот не трогаем.
    # Импорты ленивые — см. комментарий у секции импортов (тяжёлый аудио-стек,
    # грузим только здесь, где он реально нужен).
    import whisperx
    import pandas as pd
    from pyannote.audio import Pipeline

    audio_file = data["filename"]
    device = "cuda"

    # 0. Redefine torch.load (восстанавливаем в finally — процесс живёт долго)
    _original_torch_load = torch.load

    def _trusted_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _trusted_load

    try:
        # 1. Transcribe with original whisper (batched)
        model = whisperx.load_model("large-v3", device, compute_type="float16", vad_method="silero")

        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=16)

        language_code = result["language"]

        # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(
            language_code=language_code, device=device)
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        # 3. Assign speaker labels
        diarize_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=os.getenv("HF_API_KEY"),          # в 4.x параметр называется token
        ).to(torch.device(device))

        diarization = diarize_pipeline({
            "waveform": torch.from_numpy(audio[None, :]),
            "sample_rate": 16000,
        })

        annotation = diarization.speaker_diarization

        # конвертим Annotation → DataFrame в формате, который ждёт assign_word_speakers
        diarize_df = pd.DataFrame(
            [(t.start, t.end, spk) for t, _, spk in annotation.itertracks(yield_label=True)],
            columns=["start", "end", "speaker"],
        )
        print("SPEAKERS:", diarize_df["speaker"].unique())

        result = whisperx.assign_word_speakers(diarize_df, result)
        result["language"] = language_code

        return result
    finally:
        torch.load = _original_torch_load
        _free_cuda()


def gpu_worker(job_q, res_q):
    """Единственный процесс, владеющий GPU. Держит видео-модель резидентно."""
    print("GPU worker started", flush=True)

    while True:
        job = job_q.get()
        if job == "BREAK":
            break

        id = job["id"]
        ptype = job["type"]
        data = job["data"]

        try:
            if ptype == ProcessType.T2V:
                res = _run_t2v(data)
            elif ptype == ProcessType.I2V:
                res = _run_i2v(data)
            elif ptype == ProcessType.IMAGE_GENERATION:
                res = _run_image(data)
            elif ptype == ProcessType.IMAGE_EDIT:
                res = _run_image_edit(data)
            elif ptype == ProcessType.TRANSCRIPTION:
                res = _run_transcription(data)
            else:
                res = {"error": f"unknown type {ptype}"}
        except Exception as e:
            traceback.print_exc()
            res = {"error": str(e)}

        res_q.put((id, res))


# ==========================================================================
#  ПЛАНИРОВЩИК ОЧЕРЕДИ (host-сторона)
#  Логика вынесена в scheduler.py (без тяжёлых зависимостей, покрыта тестами).
#  По умолчанию FIFO, но пока видео-модель тёплая — добиваем задачи того же
#  видео-подтипа, чтобы не перегружать 50 ГБ.
# ==========================================================================

MAX_VIDEO_BATCH = 10         # макс. видео-задач одного подтипа подряд, если ждёт другой тип
MAX_WAIT_SECS = 900          # 15 мин: ждущую дольше задачу обслуживаем вне батчинга
MAX_VIDEOS_BEFORE_CHEAP = 3  # не больше N видео подряд, если ждут картинки/транскрипции
MAX_USER_BATCH = 2           # задач одного человека подряд, пока в очереди есть другие
MAX_USER_INFLIGHT = 5        # задач одного человека в работе; сверх этого — отказ

# Что говорим, когда у человека уже полно задач. Текст уходит в detail 429;
# бот показывает свою фразу, а этот нужен людям в логах и при отладке руками.
QUEUE_FULL_DETAIL = f"уже {MAX_USER_INFLIGHT} задач в работе, дождитесь их"

# Сколько VRAM должно освободиться после comfy.free(), прежде чем грузить свою
# модель. Z-Image берёт пиком 23.2 ГБ из 23.5 доступных, так что ждать «почти всё»
# бессмысленно — порог отделяет «ComfyUI отпустил» (замер: 23858 МБ) от «ещё
# держит» (замер: 1217 МБ).
FREE_VRAM_TARGET_MB = 20000

# Карты пула и адреса их ComfyUI — по одному экземпляру на карту, парами по
# порядку. GPU_DEVICES="0,1" + COMFYUI_URLS="http://comfyui-0:8188,http://comfyui-1:8188".
# Один COMFYUI_URL и одна карта — прежняя однокарточная конфигурация.
GPU_DEVICES = [int(d) for d in
               os.getenv("GPU_DEVICES", "0").replace(" ", "").split(",") if d]
COMFYUI_URLS = [u for u in
                os.getenv("COMFYUI_URLS", os.getenv("COMFYUI_URL", "")).split(",") if u]

# Сколько оперативной памяти просит одно видео.
#
# Цифра верна только вместе с --fast-disk (см. gen-gpu-override.sh). Без флага
# ComfyUI держит веса H3 в RAM целиком — трансформер 20 ГБ, текст-энкодер 15,
# VAE 5.5; замер: ram_avail падал с 55.9 до 15.2 ГБ, то есть 40.7 ГБ на ролик.
# На 60 ГБ это давало потолок в ОДНО видео при двух картах, то есть вторая
# карта простаивала, а ролики стояли в очереди по 4-10 минут.
#
# 25.09.2026 флаг вернули, и замер на этот раз честный: боевые 576x1024 на
# обеих ветках, page cache сброшен перед каждым прогоном.
#
#     без флага, два ролика по одному   363 с   (181 с на ролик)
#     с флагом,  один ролик             189 с
#     с флагом,  два одновременно       170 с   (155 и 169 с)
#
# Пропускная вдвое, а шаг семплера одинаковый на обеих картах разом — 15.7 с
# против тех же 15.7 у одиночного. Драки за диск нет, и вот почему: экземпляры
# читают ОДНИ И ТЕ ЖЕ файлы весов, а page cache общий по файлу, а не по
# процессу. Этим --fast-disk и отличается от приватной page-locked памяти,
# которой действительно нужно вдвое. Расход RAM на ролик — 6.7 ГБ вместо 40.7,
# минимум доступной за прогон 44 ГБ, своп не шевельнулся.
#
# Почему прошлый вывод (22.09.2026, «два ролика упёрлись в диск, 358-405 с»)
# был неверным — в шапке gen-gpu-override.sh.
#
# Снимете флаг — верните сюда 40, иначе планировщик разрешит два видео там,
# где памяти хватает на одно, и машина уйдёт в своп.
VIDEO_RAM_BUDGET_GB = 10


def _max_concurrent_video(devices):
    """Сколько видео тянет оперативка. Ограничение общее на пул.

    Упирается не в карты, а в RAM: на 78 ГБ помещается одно видео, на 256 —
    четыре. Считаем здесь, а не в планировщике: тот намеренно ничего не знает
    ни про /proc, ни про железо. ``MAX_CONCURRENT_VIDEO`` в окружении
    перекрывает расчёт, "0" снимает потолок совсем.
    """
    override = os.getenv("MAX_CONCURRENT_VIDEO")
    if override is not None:
        return int(override) or None

    total_gb = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    total_gb = int(line.split()[1]) / 1024 / 1024
                    break
    except OSError:      # не Linux — на потолок не претендуем
        return None

    fits = int(total_gb // VIDEO_RAM_BUDGET_GB)
    return max(1, min(fits, len(devices)))


scheduler = Scheduler(
    VIDEO_TYPES,
    devices=tuple(GPU_DEVICES),
    max_video_batch=MAX_VIDEO_BATCH,
    max_wait_secs=MAX_WAIT_SECS,
    max_videos_before_cheap=MAX_VIDEOS_BEFORE_CHEAP,
    max_user_batch=MAX_USER_BATCH,
    max_user_inflight=MAX_USER_INFLIGHT,
    max_concurrent_video=_max_concurrent_video(GPU_DEVICES),
)


class Card:
    """Одна карта пула: закреплённый за ней GPU-процесс и свой ComfyUI.

    Соседние карты про неё ничего не знают: у каждой свой diffusers-процесс
    (привязан через CUDA_VISIBLE_DEVICES) и свой экземпляр ComfyUI, поэтому
    ни ``free()``, ни ``recycle()`` соседа не задевают.
    """

    def __init__(self, index, comfy_url=None):
        self.index = index
        self.comfy = ComfyClient(comfy_url)
        self.gpu = None          # GpuRunner поднимается в lifespan
        self.thread = None

    def __repr__(self):
        return f"<Card {self.index} comfy={self.comfy.base}>"


cards = [Card(index, COMFYUI_URLS[i] if i < len(COMFYUI_URLS) else None)
         for i, index in enumerate(GPU_DEVICES)]

_comfy_templates = {}


def _comfy_template(name):
    if name not in _comfy_templates:
        _comfy_templates[name] = load_template(name)
    return _comfy_templates[name]


def _run_video_comfy(card, ptype, data):
    """Гонит видео через ComfyUI: подставляет параметры в воркфлоу → run → mp4 bytes."""
    kind = "t2v" if ptype == ProcessType.T2V else "i2v"
    # t2v приходит объектом Item, i2v — dict (там ещё байты картинки из формы)
    get = (lambda k: getattr(data, k)) if kind == "t2v" else data.get

    model = get("model") or DEFAULT_MODEL
    image_name = None
    if kind == "i2v":  # стартовую картинку подогнать под холст и загрузить в ComfyUI
        image = prepare_image(model, data["image"], get("width"), get("height"))
        image_name = card.comfy.upload_image(image)

    wf = build_video_workflow(
        model, kind, _comfy_template(template_name(model, kind)),
        prompt=get("prompt"), image_name=image_name,
        width=get("width"), height=get("height"), fps=get("fps"))
    return card.comfy.run(wf)


def _mem_snapshot():
    """Свободная RAM/своп с точностью до МБ (в контейнере /proc/meminfo — хостовый)."""
    want = {"MemAvailable", "Cached", "SwapTotal", "SwapFree"}
    out = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                if k in want:
                    out[k] = int(v.split()[0]) // 1024
    except OSError:
        pass
    return out


def _log_resources(card, tag):
    """Сколько было свободно на входе в задачу.

    Без этого медленный прогон неотличим от быстрого задним числом: тайминги
    показывают, ЧТО тормозило, а этот снимок — почему. VRAM спрашиваем у своего
    ComfyUI: у каждой карты она своя, общая цифра тут ничего не значила бы.
    """
    vram = card.comfy.vram_free_mb()
    m = _mem_snapshot()
    swap_used = (m.get("SwapTotal", 0) - m.get("SwapFree", 0)) or 0
    print(f"[res] gpu{card.index} {tag} | vram_free {vram if vram is not None else '?'} MB"
          f" | ram_avail {m.get('MemAvailable', '?')} MB"
          f" | cached {m.get('Cached', '?')} MB"
          f" | swap_used {swap_used} MB", flush=True)


def _stat_field(data, name):
    """Одно поле задачи независимо от того, Item это или dict.

    t2v и txt2img приходят объектом Item, остальные — словарём из формы.
    """
    if isinstance(data, dict):
        return data.get(name)
    return getattr(data, name, None)


def _b64_size(payload):
    """Размер исходных байт по длине base64, не раскодируя её.

    Раскодировать ради одного числа было бы жалко: у видео это десятки
    мегабайт на каждую задачу.
    """
    if not isinstance(payload, (bytes, bytearray)) or not payload:
        return None
    return len(payload) // 4 * 3 - payload[-2:].count(b"=")


def _record_job(job, card, started, finished):
    """Строка статистики о завершившейся задаче.

    Исход читаем из results. Там его может уже не быть: /api/result удаляет
    запись при первом же чтении, и дотошный клиент успевает забрать её раньше,
    чем мы сюда дойдём. Окно крохотное, но существует, поэтому отсутствие
    записи считаем успехом без размера, а не ошибкой.
    """
    id = job.get("id")
    data = job.get("data")
    with lock:
        outcome = results.get(id) or {}
    payload = outcome.get("data")
    failed = outcome.get("status") == Status.ERROR
    enqueued = job.get("ts", started)
    stats.record(
        job_id=id,
        user_id=job.get("user"),
        kind=job.get("type").value,
        model=_stat_field(data, "model"),
        width=_stat_field(data, "width"),
        height=_stat_field(data, "height"),
        fps=_stat_field(data, "fps"),
        source_prompt=_stat_field(data, "source_prompt"),
        prompt=_stat_field(data, "prompt"),
        style=_stat_field(data, "style"),
        enqueued_at=enqueued,
        started_at=started,
        finished_at=finished,
        waited_s=started - enqueued,
        duration_s=finished - started,
        card=card.index,
        status="error" if failed else "done",
        error_class=(str(payload)[:200] if failed else None),
        out_bytes=None if failed else _b64_size(payload),
    )


def worker(results, lock, card):
    """Поток обслуживания одной карты. Потоков столько же, сколько карт.

    Состояние ниже — про эту карту и только про неё: у соседней свои тёплая
    модель и свой бэкенд, и планировщик учитывает это отдельно.
    """
    gpu = card.gpu
    print(f"Worker started on gpu{card.index}", flush=True)

    loaded = None    # ProcessType в diffusers-процессе (для recycle при смене модели)
    backend = None   # "comfy" | "diffusers" — кто последним держал VRAM

    while True:
        job = scheduler.next_job(card.index)
        if job is None:  # остановка
            break

        id = job.get("id")
        type = job.get("type")
        data = job.get("data")

        job_backend = "comfy" if (type in VIDEO_TYPES and USE_COMFYUI) else "diffusers"

        # diffusers-откат собран только вокруг Wan; H3 живёт исключительно в ComfyUI.
        # Молча подменить модель нельзя — вернём ошибку, не трогая GPU.
        if type in VIDEO_TYPES and job_backend == "diffusers":
            requested = (data.model if type == ProcessType.T2V else data.get("model"))
            if (requested or DEFAULT_MODEL) != "wan":
                with lock:
                    results[id] = {"status": Status.ERROR, "data":
                                   f"модель '{requested}' работает только через ComfyUI, "
                                   f"а сейчас USE_COMFYUI=False"}
                # ранний выход мимо try/finally ниже — место в допуске
                # освобождаем здесь, иначе оно останется занятым навсегда
                scheduler.finish(job)
                now = time.time()
                _record_job(job, card, now, now)
                continue

        # На границе бэкендов освобождаем VRAM у того, кто её держал (одна карта):
        # comfy→diffusers — просим ComfyUI выгрузить (/free); diffusers→comfy —
        # убиваем diffusers-процесс (recycle), чтобы отдать VRAM ComfyUI.
        # backend=None — воркер только что стартовал и не знает, кто держит VRAM.
        # ComfyUI живёт в своём контейнере и переживает рестарт API с загруженной
        # моделью, поэтому перед ЛЮБОЙ первой diffusers-задачей просим его
        # освободиться. Иначе pipe.to("cuda") падает с CUDA OOM (проверено на
        # боксе: ComfyUI держал 18.8 ГБ, свободно оставалось 3 МБ).
        t_sw = time.time()
        if job_backend == "diffusers" and backend != "diffusers":
            card.comfy.free(wait_vram_mb=FREE_VRAM_TARGET_MB)
        elif backend == "diffusers" and job_backend == "comfy":
            gpu.recycle()
            loaded = None
        backend = job_backend
        switch = time.time() - t_sw

        # Тайминги на границе задач: сколько задача пролежала в очереди,
        # сколько стоило переключение бэкенда и сколько заняла целиком.
        # Расхождение total и inference из логов GPU-процесса = загрузка
        # модели заново (recycle/free) или ожидание чужой задачи на карте.
        t_job = time.time()
        waited = t_job - job.get("ts", t_job)
        print(f"[worker] start {type.value} {id} | waited {waited:.1f}s"
              f" | backend {job_backend} | switch {switch:.1f}s", flush=True)
        _log_resources(card, f"before {type.value}")
        try:
            with lock:
                results[id] = {"status": Status.IN_PROGRESS}
                current[card.index] = {
                    "id": id, "type": type, "backend": job_backend,
                    "user": job.get("user"), "started": time.time()}

            # --- видео через ComfyUI (host-сторона, без diffusers-процесса) ---
            if job_backend == "comfy":
                try:
                    res = _run_video_comfy(card, type, data)
                    with lock:
                        results[id] = {"status": Status.DONE, "data": base64.b64encode(res)}
                except Exception as e:
                    with lock:
                        results[id] = {"status": Status.ERROR, "data": str(e)}
                continue

            # --- diffusers-бэкенд: смена модели внутри процесса → жёсткий сброс VRAM ---
            if loaded is not None and type != loaded:
                t_rc = time.time()
                gpu.recycle()
                # после recycle модель грузится с нуля: следующий "[<type>] load:"
                # в логах GPU-процесса — цена этой смены, а не медленный инференс
                print(f"[worker] recycle {loaded.value}->{type.value}: "
                      f"{time.time() - t_rc:.1f}s", flush=True)
            loaded = type

            if type == ProcessType.TRANSCRIPTION:
                filename = data.get("filename")
                try:
                    res = gpu.submit_and_wait(job)
                    if isinstance(res, dict) and res.get("error"):
                        with lock:
                            results[id] = {"status": Status.ERROR, "data": res.get("error")}
                    else:
                        with lock:
                            results[id] = {"status": Status.DONE, "data": res}
                finally:
                    if filename and os.path.exists(filename):
                        os.unlink(filename)

            elif type in (ProcessType.IMAGE_GENERATION, ProcessType.IMAGE_EDIT):
                res = gpu.submit_and_wait(job)
                if isinstance(res, dict):  # {"error": ...}
                    with lock:
                        results[id] = {"status": Status.ERROR, "data": res.get("error")}
                else:
                    filtered_image = BytesIO()
                    res.save(filtered_image, "PNG")
                    filtered_image.seek(0)
                    with lock:
                        results[id] = {"status": Status.DONE,
                                       "data": base64.b64encode(filtered_image.read())}

            else:  # T2V / I2V на diffusers (USE_COMFYUI=False, откат на nf4)
                res = gpu.submit_and_wait(job)
                if isinstance(res, dict):  # {"error": ...}
                    with lock:
                        results[id] = {"status": Status.ERROR, "data": res.get("error")}
                else:
                    with lock:
                        results[id] = {"status": Status.DONE,
                                       "data": base64.b64encode(res)}
        finally:
            # задача досчитана (или упала) — освобождаем место под следующую
            # задачу этого человека
            scheduler.finish(job)
            finished = time.time()
            print(f"[worker] done  {type.value} {id} |"
                  f" total {finished - t_job:.1f}s", flush=True)
            _record_job(job, card, t_job, finished)
            _log_resources(card, f"after  {type.value}")


load_dotenv()
results = {}
lock = threading.Lock()
# Последняя взятая задача КАЖДОЙ карты — для /api/queue: карта -> задача.
# Специально НЕ чистим по завершении: признак «ещё выполняется» — статус
# IN_PROGRESS в results, который воркер и так проставляет. Иначе пришлось бы
# оборачивать всё тело цикла в try/finally ради одного поля.
current = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Карты: {cards} | потолок одновременных видео: "
          f"{scheduler.max_concurrent_video}", flush=True)

    stats.init()
    stats.start_exporter()

    for card in cards:
        card.gpu = GpuRunner(gpu_worker, device=card.index)
        card.thread = threading.Thread(target=worker, args=(results, lock, card),
                                       daemon=True, name=f"worker-gpu{card.index}")
        card.thread.start()

    yield

    scheduler.stop()
    for card in cards:
        card.gpu.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/api")
async def root():
    return {"status": "ok"}


def enqueue_or_reject(job):
    """Ставит задачу в очередь либо отвечает 429, если у человека их уже полно.

    Место в ``results`` занимается ДО постановки (иначе воркер успеет перевести
    задачу в IN_PROGRESS, а мы затрём это обратно в PENDING), поэтому при отказе
    его надо освободить — задачи-то не будет.
    """
    if scheduler.enqueue(job):
        return

    with lock:
        results.pop(job["id"], None)
    raise HTTPException(status_code=429, detail=QUEUE_FULL_DETAIL)


@app.post("/api/txt2img")
async def txt2img(item: Item):
    id = str(uuid.uuid4())
    print("img", id)
    with lock:
        results[id] = {"status": Status.PENDING}
    enqueue_or_reject({"id": id, "type": ProcessType.IMAGE_GENERATION,
                       "data": item, "user": item.user})

    return {"id": id}


@app.post("/api/edit")
async def edit(
    files: list[UploadFile],
    prompt: str = Form(...),
    steps: int | None = Form(None),
    seed: int | None = Form(None),
    user: int | None = Form(None),
):
    """Правка изображений по инструкции. Одна картинка — правка, несколько —
    микс: «возьми женщину со второго кадра и посади за верстак с первого».

    prompt — ИНСТРУКЦИЯ, а не описание желаемого кадра. Это ровно наоборот к
    /api/txt2img, и разница существенная: модель обучена на парах «было →
    стало».
    """
    if not files:
        raise HTTPException(status_code=400, detail="нужен хотя бы один файл")
    if len(files) > EDIT_MAX_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"максимум {EDIT_MAX_IMAGES} картинки, пришло {len(files)}")

    id = str(uuid.uuid4())
    print("edit", id, len(files), "файл(ов)")
    # Читаем здесь, а не в воркере: UploadFile живёт только внутри запроса, а до
    # GPU-процесса задача едет через pickle — туда должны уехать уже байты.
    images = [await f.read() for f in files]
    with lock:
        results[id] = {"status": Status.PENDING}
    enqueue_or_reject({
        "id": id,
        "type": ProcessType.IMAGE_EDIT,
        "user": user,
        "data": {"prompt": prompt, "images": images, "steps": steps, "seed": seed},
    })

    return {"id": id}


@app.post("/api/transcription")
async def transcription(file: UploadFile, user: int | None = Form(None)):
    if not os.path.exists("files"):
        os.mkdir("files")

    _, extension = os.path.splitext(file.filename)
    id = str(uuid.uuid4())
    print("trans", id)

    filename = f"files/{id}{extension}"
    with open(filename, "wb") as f:
        f.write(file.file.read())

    with lock:
        results[id] = {"status": Status.PENDING}
    try:
        enqueue_or_reject({"id": id, "type": ProcessType.TRANSCRIPTION,
                           "data": {"filename": filename}, "user": user})
    except HTTPException:
        # задача не встала — файл убираем за собой, чистить его больше некому
        if os.path.exists(filename):
            os.unlink(filename)
        raise

    return {"id": id}


@app.post("/api/t2v")
async def t2v(item: Item):
    id = str(uuid.uuid4())
    print("t2v", id, item.model)
    with lock:
        results[id] = {"status": Status.PENDING}
    enqueue_or_reject({"id": id, "type": ProcessType.T2V,
                       "data": item, "user": item.user})

    return {"id": id}


@app.post("/api/i2v")
async def i2v(
    file: UploadFile,
    prompt: str = Form(...),
    width: int = Form(832),
    height: int = Form(480),
    fps: int | None = Form(None),
    model: Literal[tuple(VIDEO_MODELS)] = Form(DEFAULT_MODEL),
    user: int | None = Form(None),
    # см. комментарий у Item.source_prompt
    source_prompt: str | None = Form(None),
    style: str | None = Form(None),
):
    id = str(uuid.uuid4())
    print("i2v", id, model)
    image = await file.read()
    with lock:
        results[id] = {"status": Status.PENDING}
    enqueue_or_reject({
        "id": id,
        "type": ProcessType.I2V,
        "user": user,
        "data": {"prompt": prompt, "image": image, "width": width,
                 "height": height, "fps": fps, "model": model,
                 "source_prompt": source_prompt, "style": style},
    })

    return {"id": id}


@app.get("/api/queue")
def get_queue():
    """Текущее состояние очереди: что считается, что ждёт и почему в таком порядке.

    Диагностический эндпойнт, задачи не трогает (в отличие от /api/result,
    который забирает и удаляет результат).
    """
    now = time.time()
    snap = scheduler.snapshot(now=now)

    with lock:
        # current[card] — последняя ВЗЯТАЯ карты задача; выполняется она, только
        # пока воркер не сменил её статус на DONE/ERROR (либо пока результат не
        # забрали)
        running = {}
        for card in cards:
            run = current.get(card.index)
            if run and results.get(run["id"], {}).get("status") == Status.IN_PROGRESS:
                running[card.index] = run

        awaiting_pickup = sum(1 for r in results.values()
                              if r["status"] in (Status.DONE, Status.ERROR))

    by_type = {}
    for j in snap["pending"]:
        by_type[j["type"].value] = by_type.get(j["type"].value, 0) + 1

    def _card_state(card):
        run = running.get(card.index)
        dev = snap["devices"].get(card.index, {})
        resident = dev.get("resident_vtype")

        return {
            "comfy": card.comfy.base,
            "alive": card.thread is not None and card.thread.is_alive(),
            "resident_vtype": resident.value if resident else None,
            "subtype_streak": dev.get("subtype_streak"),
            "video_streak": dev.get("video_streak"),
            "running": {
                "id": run["id"],
                "type": run["type"].value,
                "backend": run["backend"],      # comfy | diffusers
                "user": run.get("user"),
                "elapsed": round(now - run["started"], 1),
            } if run else None,
        }

    return {
        # Что считает каждая карта. Ключ — её номер в GPU_DEVICES.
        "devices": {card.index: _card_state(card) for card in cards},
        # Первая занятая карта — чтобы старые читатели снимка не сломались.
        "running": next((_card_state(c)["running"] for c in cards
                         if running.get(c.index)), None),
        # В порядке ОБСЛУЖИВАНИЯ, а не постановки: очередь не FIFO — порядок
        # задают круг по людям и батчинг. Бот по этому списку считает «ты N-й»,
        # так что хронология тут была бы враньём.
        "pending": [
            {"id": j["id"], "type": j["type"].value, "user": j["user"],
             "waiting": round(now - j["ts"], 1)}
            for j in snap["pending"]
        ],
        "counts": {
            "pending": len(snap["pending"]),
            "by_type": by_type,
            "awaiting_pickup": awaiting_pickup,  # готовые, за которыми не пришли
        },
        "scheduler": {
            "video_running": snap["video_running"],
            "max_concurrent_video": snap["max_concurrent_video"],
            "next_id": snap["next_id"],
            "current_user": snap["current_user"],
            "user_streak": snap["user_streak"],
            "inflight": snap["inflight"],       # пользователь -> задач в работе
            "limits": {"max_video_batch": MAX_VIDEO_BATCH,
                       "max_wait_secs": MAX_WAIT_SECS,
                       "max_videos_before_cheap": MAX_VIDEOS_BEFORE_CHEAP,
                       "max_user_batch": MAX_USER_BATCH,
                       "max_user_inflight": MAX_USER_INFLIGHT,
                       "max_concurrent_video": scheduler.max_concurrent_video},
        },
        # False при живой очереди = все воркеры умерли, задачи не разгребаются.
        # По картам живость видна в devices[N].alive.
        "worker_alive": any(c.thread is not None and c.thread.is_alive()
                            for c in cards),
    }


@app.get("/api/stats")
def get_stats(hours: int = 24):
    """Сводка по задачам за окно. Источник — локальная база, не Postgres:
    эндпоинт обязан отвечать, даже когда наблюдательная машина лежит.
    """
    return stats.summary(hours)


@app.get("/api/result")
def get_result(id: str):
    response = None
    with lock:
        response = results.get(id)
        if response == None:
            response = Response(status_code=404, content="")
        else:
            status = response.get("status")
            if status == Status.DONE or status == Status.ERROR:
                del results[id]

    return response
