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

from diffusers import ZImagePipeline, WanPipeline, AutoencoderKLWan, WanImageToVideoPipeline, UniPCMultistepScheduler, WanTransformer3DModel, BitsAndBytesConfig
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
    # Use bfloat16 for optimal performance on supported GPUs
    # low_cpu_mem_usage=True — дефолт diffusers при установленном accelerate;
    # стоявший здесь False заставлял сперва собрать пустую модель в RAM, а потом
    # залить в неё state dict, то есть держать ~20 ГБ лишних и выбивать page cache
    # (после чего веса H3 перечитывались с диска на 15 МБ/с). На инференс не
    # влияет — только на загрузку. Если Z-Image когда-то ломался без False,
    # это вылезет сразу на первой загрузке.
    pipe = _from_cache_first(
        ZImagePipeline,
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if IMAGE_CPU_OFFLOAD:
        # accelerate двигает модули по одному; в VRAM живёт самый большой из них
        # плюс активации — это и покажет "peak VRAM" в логе инференса.
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe, {}


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
    pipe, _ = _get_pipe(ProcessType.IMAGE_GENERATION, _build_image_pipe)

    t_inf = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    image = pipe(
        prompt=data.prompt,
        height=896,
        width=1152,
        num_inference_steps=9,  # This actually results in 8 DiT forwards
        guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
        generator=torch.Generator("cuda").manual_seed(
            random.randint(0, sys.maxsize)),
    ).images[0]

    # 8 шагов Z-Image Turbo на 3090 — единицы секунд. Десятки/сотни секунд при
    # нормальном peak VRAM = карту делят или троттлит; peak ~0 = инференс уехал
    # на CPU (VRAM занял ComfyUI, /free не сработал).
    peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
    print(f"[img_gen] inference: {time.time() - t_inf:.1f}s | peak VRAM {peak:.1f} GB "
          f"| cuda={torch.cuda.is_available()}", flush=True)
    return image  # PIL.Image — в base64/PNG превращает host-сторона


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

scheduler = Scheduler(
    VIDEO_TYPES,
    max_video_batch=MAX_VIDEO_BATCH,
    max_wait_secs=MAX_WAIT_SECS,
    max_videos_before_cheap=MAX_VIDEOS_BEFORE_CHEAP,
    max_user_batch=MAX_USER_BATCH,
    max_user_inflight=MAX_USER_INFLIGHT,
)

comfy = ComfyClient()   # HTTP-клиент ComfyUI (соединение только при первом запросе)
_comfy_templates = {}


def _comfy_template(name):
    if name not in _comfy_templates:
        _comfy_templates[name] = load_template(name)
    return _comfy_templates[name]


def _run_video_comfy(ptype, data):
    """Гонит видео через ComfyUI: подставляет параметры в воркфлоу → run → mp4 bytes."""
    kind = "t2v" if ptype == ProcessType.T2V else "i2v"
    # t2v приходит объектом Item, i2v — dict (там ещё байты картинки из формы)
    get = (lambda k: getattr(data, k)) if kind == "t2v" else data.get

    model = get("model") or DEFAULT_MODEL
    image_name = None
    if kind == "i2v":  # стартовую картинку подогнать под холст и загрузить в ComfyUI
        image = prepare_image(model, data["image"], get("width"), get("height"))
        image_name = comfy.upload_image(image)

    wf = build_video_workflow(
        model, kind, _comfy_template(template_name(model, kind)),
        prompt=get("prompt"), image_name=image_name,
        width=get("width"), height=get("height"), fps=get("fps"))
    return comfy.run(wf)


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


def _log_resources(tag):
    """Сколько было свободно на входе в задачу.

    Без этого медленный прогон неотличим от быстрого задним числом: тайминги
    показывают, ЧТО тормозило, а этот снимок — почему.
    """
    vram = comfy.vram_free_mb()
    m = _mem_snapshot()
    swap_used = (m.get("SwapTotal", 0) - m.get("SwapFree", 0)) or 0
    print(f"[res] {tag} | vram_free {vram if vram is not None else '?'} MB"
          f" | ram_avail {m.get('MemAvailable', '?')} MB"
          f" | cached {m.get('Cached', '?')} MB"
          f" | swap_used {swap_used} MB", flush=True)


def worker(results, lock, gpu):
    print("Worker started", flush=True)

    loaded = None    # ProcessType в diffusers-процессе (для recycle при смене модели)
    backend = None   # "comfy" | "diffusers" — кто последним держал VRAM

    while True:
        job = scheduler.next_job()
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
            comfy.free(wait_vram_mb=FREE_VRAM_TARGET_MB)
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
        _log_resources(f"before {type.value}")
        try:
            with lock:
                results[id] = {"status": Status.IN_PROGRESS}
                current.update({"id": id, "type": type, "backend": job_backend,
                                "user": job.get("user"), "started": time.time()})

            # --- видео через ComfyUI (host-сторона, без diffusers-процесса) ---
            if job_backend == "comfy":
                try:
                    res = _run_video_comfy(type, data)
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

            elif type == ProcessType.IMAGE_GENERATION:
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
            print(f"[worker] done  {type.value} {id} |"
                  f" total {time.time() - t_job:.1f}s", flush=True)
            _log_resources(f"after  {type.value}")


load_dotenv()
results = {}
lock = threading.Lock()
# Последняя взятая воркером задача — для /api/queue. Специально НЕ чистим по
# завершении: признак «ещё выполняется» — статус IN_PROGRESS в results, который
# воркер и так проставляет. Иначе пришлось бы оборачивать всё тело цикла в
# try/finally ради одного поля.
current = {}
gpu = None
worker_thread = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global gpu, worker_thread
    gpu = GpuRunner(gpu_worker)
    worker_thread = threading.Thread(target=worker, args=(results, lock, gpu),
                                     daemon=True)
    worker_thread.start()

    yield

    scheduler.stop()
    gpu.stop()


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
                 "height": height, "fps": fps, "model": model},
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
        run = dict(current) if current else None
        # current — последняя ВЗЯТАЯ задача; выполняется она, только пока воркер
        # не сменил её статус на DONE/ERROR (либо пока результат не забрали)
        if run and results.get(run["id"], {}).get("status") != Status.IN_PROGRESS:
            run = None
        awaiting_pickup = sum(1 for r in results.values()
                              if r["status"] in (Status.DONE, Status.ERROR))

    by_type = {}
    for j in snap["pending"]:
        by_type[j["type"].value] = by_type.get(j["type"].value, 0) + 1

    resident = snap["resident_vtype"]
    return {
        "running": {
            "id": run["id"],
            "type": run["type"].value,
            "backend": run["backend"],          # comfy | diffusers
            "user": run.get("user"),
            "elapsed": round(now - run["started"], 1),
        } if run else None,
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
            "resident_vtype": resident.value if resident else None,
            "subtype_streak": snap["subtype_streak"],
            "video_streak": snap["video_streak"],
            "next_id": snap["next_id"],
            "current_user": snap["current_user"],
            "user_streak": snap["user_streak"],
            "inflight": snap["inflight"],       # пользователь -> задач в работе
            "limits": {"max_video_batch": MAX_VIDEO_BATCH,
                       "max_wait_secs": MAX_WAIT_SECS,
                       "max_videos_before_cheap": MAX_VIDEOS_BEFORE_CHEAP,
                       "max_user_batch": MAX_USER_BATCH,
                       "max_user_inflight": MAX_USER_INFLIGHT},
        },
        # False при живой очереди = воркер умер, задачи не разгребаются
        "worker_alive": worker_thread is not None and worker_thread.is_alive(),
    }


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
