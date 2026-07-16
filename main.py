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
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import Response
from io import BytesIO
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from enum import Enum

# whisperx / pyannote / pandas импортируются ЛЕНИВО внутри _run_transcription.
# Это тяжёлый и капризный аудио-стек (torchcodec/ffmpeg): грузим только когда
# реально нужен. Плюсы — spawn-потомок сборки fp8-кэша не тащит его в память,
# и поломка аудио-стека не роняет весь сервер на старте (страдает только
# транскрипция, видео работает). Цена — ~пара секунд на первой транскрипции.

import huggingface_hub
from huggingface_hub.utils import http_backoff

from scheduler import Scheduler
from gpu_runner import GpuRunner
from comfy_client import ComfyClient, build_t2v_workflow, build_i2v_workflow, load_template

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
    width: int = 832
    height: int = 480
    fps: int = 30


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


def _build_image_pipe():
    # Use bfloat16 for optimal performance on supported GPUs
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )
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

    return _video_to_bytes(video, data.fps)


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

    return _video_to_bytes(video, data["fps"])


def _run_image(data):
    pipe, _ = _get_pipe(ProcessType.IMAGE_GENERATION, _build_image_pipe)

    image = pipe(
        prompt=data.prompt,
        height=896,
        width=1152,
        num_inference_steps=9,  # This actually results in 8 DiT forwards
        guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
        generator=torch.Generator("cuda").manual_seed(
            random.randint(0, sys.maxsize)),
    ).images[0]
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

scheduler = Scheduler(
    VIDEO_TYPES,
    max_video_batch=MAX_VIDEO_BATCH,
    max_wait_secs=MAX_WAIT_SECS,
    max_videos_before_cheap=MAX_VIDEOS_BEFORE_CHEAP,
)

comfy = ComfyClient()   # HTTP-клиент ComfyUI (соединение только при первом запросе)
_comfy_templates = {}


def _comfy_template(kind):
    if kind not in _comfy_templates:
        _comfy_templates[kind] = load_template(kind)
    return _comfy_templates[kind]


def _run_video_comfy(ptype, data):
    """Гонит видео через ComfyUI: подставляет параметры в воркфлоу → run → mp4 bytes."""
    if ptype == ProcessType.T2V:
        wf = build_t2v_workflow(_comfy_template("t2v"), prompt=data.prompt,
                                width=data.width, height=data.height, fps=data.fps)
    else:  # I2V — сперва загрузить стартовую картинку в ComfyUI
        image_name = comfy.upload_image(data["image"])
        wf = build_i2v_workflow(_comfy_template("i2v"), prompt=data["prompt"],
                                image_name=image_name, width=data["width"],
                                height=data["height"], fps=data["fps"])
    return comfy.run(wf)


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

        # На границе бэкендов освобождаем VRAM у того, кто её держал (одна карта):
        # comfy→diffusers — просим ComfyUI выгрузить (/free); diffusers→comfy —
        # убиваем diffusers-процесс (recycle), чтобы отдать VRAM ComfyUI.
        if backend == "comfy" and job_backend == "diffusers":
            comfy.free()
        elif backend == "diffusers" and job_backend == "comfy":
            gpu.recycle()
            loaded = None
        backend = job_backend

        with lock:
            results[id] = {"status": Status.IN_PROGRESS}

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
            gpu.recycle()
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


load_dotenv()
results = {}
lock = threading.Lock()
gpu = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global gpu
    gpu = GpuRunner(gpu_worker)
    threading.Thread(target=worker, args=(results, lock, gpu), daemon=True).start()

    yield

    scheduler.stop()
    gpu.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/api")
async def root():
    return {"status": "ok"}


@app.post("/api/txt2img")
async def txt2img(item: Item):
    id = str(uuid.uuid4())
    print("img", id)
    with lock:
        results[id] = {"status": Status.PENDING}
    scheduler.enqueue({"id": id, "type": ProcessType.IMAGE_GENERATION, "data": item})

    return {"id": id}


@app.post("/api/transcription")
async def transcription(file: UploadFile):
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
    scheduler.enqueue({"id": id, "type": ProcessType.TRANSCRIPTION,
             "data": {"filename": filename}})

    return {"id": id}


@app.post("/api/t2v")
async def t2v(item: Item):
    id = str(uuid.uuid4())
    print("t2v", id)
    with lock:
        results[id] = {"status": Status.PENDING}
    scheduler.enqueue({"id": id, "type": ProcessType.T2V, "data": item})

    return {"id": id}


@app.post("/api/i2v")
async def i2v(
    file: UploadFile,
    prompt: str = Form(...),
    width: int = Form(832),
    height: int = Form(480),
    fps: int = Form(30),
):
    id = str(uuid.uuid4())
    print("i2v", id)
    image = await file.read()
    with lock:
        results[id] = {"status": Status.PENDING}
    scheduler.enqueue({
        "id": id,
        "type": ProcessType.I2V,
        "data": {"prompt": prompt, "image": image,
                 "width": width, "height": height, "fps": fps},
    })

    return {"id": id}


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
