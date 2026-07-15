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

from diffusers import ZImagePipeline, WanPipeline, AutoencoderKLWan, WanImageToVideoPipeline, UniPCMultistepScheduler, WanTransformer3DModel, TorchAoConfig
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

# fp8-квантизация трансформеров Wan (по образцу ComfyUI): эксперт ужимается
# ~28→14 ГБ и влезает в VRAM целиком, поэтому model_cpu_offload свопает эксперты
# по одному, а не стримит веса послойно (как sequential). Это и есть тот 4x на
# инференсе (~300с → ~100с по цифрам ComfyUI). False = рабочий bf16-путь.
USE_FP8 = True

# Wan VAE в fp32 стабильнее (меньше артефактов), но декод медленнее и жрёт память.
# ComfyUI гоняет VAE в bf16. Ставим False ради скорости — верни True, если видео поплывёт.
VAE_FP32 = False


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


def _fp8_quant_config():
    # diffusers ждёт объект torchao AOBaseConfig. Weight-only fp8: веса в fp8,
    # матмул в bf16 — работает и на Ampere (3090), и на Ada (4090).
    from torchao.quantization import Float8WeightOnlyConfig
    return TorchAoConfig(Float8WeightOnlyConfig())


def _quantize_transformer(model_id, subfolder):
    """fp8-квантованный трансформер (свежая квантизация через TorchAoConfig).

    БЕЗ дискового кэша: роундтрип квантованной модели в этом стеке принципиально
    сломан — save_pretrained шардит без индекса, torch.save(module) не пиклит
    (torchao патчит forward через functools.partial), а from_config+assign теряет
    метаданные квантизации (hf_quantizer), из-за чего падает инъекция LoRA
    (TorchaoLoraLinear ... missing get_apply_tensor_subclass). Только свежая
    квантизация даёт полностью рабочую модель (LoRA + offload на GPU).
    """
    print(f"[fp8] quantize {model_id.split('/')[-1]}/{subfolder}", flush=True)
    return WanTransformer3DModel.from_pretrained(
        model_id, subfolder=subfolder,
        quantization_config=_fp8_quant_config(), torch_dtype=torch.bfloat16)


def _build_wan_fp8(kind):
    """fp8-путь: свежеквантованные трансформеры + LoRA поверх + model_cpu_offload.
    На torch 2.11 / torchao 0.17 diffusers знает про квантизацию, поэтому offload
    корректно двигает fp8-веса на GPU (а не оставляет на CPU)."""
    m = WAN_MODELS[kind]
    transformer = _quantize_transformer(m["model_id"], "transformer")
    transformer_2 = _quantize_transformer(m["model_id"], "transformer_2")

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


def worker(results, lock, gpu):
    print("Worker started", flush=True)

    loaded = None   # какая модель сейчас в GPU-процессе (по типу задачи)

    while True:
        job = scheduler.next_job()
        if job is None:  # остановка
            break

        id = job.get("id")
        type = job.get("type")
        data = job.get("data")

        # Смена модели → жёсткий сброс VRAM смертью процесса (на 24 ГБ иначе OOM:
        # CUDA не отдаёт «хвост» внутри живого процесса). Однотипные задачи подряд
        # идут по тёплой модели без пересоздания.
        if loaded is not None and type != loaded:
            gpu.recycle()
        loaded = type

        with lock:
            results[id] = {"status": Status.IN_PROGRESS}

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

        else:  # T2V / I2V
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
