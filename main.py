import torch
import sys
import queue
import random
import whisperx
import os
import threading
import multiprocessing
import uuid
import base64
import tempfile
import time
import gc
import traceback

from diffusers import ZImagePipeline, WanPipeline, AutoencoderKLWan, WanImageToVideoPipeline, UniPCMultistepScheduler
from diffusers.utils import export_to_video, load_image
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import Response
from io import BytesIO
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from enum import Enum

import torch, pandas as pd
from pyannote.audio import Pipeline

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


# ==========================================================================
#  GPU-СТОРОНА: исполняется в одном долгоживущем процессе.
#  - видео-модель (t2v ИЛИ i2v) кэшируется резидентно в _video_slot;
#  - картинки и транскрипция грузятся на время запроса и освобождаются,
#    НЕ вытесняя видео-модель.
# ==========================================================================

_video_slot = {"type": None, "pipe": None, "meta": None}


def _get_video_pipe(ptype, builder):
    """Резидентный видео-слот. Перезагрузка только при смене t2v<->i2v."""
    if _video_slot["type"] != ptype:
        if _video_slot["pipe"] is not None:
            print(f"[gpu] swap video model {_video_slot['type']} -> {ptype}", flush=True)
        # выгружаем прежнюю видео-модель (обе ~50 ГБ, одновременно не держим)
        _video_slot["type"] = None
        _video_slot["pipe"] = None
        _video_slot["meta"] = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        t0 = time.time()
        pipe, meta = builder()
        _video_slot["type"] = ptype
        _video_slot["pipe"] = pipe
        _video_slot["meta"] = meta
        print(f"[{ptype.value}] load: {time.time() - t0:.1f}s", flush=True)
    return _video_slot["pipe"], _video_slot["meta"]


def _build_t2v_pipe():
    MODEL_ID = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
    USE_ACCELERATOR_LORA = True  # False = медленнее, но без LoRA

    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.bfloat16)

    if USE_ACCELERATOR_LORA:
        # LoRA грузится ОТДЕЛЬНО на high-noise (pipe.transformer) и low-noise (pipe.transformer_2)
        pipe.load_lora_weights(
            "lightx2v/Wan2.2-Lightning",
            weight_name="Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors",
            adapter_name="high",
        )
        pipe.load_lora_weights(
            "lightx2v/Wan2.2-Lightning",
            weight_name="Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors",
            adapter_name="low",
        )
        pipe.set_adapters(["high"])
        pipe.transformer_2.set_adapters(["low"])

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)
        meta = {"num_steps": 4, "guidance": 1.0, "guidance_2": 1.0}
    else:
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)
        meta = {"num_steps": 40, "guidance": 4.0, "guidance_2": 3.0}

    # Критично для 24 ГБ: выгружаем неактивные части на CPU по мере необходимости
    pipe.enable_sequential_cpu_offload()
    return pipe, meta


def _build_i2v_pipe():
    MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
    USE_ACCELERATOR_LORA = True

    vae = AutoencoderKLWan.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=torch.bfloat16)

    if USE_ACCELERATOR_LORA:
        # ВНИМАНИЕ: проверь точные имена I2V-весов в репо lightx2v/Wan2.2-Lightning
        pipe.load_lora_weights(
            "lightx2v/Wan2.2-Lightning",
            weight_name="Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/high_noise_model.safetensors",
            adapter_name="high",
        )
        pipe.load_lora_weights(
            "lightx2v/Wan2.2-Lightning",
            weight_name="Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/low_noise_model.safetensors",
            adapter_name="low",
        )
        pipe.set_adapters(["high"])
        pipe.transformer_2.set_adapters(["low"])

        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)
        meta = {"num_steps": 4, "guidance": 1.0, "guidance_2": 1.0}
    else:
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)
        meta = {"num_steps": 40, "guidance": 3.5, "guidance_2": 3.5}

    pipe.enable_sequential_cpu_offload()
    return pipe, meta


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
    pipe, meta = _get_video_pipe(ProcessType.T2V, _build_t2v_pipe)

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
    print(f"[t2v] inference: {time.time() - t_inf:.1f}s", flush=True)

    out = _video_to_bytes(video, data.fps)
    _free_cuda()  # освобождаем VRAM под возможные картинку/аудио (веса видео остаются в RAM)
    return out


def _run_i2v(data):
    from PIL import Image

    pipe, meta = _get_video_pipe(ProcessType.I2V, _build_i2v_pipe)

    image = Image.open(BytesIO(data["image"])).convert("RGB")

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
    print(f"[i2v] inference: {time.time() - t_inf:.1f}s", flush=True)

    out = _video_to_bytes(video, data["fps"])
    _free_cuda()
    return out


def _run_image(data):
    # Транзиентно: грузим на время запроса и освобождаем, видео-слот не трогаем
    pipe, _ = _build_image_pipe()
    try:
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
    finally:
        del pipe
        _free_cuda()


def _run_transcription(data):
    # Транзиентно: whisperx грузит свои модели и освобождает после, видео-слот не трогаем
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

    while True:
        job = scheduler.next_job()
        if job is None:  # остановка
            break

        id = job.get("id")
        type = job.get("type")
        data = job.get("data")

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
