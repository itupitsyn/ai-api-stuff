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

from diffusers import ZImagePipeline
from fastapi import FastAPI, UploadFile
from fastapi.responses import Response
from io import BytesIO
from pydantic import BaseModel
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from enum import Enum

class ProcessType(Enum):
    IMAGE_GENERATION = "img_gen"
    TRANSCRIPTION = "trans"


class Status(Enum):
    IN_PROGRESS = "in_progress"
    ERROR = "error"
    DONE = "done"
    PENDING = "pending"


class Item(BaseModel):
    prompt: str


def worker(results, lock, q):
    print("Worker started")

    # Используем контекст с методом 'spawn'
    mp_context = multiprocessing.get_context('spawn')

    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    while True:
        item = q.get()

        if item == "BREAK":
            break

        id = item.get("id")
        type = item.get("type")
        data = item.get("data")

        if type == ProcessType.IMAGE_GENERATION:
            with lock:
                results[id] = {"status": Status.IN_PROGRESS}

            process = mp_context.Process(
                target=generate_image, args=(data.prompt, return_dict))
            process.start()
            process.join()

            image_bytes = return_dict.get("res")

            if isinstance(image_bytes, dict):
                with lock:
                    results[id] = {"status": Status.ERROR,
                                   "data": image_bytes.get("error")}
            else:
                filtered_image = BytesIO()
                image_bytes.save(filtered_image, "PNG")
                filtered_image.seek(0)

                with lock:
                    results[id] = {"status": Status.DONE,
                                   "data": base64.b64encode(filtered_image.read())}

        elif type == ProcessType.TRANSCRIPTION:
            with lock:
                results[id] = {"status": Status.IN_PROGRESS}

            try:
                filename = data.get("filename")
                process = mp_context.Process(
                    target=generate_transcription, args=(filename, return_dict))
                process.start()
                process.join()

                result = return_dict.get("res")

                if result.get("error"):
                    with lock:
                        results[id] = {"status": Status.ERROR,
                                       "data": result.get("error")}
                else:
                    with lock:
                        results[id] = {"status": Status.DONE, "data": result}
            except Exception as e:
                with lock:
                    results[id] = {"status": Status.ERROR, "data": e}
            finally:
                os.unlink(filename)

        q.task_done()


q = queue.Queue()

load_dotenv()
results = {}
lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=worker, args=(results, lock, q)).start()

    yield
    q.put("BREAK")


app = FastAPI(lifespan=lifespan)


@app.get("/api")
async def root():
    return {"status": "ok"}


@app.post("/api/txt2img")
async def txt2img(item: Item):
    id = str(uuid.uuid4())
    print("img", id)
    q.put({"id": id, "type": ProcessType.IMAGE_GENERATION, "data": item})
    with lock:
        results[id] = {"status": Status.PENDING}

    return {"id": id}


@app.post("/api/transcription")
async def transcription(file: UploadFile):
    if not os.path.exists("files"):
        os.mkdir("files")

    _, extension = os.path.splitext(file.filename)
    id = str(uuid.uuid4())
    print("trans", id)

    filename = f"files/{id}.{extension}"
    with open(filename, "wb") as f:
        f.write(file.file.read())

    q.put({"id": id, "type": ProcessType.TRANSCRIPTION,
          "data": {"filename": filename}})
    with lock:
        results[id] = {"status": Status.PENDING}

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


def generate_image(prompt: str, return_dict):
    try:
        # 1. Load the pipeline
        # Use bfloat16 for optimal performance on supported GPUs
        pipe = ZImagePipeline.from_pretrained(
            "Tongyi-MAI/Z-Image-Turbo",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        pipe.to("cuda")

        # 2. Generate Image
        image = pipe(
            prompt=prompt,
            height=896,
            width=1152,
            num_inference_steps=9,  # This actually results in 8 DiT forwards
            guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
            generator=torch.Generator("cuda").manual_seed(
                random.randint(0, sys.maxsize)),
        ).images[0]

        return_dict["res"] = image
    except Exception as e:
        print(e)
        return_dict["res"] = {"error": e}


def generate_transcription(audio_file: str, return_dict):
    try:
        device = "cuda"

        # 0. Redefine torch.load
        _original_torch_load = torch.load

        def _trusted_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return _original_torch_load(*args, **kwargs)
        torch.load = _trusted_load

        # 1. Transcribe with original whisper (batched)
        model = whisperx.load_model("large-v3", device, compute_type="float16")

        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=16)

        # 2. Align whisper output
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device)
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

        # 3. Assign speaker labels
        diarize_model = whisperx.diarize.DiarizationPipeline(
            use_auth_token=os.getenv("HF_API_KEY"), device=device)

        diarize_segments = diarize_model(audio)

        result = whisperx.assign_word_speakers(diarize_segments, result)

        return_dict["res"] = result

    except Exception as e:
        print(e)
        return_dict["res"] = {"error": e}
