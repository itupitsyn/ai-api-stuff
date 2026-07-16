"""Клиент ComfyUI для видео-генерации.

Гоняет готовые воркфлоу из comfy_workflows/{t2v,i2v}.json через HTTP API ComfyUI:
submit → poll history → download mp4. Плюс upload картинки (i2v) и /free (сброс
VRAM на границах, чтобы делить одну карту с diffusers-генерацией картинок/аудио).

Чистые функции (build_*_workflow, find_output_file) вынесены отдельно и покрыты
тестами; сетевой ComfyClient тестируется по желанию с поднятым ComfyUI.
"""
import copy
import io
import json
import os
import random
import time
import uuid

import requests

DEFAULT_BASE = os.getenv("COMFYUI_URL", "http://comfyui:8188")
WORKFLOW_DIR = os.getenv("COMFY_WORKFLOW_DIR",
                         os.path.join(os.path.dirname(__file__), "comfy_workflows"))


def load_template(kind):
    with open(os.path.join(WORKFLOW_DIR, f"{kind}.json"), encoding="utf-8") as f:
        return json.load(f)


def _seed(seed):
    return seed if seed is not None else random.randint(0, 2 ** 63 - 1)


def build_t2v_workflow(template, *, prompt, width, height, fps, num_frames=81, seed=None):
    """Подставляет параметры в t2v-граф (ноды: 89 текст, 74 латент, 88 fps, 81 seed)."""
    wf = copy.deepcopy(template)
    wf["89"]["inputs"]["text"] = prompt
    wf["74"]["inputs"]["width"] = width
    wf["74"]["inputs"]["height"] = height
    wf["74"]["inputs"]["length"] = num_frames
    wf["88"]["inputs"]["fps"] = fps
    wf["81"]["inputs"]["noise_seed"] = _seed(seed)   # high-noise сэмплер (add_noise=enable)
    return wf


def build_i2v_workflow(template, *, prompt, image_name, width, height, fps, num_frames=81, seed=None):
    """Подставляет параметры в i2v-граф (ноды: 93 текст, 97 картинка, 98 латент, 94 fps, 86 seed)."""
    wf = copy.deepcopy(template)
    wf["93"]["inputs"]["text"] = prompt
    wf["97"]["inputs"]["image"] = image_name
    wf["98"]["inputs"]["width"] = width
    wf["98"]["inputs"]["height"] = height
    wf["98"]["inputs"]["length"] = num_frames
    wf["94"]["inputs"]["fps"] = fps
    wf["86"]["inputs"]["noise_seed"] = _seed(seed)
    return wf


def find_output_file(history_entry):
    """Находит (filename, subfolder, type) выходного файла в outputs history-записи."""
    for node_out in history_entry.get("outputs", {}).values():
        for key in ("videos", "gifs", "images"):
            files = node_out.get(key)
            if files:
                f = files[0]
                return f["filename"], f.get("subfolder", ""), f.get("type", "output")
    return None


class ComfyClient:
    def __init__(self, base_url=None, client_id=None, poll_interval=2.0):
        self.base = (base_url or DEFAULT_BASE).rstrip("/")
        self.client_id = client_id or uuid.uuid4().hex
        self.poll_interval = poll_interval

    def upload_image(self, image_bytes, filename="input.png"):
        r = requests.post(
            f"{self.base}/upload/image",
            files={"image": (filename, io.BytesIO(image_bytes), "image/png")},
            data={"overwrite": "true"}, timeout=60)
        r.raise_for_status()
        info = r.json()
        name = info["name"]
        return f"{info['subfolder']}/{name}" if info.get("subfolder") else name

    def submit(self, workflow):
        r = requests.post(f"{self.base}/prompt",
                          json={"prompt": workflow, "client_id": self.client_id}, timeout=60)
        r.raise_for_status()
        return r.json()["prompt_id"]

    def wait(self, prompt_id, timeout=1800):
        deadline = time.time() + timeout
        while time.time() < deadline:
            r = requests.get(f"{self.base}/history/{prompt_id}", timeout=30)
            r.raise_for_status()
            hist = r.json()
            entry = hist.get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"ComfyUI prompt {prompt_id} failed: {status}")
                if entry.get("outputs"):
                    return entry
            time.sleep(self.poll_interval)
        raise TimeoutError(f"ComfyUI prompt {prompt_id} timed out after {timeout}s")

    def download(self, filename, subfolder, ftype):
        r = requests.get(f"{self.base}/view",
                         params={"filename": filename, "subfolder": subfolder, "type": ftype},
                         timeout=120)
        r.raise_for_status()
        return r.content

    def free(self):
        """Сброс VRAM: модели ComfyUI уходят в RAM, кэш чистится."""
        try:
            requests.post(f"{self.base}/free",
                          json={"unload_models": True, "free_memory": True}, timeout=30)
        except Exception as e:
            print(f"[comfy] /free failed: {e}", flush=True)

    def run(self, workflow):
        """Полный цикл: submit → wait → скачать байты видео."""
        pid = self.submit(workflow)
        entry = self.wait(pid)
        found = find_output_file(entry)
        if not found:
            raise RuntimeError(f"ComfyUI prompt {pid}: no output file in history")
        return self.download(*found)
