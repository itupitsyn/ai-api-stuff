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


# Карты подстановки: какой (node_id, field) получает каждый параметр — СВОЯ на модель.
# Wan-графы из prompts.go. Для LTX добавить LTX_*_MAP с его node ID (см. build_workflow).
WAN_T2V_MAP = {
    "prompt": ("89", "text"), "width": ("74", "width"), "height": ("74", "height"),
    "num_frames": ("74", "length"), "fps": ("88", "fps"), "seed": ("81", "noise_seed"),
}
WAN_I2V_MAP = {
    "prompt": ("93", "text"), "image": ("97", "image"), "width": ("98", "width"),
    "height": ("98", "height"), "num_frames": ("98", "length"), "fps": ("94", "fps"),
    "seed": ("86", "noise_seed"),
}


def build_workflow(template, mapping, *, prompt=None, image_name=None, width=None,
                   height=None, fps=None, num_frames=None, seed=None):
    """Модель-агностичная подстановка: кладёт значения в ноды по `mapping`.

    mapping — dict {параметр: (node_id, field)}. Задан только для тех параметров,
    что есть в конкретном графе. Чтобы подключить новую модель (LTX и т.п.) —
    достаточно её workflow-JSON + такой карты, код менять не надо.
    """
    wf = copy.deepcopy(template)
    values = {
        "prompt": prompt, "image": image_name, "width": width, "height": height,
        "fps": fps, "num_frames": num_frames,
        "seed": _seed(seed) if "seed" in mapping else None,  # None → случайный
    }
    for key, value in values.items():
        if value is not None and key in mapping:
            node_id, field = mapping[key]
            wf[node_id]["inputs"][field] = value
    return wf


def build_t2v_workflow(template, *, prompt, width, height, fps, num_frames=81, seed=None):
    return build_workflow(template, WAN_T2V_MAP, prompt=prompt, width=width, height=height,
                          fps=fps, num_frames=num_frames, seed=seed)


def build_i2v_workflow(template, *, prompt, image_name, width, height, fps, num_frames=81, seed=None):
    return build_workflow(template, WAN_I2V_MAP, prompt=prompt, image_name=image_name,
                          width=width, height=height, fps=fps, num_frames=num_frames, seed=seed)


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
