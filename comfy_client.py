"""Клиент ComfyUI для видео-генерации.

Имена шаблонов в comfy_workflows/ читаются слева направо:
    {модель}_{режим}[_turbo]_{формат}.json
Модель совпадает с параметром `model` в API (wan / minimax_h3), формат — это
_api (плоский граф для POST /prompt, его грузит код) или _ui (граф с нодами и
связями, его открывают в браузере). Без _turbo — базовый 20-шаговый вариант,
держим для сравнения качества, в API он не используется.

Гоняет готовые воркфлоу через HTTP API ComfyUI:
submit → poll history → download mp4. Плюс upload картинки (i2v) и /free (сброс
VRAM на границах, чтобы делить одну карту с diffusers-генерацией картинок/аудио).

Моделей две (см. MODELS): wan (Wan2.2 fp8 + lightx2v) и minimax_h3 (MiniMax H3 +
turbo-LoRA, со звуком). Выбор — параметром `model` в API, по умолчанию minimax_h3.

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
    path = os.path.join(WORKFLOW_DIR, f"{kind}.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"нет шаблона {path}. Если он потерялся — открой в ComfyUI парный "
            f"{kind.replace('_api', '_ui')}.json и сохрани через "
            f"Workflow -> Export (API), см. README") from None


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


# --------------------------------------------------------------------------
#  MiniMax H3
#  Граф H3 собран из сабграфа, и при экспорте в API-формат ComfyUI нумерует
#  ноды заново — id непредсказуемы и меняются при каждом ре-экспорте. Поэтому
#  адресуемся не по id, как у Wan, а по class_type: он стабилен.
# --------------------------------------------------------------------------
H3_T2V_CLASSES = {
    "prompt": ("MiniMaxH3ImageToVideo", "prompt"),
    "width": ("MiniMaxH3ImageToVideo", "width"),
    "height": ("MiniMaxH3ImageToVideo", "height"),
    "num_frames": ("MiniMaxH3ImageToVideo", "length"),
    "fps": ("CreateVideo", "fps"),
    "seed": ("RandomNoise", "noise_seed"),
}
H3_I2V_CLASSES = dict(H3_T2V_CLASSES, image=("LoadImage", "image"))

# Модель обучена на длинах вида 17k+5 кадров при 24 fps (124..362 = 5..15 сек).
# Ровно эту арифметику делает ComfyMathExpression внутри штатного графа —
# повторяем её здесь, потому что в API-формате мы кладём готовое число в length.
H3_FRAME_STEP = 17
H3_FRAME_BASE = 5
H3_MAX_FRAMES = 362      # 15 сек — потолок модели
H3_FPS = 24              # нативная частота; менять её = менять скорость видео


def h3_num_frames(seconds, fps=H3_FPS):
    """Ближайшая сверху валидная для H3 длина в кадрах (17k+5, не больше 362)."""
    n = max(H3_FRAME_BASE, round(seconds * fps))
    n += (H3_FRAME_BASE - n % H3_FRAME_STEP) % H3_FRAME_STEP
    return min(n, H3_MAX_FRAMES)


# Нативный холст H3: короткая сторона 768, потолок 1344x768 (~1.0 MP), сетка 32.
H3_SIZE_STEP = 32
H3_MAX_PIXELS = 1344 * 768


def h3_snap_size(width, height):
    """Приводит размер к сетке H3: кратно 32 и не крупнее нативного холста.

    Нода объявляет step=32, но ComfyUI шаг не валидирует — некратный размер
    спокойно доедет до модели и развалится уже внутри. Пропорции сохраняем:
    размеры к нам приходят из исходной картинки (i2v), и перекос там заметен.
    """
    w, h = max(1, int(width)), max(1, int(height))
    if w * h > H3_MAX_PIXELS:
        scale = (H3_MAX_PIXELS / (w * h)) ** 0.5
        w, h = w * scale, h * scale
    return _snap32(w), _snap32(h)


def center_crop_box(src_w, src_h, target_w, target_h):
    """(left, top, right, bottom) — центральный кроп src под соотношение target.

    Нода H3 растягивает первый кадр в холст без кропа (crop="disabled"), поэтому
    соотношение подгоняем заранее. Снап к сетке 32 двигает пропорции на 1-3%,
    и лучше отрезать эти проценты, чем сплющить весь кадр.
    """
    if min(src_w, src_h, target_w, target_h) <= 0:
        raise ValueError(f"неположительный размер: {src_w}x{src_h} -> {target_w}x{target_h}")
    target_ar = target_w / target_h
    if src_w / src_h > target_ar:            # исходник шире цели — режем по бокам
        crop_w = min(src_w, int(round(src_h * target_ar)))
        left = (src_w - crop_w) // 2
        return (left, 0, left + crop_w, src_h)
    crop_h = min(src_h, int(round(src_w / target_ar)))   # уже цели — режем сверху/снизу
    top = (src_h - crop_h) // 2
    return (0, top, src_w, top + crop_h)


def h3_fit_image(image_bytes, width, height):
    """Кроп по центру под соотношение холста + ресайз ровно в него.

    После этого «растягивание» внутри ноды становится тождественным.
    """
    from PIL import Image   # тяжёлая зависимость: тянем только когда реально нужна

    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        img = img.crop(center_crop_box(img.width, img.height, width, height))
        img = img.resize((width, height), Image.LANCZOS)
        out = io.BytesIO()
        img.save(out, format="PNG")
    return out.getvalue()


def prepare_image(model, image_bytes, width, height):
    """Готовит стартовый кадр под модель (для Wan — отдаёт байты без изменений)."""
    spec = MODELS[model]
    if not spec["fit_image"]:
        return image_bytes
    if spec["snap"]:
        width, height = spec["snap"](width, height)
    return spec["fit_image"](image_bytes, width, height)


def _snap32(value):
    # округление к ближайшему, половинки вверх: round() у Python банковское
    # (720 -> 704, а не 736), и на глаз это выглядит как случайность
    return max(H3_SIZE_STEP, int(value / H3_SIZE_STEP + 0.5) * H3_SIZE_STEP)


def resolve_class_map(workflow, class_map):
    """{параметр: (class_type, поле)} → {параметр: (node_id, поле)} для этого графа.

    Падаем с внятной ошибкой, если ноды нет или их несколько: молча пропустить
    подстановку хуже — получим видео с дефолтным промптом из шаблона.
    """
    resolved = {}
    for key, (class_type, field) in class_map.items():
        ids = [nid for nid, node in workflow.items()
               if isinstance(node, dict) and node.get("class_type") == class_type]
        if not ids:
            raise KeyError(f"в воркфлоу нет ноды {class_type} (нужна для '{key}')")
        if len(ids) > 1:
            raise KeyError(f"в воркфлоу {len(ids)} нод {class_type} — неоднозначно, "
                           f"не знаю, в какую класть '{key}'")
        resolved[key] = (ids[0], field)
    return resolved


# --------------------------------------------------------------------------
#  Реестр моделей. Добавить новую = дописать сюда шаблоны + карту подстановки.
# --------------------------------------------------------------------------
MODELS = {
    "wan": {
        "templates": {"t2v": "wan_t2v_api", "i2v": "wan_i2v_api"},
        "maps": {"t2v": WAN_T2V_MAP, "i2v": WAN_I2V_MAP},
        "by_class": False,
        "fps": 24,        # было 30; 81 кадр теперь играется 3.4 сек вместо 2.7
        "num_frames": 81,
        "snap": None,        # Wan к сетке не привязан, размеры отдаём как есть
        "fit_image": None,   # и картинку не трогаем — поведение как было
        "audio": False,
    },
    "minimax_h3": {
        # только турбо: базовые 20-шаговые графы в 2.2 раза медленнее
        "templates": {"t2v": "minimax_h3_t2v_turbo_api",
                      "i2v": "minimax_h3_i2v_turbo_api"},
        "maps": {"t2v": H3_T2V_CLASSES, "i2v": H3_I2V_CLASSES},
        "by_class": True,
        "fps": H3_FPS,
        "num_frames": h3_num_frames(5),   # 124 кадра ≈ 5 сек
        "snap": h3_snap_size,             # размер обязан лечь на сетку 32
        "fit_image": h3_fit_image,        # кроп под холст вместо растягивания в ноде
        "audio": True,                    # генерит нативное стерео в тот же проход
    },
}

DEFAULT_MODEL = "minimax_h3"


def template_name(model, kind):
    """Имя файла шаблона (без .json) для пары модель+режим."""
    return MODELS[model]["templates"][kind]


def build_video_workflow(model, kind, template, *, prompt, image_name=None,
                         width, height, fps=None, num_frames=None, seed=None):
    """Единая точка сборки графа для любой модели.

    fps и num_frames, если не заданы, берутся из дефолтов модели: у H3 они
    жёстко завязаны на 24 fps и длину 17k+5, у Wan свои.
    """
    if model not in MODELS:
        raise ValueError(f"неизвестная модель '{model}', доступны: {sorted(MODELS)}")
    spec = MODELS[model]
    if kind not in spec["maps"]:
        raise ValueError(f"модель '{model}' не умеет режим '{kind}'")

    mapping = spec["maps"][kind]
    if spec["by_class"]:
        mapping = resolve_class_map(template, mapping)
    if spec["snap"]:
        width, height = spec["snap"](width, height)

    return build_workflow(
        template, mapping, prompt=prompt, image_name=image_name,
        width=width, height=height,
        fps=spec["fps"] if fps is None else fps,
        num_frames=spec["num_frames"] if num_frames is None else num_frames,
        seed=seed,
    )


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

    def vram_free_mb(self):
        """Свободная VRAM по данным ComfyUI (МБ). None, если он недоступен.

        Берём отсюда, а не из torch: API-процесс не должен поднимать свой
        CUDA-контекст ради одной цифры.
        """
        try:
            r = requests.get(f"{self.base}/system_stats", timeout=10)
            r.raise_for_status()
            dev = (r.json().get("devices") or [{}])[0]
            return int(dev.get("vram_free", 0)) // (1024 * 1024)
        except Exception:
            return None

    def free(self, wait_vram_mb=0, timeout=60):
        """Сброс VRAM: модели ComfyUI уходят в RAM, кэш чистится.

        ``/free`` отвечает 200 ДО того, как память реально отдана драйверу.
        Замер на боксе: сразу после вызова ``vram_free`` было 1217 МБ, и
        Z-Image (пик 23.2 ГБ) грузился в занятую карту 201 с вместо 20; в другой
        раз тот же расклад дал CUDA OOM. Поэтому при ``wait_vram_mb`` ждём по
        факту, а не по коду ответа.

        По истечении ``timeout`` не бросаем исключение, а идём дальше с
        предупреждением: лучше попытаться и упасть с внятной ошибкой загрузки,
        чем заблокировать очередь навсегда.
        """
        try:
            requests.post(f"{self.base}/free",
                          json={"unload_models": True, "free_memory": True}, timeout=30)
        except Exception as e:
            print(f"[comfy] /free failed: {e}", flush=True)
            return None

        if not wait_vram_mb:
            return None

        t0 = time.time()
        while time.time() - t0 < timeout:
            got = self.vram_free_mb()
            if got is None:      # ComfyUI не отвечает — ждать нечего
                break
            if got >= wait_vram_mb:
                print(f"[comfy] VRAM отдана: {got} MB за {time.time() - t0:.1f}s",
                      flush=True)
                return got
            time.sleep(self.poll_interval)

        got = self.vram_free_mb()
        print(f"[comfy] ВНИМАНИЕ: за {timeout}s освободилось только {got} MB "
              f"(ждали {wait_vram_mb}); продолжаем", flush=True)
        return got

    def run(self, workflow):
        """Полный цикл: submit → wait → скачать байты видео."""
        pid = self.submit(workflow)
        entry = self.wait(pid)
        found = find_output_file(entry)
        if not found:
            raise RuntimeError(f"ComfyUI prompt {pid}: no output file in history")
        return self.download(*found)
