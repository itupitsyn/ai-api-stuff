# -*- coding: utf-8 -*-
"""Вырезание фона и подготовка картинки под стикер Telegram.

Зачем отдельный модуль: генерация отдаёт фотографию с фоном, а стикер без
прозрачности выглядит наклеенным прямоугольником. Плюс правки съезжают по
кадру — модель то приближает лицо, то отдаляет, — и в наборе рядом это видно.
Оба вопроса закрываются одним проходом: маска, обрезка по найденному объекту,
общий квадрат.

Почему ONNX, а не rembg: rembg тянет numba, opencv и scikit-image — под 400 МБ
ради одной маски. onnxruntime у нас уже стоит ради whisperx, numpy и PIL тоже,
поэтому модель гоняем напрямую. Веса (isnet-general-use, ~170 МБ) лежат рядом с
сервисом в ./models и в гит не идут.

Считает на процессоре: маска для одной картинки — доли секунды, а карту в это
время держать незачем.
"""
import os
import threading

import numpy as np
from PIL import Image

# Путь к весам. Файл монтируется с хоста вместе с каталогом сервиса, поэтому
# переживает пересборку образа.
MODEL_PATH = os.getenv("CUTOUT_MODEL", "/root/models/isnet-general-use.onnx")

# Сторона квадрата на выходе. 512 — требование Telegram к стикеру.
CANVAS = int(os.getenv("STICKER_SIZE", "512"))

# Поля вокруг объекта, доля от его большей стороны. Без них голова упирается в
# край и стикер выглядит обрезанным.
MARGIN = 0.06

# Порог маски. Ниже — прозрачно. Модель отдаёт мягкие края, и резать их жёстко
# нельзя: волосы держатся как раз на полутонах.
_ALPHA_FLOOR = 8

_session = None
_lock = threading.Lock()


def available():
    """Есть ли чем резать. False — стикеры просто не предлагаем."""
    return os.path.exists(MODEL_PATH)


def _get_session():
    """Сессия onnxruntime, одна на процесс.

    Создаётся лениво: модуль импортируется в каждом воркере, а весит модель
    170 МБ, и поднимать её тем, кто стикеров не считает, незачем.
    """
    global _session
    with _lock:
        if _session is None:
            import onnxruntime

            opts = onnxruntime.SessionOptions()
            # Матирование идёт рядом с генерацией на той же машине, и отдавать
            # ему все ядра — значит отнимать их у того, что считает карта.
            opts.intra_op_num_threads = 4
            _session = onnxruntime.InferenceSession(
                MODEL_PATH, opts, providers=["CPUExecutionProvider"])

    return _session


def _mask(image):
    """Маска объекта в размер исходной картинки, uint8 0..255."""
    session = _get_session()
    inp = session.get_inputs()[0]
    # У isnet вход 1024x1024; берём из модели, чтобы не зашивать число.
    _, _, h, w = inp.shape
    h, w = int(h), int(w)

    small = image.convert("RGB").resize((w, h), Image.LANCZOS)
    x = np.asarray(small, dtype=np.float32) / 255.0
    # Нормализация как в обучении isnet: среднее 0.5, разброс 1.0.
    x = (x - 0.5) / 1.0
    x = np.transpose(x, (2, 0, 1))[None]

    out = session.run(None, {inp.name: x})[0]
    out = out[0][0] if out.ndim == 4 else out[0]

    # Сеть отдаёт произвольный диапазон — растягиваем в 0..1 по самой картинке.
    lo, hi = float(out.min()), float(out.max())
    out = (out - lo) / (hi - lo) if hi > lo else np.zeros_like(out)

    return np.asarray(
        Image.fromarray((out * 255).astype(np.uint8)).resize(image.size,
                                                             Image.LANCZOS))


def _bbox(alpha):
    """Границы непрозрачного. None — маска пустая."""
    ys, xs = np.where(alpha > _ALPHA_FLOOR)
    if not len(xs):
        return None

    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def to_sticker(png_bytes):
    """Байты картинки -> байты PNG 512x512 с прозрачным фоном.

    Возвращает None, если резать нечем или маска пустая: тогда вызывающий
    отдаёт исходную картинку, а не роняет задачу. Стикер — украшение поверх
    генерации, и ронять из-за него уже посчитанное нельзя.
    """
    if not available():
        return None

    try:
        import io

        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        alpha = _mask(image)
        box = _bbox(alpha)
        if box is None:
            return None

        rgba = image.convert("RGBA")
        rgba.putalpha(Image.fromarray(alpha))

        # Квадрат вокруг объекта: берём большую сторону, добавляем поля и
        # центрируем. Именно это выравнивает кадр между стикерами набора —
        # что бы модель ни сделала с масштабом лица, на выходе объект занимает
        # одну и ту же долю квадрата.
        x0, y0, x1, y1 = box
        side = max(x1 - x0, y1 - y0)
        side = int(side * (1 + 2 * MARGIN))
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        crop = (cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2)

        # Вылезающий за края квадрат дорисовываем прозрачным, а не двигаем
        # внутрь: сдвиг сбил бы центровку, ради которой всё и затевалось.
        square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        square.paste(rgba.crop(crop), (0, 0))

        out = io.BytesIO()
        square.resize((CANVAS, CANVAS), Image.LANCZOS).save(out, "PNG",
                                                           optimize=True)
        return out.getvalue()
    except Exception as e:
        print(f"[cutout] не вырезал фон: {type(e).__name__}: {e}", flush=True)
        return None
