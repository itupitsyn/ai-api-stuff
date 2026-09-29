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
#
# BiRefNet lite, а не isnet, с которого начинали. isnet терял бледную руку на
# бледной стене: маска в этом месте проваливалась, и дыра на тёмной панели
# Telegram читалась как чёрная рука. Лечить это обработкой маски невозможно —
# четыре подхода (сырая альфа, два порога, усиление) дали четыре разных вида
# одного провала. BiRefNet справляется на всех кадрах замера.
#
# Из трёх вариантов BiRefNet взят lite: на наших картинках он неотличим от
# полного, но вдвое быстрее (7.6 с против 12.5) и впятеро меньше (224 МБ против
# 972). Полный и portrait лежат рядом в models/, переключаются этой переменной.
MODEL_PATH = os.getenv("CUTOUT_MODEL", "/root/models/birefnet-lite.onnx")

# Нормализация входа как при обучении BiRefNet — ImageNet.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Сторона входа сети.
_INPUT = 1024

# Сторона квадрата на выходе. 512 — требование Telegram к стикеру.
CANVAS = int(os.getenv("STICKER_SIZE", "512"))

# Поля вокруг объекта, доля от его большей стороны. Без них голова упирается в
# край и стикер выглядит обрезанным.
MARGIN = 0.06

# Порог маски для поиска границ объекта. Ниже — считаем прозрачным.
_ALPHA_FLOOR = 8

# Усиление альфы. Сырой выход сети только УМНОЖАЕТСЯ и обрезается единицей:
# уверенная середина объекта становится полностью непрозрачной, а всё
# сомнительное лишь подрастает и никогда не обнуляется.
#
# История двух неудачных попыток, чтобы никто не повторил. Сперва альфа бралась
# сырой — сеть отдаёт середину объекта значением около 230, и человек целиком
# выходил полупрозрачным: на тёмной панели Telegram сквозь него просвечивал фон.
# Потом я поставил ступеньку 0.35-0.65 — стало хуже: всё, в чём сеть не уверена,
# уходило в ноль, и на косплейном фото у человека вырезало руку начисто, а
# чёрная дыра на её месте читалась как «чёрная кожа». Ступенька 0.20-0.50 вела
# себя так же на трёх кадрах из четырёх.
#
# Умножение свободно от этого по устройству: оно монотонно и не может убрать
# то, что сеть хоть как-то нашла. Слабая маска даст полупрозрачную руку —
# некрасиво, но лучше дыры.
_ALPHA_GAIN = 1 / 0.80

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
            opts.intra_op_num_threads = 8
            _session = onnxruntime.InferenceSession(
                MODEL_PATH, opts, providers=["CPUExecutionProvider"])

    return _session


def _mask(image):
    """Маска объекта в размер исходной картинки, uint8 0..255."""
    session = _get_session()
    name = session.get_inputs()[0].name

    small = image.convert("RGB").resize((_INPUT, _INPUT), Image.LANCZOS)
    x = (np.asarray(small, dtype=np.float32) / 255.0 - _MEAN) / _STD
    x = np.transpose(x, (2, 0, 1))[None].astype(np.float32)

    # Сеть отдаёт несколько карт разного масштаба, нужна последняя — самая
    # подробная. И это логиты, поэтому сигмоида.
    out = session.run(None, {name: x})[-1]
    out = out[0][0] if out.ndim == 4 else out[0]
    out = 1.0 / (1.0 + np.exp(-out))

    # Усиление, а не порог. См. _ALPHA_GAIN.
    out = np.clip(out * _ALPHA_GAIN, 0.0, 1.0)

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
