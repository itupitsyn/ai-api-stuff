# -*- coding: utf-8 -*-
"""Подготовка входов для замены человека в видео (Wan 2.2 Animate).

Модель ждёт четыре потока кадров, и три из них устроены неочевидно. Каждое
правило здесь выстрадано отдельным прогоном, поэтому записано вместе с тем,
что ломается без него:

* ``mask`` — ПРЯМОУГОЛЬНИК по габаритам человека, а не силуэт. Внутри маски
  модель рисует только персонажа и НЕ дорисовывает вокруг него фон: при
  силуэте остаток формы остаётся таким, каким его подали, и расширение маски
  это не лечит, а только увеличивает пятно.
* ``bg`` — исходный кадр, где область маски залита ЧЁРНЫМ (``frame * (1-m)``).
  Серый 127.5 — значение, которым нода заполняет пустоту по умолчанию, —
  модель воспроизводит буквально, как содержимое кадра.
* ``pose`` — отрисованный скелет, и обязательно в формате препроцессора Wan
  (``draw_aapose_by_meta_new``). На скелетах OpenPose персонаж плывёт заметно
  сильнее: модель обучалась не на них.
* ``face`` — кроп головы; нода жёстко режет его в 512x512.

Всё считается на карте: ViTPose и YOLO через onnxruntime, SAM2 через torch.
Порядок ровно такой, как ниже: сначала поза, потом по её ключевым точкам
рамка для SAM2. Отдельный детектор для рамки не нужен — раньше её давал
BiRefNet, теперь она берётся бесплатно из уже посчитанных точек.
"""
import gc
import os
import sys
import tempfile
import time

import cv2
import numpy as np

# Вендоренные модули Wan лежат рядом и импортируют друг друга по короткому
# имени (``from pose2d_utils import ...``), поэтому их каталог нужен в path.
_WAN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wan_preprocess")
if _WAN not in sys.path:
    sys.path.insert(0, _WAN)

# Веса препроцессора монтируются с хоста вместе с каталогом сервиса, поэтому
# переживают пересборку образа и в него не попадают (2.5 ГБ).
CKPT = os.getenv("WAN_PREPROCESS", "/root/models/wan-preprocess")

_DILATE = np.ones((7, 7), np.uint8)   # 3 прохода этим ядром — как в оригинале
_FACE_SIDE_K = 0.42                   # доля высоты силуэта под кроп головы
_FACE_TOP_K = 0.18                    # верхняя доля силуэта, где ищем голову
_SAM2_MODEL = "facebook/sam2.1-hiera-large"


def available():
    """Есть ли веса препроцессора на месте."""
    return os.path.isfile(os.path.join(
        CKPT, "pose2d", "vitpose_h_wholebody.onnx", "end2end.onnx"))


def _free_cuda():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                                    # noqa: BLE001
        pass


def _pose(frames, width, height):
    """Скелеты в размер генерации + ключевые точки первого кадра.

    Координаты в метах нормированы, на пиксели их умножает сам
    ``from_humanapi_meta`` по width/height ИЗ МЕТЫ — поэтому, чтобы рисовать
    сразу в размере генерации, достаточно подменить там эти два числа.
    """
    from pose2d import Pose2d
    from pose2d_utils import AAPoseMeta
    from human_visualization import draw_aapose_by_meta_new

    pose = Pose2d(os.path.join(CKPT, "pose2d", "vitpose_h_wholebody.onnx"),
                  os.path.join(CKPT, "det", "yolov10m.onnx"), device="cuda")
    metas = pose(np.asarray(frames))
    del pose
    _free_cuda()

    first_kps = metas[0]["keypoints_body"].copy() if metas else None
    drawn = []
    for meta in metas:
        meta["width"], meta["height"] = width, height
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        # Их рендерер отдаёт RGB (дальше у них moviepy), у нас всё в BGR.
        drawn.append(draw_aapose_by_meta_new(
            canvas, AAPoseMeta.from_humanapi_meta(meta))[:, :, ::-1])
    return drawn, first_kps


def _seed_box(kps, width, height, margin=0.06):
    """Рамка человека по ключевым точкам первого кадра — затравка для SAM2."""
    pts = kps[kps[:, 2] > 0.3][:, :2] if kps is not None and len(kps) else None
    if pts is None or not len(pts):
        return np.array([0, 0, width - 1, height - 1], dtype=np.float32)
    x0, y0 = pts.min(axis=0) * (width, height)
    x1, y1 = pts.max(axis=0) * (width, height)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return np.array([max(x0 - mx, 0), max(y0 - my, 0),
                     min(x1 + mx, width - 1), min(y1 + my, height - 1)],
                    dtype=np.float32)


def _masks(frames, box):
    """Маски человека по всему ролику: SAM2 ВЕДЁТ объект, а не ищет заново.

    Посегментное вырезание (BiRefNet) иногда перескакивало на второго человека
    в кадре, и приходилось оставлять связную область, пересекающуюся с
    предыдущим кадром. Здесь перескок невозможен по устройству: объект задан
    один раз рамкой на первом кадре.
    """
    import torch
    from sam2.sam2_video_predictor import SAM2VideoPredictor

    with tempfile.TemporaryDirectory() as tmp:
        # SAM2 читает кадры только как JPEG с именами вида 00000.jpg.
        for i, frame in enumerate(frames):
            cv2.imwrite(os.path.join(tmp, f"{i:05d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        predictor = SAM2VideoPredictor.from_pretrained(_SAM2_MODEL, device="cuda")
        out = [None] * len(frames)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(video_path=tmp)
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=box)
            for idx, _ids, logits in predictor.propagate_in_video(state):
                out[idx] = (logits[0] > 0).cpu().numpy()[0]
        del predictor, state
    _free_cuda()
    # Кадры, до которых трекер не дошёл, лучше не заменять вовсе, чем
    # заменять наугад: пустая маска означает «оставить как в исходнике».
    return [m if m is not None else np.zeros(frames[0].shape[:2], bool)
            for m in out]


def _face_box(masks, shape):
    """Общий на всю последовательность кроп головы.

    Общий, а не покадровый, намеренно: дёргающийся кроп ломает артикуляцию —
    модель читает по нему мимику, и если он ездит, губы плывут вместе с ним.
    """
    h, w = shape
    boxes = []
    for m in masks:
        ys, xs = np.where(m)
        if not len(ys):
            continue
        y0, y1 = ys.min(), ys.max()
        head = m[y0:y0 + max(1, int((y1 - y0) * _FACE_TOP_K))]
        hx = np.where(head.any(axis=0))[0]
        cx = (hx.min() + hx.max()) / 2 if len(hx) else (xs.min() + xs.max()) / 2
        boxes.append((cx, y0, (y1 - y0) * _FACE_SIDE_K))
    if not boxes:
        side = min(w, h)
        return 0, 0, side
    side = int(round(min(np.median([b[2] for b in boxes]), w, h)))
    left = int(round(min(max(np.median([b[0] for b in boxes]) - side / 2, 0), w - side)))
    top = int(round(min(max(np.median([b[1] for b in boxes]) - side * 0.10, 0), h - side)))
    return left, top, side


def prepare(frames, width, height, log=print):
    """Кадры исходника -> (pose, bg, mask, face) в BGR, готовые для Wan.

    frames — список кадров BGR в РОДНОМ размере ролика. Маска, фон и лицо
    считаются в нём же: их ужимает уже сама нода, а нам важнее не потерять
    точность на промежуточном ресайзе. В размер генерации приводится только
    скелет — его рисуем сразу куда надо.
    """
    if not available():
        raise RuntimeError(f"нет весов препроцессора Wan в {CKPT}")

    t0 = time.time()
    pose, kps = _pose(frames, width, height)
    log(f"[swap] скелет: {len(pose)} кадров за {time.time() - t0:.0f} c")

    h, w = frames[0].shape[:2]
    t1 = time.time()
    masks = _masks(frames, _seed_box(kps, w, h))
    log(f"[swap] маска: {len(masks)} кадров за {time.time() - t1:.0f} c")

    left, top, side = _face_box(masks, (h, w))
    bg, mask_frames, face = [], [], []
    for frame, m in zip(frames, masks):
        grown = cv2.dilate(m.astype(np.uint8), _DILATE, iterations=3)
        ys, xs = np.where(grown > 0)
        box = np.zeros_like(grown)
        if len(ys):
            box[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 1
        mask_frames.append(np.repeat((box * 255)[:, :, None], 3, axis=2))
        bg.append(frame * (1 - box[:, :, None]))
        face.append(cv2.resize(frame[top:top + side, left:left + side],
                               (512, 512), interpolation=cv2.INTER_LANCZOS4))

    log(f"[swap] подготовка целиком {time.time() - t0:.0f} c")
    return pose, bg, mask_frames, face


def _encode(frames, path, fps):
    """Кадры -> mp4 БЕЗ ПОТЕРЬ.

    Именно без потерь: маска бинарная, и артефакты сжатия на её краю модель
    читает как полупрозрачность — ореол лезет прямо в результат.
    """
    import subprocess
    import imageio_ffmpeg

    h, w = frames[0].shape[:2]
    proc = subprocess.Popen(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
         "-r", f"{fps:.6f}", "-i", "-",
         "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", path],
        stdin=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg не смог записать {path}")
    return path


def worker(job_q, res_q):
    """Контракт gpu_runner: задача -> четыре готовых mp4 на диске.

    Живёт ровно одну задачу: подготовка берёт 4.3 ГБ, а следом ComfyUI просит
    22 — вместе не помещаются, и вернуть память можно только смертью процесса.
    Поэтому наружу отдаются ПУТИ, а не кадры: родителю остаётся их прочитать,
    и держать в очереди сотни мегабайт кадров не приходится.
    """
    while True:
        job = job_q.get()
        if job == "BREAK":
            return
        try:
            cap = cv2.VideoCapture(job["video_path"])
            frames = []
            while len(frames) < job["length"]:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            cap.release()
            if len(frames) < job["length"]:
                raise RuntimeError(
                    f"в ролике {len(frames)} кадров, а нужно {job['length']}")

            pose, bg, mask, face = prepare(frames, job["width"], job["height"])
            out = {}
            for name, seq in (("pose", pose), ("bg", bg),
                              ("mask", mask), ("face", face)):
                out[name] = _encode(seq, os.path.join(job["out_dir"],
                                                      f"{name}.mp4"), job["fps"])
            res_q.put((job["id"], {"files": out}))
        except Exception as exc:                          # noqa: BLE001
            res_q.put((job["id"], {"error": f"{type(exc).__name__}: {exc}"}))
