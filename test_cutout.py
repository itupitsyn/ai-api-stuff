# -*- coding: utf-8 -*-
"""Тесты арифметики тона в cutout: ONNX и веса не нужны.

Маску и эталон подменяем, потому что проверяется именно коррекция, а не сеть.
Запуск:  pytest test_cutout.py -v
"""
import io

import numpy as np
import pytest
from PIL import Image

import cutout


def skin_patch(lum_scale=1.0, size=64, bg=None):
    """Картинка: квадрат телесного цвета, вокруг — фон.

    Фон по умолчанию бежевый НАРОЧНО: он проходит по цветности как кожа, и на
    этом ловится ошибка «посчитали тон вместе с фоном».
    """
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[:, :] = bg if bg is not None else (235.0, 215.0, 195.0)
    img[8:56, 8:56] = np.array([200.0, 160.0, 140.0]) * lum_scale
    return np.clip(img, 0, 255)


def rgba(arr):
    """RGBA-картинка: объект непрозрачный, фон прозрачный — как готовый квадрат."""
    a = np.where(subject_mask(arr.shape[0]), 255, 0).astype(np.uint8)
    return Image.fromarray(
        np.dstack([arr.astype(np.uint8), a]), "RGBA")


def subject_mask(size=64):
    """Объект — только центральный квадрат, фон в маску не входит."""
    m = np.zeros((size, size), dtype=bool)
    m[8:56, 8:56] = True
    return m


def test_skin_ignores_background():
    """Фон в кожу не попадает, даже если он телесный по цветности."""
    rgb = skin_patch()
    inside = cutout._skin(rgb, subject_mask())
    whole = cutout._skin(rgb, np.ones((64, 64), dtype=bool))

    assert inside.sum() == 48 * 48
    # Без маски объекта бежевый фон затекает в «кожу» — ровно та ошибка,
    # из-за которой замер однажды дал неверный вывод.
    assert whole.sum() > inside.sum()


def test_skin_mean_none_when_too_little():
    rgb = skin_patch()
    tiny = np.zeros((64, 64), dtype=bool)
    tiny[0:5, 0:5] = True

    assert cutout._skin_mean(rgb, tiny) is None


def test_match_tone_returns_source_level(monkeypatch):
    """Затемнённая картинка возвращается к тону исходника."""
    bright = skin_patch(1.0)
    dark = skin_patch(0.90)        # 11% — внутри потолка TONE_LIMIT
    monkeypatch.setattr(cutout, "_skin_ref",
                        lambda b: cutout._skin_mean(bright, subject_mask()))

    out = cutout._match_tone(rgba(dark), b"src")
    got = cutout._skin_mean(np.asarray(out, dtype=np.float32)[..., :3],
                            subject_mask())
    want = cutout._skin_mean(bright, subject_mask())

    assert got == pytest.approx(want, abs=1.5)


def test_match_tone_respects_limit(monkeypatch):
    """Коррекция не выходит за потолок, как бы ни врал эталон."""
    dark = skin_patch(0.5)        # ниже 0.4 кожа не проходит порог y>40
    monkeypatch.setattr(cutout, "_skin_ref",
                        lambda b: np.array([255.0, 255.0, 255.0]))

    out = cutout._match_tone(rgba(dark), b"src")
    got = cutout._skin_mean(np.asarray(out, dtype=np.float32)[..., :3],
                            subject_mask())
    was = cutout._skin_mean(dark, subject_mask())

    assert (got / was).max() == pytest.approx(cutout.TONE_LIMIT, abs=0.01)


def test_match_tone_passes_through_without_reference(monkeypatch):
    """Эталон не посчитался — картинка возвращается как была, без исключения."""
    dark = skin_patch(0.8)
    monkeypatch.setattr(cutout, "_skin_ref", lambda b: None)

    src = rgba(dark)
    out = cutout._match_tone(src, b"src")

    assert out is src


def test_skin_ref_cached(monkeypatch):
    """Эталон считается один раз на набор: десять стикеров — один проход маски."""
    cutout._ref_cache.clear()
    calls = []

    def fake_mask(image):
        calls.append(1)
        return np.where(subject_mask(), 255, 0).astype(np.uint8)

    monkeypatch.setattr(cutout, "_mask", fake_mask)
    buf = io.BytesIO()
    Image.fromarray(skin_patch().astype(np.uint8)).save(buf, "PNG")
    data = buf.getvalue()

    first = cutout._skin_ref(data)
    for _ in range(9):
        cutout._skin_ref(data)

    assert len(calls) == 1
    assert first is not None
