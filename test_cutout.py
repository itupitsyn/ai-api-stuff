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


def alpha_curve(raw):
    """Та же арифметика, что в _mask: усиление и завал неуверенного края."""
    return np.clip(raw * cutout._ALPHA_GAIN, 0.0, 1.0) ** cutout._ALPHA_FALLOFF


def test_alpha_curve_keeps_the_confident_subject():
    """Уверенная середина объекта обязана остаться полностью непрозрачной.

    Это и есть страховка от двух старых провалов: как только уверенный пиксель
    перестаёт быть единицей, человек начинает просвечивать.
    """
    confident = np.array([0.80, 0.85, 0.95, 1.0])

    assert np.allclose(alpha_curve(confident), 1.0)


def test_alpha_curve_never_zeroes_what_the_net_found():
    """Монотонность и отсутствие обнуления — разница между завалом и ступенькой.

    Ступенька 0.35-0.65 однажды вырезала бледную руку начисто, и дыра на тёмной
    панели читалась как чёрная кожа. Завал слабое место лишь ослабляет.
    """
    raw = np.linspace(0.001, 1.0, 200)
    out = alpha_curve(raw)

    assert (out > 0).all()
    assert (np.diff(out) >= 0).all()


def test_alpha_curve_thins_the_uncertain_fringe():
    """Между прядями сеть даёт 0.2-0.5, и это чистый фон: он должен бледнеть.

    Без завала фон там оставался заметным — на тёмной панели Telegram светлое
    небо под альфой 0.3 выглядит серым пятном между волосами.
    """
    fringe = np.array([0.2, 0.3, 0.4, 0.5])
    gain_only = np.clip(fringe * cutout._ALPHA_GAIN, 0.0, 1.0)

    assert (alpha_curve(fringe) < gain_only * 0.8).all()


GREEN = (30.0, 200.0, 60.0)
BLUE = (40.0, 60.0, 210.0)


def green_frame(size=64, subject=None, screen=GREEN):
    """Кадр: сплошной экран, в середине объект. subject — цвет объекта."""
    img = np.zeros((size, size, 3), dtype=np.float32)
    img[:, :] = screen
    img[8:56, 8:56] = subject if subject is not None else (200.0, 160.0, 140.0)
    return img


def test_tint_alpha_separates_screen_from_skin():
    a = cutout._tint_alpha(green_frame(), 1)

    assert a[0, 0] == pytest.approx(0.0)        # фон — прозрачный
    assert a[32, 32] == pytest.approx(1.0)      # кожа — непрозрачная


def test_screen_colour_reads_the_border():
    """Цвет экрана не зашит: он берётся с рамки, какой бы он ни был."""
    assert cutout._screen_colour(green_frame()) == pytest.approx(GREEN)
    assert cutout._screen_colour(green_frame(screen=BLUE)) == pytest.approx(BLUE)


def test_screen_colour_needs_the_border():
    """Экран от кофты в тот же цвет отличается тем, что он по краям кадра."""
    shirt = np.zeros((64, 64, 3), dtype=np.float32)
    shirt[:, :] = (235.0, 215.0, 195.0)
    shirt[8:56, 8:56] = GREEN

    assert cutout._screen_colour(shirt) is None


def test_screen_colour_ignores_a_plain_wall():
    """Ровная, но блёклая стена — не экран: по цвету на ней резать нечего.

    Без этого условия любой снимок у белой стены поехал бы в хромакей, и
    бледная рука на ней стала бы прозрачной.
    """
    wall = np.zeros((64, 64, 3), dtype=np.float32)
    wall[:, :] = (235.0, 230.0, 225.0)
    wall[8:56, 8:56] = (200.0, 160.0, 140.0)

    assert cutout._screen_colour(wall) is None


def test_cut_keeps_green_clothes(monkeypatch):
    """Человек в зелёной толстовке обязан уцелеть.

    Голый хромакей съедал его на 70%, оставляя голову и кисти. Уверенная часть
    маски непрозрачна, что бы ни думал про неё цвет.

    Одежда взята зелёной, но НЕ цвета экрана, и это не поблажка тесту: замер
    02.10.2026 показал, что настоящая зелёная толстовка отстоит от цвета экрана
    минимум на 35 единиц, а сам экран от своей медианы — на 2-4.
    """
    arr = green_frame(subject=(20.0, 140.0, 50.0))
    monkeypatch.setattr(cutout, "_mask",
                        lambda im: np.where(subject_mask(64), 255, 0).astype(np.uint8))

    alpha, _ = cutout._cut(Image.fromarray(arr.astype(np.uint8), "RGB"))

    assert alpha[subject_mask(64)].min() == 255
    assert alpha[0, 0] == 0


def test_cut_does_not_despill_the_subject(monkeypatch):
    """Зелёная одежда не должна посереть: налёт давим только по кромке."""
    arr = green_frame(subject=(20.0, 140.0, 50.0))
    monkeypatch.setattr(cutout, "_mask",
                        lambda im: np.where(subject_mask(64), 255, 0).astype(np.uint8))

    _, clean = cutout._cut(Image.fromarray(arr.astype(np.uint8), "RGB"))

    assert np.asarray(clean)[32, 32, 1] == 140


def test_cut_falls_back_to_the_mask_without_green(monkeypatch):
    """Нет зелёного экрана — работаем как раньше, через маску."""
    called = []
    monkeypatch.setattr(cutout, "_mask",
                        lambda im: called.append(1) or np.zeros((64, 64), np.uint8))

    img = Image.fromarray(skin_patch().astype(np.uint8), "RGB")
    alpha, same = cutout._cut(img)

    assert same is img
    assert len(called) == 1


def test_cut_despills_the_edge_of_the_subject(monkeypatch):
    """Налёт давится и там, где сеть уверена, но близко к краю.

    Зелёный подсвечивает волосы на пару пикселей вглубь контура, и без этой
    полосы по краю причёски оставалась зелёная кайма.
    """
    arr = green_frame(subject=(40.0, 40.0, 40.0))
    arr[8:11, 8:56] = (40.0, 120.0, 40.0)        # зелёный налёт по верхней кромке
    monkeypatch.setattr(cutout, "_mask",
                        lambda im: np.where(subject_mask(64), 255, 0).astype(np.uint8))

    _, clean = cutout._cut(Image.fromarray(arr.astype(np.uint8), "RGB"))
    out = np.asarray(clean)

    assert out[9, 32, 1] <= 48                   # кайму сняли
    assert out[32, 32, 1] == 40                  # середину не тронули


def test_cut_removes_the_screen_even_inside_the_subject(monkeypatch):
    """Дыра цвета экрана внутри объекта обязана стать прозрачной.

    Это тот самый дефект, который вылез на живом наборе 02.10.2026: сеть
    уверенно заливает человеком просветы между прядями, страховка их берегла, и
    в стикере между волосами оставался ярко-зелёный фон. Пиксель, который в
    точности цвет экрана, человеком быть не может.
    """
    arr = green_frame(subject=(20.0, 140.0, 50.0))
    arr[20:28, 20:28] = (30.0, 200.0, 60.0)      # просвет внутри объекта
    monkeypatch.setattr(cutout, "_mask",
                        lambda im: np.where(subject_mask(64), 255, 0).astype(np.uint8))

    alpha, _ = cutout._cut(Image.fromarray(arr.astype(np.uint8), "RGB"))

    assert alpha[24, 24] == 0                    # просвет убрали
    assert alpha[40, 40] == 255                  # одежду не тронули


def test_cut_in_net_mode_takes_alpha_from_the_net(monkeypatch):
    """В режиме сетки цвет про прозрачность не решает ничего.

    Тот же кадр, что в тесте выше: внутри объекта просвет цвета экрана, и сеть
    уверенно заливает его человеком. Хромакей просвет убирает, сетка — нет, и
    это ровно та разница, которую стенд даёт посмотреть глазами.
    """
    arr = green_frame(subject=(20.0, 140.0, 50.0))
    arr[20:28, 20:28] = (30.0, 200.0, 60.0)
    monkeypatch.setattr(cutout, "_mask",
                        lambda im: np.where(subject_mask(64), 255, 0).astype(np.uint8))

    img = Image.fromarray(arr.astype(np.uint8), "RGB")
    alpha, _ = cutout._cut(img, cutout.CUT_NET)

    assert alpha[24, 24] == 255                  # сеть сказала «человек»
    assert alpha[0, 0] == 0                      # фон всё равно прозрачный


def test_cut_in_net_mode_still_despills_the_edge(monkeypatch):
    """Налёт по кромке глушим и в режиме сетки.

    Иначе способ был бы оболган: кадр-то приходит с зелёным экраном, пока его
    просит промт, и зелёная кайма на волосах — вина не сетки.
    """
    arr = green_frame(subject=(40.0, 40.0, 40.0))
    arr[8:11, 8:56] = (40.0, 120.0, 40.0)
    monkeypatch.setattr(cutout, "_mask",
                        lambda im: np.where(subject_mask(64), 255, 0).astype(np.uint8))

    _, clean = cutout._cut(Image.fromarray(arr.astype(np.uint8), "RGB"),
                           cutout.CUT_NET)
    out = np.asarray(clean)

    assert out[9, 32, 1] <= 48                   # кайму сняли
    assert out[32, 32, 1] == 40                  # середину не тронули


def test_cut_in_net_mode_leaves_a_clean_frame_alone(monkeypatch):
    """Нет зелёного — картинка возвращается той же, без лишней работы."""
    monkeypatch.setattr(cutout, "_mask",
                        lambda im: np.where(subject_mask(64), 255, 0).astype(np.uint8))

    img = Image.fromarray(skin_patch().astype(np.uint8), "RGB")
    _, same = cutout._cut(img, cutout.CUT_NET)

    assert same is img


def test_cut_works_on_a_screen_of_another_colour(monkeypatch):
    """Синий экран режется ровно так же, как зелёный: цвет нигде не зашит."""
    arr = green_frame(screen=BLUE)
    monkeypatch.setattr(cutout, "_mask",
                        lambda im: np.where(subject_mask(64), 255, 0).astype(np.uint8))

    alpha, _ = cutout._cut(Image.fromarray(arr.astype(np.uint8), "RGB"))

    assert alpha[0, 0] == 0                      # экран убрали
    assert alpha[32, 32] == 255                  # кожу оставили


def test_cut_despills_the_colour_of_the_screen(monkeypatch):
    """Налёт давится по главному каналу экрана, а не по зелёному всегда."""
    arr = green_frame(subject=(40.0, 40.0, 40.0), screen=BLUE)
    arr[8:11, 8:56] = (40.0, 40.0, 120.0)        # синий налёт по верхней кромке
    monkeypatch.setattr(cutout, "_mask",
                        lambda im: np.where(subject_mask(64), 255, 0).astype(np.uint8))

    _, clean = cutout._cut(Image.fromarray(arr.astype(np.uint8), "RGB"))
    out = np.asarray(clean)

    assert out[9, 32, 2] <= 48                   # кайму сняли
    assert out[32, 32, 2] == 40                  # середину не тронули
