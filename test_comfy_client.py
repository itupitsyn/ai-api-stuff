"""Тесты чистой логики comfy_client (подстановка в воркфлоу, парсинг вывода).

Сеть не трогаем — только сборка графа из шаблонов и разбор history.
Запуск:  pytest test_comfy_client.py -v
"""
import io

import pytest

from comfy_client import (
    build_i2v_workflow,
    build_t2v_workflow,
    build_video_workflow,
    center_crop_box,
    find_output_file,
    h3_fit_image,
    h3_num_frames,
    h3_snap_size,
    load_template,
    prepare_image,
    resolve_class_map,
    template_name,
)


# H3-граф экспортируется из сабграфа, поэтому его node id непредсказуемы. Здесь
# нарочно взяты «чужие» номера: тесты должны проходить при любой нумерации.
def h3_template():
    return {
        "41": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["7", 0], "vae": ["8", 0],
            "first_frame": ["19", 0], "last_frame": None,
            "prompt": "шаблонный промпт", "width": 1344, "height": 768, "length": 124}},
        "62": {"class_type": "CreateVideo", "inputs": {
            "images": ["55", 0], "audio": ["56", 0], "fps": 24}},
        "13": {"class_type": "RandomNoise", "inputs": {"noise_seed": 1}},
        "19": {"class_type": "LoadImage", "inputs": {"image": "example.png"}},
        "7": {"class_type": "CLIPLoader", "inputs": {}},
        "8": {"class_type": "VAELoader", "inputs": {}},
        "9": {"class_type": "VAELoader", "inputs": {}},   # второй VAE (аудио)
    }


def test_t2v_substitution():
    tpl = load_template("wan_t2v_api")
    wf = build_t2v_workflow(tpl, prompt="a cat", width=832, height=480, fps=30, seed=123)
    assert wf["89"]["inputs"]["text"] == "a cat"
    assert wf["74"]["inputs"]["width"] == 832
    assert wf["74"]["inputs"]["height"] == 480
    assert wf["74"]["inputs"]["length"] == 81
    assert wf["88"]["inputs"]["fps"] == 30
    assert wf["81"]["inputs"]["noise_seed"] == 123


def test_i2v_substitution():
    tpl = load_template("wan_i2v_api")
    wf = build_i2v_workflow(tpl, prompt="a dog", image_name="sub/pic.png",
                            width=800, height=496, fps=24, seed=7)
    assert wf["93"]["inputs"]["text"] == "a dog"
    assert wf["97"]["inputs"]["image"] == "sub/pic.png"
    assert wf["98"]["inputs"]["width"] == 800
    assert wf["98"]["inputs"]["height"] == 496
    assert wf["94"]["inputs"]["fps"] == 24
    assert wf["86"]["inputs"]["noise_seed"] == 7


def test_build_does_not_mutate_template():
    tpl = load_template("wan_t2v_api")
    orig = tpl["89"]["inputs"]["text"]
    build_t2v_workflow(tpl, prompt="changed", width=640, height=640, fps=24)
    assert tpl["89"]["inputs"]["text"] == orig  # шаблон не тронут (deepcopy)


def test_random_seed_when_none():
    tpl = load_template("wan_t2v_api")
    a = build_t2v_workflow(tpl, prompt="x", width=640, height=640, fps=24)
    b = build_t2v_workflow(tpl, prompt="x", width=640, height=640, fps=24)
    # два вызова без seed дают разные seed (иначе одинаковые видео)
    assert a["81"]["inputs"]["noise_seed"] != b["81"]["inputs"]["noise_seed"]


def test_find_output_file_videos():
    entry = {"outputs": {"108": {"videos": [
        {"filename": "ComfyUI_00001.mp4", "subfolder": "video", "type": "output"}]}}}
    assert find_output_file(entry) == ("ComfyUI_00001.mp4", "video", "output")


def test_find_output_file_gifs_fallback():
    entry = {"outputs": {"9": {"gifs": [{"filename": "a.webp"}]}}}
    assert find_output_file(entry) == ("a.webp", "", "output")


def test_find_output_file_none():
    assert find_output_file({"outputs": {"3": {"text": ["nope"]}}}) is None
    assert find_output_file({}) is None


# --- MiniMax H3 -----------------------------------------------------------

def test_h3_frame_quantization():
    # модель обучена на длинах 17k+5, ближайшая сверху
    assert h3_num_frames(5) == 124
    assert h3_num_frames(15) == 362
    assert h3_num_frames(2) == 56
    assert all(h3_num_frames(s) % 17 == 5 for s in (1, 3, 5, 7, 9, 11, 13, 15))
    assert h3_num_frames(0) == 5           # не уходим в ноль/отрицательные
    assert h3_num_frames(60) == 362        # потолок модели


def test_h3_snap_size_keeps_valid_sizes():
    assert h3_snap_size(832, 480) == (832, 480)     # уже на сетке 32
    assert h3_snap_size(1344, 768) == (1344, 768)   # нативный холст


def test_h3_snap_size_rounds_to_grid():
    assert h3_snap_size(1080, 720) == (1088, 736)
    for w, h in ((1000, 700), (513, 999), (100, 100), (1, 1)):
        sw, sh = h3_snap_size(w, h)
        assert sw % 32 == 0 and sh % 32 == 0, (w, h, sw, sh)
        assert sw >= 32 and sh >= 32


def test_h3_snap_size_downscales_and_keeps_aspect():
    # телефонный вертикальный кадр: обязан ужаться под ~1 MP, пропорции сохранить
    w, h = h3_snap_size(1080, 1920)
    assert w * h <= 1344 * 768 * 1.05
    assert abs((w / h) - (1080 / 1920)) < 0.05
    assert w % 32 == 0 and h % 32 == 0


def test_h3_snapped_size_lands_in_workflow():
    wf = build_video_workflow("minimax_h3", "i2v", h3_template(), prompt="p",
                              image_name="a.png", width=1080, height=1920)
    node = wf["41"]["inputs"]
    assert node["width"] % 32 == 0 and node["height"] % 32 == 0
    assert node["width"] * node["height"] <= 1344 * 768 * 1.05


def test_wan_size_passed_through_untouched():
    wf = build_video_workflow("wan", "i2v", load_template("wan_i2v_api"), prompt="p",
                              image_name="a.png", width=1080, height=1920)
    assert wf["98"]["inputs"]["width"] == 1080
    assert wf["98"]["inputs"]["height"] == 1920


def test_h3_snap_is_idempotent():
    # prepare_image снапает отдельно от build_video_workflow — размеры обязаны совпасть
    for w, h in ((1080, 1920), (3024, 4032), (1080, 720), (832, 480), (100, 100)):
        once = h3_snap_size(w, h)
        assert h3_snap_size(*once) == once


def test_center_crop_box_trims_the_wider_axis():
    # исходник шире цели -> режем по бокам, высота целиком
    assert center_crop_box(1000, 500, 1, 1) == (250, 0, 750, 500)
    # исходник уже цели -> режем сверху/снизу, ширина целиком
    assert center_crop_box(500, 1000, 1, 1) == (0, 250, 500, 750)


def test_center_crop_box_keeps_everything_when_aspect_matches():
    assert center_crop_box(1920, 1080, 1344, 768) != (0, 0, 1920, 1080)  # 16:9 vs 1.75
    assert center_crop_box(640, 480, 640, 480) == (0, 0, 640, 480)
    assert center_crop_box(1280, 960, 640, 480) == (0, 0, 1280, 960)     # то же соотношение


def test_center_crop_box_never_exceeds_source():
    for src in ((113, 78), (4032, 3024), (33, 4000), (1, 1)):
        for tgt in ((1344, 768), (768, 1344), (32, 32)):
            left, top, right, bottom = center_crop_box(*src, *tgt)
            assert 0 <= left < right <= src[0], (src, tgt)
            assert 0 <= top < bottom <= src[1], (src, tgt)


def test_center_crop_box_rejects_degenerate_size():
    with pytest.raises(ValueError):
        center_crop_box(0, 100, 64, 64)


def test_h3_fit_image_returns_exact_canvas():
    Image = pytest.importorskip("PIL.Image")
    src = io.BytesIO()
    Image.new("RGB", (3024, 4032), "red").save(src, format="PNG")

    w, h = h3_snap_size(3024, 4032)
    fitted = h3_fit_image(src.getvalue(), w, h)
    with Image.open(io.BytesIO(fitted)) as out:
        assert out.size == (w, h)   # ровно холст -> «растягивание» в ноде тождественно


def test_prepare_image_leaves_wan_untouched():
    raw = b"not-even-an-image"
    assert prepare_image("wan", raw, 1080, 1920) is raw


def test_h3_resolve_by_class_ignores_ids():
    mapping = resolve_class_map(h3_template(), {"prompt": ("MiniMaxH3ImageToVideo", "prompt"),
                                                "fps": ("CreateVideo", "fps")})
    assert mapping == {"prompt": ("41", "prompt"), "fps": ("62", "fps")}


def test_h3_resolve_raises_on_missing_node():
    with pytest.raises(KeyError, match="MiniMaxH3ReferenceToVideo"):
        resolve_class_map(h3_template(), {"prompt": ("MiniMaxH3ReferenceToVideo", "prompt")})


def test_h3_resolve_raises_on_ambiguous_node():
    # два VAELoader в графе — молча выбрать первый нельзя
    with pytest.raises(KeyError, match="неоднозначно"):
        resolve_class_map(h3_template(), {"x": ("VAELoader", "vae_name")})


def test_h3_t2v_substitution():
    wf = build_video_workflow("minimax_h3", "t2v", h3_template(), prompt="кот в шляпе",
                              width=864, height=480, seed=42)
    node = wf["41"]["inputs"]
    assert node["prompt"] == "кот в шляпе"
    assert (node["width"], node["height"]) == (864, 480)
    assert node["length"] == 124                  # дефолт модели: 5 сек
    assert wf["62"]["inputs"]["fps"] == 24        # нативные 24 fps
    assert wf["13"]["inputs"]["noise_seed"] == 42


def test_h3_i2v_puts_image_into_loadimage():
    wf = build_video_workflow("minimax_h3", "i2v", h3_template(), prompt="p",
                              image_name="sub/pic.png", width=832, height=480)
    assert wf["19"]["inputs"]["image"] == "sub/pic.png"
    assert wf["41"]["inputs"]["prompt"] == "p"


def test_h3_explicit_fps_and_frames_win_over_defaults():
    wf = build_video_workflow("minimax_h3", "t2v", h3_template(), prompt="p",
                              width=832, height=480, fps=30, num_frames=362)
    assert wf["62"]["inputs"]["fps"] == 30
    assert wf["41"]["inputs"]["length"] == 362


def test_h3_build_does_not_mutate_template():
    tpl = h3_template()
    build_video_workflow("minimax_h3", "t2v", tpl, prompt="изменено", width=640, height=640)
    assert tpl["41"]["inputs"]["prompt"] == "шаблонный промпт"


def test_wan_goes_through_the_same_entry_point():
    wf = build_video_workflow("wan", "t2v", load_template("wan_t2v_api"), prompt="a cat",
                              width=832, height=480, seed=5)
    assert wf["89"]["inputs"]["text"] == "a cat"
    assert wf["88"]["inputs"]["fps"] == 24
    assert wf["74"]["inputs"]["length"] == 81
    assert wf["81"]["inputs"]["noise_seed"] == 5


def test_unknown_model_rejected():
    with pytest.raises(ValueError, match="неизвестная модель"):
        build_video_workflow("sora", "t2v", h3_template(), prompt="p", width=1, height=1)


def test_template_names():
    assert template_name("wan", "t2v") == "wan_t2v_api"
    assert template_name("minimax_h3", "i2v") == "minimax_h3_i2v_turbo_api"
