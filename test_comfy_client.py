"""Тесты чистой логики comfy_client (подстановка в воркфлоу, парсинг вывода).

Сеть не трогаем — только сборка графа из шаблонов и разбор history.
Запуск:  pytest test_comfy_client.py -v
"""
from comfy_client import (
    build_i2v_workflow,
    build_t2v_workflow,
    find_output_file,
    load_template,
)


def test_t2v_substitution():
    tpl = load_template("t2v")
    wf = build_t2v_workflow(tpl, prompt="a cat", width=832, height=480, fps=30, seed=123)
    assert wf["89"]["inputs"]["text"] == "a cat"
    assert wf["74"]["inputs"]["width"] == 832
    assert wf["74"]["inputs"]["height"] == 480
    assert wf["74"]["inputs"]["length"] == 81
    assert wf["88"]["inputs"]["fps"] == 30
    assert wf["81"]["inputs"]["noise_seed"] == 123


def test_i2v_substitution():
    tpl = load_template("i2v")
    wf = build_i2v_workflow(tpl, prompt="a dog", image_name="sub/pic.png",
                            width=800, height=496, fps=24, seed=7)
    assert wf["93"]["inputs"]["text"] == "a dog"
    assert wf["97"]["inputs"]["image"] == "sub/pic.png"
    assert wf["98"]["inputs"]["width"] == 800
    assert wf["98"]["inputs"]["height"] == 496
    assert wf["94"]["inputs"]["fps"] == 24
    assert wf["86"]["inputs"]["noise_seed"] == 7


def test_build_does_not_mutate_template():
    tpl = load_template("t2v")
    orig = tpl["89"]["inputs"]["text"]
    build_t2v_workflow(tpl, prompt="changed", width=640, height=640, fps=24)
    assert tpl["89"]["inputs"]["text"] == orig  # шаблон не тронут (deepcopy)


def test_random_seed_when_none():
    tpl = load_template("t2v")
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
