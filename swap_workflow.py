# -*- coding: utf-8 -*-
"""Граф ComfyUI для замены человека в видео (Wan 2.2 Animate, режим replacement).

Почему отдельным модулем, а не шаблоном рядом с видео: у видео шаблон грузится
файлом и в него подставляются значения, а здесь узлов немного и половина из них
существует только ради связей. Собрать словарь короче и честнее, чем держать
json, в котором всё равно меняется почти каждое поле.

Длина держится СКОЛЬЗЯЩИМ ОКНОМ (WanContextWindowsManual), а не увеличением
length: окно внимания остаётся прежним, окна ездят по всей длине внутри одного
прохода семплера. Поэтому шва не возникает — второго независимого
семплирования, которому было бы с чем разъехаться, просто нет.
"""

# Негатив из шаблона Wan, китайский. Не переводить и не сокращать: модель
# обучена именно на нём, на переводе качество заметно ниже.
NEGATIVE = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")

UNET = "wan2.2_animate_14B_int8_convrot.safetensors"
# relight подгоняет освещение персонажа под фон — она существует ровно ради
# режима замены. lightx2v дистилляционная: 6 шагов при cfg 1 вместо двадцати
# с полноценным guidance.
LORA_RELIGHT = "wan2.2_animate_14B_relight_lora_bf16.safetensors"
LORA_SPEED = "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
CLIP = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE = "Wan2_1_VAE_bf16.safetensors"
CLIP_VISION = "clip_vision_h.safetensors"

STEPS = 6
CONTEXT_OVERLAP = 30


def context_for(width, height):
    """Кадров в окне внимания под размер кадра.

    Разрешение растит ПРОСТРАНСТВЕННЫЕ токены, окно — ВРЕМЕННЫЕ, и в 24 ГБ они
    делят одну память. Замеры на 3090:

        464x832   окно 81 -> 20.9 ГБ, 279 с
        576x1024  окно 81 -> не пробовали, окно 65 -> 22.0 ГБ, 575 с
        576x1024  окно 49 -> 21.2 ГБ, 506 с

    Отсюда правило: до ~400 тысяч пикселей берём 81, дальше 49. Константой это
    держать нельзя — на 464x832 фиксированные 49 дают шесть окон вместо двух и
    задачу вдвое дороже.
    """
    return 81 if width * height <= 400000 else 49


def build(*, ref_image, pose_video, bg_video, mask_video, face_video,
          character_text, width, height, length, fps, seed=7,
          steps=STEPS, context_length=None,
          context_overlap=CONTEXT_OVERLAP, prefix="swap"):
    """Готовый граф. length обязан лежать на сетке 4n+1.

    context_length подобран замерами на 576x1024: 49 кадров дают 506 с и
    21.2 ГБ, 65 — 575 с и 22.0, 81 — 548 с и 22.4. Больше окно — дороже каждый
    кадр внутри него (внимание растёт быстрее линейного), меньше — перехлёст
    съедает выигрыш. На качество окно не влияет.
    """
    if context_length is None:
        context_length = context_for(width, height)
    g = {}

    def n(name, cls, **inputs):
        g[name] = {"class_type": cls, "inputs": inputs}

    n("unet", "UNETLoader", unet_name=UNET, weight_dtype="default")
    n("relight", "LoraLoaderModelOnly", model=["unet", 0],
      lora_name=LORA_RELIGHT, strength_model=1.0)
    n("fast", "LoraLoaderModelOnly", model=["relight", 0],
      lora_name=LORA_SPEED, strength_model=1.0)
    n("shift", "ModelSamplingSD3", model=["fast", 0], shift=5.0)
    # retain_first_frame=True — ЯКОРЬ на референс. Без него референс попадает
    # только в первое окно, и со второго персонаж переодевается: в замере к
    # 81-му кадру костюм менялся целиком вместе с причёской.
    n("ctx", "WanContextWindowsManual", model=["shift", 0],
      context_length=context_length, context_overlap=context_overlap,
      context_schedule="standard_static", context_stride=1, closed_loop=False,
      fuse_method="pyramid", freenoise=True, retain_first_frame=True,
      split_conds_to_windows=False)

    n("clip", "CLIPLoader", clip_name=CLIP, type="wan", device="default")
    n("pos", "CLIPTextEncode", clip=["clip", 0], text=character_text)
    n("neg", "CLIPTextEncode", clip=["clip", 0], text=NEGATIVE)
    n("vae", "VAELoader", vae_name=VAE)
    n("cvl", "CLIPVisionLoader", clip_name=CLIP_VISION)

    n("refimg", "LoadImage", image=ref_image)
    n("ref", "ImageScale", image=["refimg", 0], upscale_method="area",
      width=width, height=height, crop="center")
    n("cvref", "CLIPVisionEncode", clip_vision=["cvl", 0], image=["ref", 0],
      crop="none")

    for tag, fname in (("pv", pose_video), ("bgv", bg_video),
                       ("mv", mask_video), ("fv", face_video)):
        n(tag, "LoadVideo", file=fname)
        n(tag + "c", "GetVideoComponents", video=[tag, 0])
    n("mask", "ImageToMask", image=["mvc", 0], channel="red")

    n("w2v", "WanAnimateToVideo",
      positive=["pos", 0], negative=["neg", 0], vae=["vae", 0],
      width=width, height=height, length=length, batch_size=1,
      continue_motion_max_frames=5, video_frame_offset=0,
      clip_vision_output=["cvref", 0], reference_image=["ref", 0],
      face_video=["fvc", 0], pose_video=["pvc", 0],
      background_video=["bgvc", 0], character_mask=["mask", 0])

    n("sampler", "KSamplerSelect", sampler_name="lcm")
    n("sigmas", "BasicScheduler", model=["ctx", 0], scheduler="simple",
      steps=steps, denoise=1.0)
    # cfg=1: дистилляционная lightx2v делает своё за 6 шагов, guidance не нужен.
    n("sample", "SamplerCustom", model=["ctx", 0], add_noise=True,
      noise_seed=seed, cfg=1.0, positive=["w2v", 0], negative=["w2v", 1],
      sampler=["sampler", 0], sigmas=["sigmas", 0], latent_image=["w2v", 2])

    n("trim", "TrimVideoLatent", samples=["sample", 0], trim_amount=["w2v", 3])
    n("dec", "VAEDecode", samples=["trim", 0], vae=["vae", 0])
    # Ведущие кадры, перегенерированные ради стыка, выбрасываем: их число знает
    # только нода (выход 4), поэтому берём связью, а не константой.
    n("cut", "ImageFromBatch", image=["dec", 0], batch_index=["w2v", 4],
      length=4096)
    n("mk", "CreateVideo", images=["cut", 0], fps=fps)
    n("save", "SaveVideo", video=["mk", 0], filename_prefix=prefix,
      format="auto", codec="auto")
    return g


def check_links(graph):
    """Все ли ссылки указывают на существующие узлы.

    ComfyUI ловит такую опечатку только на валидации, уже приняв задачу, и
    сообщает KeyError из середины чужого стека. Один неверный идентификатор
    однажды стоил полного прогона подготовки, поэтому проверяем заранее.
    """
    bad = []
    for node, body in graph.items():
        for name, value in body["inputs"].items():
            if (isinstance(value, list) and len(value) == 2
                    and isinstance(value[0], str) and value[0] not in graph):
                bad.append(f"{node}.{name} -> нет узла '{value[0]}'")
    return bad
