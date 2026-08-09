# Before starting install this

### Ubuntu

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

sudo apt install nvidia-cuda-toolkit ffmpeg libcudnn8 libcudnn8-dev
```

```
pip install -r requirements.txt
```

# Video models

`/api/t2v` and `/api/i2v` accept a `model` parameter:

- `minimax_h3` (default) — MiniMax H3 + turbo LoRA, 8 steps, native stereo audio,
  24 fps, length quantized to 17k+5 frames (124 = 5 s, 362 = 15 s max).
- `wan` — Wan2.2 fp8 + lightx2v, 81 frames, 24 fps, no audio.

`fps` is optional and defaults to 24 for both models. Note it only sets playback
rate — frame count is fixed per model, so 81 Wan frames now play over 3.4 s
instead of the 2.7 s they did at 30 fps.

Requested `width`/`height` are snapped to H3's grid (multiples of 32, capped at
the 1344x768 native canvas, aspect preserved) — a 1080x1920 phone shot becomes
768x1344. Wan takes the size as given.

Snapping shifts the aspect ratio by 1-3%, and `MiniMaxH3ImageToVideo` stretches
the first frame into the canvas without cropping. So for H3 the i2v image is
center-cropped to the target aspect and resized to the exact canvas before
upload, trading 1-3% of the edges for no distortion. Below roughly 200 px the
32-grid gets coarse relative to the image and the crop grows (a 113x78 thumbnail
loses 28%) — such inputs are too small to generate from anyway. Wan images are
uploaded untouched.

Adding a model means one workflow template plus one substitution map in
`comfy_client.MODELS` — no other code changes.

Weights: `bash comfyui/download_models_h3.sh` (~43 GB). ComfyUI 0.30.0+ required,
launched with `--disable-pinned-memory` (see the `command:` override in
docker-compose) — without it ComfyUI pins most of system RAM and the OOM killer
takes the container down on long clips.

Templates in `comfy_workflows/` are named `{model}_{kind}[_turbo]_{format}.json`,
read left to right:

- `{model}` matches the `model` API parameter — `wan` or `minimax_h3`.
- `_turbo` means the distilled 8-step variant. Without it, the base 20-step
  graph, kept for quality comparison and never used by the API.
- `_api` is the flat graph posted to `/prompt` (what the code loads); `_ui` is
  the node graph you open in the browser.

So `minimax_h3_t2v_turbo_api.json` is what `/api/t2v` runs by default, and
`minimax_h3_t2v_turbo_ui.json` is the same thing to open in ComfyUI.

The `_api` graphs drop the calculator nodes (ResolutionSelector,
ComfyMathExpression, PrimitiveFloat) since width/height/length are substituted
directly. Substitution targets nodes by `class_type`, not by id, so re-exporting
a template from ComfyUI with different ids is fine.

# Tests

The queue scheduler and the GPU-process wrapper are isolated in `scheduler.py`
and `gpu_runner.py` — neither imports torch/diffusers, so their tests run
anywhere (no GPU required):

```
pip install pytest
pytest
```

What is covered:

- `test_scheduler.py` — queue policy: FIFO baseline, batching video jobs by
  subtype (to avoid reloading the ~50 GB model), the "no more than 3 videos in a
  row while an image/transcription is waiting" rule, keeping the video slot warm
  across cheap jobs, and the max-wait anti-starvation guard.
- `test_gpu_runner.py` — GPU-process mechanics with fake workers: normal
  submit/return, and respawn after a hard process death (OOM/segfault) so a dead
  worker returns a clear error instead of hanging the queue.

Not covered (needs the actual GPU box): model loading/offload and inference in
`main.py`. Verify those with a real run on the server.

