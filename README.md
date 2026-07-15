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

